"""Bounded inference/resource audit of the 16 existing HD-EMG decoder versions.

Uses participant 01, seed 260911 and the first four validation windows only.
No fitting, accuracy scoring, test access, checkpoint writes or compilation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import torch
from paper.common.paths import REPO, ROOT


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "bytes": value.numel() * value.element_size(),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_inputs(runs):
    records = {}
    with h5py.File(runs / "hdemg_within_v1/subject_01/features.h5", "r") as handle:
        assert bool(handle.attrs["complete"])
        group = handle["val"]
        values = {
            name: torch.from_numpy(np.array(group[name][:4]))
            for name in ["tokens", "continuous", "raw"]
        }
        values["book"] = torch.from_numpy(handle["codebook"][:]).float()
        values["descriptors"] = torch.from_numpy(handle["descriptors"][:]).float()
        identities = {name: np.asarray(group[name][:4]) for name in ["subjects", "tasks", "starts"]}
        assert set(identities["subjects"]) == {1}
        records["reference"] = {
            "path": str(handle.filename), "group": "val", "indices": list(range(4)),
            "identities": {k: v.tolist() for k, v in identities.items()},
            "arrays": {k: tensor_record(v) for k, v in values.items()},
        }
    with h5py.File(runs / "hdemg_v20_continuous_v1/subject_01/features.h5", "r") as handle:
        assert bool(handle.attrs["complete"])
        values["continuous_v20"] = torch.from_numpy(handle["val/continuous"][:4])
        for name, expected in identities.items():
            np.testing.assert_array_equal(handle[f"val/{name}"][:4], expected)
        records["continuous_v20"] = tensor_record(values["continuous_v20"])
    with h5py.File(runs / "hdemg_raulnet_v1/subject_01/native.h5", "r") as handle:
        assert bool(handle.attrs["complete"])
        values["native_raw"] = torch.from_numpy(handle["val/raw"][:4, -1])
        for name, expected in identities.items():
            np.testing.assert_array_equal(handle[f"val/{name}"][:4], expected)
        records["native_raw"] = tensor_record(values["native_raw"])
    assert values["tokens"].shape == (4, 320, 100)
    assert values["continuous"].dtype == torch.float16
    assert values["continuous_v20"].dtype == torch.float32
    assert values["raw"].shape == (4, 320, 6000)
    assert values["native_raw"].shape == (4, 320, 320)
    return values, records


def representation_cost(representation, input_dtype, source_hz=2000):
    # Array/payload arithmetic, not measured networking or HDF5 file sizes.
    if representation == "tokens":
        items_per_second, payload_bits, storage_bits = 320 * 50, 10, 16
        encoding = "1024-code IDs: packed 10-bit payload; uint16 native arrays"
    elif representation == "continuous":
        items_per_second = 320 * 50 * 256
        payload_bits = storage_bits = 16 if input_dtype == torch.float16 else 32
        encoding = f"256 features/channel/frame; float{payload_bits} cache precision"
    else:
        items_per_second, payload_bits, storage_bits = 320 * source_hz, 32, 32
        encoding = f"320-channel float32 raw stream at {source_hz} Hz; derived low-pass plane excluded"
    return {
        "channels": 320,
        "items_per_second": items_per_second,
        "payload_bits_per_item": payload_bits,
        "native_array_bits_per_item": storage_bits,
        "payload_kbit_s": items_per_second * payload_bits / 1000,
        "packed_payload_mb_min": items_per_second * payload_bits / 8 * 60 / 1e6,
        "native_array_mb_min": items_per_second * storage_bits / 8 * 60 / 1e6,
        "encoding": encoding,
    }


def storage_record(model, extras):
    # Count unique tensor storages, including persistent state and receiver assets.
    seen = set()
    components = {}
    for name, value in list(model.state_dict().items()) + list(extras.items()):
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            components[name] = storage.nbytes()
    return {
        "model_tensor_bytes": sum(components.values()),
        "model_tensor_mb": sum(components.values()) / 1e6,
        "component_tensor_bytes": components,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "excludes": "shared upstream EMG tokenizer, activations, allocator, optimizer, file/container metadata",
    }


def measure(call, samples=48):
    with torch.inference_mode():
        for index in range(12):
            result = call(index % 4)
            assert result.shape == (1, 60) and torch.isfinite(result).all()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for index in range(samples):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = call(index % 4)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        assert result.shape == (1, 60) and torch.isfinite(result).all()
    median = float(np.median(times))
    return {
        "warmup_calls": 12, "measured_calls": samples,
        "p50_ms": median, "p95_ms": float(np.percentile(times, 95)),
        "updates_per_second": 1000 / median,
        "samples_ms": times,
        "peak_process_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = args.project / "results"
    from paper.receivers.hdemg_raulnet_model import CONFIG, RaulNetV19, configure_eager, dual_input
    from paper.receivers import hdemg_matched_core as matched
    from paper.hdemg import run_hdemg_pilot as small
    from paper.receivers import hdemg_llm_core as llm
    from myoji._vendor.raulnet_v20 import RaulNetV20

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(260911)
    assert torch.cuda.device_count() == 1, "Expose only the previously idle GPU"
    free, total = torch.cuda.mem_get_info()
    assert total - free < 1.5 * 2**30, "Selected GPU is already occupied"
    cpu, input_records = read_inputs(runs)
    data = {k: v.to("cuda") for k, v in cpu.items()}
    # Device-resident decoder inputs, including V19's derived second plane.
    data["native"] = dual_input(data["native_raw"])
    zeros = torch.zeros(1, 320, 1, device="cuda")
    ones = torch.ones_like(zeros)
    specs = [
        ("A", "hdemg_within_v1", "tokens", "small", "tokens"),
        ("A", "hdemg_within_v1", "continuous", "small", "continuous"),
        ("A", "hdemg_transformer_v1", "tokens", "small", "tokens"),
        ("A", "hdemg_within_v1", "raw", "small", "raw"),
        ("B", "hdemg_matched_v1", "tcn_tokens", "matched", "tokens"),
        ("B", "hdemg_matched_v1", "tcn_continuous", "matched", "continuous"),
        ("B", "hdemg_matched_v1", "transformer_tokens", "matched", "tokens"),
        ("B", "hdemg_matched_v1", "transformer_continuous", "matched", "continuous"),
        ("B", "hdemg_matched_v1", "v19", "native", "raw"),
        ("C", "hdemg_raulnet_v1", "raulnet_v19_q32", "native", "raw"),
        ("C", "hdemg_v20_v1", "tokens", "v20", "tokens"),
        ("C", "hdemg_v20_continuous_v1", "continuous", "v20", "continuous"),
    ] + [("D", "hdemg_llm_pilot_v1", arm, "llm", "continuous" if arm.startswith("continuous") else "tokens") for arm in llm.ARMS]
    sources = [Path(__file__), Path(small.__file__), Path(matched.__file__), Path(llm.__file__)]
    sources.extend(sorted((REPO / "src/myoji").rglob("*.py")))
    sources.extend(sorted((ROOT / "vendor").rglob("*.py")))
    report = {
        "complete": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Decoder-only full-context latest-pose updates; one fitted participant/seed and four validation windows; no accuracy or test evaluation.",
        "hardware": {"gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                     "host": platform.node(), "cpu_threads": 4},
        "precision": "All models and floating receiver state float32, no AMP/TF32; input caches retain original dtype until the timed forward conversion.",
        "timing": "One batch-1 call returns the latest 60-coordinate pose. Recompute native context; 100 frames for feature decoders, 6000 raw samples for initial raw model, 320 native samples/two planes for V19. No KV/streaming cache or torch.compile. Inputs already on GPU; synchronized wall time includes ID lookup and conversion but excludes shared EMG encoding, preprocessing, transfers, I/O and target denormalization. Inverse median latency is update throughput, not saturated batch throughput.",
        "space": "Decimal MB. Model tensor payload includes loaded decoder weights/buffers, frozen Qwen blocks, target normalization and receiver codebook/geometry where used, but no shared upstream EMG encoder. Native-array MB/min counts one nonoverlapping steady stream with the observed dtype; packed token payload uses 10 bits whereas current token arrays use uint16. No packet headers, setup, labels, overlapping-window or container overhead.",
        "dependencies": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "h5py", "transformers", "myoji"]},
        "source_sha256": {str(path): digest_file(path) for path in sources},
        "inputs": input_records, "test_evaluated": False, "training_performed": False,
        "results": [],
    }
    attempt = {key: report[key] for key in ["started_at", "hardware", "source_sha256"]}
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        for key in ["inputs", "precision", "timing", "space", "dependencies"]:
            assert previous[key] == report[key], f"Cannot resume a changed benchmark: {key}"
        for key in ["gpu", "visible_devices", "host", "cpu_threads"]:
            assert previous["hardware"][key] == report["hardware"][key]
        report["attempts"] = previous.get("attempts", [{key: previous[key] for key in ["started_at", "hardware", "source_sha256"]}])
        report["results"] = previous["results"]
        for row in report["results"]:
            row.setdefault("benchmark_attempt_index", 0)
        report["started_at"] = previous["started_at"]
    else:
        report["attempts"] = []
    attempt_index = len(report["attempts"])
    report["attempts"].append(attempt)
    completed = {(row["campaign"], row["arm"]) for row in report["results"]}
    expected = {(campaign, arm) for _, campaign, arm, _, _ in specs}
    assert completed <= expected and len(completed) == len(report["results"])
    snapshot = args.output.parent / f"benchmark_step_{os.environ.get('SLURM_STEP_ID', 'local')}.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(args.output, report)
    for panel, campaign, arm, family, representation in specs:
        if (campaign, arm) in completed:
            continue
        folder = runs / campaign
        if family == "llm":
            checkpoint = folder / arm / "best.pt"
        else:
            fit_arm = "raulnet" if campaign == "hdemg_raulnet_v1" else arm
            checkpoint = folder / "subject_01/fits" / f"{fit_arm}_260911/best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        extras = {}
        backbone_verified = None
        if family == "small":
            model = small.build(arm, state["protocol"], state["config"])
            model.load_state_dict(state["model"], strict=True)
        elif family == "matched":
            model = matched.build(arm)
            model.load_state_dict(state["model"], strict=True)
        elif family == "native":
            model = configure_eager(RaulNetV19(**CONFIG))
            model.load_state_dict(state["state_dict"] if campaign == "hdemg_raulnet_v1" else state["model"], strict=True)
        elif family == "v20":
            config = json.loads((folder / "protocol.json").read_text())["readout"]
            model = RaulNetV20(cpu["book"], **{k: v for k, v in config.items() if k != "kind"})
            model.load_state_dict(state["model"], strict=True)
        else:
            # Reconstruct the actual BF16 frozen weights used in the pilot,
            # including random-backbone rounding, before the common FP32 timing.
            model, loading = llm.build_model(folder / "model", arm, 260911, "cpu", torch.bfloat16)
            mismatch = model.load_state_dict(state["adapter"], strict=False)
            assert not mismatch.unexpected_keys
            assert all(k.startswith("backbone.") for k in mismatch.missing_keys)
            if model.backbone is not None:
                saved_result = json.loads((folder / arm / "result.json").read_text())
                assert llm.tensor_digest(model.backbone.state_dict().items()) == saved_result["frozen_backbone_sha256"]
                backbone_verified = True
        model = model.float().eval().to("cuda")
        for key in ["mean", "scale"]:
            if key in state:
                extras["target_" + key] = state[key].float().to("cuda")
        if family in {"small", "matched", "llm"}:
            extras["geometry"] = data["descriptors"]
        if representation == "tokens" and family != "v20":
            extras["emg_codebook"] = data["book"]
        input_key = "native" if family == "native" else "continuous_v20" if family == "v20" and representation == "continuous" else representation

        def forward(index):
            x = data[input_key][index:index + 1]
            if family == "native":
                return model(x)
            if family == "v20":
                return (model(x.long()) if representation == "tokens" else model.forward_embeddings(x.float()))[:, -1]
            if representation == "tokens":
                x = data["book"][x.long()]
            else:
                x = x.float()
            if family == "llm":
                return model(x, data["descriptors"])[:, -1]
            return model(x, data["descriptors"], zeros, ones)[:, -1]

        initial_versions = {name: value._version for name, value in model.state_dict().items()}
        row = {
            "panel": panel, "campaign": campaign, "arm": arm, "representation": representation,
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint),
            "model_source_sha256": digest_file(
                ROOT / "vendor/raulnet_v19.py"
                if family == "native" else inspect.getfile(type(model))
            ),
            "benchmark_attempt_index": attempt_index,
            "frozen_backbone_matches_fitted_pilot": backbone_verified,
            "subject": 1, "seed": 260911,
            "receiver_input_shape": [1] + list(data[input_key].shape[1:]),
            "receiver_input_dtype": str(data[input_key].dtype),
            **storage_record(model, extras),
            **representation_cost(representation, data[input_key].dtype, 2048 if family == "native" else 2000),
            **measure(forward),
        }
        assert initial_versions == {name: value._version for name, value in model.state_dict().items()}
        assert not model.training
        row["loaded_state_unchanged"] = True
        report["results"].append(row)
        write_json(args.output, report)
        print(json.dumps({k: row[k] for k in ["panel", "arm", "p50_ms", "updates_per_second", "model_tensor_mb", "payload_kbit_s", "native_array_mb_min"]}), flush=True)
        del forward, model, state, extras
        gc.collect()
        torch.cuda.empty_cache()
    assert len(report["results"]) == 16
    report["complete"] = True
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, report)
    print("Completed all 16 inference/resource measurements.", flush=True)


if __name__ == "__main__":
    main()
