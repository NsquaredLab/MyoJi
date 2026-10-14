"""Freeze real validation inputs and audit release determinism and perturbations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import paths
import torch

from myoji.data import (
    EMG2POSE_ARCHIVE_URL,
    EMG2PoseWindowDataset,
    RemoteTarHDF5Cache,
    recording_stratified_indices,
)
from myoji.learned.stream import Tokenizer, sha256
from myoji.preprocessing import calibration_normalize


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    remote = None
    if args.tar_index:
        remote = RemoteTarHDF5Cache(
            EMG2POSE_ARCHIVE_URL,
            args.tar_index,
            args.data_root,
            max_cache_bytes=20 * 1024**3,
        )
    dataset = EMG2PoseWindowDataset(
        args.data_root,
        args.metadata,
        split=args.split,
        window_samples=6000,
        stride_samples=6000,
        jitter=False,
        remote_cache=remote,
    )
    indices = recording_stratified_indices(
        dataset, args.windows, args.recordings, seed=260905, random_recordings=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    seen, provenance = set(), []
    with h5py.File(temporary, "w") as h5:
        raw = h5.create_dataset(
            "emg",
            shape=(0, 16, 6000),
            maxshape=(None, 16, 6000),
            dtype="f4",
            chunks=(1, 16, 6000),
        )
        for index in indices:
            item = dataset[index]
            identity = (item["filename"], int(item["start_sample"]))
            if identity in seen:
                continue
            seen.add(identity)
            n = raw.shape[0]
            raw.resize(n + 1, axis=0)
            raw[n] = item["emg"].numpy()
            provenance.append(
                {
                    "filename": identity[0],
                    "start_sample": identity[1],
                    "user": item["user"],
                }
            )
        manifest = {
            "split": args.split,
            "sample_rate_hz": 2000,
            "metadata_sha256": sha256(args.metadata),
            "windows": len(provenance),
            "recordings": len({p["filename"] for p in provenance}),
            "users": len({p["user"] for p in provenance}),
            "items": provenance,
        }
        h5.attrs["manifest"] = json.dumps(manifest)
    temporary.replace(args.output)
    write_json(args.output.with_suffix(".json"), manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "items"}), flush=True)


@torch.inference_mode()
def batch_ids(tokenizer, raw):
    x = torch.as_tensor(raw, dtype=torch.float32, device=tokenizer.device)
    normalized, _, _ = calibration_normalize(x, tokenizer.calibration_samples)
    descriptors = torch.zeros(
        x.shape[1], tokenizer.config.geometry_dim, device=tokenizer.device
    )
    encoded = tokenizer.model(
        normalized,
        sensor="wrist",
        channel_descriptors=descriptors,
        include_spatial=False,
    )
    return encoded.tokens[..., tokenizer.first_token_index :].cpu().numpy()


def mismatch(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"token shape mismatch: {reference.shape} vs {actual.shape}")
    changed = reference != actual
    return {
        "mismatched_tokens": int(changed.sum()),
        "total_tokens": int(changed.size),
        "requests_with_mismatch": int(changed.reshape(len(changed), -1).any(1).sum()),
        "requests": len(changed),
        "exact": not bool(changed.any()),
    }


def worker(args):
    t = Tokenizer(args.model, device=args.device)
    with h5py.File(args.inputs) as data:
        outputs = [
            batch_ids(t, data["emg"][i : i + args.batch_size])
            for i in range(0, len(data["emg"]), args.batch_size)
        ]
    np.save(args.output, np.concatenate(outputs))


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    t = Tokenizer(args.model, device=args.device)
    state_before = {k: v.cpu().clone() for k, v in t.model.state_dict().items()}
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if args.device == "cuda" else None
    with h5py.File(args.inputs) as data:
        raw = data["emg"][:]
        manifest = json.loads(data.attrs["manifest"])
    rate = manifest["sample_rate_hz"]
    references, repeated, streaming, latencies = [], [], [], []
    for i, chunk in enumerate(raw):
        started = time.perf_counter()
        result = t.encode(chunk, sample_rate_hz=rate)
        latencies.append(time.perf_counter() - started)
        references.append(np.asarray(result["tokens"], dtype=np.uint16))
        repeated.append(
            np.asarray(t.encode(chunk, sample_rate_hz=rate)["tokens"], dtype=np.uint16)
        )
        stream, pieces, ends, start = t.stream(sample_rate_hz=rate), [], [], 0
        for stop in [271, 1103, 3027, chunk.shape[-1]]:
            piece = stream.push(chunk[:, start:stop])
            pieces.append(np.asarray(piece["tokens"], dtype=np.uint16))
            ends.extend(piece["end_sample_indices"])
            start = stop
        if ends != result["end_sample_indices"]:
            raise ValueError("stream timestamp mismatch")
        streaming.append(np.concatenate(pieces, axis=1))
        if (i + 1) % 128 == 0:
            print(json.dumps({"completed": i + 1, "total": len(raw)}), flush=True)
    reference = np.asarray(references)
    batched = np.concatenate(
        [
            batch_ids(t, raw[i : i + args.batch_size])
            for i in range(0, len(raw), args.batch_size)
        ]
    )
    # A genuinely new process reloads the release file; saved IDs are then compared.
    worker_path = args.output.with_suffix(".worker.npy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model",
            str(args.model),
            "--inputs",
            str(args.inputs),
            "--output",
            str(worker_path),
            "--device",
            args.device,
            "--threads",
            str(args.threads),
            "--batch-size",
            "1",
        ],
        check=True,
    )
    reloaded = np.load(worker_path, allow_pickle=False)
    counts = np.bincount(reference.ravel(), minlength=t.config.codebook_size)
    p = counts[counts > 0] / counts.sum()
    result = {
        "model": t.metadata(),
        "input_sha256": sha256(args.inputs),
        "data": {k: v for k, v in manifest.items() if k != "items"},
        "implementation_sha256": {
            p.name: sha256(p)
            for p in [
                Path(__file__),
                paths.REPO / "src/myoji/learned/stream.py",
            ]
        },
        "repeat": mismatch(reference, np.asarray(repeated)),
        "stream": mismatch(reference, np.asarray(streaming)),
        "batch": mismatch(reference, batched),
        "fresh_process": mismatch(reference, reloaded),
        "state_unchanged": all(
            torch.equal(v.cpu(), state_before[k])
            for k, v in t.model.state_dict().items()
        ),
        "rng_unchanged": torch.equal(torch.get_rng_state(), cpu_rng)
        and (gpu_rng is None or torch.equal(torch.cuda.get_rng_state(), gpu_rng)),
        "active_codes": int((counts > 0).sum()),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "latency_ms": {
            "first_request": latencies[0] * 1000,
            "median_excluding_first": float(np.median(latencies[1:])) * 1000,
            "p95_excluding_first": float(np.quantile(latencies[1:], 0.95)) * 1000,
        },
        "storage": {
            "token_uint16_bytes": reference[0].nbytes,
            "token_ideal_packed_bytes": int(
                np.ceil(
                    reference[0].size * np.ceil(np.log2(t.config.codebook_size)) / 8
                )
            ),
            "calibration_stats_float32_bytes": int(2 * raw.shape[1] * 4),
            "real_preamble_float32_bytes": int(
                raw.shape[1] * t.calibration_samples * 4
            ),
            "raw_window_float32_bytes": int(raw[0].nbytes),
            "note": "Token bytes cover only post-warm-up outputs; no full-window compression ratio is asserted.",
        },
    }
    np.save(args.output.with_suffix(".tokens.npy"), reference)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "repeat",
                    "stream",
                    "batch",
                    "fresh_process",
                    "active_codes",
                    "latency_ms",
                ]
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def robustness(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    t = Tokenizer(args.model, device=args.device)
    with h5py.File(args.inputs) as h5:
        manifest = json.loads(h5.attrs["manifest"])
        # Spread the screen across the frozen input manifest, not just its first recording.
        indices = np.linspace(
            0, len(h5["emg"]) - 1, min(args.windows, len(h5["emg"])), dtype=int
        )
        raw = h5["emg"][indices]
    results = []
    for index, chunk in zip(indices, raw):
        x = torch.as_tensor(chunk[None], device=t.device)
        normalized, centre, scale = calibration_normalize(x, t.calibration_samples)
        descriptors = torch.zeros(x.shape[1], t.config.geometry_dim, device=t.device)

        def encode(values, descriptors=descriptors):
            out = t.model(
                values,
                sensor="wrist",
                channel_descriptors=descriptors,
                include_spatial=False,
            )
            return out.tokens[..., t.first_token_index :], out.continuous[
                ..., t.first_token_index :, :
            ]

        reference, latent = encode(normalized)
        conditions = []
        # Identity control plus gain applied either to all samples or after calibration.
        for gain in [0.5, 1.0, 2.0]:
            changed, _, _ = calibration_normalize(x * gain, t.calibration_samples)
            conditions.append(("gain_including_calibration", gain, 0, changed))
            shifted = x.clone()
            shifted[..., t.calibration_samples :] *= gain
            conditions.append(
                (
                    "gain_after_calibration",
                    gain,
                    0,
                    ((shifted - centre) / scale).clamp(-12, 12),
                )
            )
        # Noise only after calibration, power estimated from centered post-prefix samples.
        power = (
            (x[..., t.calibration_samples :] - centre).square().mean(-1, keepdim=True)
        )
        for seed in [260905, 260906, 260907]:
            generator = torch.Generator(device=t.device).manual_seed(seed + int(index))
            noise = torch.randn(x.shape, generator=generator, device=t.device)
            noise[..., : t.calibration_samples] = 0
            for snr in [40, 30, 20, 10]:
                noisy = x + noise * (power / (10 ** (snr / 10))).sqrt()
                conditions.append(
                    (
                        "noise_after_calibration_snr_db",
                        snr,
                        seed,
                        ((noisy - centre) / scale).clamp(-12, 12),
                    )
                )
        for condition, level, seed, values in conditions:
            ids, continuous = encode(values)
            results.append(
                {
                    "window_index": int(index),
                    "user": manifest["items"][index]["user"],
                    "filename": manifest["items"][index]["filename"],
                    "condition": condition,
                    "level": level,
                    "seed": seed,
                    "agreement": float((ids == reference).float().mean()),
                    "relative_latent_l2": float(
                        (continuous - latent).norm() / latent.norm().clamp_min(1e-8)
                    ),
                }
            )
    write_json(
        args.output,
        {
            "model": t.metadata(),
            "input_sha256": sha256(args.inputs),
            "split": manifest["split"],
            "windows": len(indices),
            "status": "validation_screen_no_downstream_robustness_claim",
            "rows": results,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--tar-index", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--windows", type=int, default=1024)
    p.add_argument("--recordings", type=int, default=128)
    for name in ["audit", "worker", "robustness"]:
        p = sub.add_parser(name)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--inputs", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--windows", type=int, default=64)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(getattr(args, "threads", 4))
    if getattr(args, "device", "cpu") == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
