"""Add the fitted TBME CNN to the existing HD-EMG inference-cost benchmark.

Uses the archived timing/storage functions, software versions, participant, seed
and final endpoints of the first four validation windows. No fitting or scoring.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import h5py
import numpy as np
import torch
from paper.common.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve completed benchmark measurements"
    started = time.monotonic()
    runs = args.project / "results"
    root = runs / "hdemg_tbme2024_v1"
    reference_root = runs / "hdemg_resource_benchmark_v1"
    reference = json.loads((reference_root / "results.json").read_text())
    helper_path = Path(__file__).resolve().with_name("benchmark_hdemg_all_variants.py")
    spec = importlib.util.spec_from_file_location("archived_cost_methods", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from paper.receivers.hdemg_tbme2024 import build, preprocess

    dependencies = {n: importlib.metadata.version(n) for n in ("torch", "numpy", "h5py", "lightning")}
    for name in ("torch", "numpy", "h5py"):
        assert dependencies[name] == reference["dependencies"][name], name
    protocol = json.loads((root / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert helper.digest_file(ROOT / name) == expected, name
    protocol_hash = helper.digest_file(root / "protocol.json")
    fit_folder = root / "subject_01/fits/tbme2024_260911"
    fit = json.loads((fit_folder / "fit.json").read_text())
    checkpoint = fit_folder / "best.pt"
    assert fit["complete"] and fit["protocol_sha256"] == protocol_hash
    assert helper.digest_file(checkpoint) == fit["best_sha256"]
    cache = json.loads((root / "subject_01/cache_receipt.json").read_text())
    assert helper.digest_file(root / "subject_01/cache_receipt.json") == fit["cache_receipt_sha256"]
    # Prepared examples flatten the 50 scored endpoints of each window.
    indices = [49, 99, 149, 199]
    with h5py.File(root / "subject_01/prepared.h5", "r") as handle:
        assert handle.attrs["complete"] and handle.attrs["protocol_sha256"] == cache["protocol_sha256"]
        inputs = torch.from_numpy(handle["val/inputs"][indices])
        extras = {f"target_{key}": torch.from_numpy(handle[f"target_{key}"][:])
                  for key in ("minimum", "span")}
        identities = {key: handle[f"val/{key}"][:4].tolist() for key in ("subjects", "tasks", "starts")}
    assert identities == reference["inputs"]["reference"]["identities"]
    with h5py.File(runs / "hdemg_raulnet_v1/subject_01/native.h5", "r") as handle:
        native = handle["val/raw"][:4, -1]
    assert helper.tensor_record(torch.from_numpy(native)) == reference["inputs"]["native_raw"]
    np.testing.assert_array_equal(inputs.numpy(), preprocess(native[..., -192:]))
    assert inputs.shape == (4, 2, 320, 192) and inputs.dtype == torch.float32
    input_record = helper.tensor_record(inputs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(state["state_dict"], strict=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(260911)
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name() == reference["hardware"]["gpu"]
    free, total = torch.cuda.mem_get_info()
    assert total - free < 1.5 * 2**30, "GPU is already occupied"
    model = model.float().eval().cuda()
    inputs = inputs.cuda()
    extras = {k: v.float().cuda() for k, v in extras.items()}
    before = {k: v._version for k, v in model.state_dict().items()}
    def gpu_sample():
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], text=True).strip()
    gpu_before = gpu_sample()
    timing_started = time.monotonic()
    timing = helper.measure(lambda index: model(inputs[index:index + 1]))
    timing_wall = time.monotonic() - timing_started
    gpu_after = gpu_sample()
    assert before == {k: v._version for k, v in model.state_dict().items()}
    assert not model.training and timing["measured_calls"] == 48
    row = dict(panel="TBME", campaign="hdemg_tbme2024_v1", arm="tbme2024_retrained",
               representation="raw", subject=1, seed=260911,
               checkpoint=str(checkpoint), checkpoint_sha256=fit["best_sha256"],
               model_source_sha256=helper.digest_file(ROOT / "vendor/tbme2024/model.py"),
               receiver_input_shape=[1, 2, 320, 192], receiver_input_dtype="torch.float32",
               loaded_state_unchanged=True,
               **helper.storage_record(model, extras),
               **helper.representation_cost("raw", torch.float32, 2048), **timing)
    assert row["parameter_count"] == fit["parameters"] == 27403873
    result = dict(complete=True, test_evaluated=False, training_performed=False,
                  finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Decoder-only latest-pose inference on the fitted TBME model; four validation endpoints.",
                  hardware=dict(gpu=torch.cuda.get_device_name(), host=platform.node(),
                                slurm_job_id=os.environ.get("SLURM_JOB_ID"), cpu_threads=4),
                  dependencies=dependencies, precision=reference["precision"],
                  timing="Same archived measure() and storage_record() as the existing figure: FP32, no AMP/TF32, batch 1, 12 warm-ups, 48 synchronized wall-time calls; input preprocessing/transfers/I/O and target denormalization excluded. The TBME native context is 192 samples, with two input planes; normalization inside its forward is timed.",
                  reference_benchmark_sha256=helper.digest_file(reference_root / "results.json"),
                  timing_helper_sha256=helper.digest_file(helper_path),
                  benchmark_script_sha256=helper.digest_file(Path(__file__)),
                  training_protocol_sha256=protocol_hash,
                  inputs=dict(group="val", window_indices=[0, 1, 2, 3],
                              flattened_endpoint_indices=indices, identities=identities,
                              preprocessed_inputs=input_record,
                              exact_native_input_and_preprocessing_verified=True),
                  monitoring=dict(gpu_before=gpu_before, gpu_after=gpu_after,
                                  timing_section_seconds=timing_wall,
                                  total_script_seconds=time.monotonic()-started),
                  results=[row])
    helper.write_json(args.output, result)
    print(json.dumps({k: row[k] for k in ("model_tensor_mb", "p50_ms", "p95_ms", "parameter_count")}), flush=True)


if __name__ == "__main__":
    main()
