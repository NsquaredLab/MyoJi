"""Nine-person, three-seed frozen-Qwen comparison with validation-selected tests.

The existing validation-only pilot remains immutable. This campaign reuses its
model and the matched study's exact sampling/learning-rate schedules. Test
inference is permitted only after every one of the 108 fits has completed.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from myoji._vendor.channel_tokenization import GeometryAwareChannelPool

from paper.receivers.hdemg_llm_core import ARMS, MODEL_ID, REVISION, autocast_for, build_model, tensor_digest
from paper.common.paths import REPO, RESULTS
from paper.receivers.hdemg_matched_core import SEEDS, EPOCHS, UPDATES_PER_EPOCH, learning_rate, sample_windows, schedule_digest
from paper.receivers.run_hdemg_matched import data_digest
from paper.receivers.run_hdemg_llm_pilot import (
    evaluate, features, file_hash, load_training_validation, restore_adapter,
    save_state, write_json,
)

SUBJECTS = tuple(range(1, 10))
STEPS = EPOCHS * UPDATES_PER_EPOCH
CELLS = [(subject, seed, arm) for subject in SUBJECTS for seed in SEEDS for arm in ARMS]
STOP = False


def stop_requested(_signum, _frame):
    global STOP
    STOP = True


def cell_path(root, index):
    subject, seed, arm = CELLS[index]
    return root / f"subject_{subject:02d}/fits/{arm}_{seed}"


def schedule(seed):
    return np.concatenate([sample_windows(seed, epoch) for epoch in range(EPOCHS)])


def source_manifest():
    base = Path(__file__).resolve().parents[1]
    names = ["receivers/run_hdemg_llm_full.py", "receivers/hdemg_llm_core.py",
             "receivers/run_hdemg_llm_pilot.py", "receivers/hdemg_matched_core.py",
             "receivers/run_hdemg_matched.py", "receivers/hdemg_raulnet_model.py",
             "vendor/raulnet_v19.py"]
    paths = [base / name for name in names] + sorted((REPO / "src/myoji").rglob("*.py"))
    return {str(path.relative_to(REPO)): file_hash(path) for path in paths}


def freeze(args):
    destination = args.root / "protocol.json"
    if destination.exists():
        raise FileExistsError(destination)
    matched = json.loads((args.matched / "protocol.json").read_text())
    pilot = json.loads((args.pilot / "protocol.json").read_text())
    assert matched["subjects"] == list(SUBJECTS) and matched["seeds"] == list(SEEDS)
    schedules = {str(seed): hashlib.sha256(schedule(seed).tobytes()).hexdigest() for seed in SEEDS}
    assert schedules == matched["sampling_sha256"]
    assert all(schedules[str(seed)] == schedule_digest(seed) for seed in SEEDS)
    assert pilot["revision"] == REVISION and pilot["model"] == MODEL_ID
    model_hashes = {p.name: file_hash(p) for p in (args.pilot / "model").iterdir() if p.is_file()}
    assert model_hashes == pilot["model_sha256"]
    digests = {}
    for subject in SUBJECTS:
        path = args.reference / f"subject_{subject:02d}/features.h5"
        digest = data_digest(path)
        assert digest == matched["cache_digests"][str(subject)]["cached"]
        digests[str(subject)] = digest
        print(json.dumps({"bound_subject": subject}), flush=True)
    write_json(destination, {
        "scope": "Exploratory personalized same-session/task chronological tests on previously inspected splits; tokenizer pretraining saw these recordings without pose labels.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID, "revision": REVISION, "subjects": list(SUBJECTS),
        "seeds": list(SEEDS), "arms": list(ARMS), "fits": len(CELLS),
        "cells": CELLS, "steps": STEPS, "windows_per_update": 2,
        "scored_endpoints_per_window": 50, "endpoint_presentations_per_fit": STEPS * 100,
        "validation_opportunities": EPOCHS, "validation_every": UPDATES_PER_EPOCH,
        "selection": "Earliest lowest mean validation 3D error among exactly 150 checkpoints; test inference only after all 108 selected checkpoint hashes are frozen.",
        "training": "Reuse matched-study sampling and LR: 500-step warmup to3e-4, cosine to3e-6, AdamW decay.01, gradient norm clip1, SmoothL1, training-only target mean/populationSD floor1e-4.",
        "precision": "Native CUDA BF16 autocast; FP32 supervised adapter/head; Qwen weights stored BF16 and frozen. No TF32.",
        "backbone": "Same pilot adapter and 28-layer frozen Qwen, paired supervised initialization per seed, no text embedding table or LM output head, input_scale.02, no LoRA or language pretraining.",
        "remaining_differences": "Qwen capacity, frozen pretraining, BF16 vs matched-study FP16, and backbone/adapter regularization differ. Matching cohort, splits, update/target exposure and selection does not make total compute or architecture equal.",
        "aggregation": "Mean over 3 decoder seeds within participant, then equally weighted mean and sampleSD over 9 participant means; also preserve all 27 per-arm scores and paired participant differences.",
        "physical_units": "Unverified HD-EMG stored-coordinate units, not asserted millimetres.",
        "sampling_sha256": schedules, "cache_digests": digests,
        "source_sha256": source_manifest(), "model_sha256": model_hashes,
        "external_source_sha256": {inspect.getfile(GeometryAwareChannelPool): file_hash(inspect.getfile(GeometryAwareChannelPool))},
        "dependencies": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "myoji")},
        "references": {"matched_protocol": file_hash(args.matched / "protocol.json"), "pilot_protocol": file_hash(args.pilot / "protocol.json")},
        "resource_policy": "At most one GPU used by this campaign. Reuse spare GPU in existing allocation first. Fit concurrency selected by measured throughput; continuous utilization/memory/progress logging, checkpointed handoff and stall abort. No reserved idle GPU after work completes.",
    })


def validate_protocol(args):
    protocol = json.loads((args.root / "protocol.json").read_text())
    assert protocol["source_sha256"] == source_manifest(), "Consumed source changed"
    assert protocol["cells"] == [list(cell) for cell in CELLS]
    for path, digest in protocol["external_source_sha256"].items():
        assert file_hash(path) == digest
    for name, version in protocol["dependencies"].items():
        assert importlib.metadata.version(name) == version
    return protocol, file_hash(args.root / "protocol.json")


def prepare(args, index, protocol):
    subject, seed, arm = CELLS[index]
    path = args.reference / f"subject_{subject:02d}/features.h5"
    # Full cache integrity was checked on CPU before GPU launch and is rechecked
    # before the test gate. Fit loading touches only train and validation groups.
    splits, book, descriptors, hashes = load_training_validation(path, args.device)
    assert all(np.all(split["subjects"] == subject) for split in splits.values())
    model, loading = build_model(args.pilot / "model", arm, seed, args.device, torch.bfloat16)
    initial = tensor_digest(model.adapter_state().items())
    frozen = tensor_digest(model.backbone.state_dict().items()) if model.backbone is not None else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 401212
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=.01, fused=False)
    mean = splits["train"]["targets"].mean((0, 1))
    scale = splits["train"]["targets"].std((0, 1), correction=0).clamp_min(1e-4)
    return model, optimizer, trainable, splits, book, descriptors, mean, scale, initial, frozen, loading, hashes


def train_step(model, optimizer, trainable, train, book, descriptors, mean, scale, arm, seed, step, windows, device):
    torch.manual_seed(seed + step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step)
    with autocast_for(device):
        prediction = model(features(train, windows, arm, book), descriptors)[:, -50:]
        target = (train["targets"][windows] - mean) / scale
        loss = F.smooth_l1_loss(prediction.float(), target)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite loss in {arm}, step {step}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
    if step == 0:
        assert model.project.weight.grad is not None and model.project.weight.grad.abs().sum() > 0
        assert model.bridge[1].weight.grad is not None and model.bridge[1].weight.grad.abs().sum() > 0
        assert model.backbone is None or all(p.grad is None for p in model.backbone.parameters())
    optimizer.step()
    return float(loss.detach()), float(norm)


def fit(args):
    protocol, ph = validate_protocol(args)
    index = args.index
    subject, seed, arm = CELLS[index]
    destination = cell_path(args.root, index)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "fit.json").exists():
        result = json.loads((destination / "fit.json").read_text())
        assert result["complete"] and result["protocol_sha256"] == ph
        assert result["best_sha256"] == file_hash(destination / "best.pt")
        return
    values = prepare(args, index, protocol)
    model, optimizer, trainable, splits, book, descriptors, mean, scale, initial, frozen, loading, hashes = values
    windows = schedule(seed)
    start, best, best_step, history = 0, float("inf"), None, []
    last = destination / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        assert state["protocol_sha256"] == ph and state["frozen_backbone_sha256"] == frozen
        assert state["cache_digests"] == hashes
        restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, best_step, history = (state[k] for k in ("step", "best_error", "best_step", "history"))
        torch.testing.assert_close(state["mean"], mean, rtol=0, atol=0)
        torch.testing.assert_close(state["scale"], scale, rtol=0, atol=0)
        del state
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"phase": "fit", "index": index, "subject": subject, "seed": seed,
                      "arm": arm, "resume_step": start}), flush=True)
    for step in range(start, STEPS):
        loss, norm = train_step(model, optimizer, trainable, splits["train"], book, descriptors,
                               mean, scale, arm, seed, step, windows[step], args.device)
        if (step + 1) % UPDATES_PER_EPOCH == 0:
            error, pred, target = evaluate(model, splits["val"], arm, book, descriptors, mean, scale, args.device)
            if error < best:
                best, best_step = error, step + 1
                save_state(destination / "best.pt", {"adapter": model.adapter_state(), "step": best_step,
                    "mean": mean, "scale": scale, "protocol_sha256": ph, "frozen_backbone_sha256": frozen})
                np.savez_compressed(destination / "validation_predictions.npz", predictions=pred,
                    targets=target, starts=splits["val"]["starts"], subjects=splits["val"]["subjects"])
            history.append({"step": step + 1, "val_error": error, "train_loss": loss,
                            "gradient_norm": norm, "attempt_seconds": time.monotonic() - started})
            save_state(last, {"adapter": model.adapter_state(), "optimizer": optimizer.state_dict(),
                "step": step + 1, "best_error": best, "best_step": best_step, "history": history,
                "mean": mean, "scale": scale, "protocol_sha256": ph,
                "frozen_backbone_sha256": frozen, "cache_digests": hashes})
            progress = {"index": index, "subject": subject, "seed": seed, "arm": arm,
                        "step": step + 1, "val_error": error, "best_val_error": best,
                        "updated_at": datetime.now(timezone.utc).isoformat()}
            write_json(destination / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            if STOP and step + 1 < STEPS:
                raise SystemExit(75)
    frozen_after = tensor_digest(model.backbone.state_dict().items()) if model.backbone is not None else None
    assert frozen_after == frozen
    assert len(history) == EPOCHS and [h["step"] for h in history] == list(range(100, STEPS + 1, 100))
    with np.load(destination / "validation_predictions.npz", allow_pickle=False) as saved:
        rescored = float(np.linalg.norm((saved["predictions"] - saved["targets"]).reshape(-1, 20, 3), axis=-1).mean())
    assert abs(rescored - best) < 1e-10
    write_json(destination / "fit.json", {
        "complete": True, "index": index, "subject": subject, "seed": seed, "arm": arm,
        "protocol_sha256": ph, "steps": STEPS, "endpoint_presentations": STEPS * 100,
        "sampling_sha256": hashlib.sha256(windows.tobytes()).hexdigest(),
        "initial_adapter_sha256": initial, "frozen_backbone_sha256": frozen,
        "frozen_weights_unchanged": True, "cache_digests": hashes,
        "best_validation_error": best, "best_step": best_step, "history": history,
        "best_sha256": file_hash(destination / "best.pt"), "loading_info": loading,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "this_attempt_seconds": time.monotonic() - started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "test_evaluated": False,
    })


def gate(args):
    protocol, ph = validate_protocol(args)
    selected = []
    paired = {}
    for index, (subject, seed, arm) in enumerate(CELLS):
        path = cell_path(args.root, index)
        result = json.loads((path / "fit.json").read_text())
        assert result["complete"] and result["protocol_sha256"] == ph and result["steps"] == STEPS
        assert result["sampling_sha256"] == protocol["sampling_sha256"][str(seed)]
        assert result["best_sha256"] == file_hash(path / "best.pt")
        assert result["frozen_weights_unchanged"]
        paired.setdefault((subject, seed), set()).add(result["initial_adapter_sha256"])
        selected.append({"index": index, "best_sha256": result["best_sha256"], "fit_sha256": file_hash(path / "fit.json")})
    assert all(len(values) == 1 for values in paired.values())
    for subject in SUBJECTS:
        assert data_digest(args.reference / f"subject_{subject:02d}/features.h5") == protocol["cache_digests"][str(subject)]
    for name, digest in protocol["model_sha256"].items():
        assert file_hash(args.pilot / "model" / name) == digest
    gate_value = {"passed": True, "protocol_sha256": ph, "selected": selected}
    destination = args.root / "test_gate.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == gate_value
    else:
        write_json(destination, gate_value)


def test(args):
    _protocol, ph = validate_protocol(args)
    g = json.loads((args.root / "test_gate.json").read_text())
    assert g["passed"] and g["protocol_sha256"] == ph and len(g["selected"]) == len(CELLS)
    index = args.index
    subject, seed, arm = CELLS[index]
    path = cell_path(args.root, index)
    if (path / "test.json").exists():
        r = json.loads((path / "test.json").read_text())
        assert r["protocol_sha256"] == ph and r["predictions_sha256"] == file_hash(path / "test_predictions.npz")
        return
    assert g["selected"][index]["best_sha256"] == file_hash(path / "best.pt")
    assert g["selected"][index]["fit_sha256"] == file_hash(path / "fit.json")
    with h5py.File(args.reference / f"subject_{subject:02d}/features.h5", "r") as handle:
        group = handle["test"]
        split = {"targets": torch.from_numpy(group["targets"][:, -50:]).float().to(args.device),
                 "subjects": group["subjects"][:], "starts": group["starts"][:]}
        assert len(split["subjects"]) == 16 and np.all(split["subjects"] == subject)
        key = "tokens" if arm.startswith("tokens") else "continuous"
        split[key] = torch.from_numpy(group[key][:].astype(np.int64 if key == "tokens" else np.float16)).to(args.device)
        book = torch.from_numpy(handle["codebook"][:]).float().to(args.device)
        descriptors = torch.from_numpy(handle["descriptors"][:]).float().to(args.device)
        train_targets = handle["train/targets"][:, -50:].astype(np.float64)
    model, _loading = build_model(args.pilot / "model", arm, seed, args.device, torch.bfloat16)
    state = torch.load(path / "best.pt", map_location=args.device, weights_only=False)
    assert state["protocol_sha256"] == ph
    frozen = tensor_digest(model.backbone.state_dict().items()) if model.backbone is not None else None
    assert frozen == state["frozen_backbone_sha256"]
    restore_adapter(model, state["adapter"])
    error, pred, targets = evaluate(model, split, arm, book, descriptors, state["mean"], state["scale"], args.device)
    np.savez_compressed(path / "test_predictions.npz", predictions=pred, targets=targets,
                        subjects=split["subjects"], starts=split["starts"])
    constant = float(np.linalg.norm((targets - train_targets.mean((0, 1))).reshape(-1, 20, 3), axis=-1).mean())
    write_json(path / "test.json", {"complete": True, "subject": subject, "seed": seed, "arm": arm,
        "mean_3d_error": error, "constant_pose_error": constant, "test_evaluated": True,
        "protocol_sha256": ph, "best_sha256": file_hash(path / "best.pt"),
        "predictions_sha256": file_hash(path / "test_predictions.npz")})


def report(args):
    _p, ph = validate_protocol(args)
    results = []
    for index, (subject, seed, arm) in enumerate(CELLS):
        path = cell_path(args.root, index)
        row = json.loads((path / "test.json").read_text())
        assert row["complete"] and row["protocol_sha256"] == ph
        assert row["predictions_sha256"] == file_hash(path / "test_predictions.npz")
        with np.load(path / "test_predictions.npz", allow_pickle=False) as saved:
            error = float(np.linalg.norm((saved["predictions"] - saved["targets"]).reshape(-1, 20, 3), axis=-1).mean())
            assert abs(error - row["mean_3d_error"]) < 1e-10
        results.append(row)
    summaries = {}
    for arm in ARMS:
        people = {str(s): float(np.mean([r["mean_3d_error"] for r in results if r["arm"] == arm and r["subject"] == s])) for s in SUBJECTS}
        summaries[arm] = {"participant_means": people, "mean": float(np.mean(list(people.values()))),
                          "participant_sd": float(np.std(list(people.values()), ddof=1)), "n_participants": 9, "n_seeds": 3}
    paired = {}
    for other in ("tokens_random", "tokens_bypass", "continuous_pretrained"):
        deltas = {s: summaries["tokens_pretrained"]["participant_means"][s] - summaries[other]["participant_means"][s] for s in summaries[other]["participant_means"]}
        paired["tokens_pretrained_minus_" + other] = {"participant_differences": deltas,
            "mean": float(np.mean(list(deltas.values()))), "sd": float(np.std(list(deltas.values()), ddof=1))}
    write_json(args.root / "results.json", {"complete": True, "protocol_sha256": ph,
        "scope": "Personalized same-session/task chronological test; previously inspected splits; stored-coordinate units.",
        "results": results, "summaries": summaries, "paired_differences": paired})


def probe(args):
    protocol, _ph = validate_protocol(args)
    values = prepare(args, args.index, protocol)
    model, optimizer, trainable, splits, book, descriptors, mean, scale, _initial, frozen, _loading, _hashes = values
    _subject, seed, arm = CELLS[args.index]
    windows = schedule(seed)
    for step in range(5):
        train_step(model, optimizer, trainable, splits["train"], book, descriptors, mean, scale,
                   arm, seed, step, windows[step], args.device)
    directory = args.root / "probes" / args.probe_name
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"ready-{args.index}.json", {"ready": True})
    deadline = time.monotonic() + 120
    while len(list(directory.glob("ready-*.json"))) < args.probe_peers:
        if time.monotonic() > deadline:
            raise TimeoutError("Probe peer did not become ready")
        time.sleep(.05)
    torch.cuda.synchronize()
    start = time.time()
    for step in range(5, 85):
        train_step(model, optimizer, trainable, splits["train"], book, descriptors, mean, scale,
                   arm, seed, step, windows[step], args.device)
    torch.cuda.synchronize()
    end = time.time()
    assert frozen == (tensor_digest(model.backbone.state_dict().items()) if model.backbone is not None else None)
    write_json(directory / f"result-{args.index}.json", {"index": args.index, "steps": 80,
        "start": start, "end": end, "seconds": end - start, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "fit", "gate", "test", "report", "probe"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=RESULTS / "hdemg_within_v1")
    parser.add_argument("--matched", type=Path, default=RESULTS / "hdemg_matched_v1")
    parser.add_argument("--pilot", type=Path, default=RESULTS / "hdemg_llm_pilot_v1")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-name", default="single")
    parser.add_argument("--probe-peers", type=int, default=1)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert 0 <= args.index < len(CELLS)
    if args.mode in ("fit", "test", "probe"):
        assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Expose exactly one BF16-capable GPU"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        signal.signal(signal.SIGTERM, stop_requested)
        signal.signal(signal.SIGINT, stop_requested)
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
