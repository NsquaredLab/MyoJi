"""One-person, validation-only pilot of an existing pretrained language model.

Uses only train/val groups from an existing frozen DPose cache. Four arms
isolate frozen language pretraining, backbone inclusion and quantization.
No test evaluation or automatic wider campaign is provided by this script.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from paper.receivers.dpose_llm_core import ARMS, MODEL_ID, REVISION, autocast_for, build_model, tensor_digest
from paper.common.paths import RESULTS


SEED = 260911
STEPS = 2000
VALIDATE_EVERY = 100


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_validation(path, device):
    splits = {}
    digests = {}
    with h5py.File(path, "r") as handle:
        assert bool(handle.attrs["complete"])
        for name in ("train", "val"):
            group = handle[name]
            arrays = {key: np.asarray(group[key]) for key in
                      ("tokens", "continuous", "targets", "subjects", "tasks", "starts")}
            assert len(arrays["tokens"]) == (64 if name == "train" else 16)
            assert arrays["tokens"].shape[1:] == (320, 100)
            assert arrays["continuous"].shape[1:] == (320, 100, 256)
            assert np.isfinite(arrays["targets"]).all()
            digest = hashlib.sha256()
            for key, value in arrays.items():
                digest.update(f"{key}:{value.shape}:{value.dtype}".encode())
                digest.update(value.tobytes())
            digests[name] = digest.hexdigest()
            splits[name] = {
                "tokens": torch.from_numpy(arrays["tokens"].astype(np.int64)).to(device),
                "continuous": torch.from_numpy(arrays["continuous"]).to(device),
                "targets": torch.from_numpy(arrays["targets"][:, -50:]).float().to(device),
                "starts": arrays["starts"],
                "subjects": arrays["subjects"],
            }
        book = torch.from_numpy(handle["codebook"][:]).float().to(device)
        descriptors = torch.from_numpy(handle["descriptors"][:]).float().to(device)
        digests["codebook"] = tensor_digest([("codebook", book)])
        digests["descriptors"] = tensor_digest([("descriptors", descriptors)])
    return splits, book, descriptors, digests


def features(split, indices, arm, book):
    if arm.startswith("tokens"):
        return book[split["tokens"][indices]]
    return split["continuous"][indices].float()


def sampling_schedule():
    return np.random.default_rng(np.random.SeedSequence([SEED, 260910])).integers(
        0, 64, size=(STEPS, 2), dtype=np.int64
    )


def learning_rate(step):
    if step < 100:
        return 3e-4 * (step + 1) / 100
    ratio = (step - 100) / (STEPS - 101)
    return 3e-6 + (3e-4 - 3e-6) * 0.5 * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def evaluate(model, split, arm, book, descriptors, mean, scale, device):
    model.eval()
    predictions = []
    for offset in range(0, len(split["targets"]), 2):
        indices = np.arange(offset, min(offset + 2, len(split["targets"])))
        with autocast_for(device):
            pred = model(features(split, indices, arm, book), descriptors)[:, -50:]
        predictions.append((pred.float() * scale + mean).cpu().numpy())
    pred = np.concatenate(predictions).astype(np.float64)
    targets = split["targets"].cpu().numpy().astype(np.float64)
    assert np.isfinite(pred).all()
    error = float(np.linalg.norm((pred - targets).reshape(-1, 20, 3), axis=-1).mean())
    return error, pred, targets


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_adapter(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("backbone.") for key in incompatible.missing_keys)


def fit(args, arm, splits, book, descriptors, protocol_hash, schedule):
    destination = args.root / arm
    destination.mkdir(exist_ok=True)
    result_path = destination / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["protocol_sha256"] == protocol_hash and result["complete"]
        return result
    model, loading = build_model(args.root / "model", arm, SEED, args.device, torch.bfloat16)
    initial_adapter = tensor_digest(model.adapter_state().items())
    frozen_before = (tensor_digest(model.backbone.state_dict().items())
                     if model.backbone is not None else None)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters = {"trainable": sum(p.numel() for p in trainable),
                  "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.01)
    mean = splits["train"]["targets"].mean((0, 1))
    scale = splits["train"]["targets"].std((0, 1), correction=0).clamp_min(1e-4)
    start_step, best_error, best_step, history = 0, float("inf"), None, []
    last = destination / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        assert state["protocol_sha256"] == protocol_hash
        assert state["frozen_backbone_sha256"] == frozen_before
        restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_error, best_step, history = (
            state[k] for k in ("step", "best_error", "best_step", "history")
        )
        del state
    initial_error = None
    if start_step == 0:
        initial_error = evaluate(model, splits["val"], arm, book, descriptors, mean, scale, args.device)[0]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"starting_arm": arm, "parameters": parameters,
                      "initial_val_error": initial_error, "resume_step": start_step}), flush=True)
    first_gradient_checked = False
    for step in range(start_step, STEPS):
        torch.manual_seed(SEED + step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        with autocast_for(args.device):
            prediction = model(features(splits["train"], schedule[step], arm, book), descriptors)[:, -50:]
            target = (splits["train"]["targets"][schedule[step]] - mean) / scale
            loss = F.smooth_l1_loss(prediction.float(), target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {arm}, step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        if not first_gradient_checked:
            assert model.project.weight.grad is not None and model.project.weight.grad.abs().sum() > 0
            assert model.bridge[1].weight.grad is not None and model.bridge[1].weight.grad.abs().sum() > 0
            if model.backbone is not None:
                assert all(p.grad is None for p in model.backbone.parameters())
            first_gradient_checked = True
        optimizer.step()
        if (step + 1) % VALIDATE_EVERY == 0:
            error, pred, targets = evaluate(model, splits["val"], arm, book, descriptors, mean, scale, args.device)
            improved = error < best_error
            if improved:
                best_error, best_step = error, step + 1
                save_state(destination / "best.pt", {"adapter": model.adapter_state(),
                           "step": best_step, "mean": mean, "scale": scale,
                           "protocol_sha256": protocol_hash})
                np.savez_compressed(destination / "validation_predictions.npz", predictions=pred,
                                    targets=targets, starts=splits["val"]["starts"],
                                    subjects=splits["val"]["subjects"])
            row = {"step": step + 1, "val_error": error, "train_loss": float(loss.detach()),
                   "gradient_norm": float(norm), "seconds": time.monotonic() - started}
            history.append(row)
            save_state(last, {"adapter": model.adapter_state(), "optimizer": optimizer.state_dict(),
                       "step": step + 1, "best_error": best_error, "best_step": best_step,
                       "history": history, "protocol_sha256": protocol_hash,
                       "frozen_backbone_sha256": frozen_before})
            write_json(destination / "progress.json", {"arm": arm, "history": history,
                       "best_val_error": best_error, "best_step": best_step})
            print(json.dumps({"arm": arm, **row, "best_val_error": best_error}), flush=True)
    frozen_after = (tensor_digest(model.backbone.state_dict().items())
                    if model.backbone is not None else None)
    assert frozen_before == frozen_after
    with np.load(destination / "validation_predictions.npz", allow_pickle=False) as saved:
        independently_recomputed = float(np.linalg.norm(
            (saved["predictions"] - saved["targets"]).reshape(-1, 20, 3), axis=-1).mean())
    assert abs(independently_recomputed - best_error) < 1e-10
    result = {"complete": True, "arm": arm, "subject": 1, "seed": SEED,
              "protocol_sha256": protocol_hash, "parameters": parameters,
              "initial_adapter_sha256": initial_adapter,
              "frozen_backbone_sha256": frozen_before, "frozen_weights_unchanged": True,
              "best_validation_error": best_error, "best_step": best_step,
              "final_validation_error": history[-1]["val_error"],
              "steps": STEPS, "endpoint_presentations": STEPS * 100,
              "history": history, "loading_info": loading,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "this_attempt_seconds": time.monotonic() - started,
              "test_evaluated": False}
    write_json(result_path, result)
    del model, optimizer, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference", default=RESULTS / "dpose_within_v1", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.device.startswith("cuda") or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU for this pilot")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pinned pilot requires native bfloat16 support")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    free, total = torch.cuda.mem_get_info()
    if total - free > 1.5 * 2**30:
        raise RuntimeError("Selected GPU is already occupied; refusing to share it")
    snapshots = args.root / "model"
    sources = [Path(__file__), Path(__file__).with_name("dpose_llm_core.py")]
    splits, book, descriptors, hashes = load_training_validation(
        args.reference / "subject_01/features.h5", args.device
    )
    schedule = sampling_schedule()
    protocol = {
        "scope": "One predetermined DPose participant (01), one seed, training and validation only. Exploratory feasibility pilot, not paper test results.",
        "model": MODEL_ID, "revision": REVISION, "arms": list(ARMS), "subject": 1, "seed": SEED,
        "steps": STEPS, "windows_per_update": 2, "scored_endpoints_per_window": 50,
        "validation_opportunities": STEPS // VALIDATE_EVERY,
        "learning_rate": "100-step warmup to 3e-4, cosine decay to 3e-6; AdamW decay .01, gradient clip 1",
        "adapter": "256D EMG embeddings -> learned width64 projection and four geometry-aware slots -> learned1024D frame; LayerNorm and .02 scaling -> causal Qwen hidden states -> LayerNorm and linear60-coordinate regression",
        "backbone": "All pretrained or randomly initialized Qwen blocks frozen; unused text embedding replaced with one frozen row; no text token IDs, prompting or language-output head",
        "pretraining": "External language checkpoint only. No new language/tokenizer pretraining; bridge and pose head supervised on DPose training labels",
        "continuous_storage": str(splits["train"]["continuous"].dtype),
        "cache_digests": hashes, "sampling_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "source_sha256": {path.name: file_hash(path) for path in sources},
        "model_sha256": {path.name: file_hash(path) for path in snapshots.iterdir() if path.is_file()},
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "myoji")},
        "test_evaluated": False,
    }
    protocol_path = args.root / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, "Protocol/source changed; use a new pilot root"
    else:
        write_json(protocol_path, protocol)
    protocol_hash = file_hash(protocol_path)
    write_json(args.root / "runtime.json", {
        "started_at": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "protocol_sha256": protocol_hash,
    })
    mean_pose = splits["train"]["targets"].mean((0, 1)).cpu().numpy().astype(np.float64)
    targets = splits["val"]["targets"].cpu().numpy().astype(np.float64)
    constant = float(np.linalg.norm((targets - mean_pose).reshape(-1, 20, 3), axis=-1).mean())
    results = []
    for arm in ARMS:
        results.append(fit(args, arm, splits, book, descriptors, protocol_hash, schedule))
    assert len({result["initial_adapter_sha256"] for result in results}) == 1
    report = {"complete": True, "protocol_sha256": protocol_hash, "results": results,
              "constant_validation_error": constant, "test_evaluated": False}
    write_json(args.root / "results.json", report)
    lines = ["# External-LLM DPose pilot", "", "One participant, one seed; validation only. Lower error is better.",
             "", "| Arm | Best validation error | Step | Trainable parameters | Frozen parameters |",
             "|---|---:|---:|---:|---:|"]
    for result in results:
        lines.append(f"| {result['arm']} | {result['best_validation_error']:.6f} | {result['best_step']} | {result['parameters']['trainable']} | {result['parameters']['frozen']} |")
    lines.extend([f"| Training-mean constant | {constant:.6f} | — | — | — |", "",
                  "These validation-selected minima are a feasibility check, not held-out test results or population evidence.",
                  "The prior 15,000-update paper experiments use different budgets and must not be directly ranked against this pilot."])
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "report": str(args.root / "RESULTS.md")}), flush=True)


if __name__ == "__main__":
    main()
