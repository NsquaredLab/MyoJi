"""Train token-input RaulNet V20 on the frozen personalized DPose split."""

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
from paper.common import paths
from paper.hdemg import run_dpose_pilot as pilot
import torch
from torch.nn import functional as F

SOURCE = paths.REPO / "src/myoji/_vendor/raulnet_v20.py"
SOURCE_SHA256 = "1a34d00851e6eb52b24f5a041637285bc2b7085bff1f217ac6ae9b3d6191e672"
assert pilot.sha256(SOURCE) == SOURCE_SHA256
spec = importlib.util.spec_from_file_location("pinned_raulnet_v20", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RaulNetV20 = module.RaulNetV20

SUBJECTS = tuple(range(1, 10))
SEEDS = (260911, 260912, 260913)
READOUT = {
    "kind": "raulnet_v20",
    "nr_of_electrode_grids": 5,
    "nr_of_electrodes_per_grid": 64,
    "nr_of_outputs": 60,
    "cnn_encoder_channels": [64, 32, 32],
    "mlp_encoder_channels": [128, 128],
    "temporal_width": 64,
    "context_frames": 100,
    "dropout": 0.1,
    "use_temporal": True,
}


def book_digest(book):
    return hashlib.sha256(book.detach().cpu().float().numpy().tobytes()).hexdigest()


def cache_digest(path):
    """Hash only the arrays relevant to token decoding and window identity."""
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        assert f.attrs["complete"]
        for name in ["codebook", "descriptors"] + [
            f"{s}/{k}"
            for s in ("train", "val", "test")
            for k in ("tokens", "targets", "subjects", "tasks", "starts")
        ]:
            data = f[name][:]
            h.update(name.encode())
            h.update(str(data.shape).encode())
            h.update(data.dtype.str.encode())
            h.update(data.tobytes())
    return h.hexdigest()


def predict_tokens(model, data, ix, arm, device):
    assert arm == "tokens"
    # V20 owns the frozen lookup. No geometry, IDs-as-amplitudes, or pose input.
    return model(data["features"][ix].to(device))


@contextmanager
def backend(book):
    """Reuse the audited fit/aggregation loop without changing other decoders."""
    old_pilot = {
        k: getattr(pilot, k) for k in ("ARMS", "SEEDS", "build", "predict_batch")
    }
    # This older module changes pilot globals on import; contain that too.
    from paper.hdemg import run_dpose_within as within

    old_within = {k: getattr(within, k) for k in ("ARMS", "SEEDS")}

    def build(arm, p, config):
        assert arm == "tokens" and p["readout"] == READOUT
        assert book_digest(book) == p["codebook_float32_sha256"]
        model = RaulNetV20(book, **{k: v for k, v in READOUT.items() if k != "kind"})
        assert sum(v.numel() for v in model.parameters()) == 704700
        return model

    pilot.ARMS = within.ARMS = ("tokens",)
    pilot.SEEDS = within.SEEDS = SEEDS
    pilot.build = build
    pilot.predict_batch = predict_tokens
    try:
        yield within
    finally:
        for key, value in old_pilot.items():
            setattr(pilot, key, value)
        for key, value in old_within.items():
            setattr(within, key, value)


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError(args.root / "protocol.json")
    old = json.loads((args.reference / "protocol.json").read_text())
    old_results = json.loads((args.reference / "results.json").read_text())
    assert old_results["complete"] and old["readout"]["kind"] == "tcn"
    assert old["subjects"] == list(SUBJECTS) and old["seeds"] == list(SEEDS)
    assert old["training"]["score_frames"] == 50
    assert pilot.sha256(args.checkpoint) == old["checkpoint_sha256"]
    book = torch.load(args.checkpoint, map_location="cpu", weights_only=True)[
        "state_dict"
    ]["quantizer.codebook"]
    assert book.shape == (1024, 256) and torch.isfinite(book).all()
    p = copy.deepcopy(old)
    p.pop("raw_arm", None)
    p.update(
        arms=["tokens"],
        readout=copy.deepcopy(READOUT),
        codebook_float32_sha256=book_digest(book),
        trainable_parameters=704700,
        scope=(
            "Exploratory architecture comparison on the previously inspected personalized, "
            "same-session/task chronological DPose split. The frozen tokenizer saw these "
            "recordings during unlabeled pretraining. Not unseen-person/session/task performance."
        ),
        models=(
            "Exact pinned MyoTorch token-input RaulNet V20, newly initialized per person/seed. "
            "Frozen 1024x256 codebook lookup; V19-style frame-local spatial CNN in five legacy "
            "groups of 64 channels; width-64 causal TCN with dilations 1..64; 128/128 pose MLP. "
            "No V19 weights, raw EMG, pose history, task IDs or calibration metadata enter V20. "
            "Legacy grid ordering is not a claim of accurate physical electrode distances."
        ),
        evaluation=(
            "27 fits: nine people x three established seeds. Identical training settings, "
            "windows and target endpoints to token TCN; different parameter/compute budgets. "
            "Validation-only checkpoint selection. All 27 fits and frozen codebooks must pass "
            "verification before any V20 test scoring. Independently recompute paired TCN "
            "errors and training-mean baseline. No test-driven tuning or reruns."
        ),
        cache="Read-only links to original features.h5; no encoding or cache modification.",
        reference_protocol_sha256=pilot.sha256(args.reference / "protocol.json"),
        reference_results_sha256=pilot.sha256(args.reference / "results.json"),
        reference_selection_sha256=pilot.sha256(args.reference / "selection.json"),
        canonical_source="src/myoji/_vendor/raulnet_v20.py (from MyoTorch)",
        canonical_source_sha256=SOURCE_SHA256,
        precision="float32; no mixed precision; same recipe as the prior token readouts",
        smoke="Training windows only; five optimizer steps, gradient/codebook/reload checks; discarded weights.",
    )
    # The library (including the vendored V20 module) is hashed through src/.
    sources = [
        str(path.relative_to(paths.REPO)) for path in (paths.REPO / "src").rglob("*.py")
    ] + [
        "hdemg/run_dpose_v20.py",
        "hdemg/run_dpose_pilot.py",
        "hdemg/run_dpose_within.py",
        "hdemg/report_dpose_v20.py",
        "src/myoji/_vendor/LICENSE-GPL-3.0",
    ]
    p["source_sha256"] = {
        name: pilot.sha256(pilot.source_path(name)) for name in sorted(sources)
    }
    p["reused_caches"] = {}
    p["reference_predictions_sha256"] = {}
    for subject in SUBJECTS:
        ref = args.reference / f"subject_{subject:02d}"
        source_protocol = pilot.sha256(ref / "protocol.json")
        with h5py.File(ref / "features.h5", "r") as f:
            assert f.attrs["complete"] and f.attrs["protocol_sha256"] == source_protocol
            assert np.array_equal(f["codebook"][:], book.numpy())
            for split, count in (("train", 64), ("val", 16), ("test", 16)):
                assert f[f"{split}/tokens"].shape == (count, 320, 100)
                assert f[f"{split}/targets"].shape == (count, 100, 60)
                assert np.all(f[f"{split}/subjects"][:] == subject)
        p["reused_caches"][str(subject)] = {
            "path": str((ref / "features.h5").resolve()),
            "source_protocol_sha256": source_protocol,
            "consumed_arrays_sha256": cache_digest(ref / "features.h5"),
        }
        for seed in SEEDS:
            path = ref / f"fits/tokens_{seed}/test_predictions.npz"
            p["reference_predictions_sha256"][str(path.relative_to(args.reference))] = (
                pilot.sha256(path)
            )
    pilot.write_json(args.root / "protocol.json", p)
    selection = json.loads((args.reference / "selection.json").read_text())
    selection["protocol_sha256"] = pilot.sha256(args.root / "protocol.json")
    pilot.write_json(args.root / "selection.json", selection)
    for subject in SUBJECTS:
        folder = args.root / f"subject_{subject:02d}"
        pilot.write_json(
            folder / "protocol.json",
            dict(
                p,
                subject=subject,
                campaign_protocol_sha256=pilot.sha256(args.root / "protocol.json"),
            ),
        )
        (folder / "features.h5").symlink_to(p["reused_caches"][str(subject)]["path"])
    print(
        json.dumps(
            {
                "fits": 27,
                "trainable_parameters": 704700,
                "protocol_sha256": pilot.sha256(args.root / "protocol.json"),
            }
        ),
        flush=True,
    )


def verify_campaign(args):
    p = json.loads((args.root / "protocol.json").read_text())
    selection = json.loads((args.root / "selection.json").read_text())
    assert selection["protocol_sha256"] == pilot.sha256(args.root / "protocol.json")
    for name, digest in p["source_sha256"].items():
        assert pilot.sha256(pilot.source_path(name)) == digest, name
    assert pilot.sha256(args.checkpoint) == p["checkpoint_sha256"]
    for name in ("protocol", "results", "selection"):
        assert (
            pilot.sha256(args.reference / f"{name}.json")
            == p[f"reference_{name}_sha256"]
        )
    for name, digest in p["reference_predictions_sha256"].items():
        assert pilot.sha256(args.reference / name) == digest, name
    return p


def verify_subject(root, p, subject):
    folder = root / f"subject_{subject:02d}"
    saved = json.loads((folder / "protocol.json").read_text())
    assert saved == dict(
        p,
        subject=subject,
        campaign_protocol_sha256=pilot.sha256(root / "protocol.json"),
    )
    ref = p["reused_caches"][str(subject)]
    assert (folder / "features.h5").resolve() == Path(ref["path"])
    with h5py.File(folder / "features.h5", "r") as f:
        assert (
            f.attrs["complete"]
            and f.attrs["protocol_sha256"] == ref["source_protocol_sha256"]
        )
    assert cache_digest(folder / "features.h5") == ref["consumed_arrays_sha256"]
    return folder


def verify_fit(folder, seed, book):
    """Gate completion on model, target statistics and immutable vocabulary."""
    out = folder / f"fits/tokens_{seed}"
    fit = json.loads((out / "fit.json").read_text())
    assert fit["complete"] and fit["parameters"] == 704700
    assert fit["protocol_sha256"] == pilot.sha256(folder / "protocol.json")
    assert fit["best_sha256"] == pilot.sha256(out / "best.pt")
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert state["protocol_sha256"] == fit["protocol_sha256"]
    assert state["arm"] == "tokens" and state["seed"] == seed
    assert state["protocol"] == json.loads((folder / "protocol.json").read_text())
    assert torch.equal(state["model"]["codebook"], book)
    assert all(torch.isfinite(v).all() for v in state["model"].values())
    assert np.isfinite(fit["validation_best"])
    with h5py.File(folder / "features.h5", "r") as f:
        target = torch.from_numpy(f["train/targets"][:, -50:]).reshape(-1, 60)
    assert torch.equal(state["mean"], target.mean(0))
    assert torch.equal(state["scale"], target.std(0, correction=0).clamp_min(1e-4))
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert last["protocol_sha256"] == fit["protocol_sha256"]
    assert torch.equal(last["model"]["codebook"], book)
    assert all(torch.isfinite(v).all() for v in last["model"].values())
    pilot.write_json(
        out / "integrity.json",
        {
            "verified": True,
            "best_sha256": fit["best_sha256"],
            "codebook_float32_sha256": book_digest(book),
            "train_only_target_statistics_exact": True,
        },
    )


def smoke(args, p, book):
    assert str(args.device).startswith("cuda") and torch.cuda.is_available()
    folder = verify_subject(args.root, p, SUBJECTS[0])
    train = pilot.load_split(folder, "train", "tokens")
    target = train["targets"][:, -50:].reshape(-1, 60)
    mean, scale = target.mean(0), target.std(0, correction=0).clamp_min(1e-4)
    torch.manual_seed(SEEDS[0])
    model = pilot.build("tokens", p, {}).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=p["training"]["lr"],
        weight_decay=p["training"]["weight_decay"],
    )
    original = model.token_projection[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    for step in range(5):
        ix = np.array([2 * step, 2 * step + 1])
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = pilot.predict_batch(model, train, ix, "tokens", args.device)
        assert output.shape == (2, 100, 60) and torch.isfinite(output).all()
        normalized = ((train["targets"][ix] - mean) / scale).to(args.device)
        loss = F.smooth_l1_loss(output[:, -50:], normalized[:, -50:])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, weight in model.named_parameters():
            assert weight.grad is not None and torch.isfinite(weight.grad).all(), name
        assert all(
            block.temporal.weight.grad.abs().sum() > 0 for block in model.temporal
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))
    assert not torch.equal(original, model.token_projection[0].weight)
    assert torch.equal(model.codebook.cpu(), book)
    model.eval()
    restored = pilot.build("tokens", p, {}).to(args.device)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    with torch.inference_mode():
        a = pilot.predict_batch(model, train, np.arange(2), "tokens", args.device)
        b = pilot.predict_batch(restored, train, np.arange(2), "tokens", args.device)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    receipt = {
        "passed": True,
        "protocol_sha256": pilot.sha256(args.root / "protocol.json"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "train_only": True,
        "input_shape": [2, 320, 100],
        "output_shape": [2, 100, 60],
        "trainable_parameters": 704700,
        "optimizer_steps": 5,
        "all_gradients_finite": True,
        "all_temporal_blocks_received_gradient": True,
        "frozen_codebook_exact": True,
        "checkpoint_state_reload_exact": True,
        "step_seconds": timings,
        "training_losses": losses,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "estimated_training_seconds_per_epoch_excluding_validation": float(
            np.median(timings[1:]) * 32
        ),
        "weights_discarded": True,
    }
    pilot.write_json(args.root / "smoke.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "smoke", "subject", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=paths.RESULTS / "dpose_within_v1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.root, args.reference = args.root.resolve(), args.reference.resolve()
    torch.set_num_threads(4)
    if args.mode == "freeze":
        freeze(args)
        return
    p = verify_campaign(args)
    book = torch.load(args.checkpoint, map_location="cpu", weights_only=True)[
        "state_dict"
    ]["quantizer.codebook"].float()
    assert book_digest(book) == p["codebook_float32_sha256"]
    with backend(book) as within:
        if args.mode == "smoke":
            smoke(args, p, book)
            return
        preflight = json.loads((args.root / "smoke.json").read_text())
        assert preflight["passed"] and preflight["protocol_sha256"] == pilot.sha256(
            args.root / "protocol.json"
        )
        if args.mode == "subject":
            assert 0 <= args.index < len(SUBJECTS)
            folder = verify_subject(args.root, p, SUBJECTS[args.index])
            subargs = copy.copy(args)
            subargs.root = folder
            for index, seed in enumerate(SEEDS):
                subargs.index = index
                pilot.fit(subargs)
                verify_fit(folder, seed, book)
                gc.collect()
                torch.cuda.empty_cache()
        else:
            for subject in SUBJECTS:
                folder = verify_subject(args.root, p, subject)
                for seed in SEEDS:
                    verify_fit(folder, seed, book)
            within.evaluate(args)
            from paper.hdemg.report_dpose_v20 import report

            report(args.root, args.reference, args.root / "report")


if __name__ == "__main__":
    main()
