"""Capacity- and optimization-matched HD-EMG token/continuous/V19 experiment."""

import argparse
import gc
import hashlib
import inspect
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from paper.receivers.hdemg_matched_core import (
    ARMS,
    COUNTS,
    EPOCHS,
    SEEDS,
    UPDATES_PER_EPOCH,
    build,
    cached_forward,
    sample_windows,
    schedule_digest,
    update,
)
from paper.receivers.hdemg_raulnet_model import dual_input
from myoji._vendor.channel_tokenization import GeometryAwareChannelPool
from paper.common.paths import REPO, RESULTS


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def data_digest(path, native=False):
    h = hashlib.sha256()
    names = [
        f"{s}/{k}"
        for s in ("train", "val", "test")
        for k in (
            ("raw", "targets", "subjects", "tasks", "starts")
            if native
            else ("tokens", "continuous", "targets", "subjects", "tasks", "starts")
        )
    ]
    if not native:
        names += ["codebook", "descriptors"]
    with h5py.File(path, "r") as f:
        assert f.attrs["complete"]
        for name in names:
            ds = f[name]
            h.update(f"{name}:{ds.shape}:{ds.dtype}".encode())
            for i in range(0, len(ds), 4):
                h.update(ds[i : i + 4].tobytes())
    return h.hexdigest()


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError(args.root / "protocol.json")
    base = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        base / "receivers/hdemg_matched_core.py",
        base / "receivers/hdemg_raulnet_model.py",
        base / "vendor/raulnet_v19.py",
        REPO / "src/myoji/decoders.py",
        base / "receivers/report_hdemg_matched.py",
    ]
    paths = sorted(set(paths + list((REPO / "src/myoji").rglob("*.py"))))
    p = {
        "scope": "Capacity- and optimization-matched personalized same-session/task study. Tokenizer pretraining excluded the validation and test windows. Match trainable capacity (within 1%), target exposure, updates and recipe; not input context, total system parameters, pretraining or FLOPs.",
        "subjects": list(range(1, 10)),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "fits": 135,
        "parameters": COUNTS,
        "readouts": "TCN width404, transformer width468/layers2; four slots/heads, 100 frames. V19 q32 native layers with dropout changed to .1. Token/continuous pairs have identical architecture and seed initialization.",
        "training": {
            "epochs": 150,
            "updates_per_epoch": 100,
            "windows_per_update": 2,
            "scored_endpoints_per_window": 50,
            "effective_endpoint_batch": 100,
            "updates_per_fit": 15000,
            "endpoint_presentations_per_fit": 1500000,
            "sampling": "Two windows sampled with replacement per update, all final 50 targets from each. Exact schedule per seed/epoch shared by every arm/subject; repeated-window targets retain multiplicity.",
            "loss": "Smooth L1, coordinates standardized with subject-training-only mean/std; std floor1e-4, population std.",
            "optimizer": "AdamW lr peak3e-4, betas(.9,.999), eps1e-8, weight_decay.01, amsgradFalse, fusedFalse; global gradient norm clip1.",
            "schedule": "500-update linear warmup; cosine decay to3e-6 at update14999. No SWA, early stopping or input noise. Dropout .1 throughout configurable dropout layers.",
            "precision": "float16 autocast + GradScaler, initial scale1024. Retry overflow with same samples/dropout RNG, count successful updates exactly. Log retries.",
            "validation": "All 800 endpoints each epoch, exactly150 opportunities; earliest lowest mean3D checkpoint. No post-fit extra candidates.",
            "microbatch": "Cached heads: two windows/one100-target loss. V19: two50-target microbatches with loss weighted1/2 before one optimizer update. No batch-statistic normalization in V19.",
        },
        "remaining_differences": "V19 native raw+20Hz input156ms versus token/continuous history and longer frozen-encoder receptive field/prefix calibration/20ms retained delay. Spatial pooling and normalization differ. Equal parameters/updates is not equal compute; measured GPU time/memory reported, tokenizer costs separate.",
        "sampling_sha256": {str(seed): schedule_digest(seed) for seed in SEEDS},
        "source_sha256": {str(path.relative_to(REPO)): sha(path) for path in paths},
        "external_source_sha256": {
            inspect.getfile(GeometryAwareChannelPool): sha(
                inspect.getfile(GeometryAwareChannelPool)
            )
        },
        "references": {
            "within_protocol": sha(args.reference / "protocol.json"),
            "native_protocol": sha(args.native / "protocol.json"),
        },
        "cache_digests": {},
    }
    for subject in p["subjects"]:
        token = args.reference / f"subject_{subject:02d}/features.h5"
        native = args.native / f"subject_{subject:02d}/native.h5"
        with h5py.File(token, "r") as a, h5py.File(native, "r") as b:
            for split in ("train", "val", "test"):
                assert np.array_equal(
                    a[f"{split}/targets"][:, -50:], b[f"{split}/targets"][:]
                )
                for key in ("subjects", "tasks", "starts"):
                    assert np.array_equal(
                        a[f"{split}/{key}"][:], b[f"{split}/{key}"][:]
                    )
        p["cache_digests"][str(subject)] = {
            "cached": data_digest(token),
            "native": data_digest(native, True),
        }
        print(json.dumps({"bound_subject": subject}), flush=True)
    write(args.root / "protocol.json", p)
    selection = json.loads((args.reference / "selection.json").read_text())
    selection["protocol_sha256"] = sha(args.root / "protocol.json")
    write(args.root / "selection.json", selection)


def load_data(args, subject, split, arm):
    path = (
        (args.native / f"subject_{subject:02d}/native.h5")
        if arm == "v19"
        else (args.reference / f"subject_{subject:02d}/features.h5")
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["complete"]
        g = f[split]
        n = len(g["subjects"])
        target = g["targets"][:] if arm == "v19" else g["targets"][:, -50:]
        data = {"targets": torch.from_numpy(target).to(args.device)}
        if arm == "v19":
            data["features"] = torch.empty(n, 50, 2, 320, 320, device=args.device)
            for i in range(n):
                data["features"][i] = dual_input(
                    torch.from_numpy(g["raw"][i]).to(args.device)
                )
        else:
            key = "tokens" if arm.endswith("tokens") else "continuous"
            data["features"] = torch.from_numpy(
                g[key][:].astype(np.int64 if key == "tokens" else np.float16)
            ).to(args.device)
            data["book"] = torch.from_numpy(f["codebook"][:]).to(args.device)
            data["descriptors"] = torch.from_numpy(f["descriptors"][:]).to(args.device)
    return data


@torch.inference_mode()
def predict(model, data, arm, mean, scale):
    model.eval()
    parts = []
    for i in range(0, len(data["targets"]), 2):
        ix = np.arange(i, min(i + 2, len(data["targets"])))
        with torch.autocast("cuda", dtype=torch.float16):
            if arm == "v19":
                pred = torch.stack([model(data["features"][j]) for j in ix])
            else:
                pred = cached_forward(model, data, ix, arm)
        parts.append((pred.float() * scale + mean).cpu().numpy())
    pred = np.concatenate(parts)
    assert np.isfinite(pred).all()
    target = data["targets"].cpu().numpy()
    error = float(np.linalg.norm((pred - target).reshape(-1, 20, 3), axis=-1).mean())
    return error, pred, target


def fit(args, subject, arm, seed, train, val):
    out = args.root / f"subject_{subject:02d}/fits/{arm}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha(args.root / "protocol.json")
    if (out / "fit.json").exists():
        receipt = json.loads((out / "fit.json").read_text())
        assert receipt["complete"] and receipt["protocol_sha256"] == protocol_hash
        return
    torch.manual_seed(seed)
    model = build(arm).to(args.device)
    initial = hashlib.sha256(
        b"".join(
            v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
        )
    ).hexdigest()
    mean = train["targets"].mean((0, 1))
    scale = train["targets"].std((0, 1), correction=0).clamp_min(1e-4)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        fused=False,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    start_epoch = 0
    best = float("inf")
    history = []
    retries = 0
    if (out / "last.pt").exists():
        state = torch.load(
            out / "last.pt", map_location=args.device, weights_only=False
        )
        assert state["protocol_sha256"] == protocol_hash
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        best = state["best"]
        history = state["history"]
        retries = state["retries"]
        del state
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, EPOCHS):
        losses = []
        for j, windows in enumerate(sample_windows(seed, epoch)):
            value, extra = update(
                model,
                train,
                windows,
                arm,
                mean,
                scale,
                opt,
                scaler,
                epoch * UPDATES_PER_EPOCH + j,
                seed,
            )
            losses.append(value)
            retries += extra
        error, _, _ = predict(model, val, arm, mean, scale)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_3d": error,
                "seconds_this_invocation": time.time() - started,
                "amp_retries_total": retries,
            }
        )
        print(
            json.dumps({"arm": arm, "subject": subject, "seed": seed, **history[-1]}),
            flush=True,
        )
        if error < best:
            best = error
            save(
                out / "best.pt",
                {
                    "model": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "mean": mean.cpu(),
                    "scale": scale.cpu(),
                    "epoch": epoch + 1,
                    "protocol_sha256": protocol_hash,
                },
            )
        save(
            out / "last.pt",
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "history": history,
                "retries": retries,
                "protocol_sha256": protocol_hash,
            },
        )
    write(
        out / "fit.json",
        {
            "complete": True,
            "subject": subject,
            "arm": arm,
            "seed": seed,
            "parameters": COUNTS[arm],
            "successful_updates": 15000,
            "endpoint_presentations": 1500000,
            "history": history,
            "initial_state_sha256": initial,
            "sampling_sha256": schedule_digest(seed),
            "protocol_sha256": protocol_hash,
            "best_sha256": sha(out / "best.pt"),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "amp_retries": retries,
            "target_stats_sha256": hashlib.sha256(
                mean.detach().cpu().numpy().tobytes()
                + scale.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "seconds_this_invocation": time.time() - started,
        },
    )
    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()


def smoke(args):
    rows = []
    for arm in ARMS:
        torch.manual_seed(42)
        model = build(arm).to(args.device)
        data = {"targets": torch.randn(2, 50, 60, device=args.device)}
        if arm == "v19":
            data["features"] = torch.randn(2, 50, 2, 320, 320, device=args.device)
        else:
            data.update(
                features=torch.randint(512, (2, 320, 100), device=args.device)
                if arm.endswith("tokens")
                else torch.randn(2, 320, 100, 256, device=args.device).half(),
                book=torch.randn(512, 256, device=args.device),
                descriptors=torch.zeros(320, 14, device=args.device),
            )
        mean = torch.zeros(60, device=args.device)
        scale = torch.ones_like(mean)
        opt = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.01, fused=False
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        times = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(4):
            torch.cuda.synchronize()
            start = time.time()
            _loss, _retries = update(
                model, data, np.array([0, 1]), arm, mean, scale, opt, scaler, step, 42
            )
            torch.cuda.synchronize()
            times.append(time.time() - start)
        error, _, _ = predict(model, data, arm, mean, scale)
        row = {
            "arm": arm,
            "parameters": COUNTS[arm],
            "seconds_per_update": float(np.mean(times[1:])),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "finite_inference": bool(np.isfinite(error)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, data, opt, scaler
        gc.collect()
        torch.cuda.empty_cache()
    write(
        args.root / "smoke/receipt.json",
        {
            "passed": True,
            "synthetic_only": True,
            "gpu": torch.cuda.get_device_name(),
            "profiles": rows,
        },
    )


def evaluate(args):
    ph = sha(args.root / "protocol.json")
    protocol = json.loads((args.root / "protocol.json").read_text())
    for subject in range(1, 10):
        for native, path in [
            (False, args.reference / f"subject_{subject:02d}/features.h5"),
            (True, args.native / f"subject_{subject:02d}/native.h5"),
        ]:
            assert (
                data_digest(path, native)
                == protocol["cache_digests"][str(subject)][
                    "native" if native else "cached"
                ]
            )
    for subject in range(1, 10):
        for arm in ARMS:
            for seed in SEEDS:
                out = args.root / f"subject_{subject:02d}/fits/{arm}_{seed}"
                r = json.loads((out / "fit.json").read_text())
                assert (
                    r["complete"]
                    and r["protocol_sha256"] == ph
                    and r["best_sha256"] == sha(out / "best.pt")
                    and r["successful_updates"] == 15000
                )
    for subject in range(1, 10):
        with h5py.File(args.reference / f"subject_{subject:02d}/features.h5", "r") as f:
            np.savez_compressed(
                args.root / f"subject_{subject:02d}/baseline_training_targets.npz",
                target=f["train/targets"][:, -50:],
            )
        for arm in ARMS:
            data = load_data(args, subject, "test", arm)
            for seed in SEEDS:
                out = args.root / f"subject_{subject:02d}/fits/{arm}_{seed}"
                state = torch.load(
                    out / "best.pt", map_location="cpu", weights_only=False
                )
                model = build(arm).to(args.device)
                model.load_state_dict(state["model"])
                error, pred, target = predict(
                    model,
                    data,
                    arm,
                    state["mean"].to(args.device),
                    state["scale"].to(args.device),
                )
                np.savez_compressed(
                    out / "test_predictions.npz",
                    prediction=pred,
                    target=target,
                    subject=np.full(len(pred), subject),
                )
                write(out / "test.json", {"mean_3d_error": error})
                del model, state
            del data
            torch.cuda.empty_cache()
    from paper.receivers.report_hdemg_matched import report

    report(args.root, args.root / "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "smoke", "fit", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=RESULTS / "hdemg_within_v1")
    parser.add_argument("--native", type=Path, default=RESULTS / "hdemg_raulnet_v1")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "freeze":
        freeze(args)
        return
    if args.mode == "smoke":
        smoke(args)
        return
    p = json.loads((args.root / "protocol.json").read_text())
    for name, h in p["source_sha256"].items():
        assert sha(REPO / name) == h
    for name, h in p["external_source_sha256"].items():
        assert sha(name) == h
    if args.mode == "fit":
        subject = args.index // len(ARMS) + 1
        arm = ARMS[args.index % len(ARMS)]
        assert subject in p["subjects"]
        native = arm == "v19"
        path = (
            (args.native / f"subject_{subject:02d}/native.h5")
            if native
            else (args.reference / f"subject_{subject:02d}/features.h5")
        )
        assert (
            data_digest(path, native)
            == p["cache_digests"][str(subject)]["native" if native else "cached"]
        )
        train = load_data(args, subject, "train", arm)
        val = load_data(args, subject, "val", arm)
        for seed in SEEDS:
            fit(args, subject, arm, seed, train, val)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
