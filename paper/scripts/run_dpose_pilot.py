"""Immutable DPose label-transfer pilot; all subjects were seen in pretraining."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import firwin, upfirdn
from torch.nn import functional as F

import paths
from myoji.decoders import CachedPoseDecoder, RawPoseDecoder
from myoji.learned.montage import hd_monopolar_montage
from myoji.learned.stream import Tokenizer, sha256
from myoji.preprocessing import calibration_normalize

ARMS = ("tokens", "continuous", "rms3", "raw")
SEEDS = (260907, 260908, 260909)
SPLITS = {"train": list(range(1, 10)), "val": [10, 11], "test": [12, 13]}
ENDS = np.arange(2039, 6000, 40)


def source_path(name):
    """Paper scripts are named relative to paper/, library files (src/...) relative to the repository."""
    return (paths.REPO if name.startswith("src/") else paths.ROOT) / name


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def protocol(root, data, checkpoint):
    if (root / "protocol.json").exists():
        raise FileExistsError(root / "protocol.json")
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert not frozen["pose_supervised"]
    p = {
        "dataset": "DPose HD, 13 subjects, 320 channels with paired 21x3 kinematics",
        "scope": "Supervised subject-held-out decoder pilot on a pretraining-seen cohort. All 13 subjects supplied unlabeled EMG to this frozen tokenizer. Not fully unseen-person or unseen-source generalization.",
        "checkpoint_sha256": sha256(checkpoint),
        "splits": SPLITS,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "window_samples": 6000,
        "max_windows_per_task": 3,
        "selection": "Three equally spaced nonoverlapping 3-second windows per subject/task, rounded integer starts, selected before features or fitting. A recording must be >=9 seconds; all EMG and scored targets must be finite.",
        "timing": {
            "source_hz": 2048,
            "encoder_hz": 2000,
            "up": 125,
            "down": 128,
            "fir_taps": 10241,
            "cutoff_hz": 800,
            "window": ["kaiser", 8.6],
            "causal": True,
            "zero_initial_history": True,
            "retained_delay_ms": 20,
            "labels": "At floor(output end sample * 128/125), no label lag fitting or delay compensation.",
        },
        "targets": "Stored points 1..20 minus stored point 0, flattened to 60 coordinates. Physical coordinate units and landmark anatomy unverified; report stored coordinate units, not mm or palm lengths.",
        "cache": "100 frames at ends 2039..5999. uint16 IDs, float16 continuous/rms features, float32 normalized raw and targets. Encode channels in fixed groups of 32. Checkpoint never updated.",
        "readout": {
            "kind": "tcn",
            "width": 64,
            "slots": 4,
            "heads": 4,
            "context_frames": 100,
            "output_dim": 60,
            "dropout": 0.1,
            "input_mode": "features_only",
        },
        "raw_arm": "Same causal readout, channel encoder initialized from scratch at width 64, hidden width 16; no pretrained weights. Different parameter counts are reported.",
        "inputs": "All 320 channels and fixed montage; no subject/task IDs, labels, kinematic initialization, or calibration side inputs. All arms share release prefix calibration of their EMG input.",
        "training": {
            "epochs": 40,
            "patience": 6,
            "batch_size": 2,
            "lr": 0.0003,
            "weight_decay": 0.01,
            "score_frames": 50,
            "loss": "Smooth L1 on train-only standardized coordinates, final 50 frames.",
            "selection": "Lowest validation macro-subject mean 3D point error, earliest epoch on ties.",
        },
        "evaluation": "Only after all 12 fits complete. Report both test subjects separately, macro-subject mean 3D point error and coordinate R2, mean/std over three seeds, train-mean constant baseline. Two test subjects cannot support population-level uncertainty claims.",
        "files": {p.name: sha256(p) for p in sorted(data.glob("Sub*_*.pkl"))},
        "source_sha256": {
            name: sha256(source_path(name))
            for name in ["scripts/run_dpose_pilot.py", "src/myoji/decoders.py"]
        },
    }
    assert len(p["files"]) == 26
    write_json(root / "protocol.json", p)
    return p


def resample(emg):
    taps = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
    return upfirdn(taps, emg.astype(np.float32), up=125, down=128, axis=-1)[
        ..., : emg.shape[-1] * 125 // 128
    ].astype(np.float32)


def target_at(kinematics, start):
    ix = ((start + ENDS) * 128) // 125
    points = kinematics[..., ix].transpose(2, 0, 1).astype(np.float32)
    return (points[:, 1:] - points[:, :1]).reshape(100, 60)


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def prepare(args):
    root = args.root
    p = json.loads((root / "protocol.json").read_text())
    assert sha256(args.checkpoint) == p["checkpoint_sha256"]
    out = root / "features.h5"
    if out.exists():
        raise FileExistsError(out)
    model = Tokenizer(args.checkpoint, device=args.device)
    before = state_hash(model.model)
    montage = hd_monopolar_montage().descriptor().numpy()
    manifests, inventory = [], []
    # Freeze source identities, alignment and all window starts before encoding.
    for subject in range(1, 14):
        paths = [args.data / f"Sub{subject}_{k}.pkl" for k in ("emg", "kinematics")]
        for path in paths:
            assert sha256(path) == p["files"][path.name]
        emg, kin = [joblib.load(path) for path in paths]
        assert set(emg) == set(kin)
        split = next(s for s, ids in SPLITS.items() if subject in ids)
        task_shapes = {}
        for task in sorted(emg, key=int):
            x, y = emg[task], kin[task]
            assert x.shape[0] == 320 and y.shape == (21, 3, x.shape[1]), (
                subject,
                task,
                x.shape,
                y.shape,
            )
            count = x.shape[1] * 125 // 128
            assert count >= 18000
            starts = np.linspace(0, count - 6000, 3).round().astype(int)
            assert np.all(np.diff(starts) >= 6000)
            task_shapes[task] = {
                "emg": list(x.shape),
                "kinematics": list(y.shape),
                "emg_dtype": str(x.dtype),
                "kinematics_dtype": str(y.dtype),
            }
            for start in starts:
                # Validity is a predeclared gate, never based on predicted error.
                source_start, source_end = (
                    int(start * 128 // 125),
                    int((start + 6000) * 128 // 125),
                )
                if (
                    not np.isfinite(x[:, source_start:source_end]).all()
                    or not np.isfinite(target_at(y, int(start))).all()
                ):
                    raise ValueError(
                        f"Nonfinite selected window: {subject}/{task}/{start}"
                    )
                manifests.append(
                    {
                        "subject": subject,
                        "split": split,
                        "task": task,
                        "start": int(start),
                    }
                )
        inventory.append({"subject": subject, "tasks": task_shapes})
        print(
            json.dumps({"inventory_subject": subject, "tasks": len(task_shapes)}),
            flush=True,
        )
        del emg, kin, x, y
        gc.collect()
    write_json(
        root / "selection.json",
        {
            "windows": manifests,
            "inventory": inventory,
            "protocol_sha256": sha256(root / "protocol.json"),
        },
    )
    with h5py.File(out, "w") as f:
        f.attrs["complete"] = False
        f.attrs["protocol_sha256"] = sha256(root / "protocol.json")
        f["codebook"] = model.model.quantizer.codebook.cpu().numpy()
        f["descriptors"] = montage
        for split, split_subjects in SPLITS.items():
            rows = [r for r in manifests if r["split"] == split]
            g = f.create_group(split)
            n = len(rows)
            for name, shape, dtype in [
                ("tokens", (n, 320, 100), "u2"),
                ("continuous", (n, 320, 100, 256), "f2"),
                ("rms3", (n, 320, 100, 3), "f2"),
                ("raw", (n, 320, 6000), "f4"),
                ("targets", (n, 100, 60), "f4"),
            ]:
                g.create_dataset(name, shape=shape, dtype=dtype)
            g["subjects"] = np.array([r["subject"] for r in rows])
            g["starts"] = np.array([r["start"] for r in rows])
            g["tasks"] = np.array([int(r["task"]) for r in rows])
            for subject in split_subjects:
                emg = joblib.load(args.data / f"Sub{subject}_emg.pkl")
                kin = joblib.load(args.data / f"Sub{subject}_kinematics.pkl")
                for task in sorted(emg, key=int):
                    signal = resample(emg[task])
                    for i, row in enumerate(rows):
                        if row["subject"] != subject or row["task"] != task:
                            continue
                        start = row["start"]
                        raw = torch.from_numpy(signal[:, start : start + 6000]).to(
                            args.device
                        )[None]
                        x, _, _ = calibration_normalize(raw, 1000)
                        g["raw"][i] = x[0].cpu().numpy()
                        g["targets"][i] = target_at(kin[task], start)
                        for c in range(0, 320, 32):
                            v = x[:, c : c + 32]
                            encoded = model.model(
                                v,
                                sensor="hd",
                                channel_descriptors=torch.zeros(
                                    32, model.config.geometry_dim, device=args.device
                                ),
                                include_spatial=False,
                            )
                            g["tokens"][i, c : c + 32] = (
                                encoded.tokens[0, :, -100:].cpu().numpy()
                            )
                            g["continuous"][i, c : c + 32] = (
                                encoded.continuous[0, :, -100:].half().cpu().numpy()
                            )
                            patches = v[
                                ...,
                                torch.tensor(ENDS, device=args.device)[:, None]
                                - torch.arange(959, -1, -1, device=args.device),
                            ]
                            features = torch.stack(
                                (
                                    patches.square().mean(-1).sqrt(),
                                    patches.abs().mean(-1),
                                    patches.diff(dim=-1).abs().mean(-1),
                                ),
                                -1,
                            ).log1p()
                            g["rms3"][i, c : c + 32] = features[0].half().cpu().numpy()
                        if i == 0:
                            again = (
                                model.model(
                                    x[:, :32],
                                    sensor="hd",
                                    channel_descriptors=torch.zeros(
                                        32,
                                        model.config.geometry_dim,
                                        device=args.device,
                                    ),
                                    include_spatial=False,
                                )
                                .tokens[0, :, -100:]
                                .cpu()
                                .numpy()
                            )
                            assert np.array_equal(again, g["tokens"][i, :32])
                        print(
                            json.dumps(
                                {
                                    "cached": split,
                                    "subject": subject,
                                    "task": task,
                                    "index": i,
                                }
                            ),
                            flush=True,
                        )
                    del signal
                del emg, kin
                gc.collect()
                f.flush()
        assert state_hash(model.model) == before
        f.attrs["complete"] = True
    write_json(
        root / "cache_receipt.json",
        {
            "complete": True,
            "windows": len(manifests),
            "split_windows": {
                s: sum(r["split"] == s for r in manifests) for s in SPLITS
            },
            "frozen_state_sha256": before,
            "protocol_sha256": sha256(root / "protocol.json"),
            "selection_sha256": sha256(root / "selection.json"),
            "cache_bytes": out.stat().st_size,
            "runtime": model.metadata(),
        },
    )


def build(arm, p, config):
    dims = {"tokens": 256, "continuous": 256, "rms3": 3, "raw": 64}
    kw = dict(p["readout"], embedding_dim=dims[arm], geometry_dim=14, channels=320)
    if arm == "raw":
        cfg = dict(config, embedding_dim=64, channel_hidden_dim=16)
        return RawPoseDecoder(raw_config=cfg, **kw)
    return CachedPoseDecoder(**kw)


def load_split(root, split, arm):
    with h5py.File(root / "features.h5", "r") as f:
        assert f.attrs["complete"]
        g = f[split]
        return {
            "features": torch.from_numpy(
                g[arm][:].astype(
                    np.int64
                    if arm == "tokens"
                    else np.float32
                    if arm == "rms3"
                    else g[arm].dtype
                )
            ),
            "targets": torch.from_numpy(g["targets"][:]),
            "subjects": g["subjects"][:],
            "book": torch.from_numpy(f["codebook"][:]),
            "descriptors": torch.from_numpy(f["descriptors"][:]),
        }


def predict_batch(model, data, ix, arm, device):
    x = (
        data["features"][ix].to(device).float()
        if arm != "tokens"
        else data["features"][ix].to(device)
    )
    if arm == "tokens":
        x = data["book"].to(device)[x]
    b = len(ix)
    centre = torch.zeros(b, 320, 1, device=device)
    scale = torch.ones_like(centre)
    return model(x, data["descriptors"].to(device), centre, scale)


@torch.inference_mode()
def evaluate_data(
    model, data, arm, mean, scale, device, batch=2, save_predictions=False
):
    model.eval()
    pred = []
    for i in range(0, len(data["subjects"]), batch):
        ix = np.arange(i, min(i + batch, len(data["subjects"])))
        pred.append(
            (predict_batch(model, data, ix, arm, device).cpu() * scale + mean)[:, -50:]
        )
    pred = torch.cat(pred).numpy()
    target = data["targets"][:, -50:].numpy()
    scores = {}
    for subject in np.unique(data["subjects"]):
        ix = data["subjects"] == subject
        error = pred[ix] - target[ix]
        denom = ((target[ix] - target[ix].mean(axis=(0, 1))) ** 2).sum()
        scores[str(subject)] = {
            "mean_3d_point_error": float(
                np.linalg.norm(error.reshape(-1, 20, 3), axis=-1).mean()
            ),
            "coordinate_r2": float(1 - (error**2).sum() / denom) if denom > 0 else None,
        }
    result = {
        "macro_subject_error": float(
            np.mean([v["mean_3d_point_error"] for v in scores.values()])
        ),
        "subjects": scores,
    }
    if save_predictions:
        return result, pred, target
    return result


def fit(args):
    p = json.loads((args.root / "protocol.json").read_text())
    arm = ARMS[args.index // 3]
    seed = SEEDS[args.index % 3]
    out = args.root / "fits" / f"{arm}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fit.json").exists():
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = args.device
    cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"]
    assert sha256(args.checkpoint) == p["checkpoint_sha256"]
    model = build(arm, p, cfg).to(device)
    train = load_split(args.root, "train", arm)
    val = load_split(args.root, "val", arm)
    targets = train["targets"][:, -50:].reshape(-1, 60)
    mean = targets.mean(0)
    scale = targets.std(0, correction=0).clamp_min(1e-4)
    opt = torch.optim.AdamW(
        [v for v in model.parameters() if v.requires_grad],
        lr=p["training"]["lr"],
        weight_decay=p["training"]["weight_decay"],
    )
    best = float("inf")
    stale = 0
    history = []
    start_epoch = 0
    started = time.time()
    if (out / "last.pt").exists():
        last = torch.load(out / "last.pt", map_location=device, weights_only=False)
        assert last["protocol_sha256"] == sha256(args.root / "protocol.json")
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["optimizer"])
        best = last["best"]
        stale = last["stale"]
        history = last["history"]
        start_epoch = last["epoch"]
    for epoch in range(start_epoch, p["training"]["epochs"]):
        model.train()
        torch.manual_seed(seed + epoch)
        order = np.random.default_rng(seed + epoch).permutation(len(train["subjects"]))
        losses = []
        for i in range(0, len(order), p["training"]["batch_size"]):
            ix = order[i : i + p["training"]["batch_size"]]
            pred = predict_batch(model, train, ix, arm, device)
            target = ((train["targets"][ix] - mean) / scale).to(device)
            loss = F.smooth_l1_loss(pred[:, -50:], target[:, -50:])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_data(model, val, arm, mean, scale, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation": metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps({"arm": arm, "seed": seed, **row}), flush=True)
        if metrics["macro_subject_error"] < best:
            best = metrics["macro_subject_error"]
            stale = 0
            save(
                out / "best.pt",
                {
                    "model": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "mean": mean,
                    "scale": scale,
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "config": cfg,
                    "protocol": p,
                    "protocol_sha256": sha256(args.root / "protocol.json"),
                },
            )
        else:
            stale += 1
        save(
            out / "last.pt",
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "stale": stale,
                "history": history,
                "protocol_sha256": sha256(args.root / "protocol.json"),
            },
        )
        if stale >= p["training"]["patience"]:
            break
    write_json(
        out / "fit.json",
        {
            "complete": True,
            "arm": arm,
            "seed": seed,
            "parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
            "validation_best": best,
            "epochs": len(history),
            "history": history,
            "best_sha256": sha256(out / "best.pt"),
            "protocol_sha256": sha256(args.root / "protocol.json"),
            "scope": p["scope"],
        },
    )


def evaluate(args):
    p = json.loads((args.root / "protocol.json").read_text())
    results = []
    for arm in ARMS:
        for seed in SEEDS:
            path = args.root / "fits" / f"{arm}_{seed}" / "fit.json"
            assert path.exists(), path
            f = json.loads(path.read_text())
            assert f["complete"] and f["protocol_sha256"] == sha256(
                args.root / "protocol.json"
            )
    baseline = None
    for arm in ARMS:
        data = load_split(args.root, "test", arm)
        for seed in SEEDS:
            out = args.root / "fits" / f"{arm}_{seed}"
            state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
            model = build(arm, p, state["config"]).to(args.device)
            model.load_state_dict(state["model"])
            metrics, pred, target = evaluate_data(
                model,
                data,
                arm,
                state["mean"],
                state["scale"],
                args.device,
                save_predictions=True,
            )
            np.savez_compressed(
                out / "test_predictions.npz",
                prediction=pred,
                target=target,
                subject=data["subjects"],
            )
            results.append({"arm": arm, "seed": seed, **metrics})
            print(json.dumps(results[-1]), flush=True)
            if baseline is None:
                errors = np.linalg.norm(
                    (target - state["mean"].numpy()).reshape(len(target), 50, 20, 3),
                    axis=-1,
                ).mean((1, 2))
                baseline = {
                    str(s): float(errors[data["subjects"] == s].mean())
                    for s in np.unique(data["subjects"])
                }
            del model
        del data
        gc.collect()
    summary = {
        arm: {
            "mean": float(
                np.mean([r["macro_subject_error"] for r in results if r["arm"] == arm])
            ),
            "seed_sd": float(
                np.std(
                    [r["macro_subject_error"] for r in results if r["arm"] == arm],
                    ddof=1,
                )
            ),
        }
        for arm in ARMS
    }
    write_json(
        args.root / "results.json",
        {
            "complete": True,
            "scope": p["scope"],
            "units": "stored kinematic coordinate units; physical unit unverified",
            "results": results,
            "summary": summary,
            "constant_train_mean_baseline": baseline,
            "protocol_sha256": sha256(args.root / "protocol.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["protocol", "prepare", "fit", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=paths.HDEMG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == "protocol":
        protocol(args.root, args.data, args.checkpoint)
    else:
        globals()[args.mode](args)


if __name__ == "__main__":
    main()
