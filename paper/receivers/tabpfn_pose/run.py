"""Frozen SPAR IDs -> spatial/causal summaries -> train-only PCA -> TabPFN 3.5 pose."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import h5py
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from paper.common import paths
from ._support import Monitor, sha256, validate_separation

DEFAULT_CACHE = paths.RESULTS / "hdemg_spar_simple_pilot_v1"
SEEDS = [260911, 260912, 260913]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def spatial_history(tokens, codebook, mask, score_start=50, spans=(1, 5, 25, 100)):
    """Preserve channel/feature slots; aggregate only the current and previous frames."""
    n, channels, frames = tokens.shape
    if mask.shape != (channels, codebook.shape[1]) or tokens.max() >= len(codebook):
        raise ValueError("Invalid codebook/mask/token geometry")
    x = (codebook[tokens] * mask[None, :, None, :]).transpose(0, 2, 1, 3)
    x = x.reshape(n, frames, -1)[..., mask.reshape(-1).astype(bool)]
    cumulative = np.concatenate([np.zeros_like(x[:, :1], dtype=np.float64),
                                 np.cumsum(x, axis=1, dtype=np.float64)], axis=1)
    ends = np.arange(score_start + 1, frames + 1)
    summaries = []
    for span in spans:
        starts = np.maximum(0, ends - span)
        summaries.append(((cumulative[:, ends] - cumulative[:, starts]) /
                          (ends - starts)[None, :, None]).astype(np.float32))
    return np.concatenate(summaries, axis=-1).reshape(n * len(ends), -1)


def error_mm(prediction, target):
    if prediction.shape != target.shape or prediction.shape[-1] != 60:
        raise ValueError("Expected matching 60-coordinate pose arrays")
    return float(np.linalg.norm((prediction - target).reshape(-1, 20, 3), axis=-1).mean())


def scores(prediction, target, tasks):
    return dict(error_mm=error_mm(prediction, target),
                per_task_mm={str(int(task)): error_mm(prediction[tasks == task], target[tasks == task])
                             for task in np.unique(tasks)},
                per_window_mm=np.linalg.norm((prediction - target).reshape(len(tasks), 50, 20, 3), axis=-1).mean((1, 2)).tolist())


def prepare(args):
    root, source = args.root, args.cache_root
    root.mkdir(parents=True, exist_ok=False)
    p = json.loads((source / "protocol.json").read_text())
    receipt = json.loads((source / "cache-receipt.json").read_text())
    if not receipt["complete"] or sha256(source / "features.h5") != receipt["sha256"]:
        raise ValueError("Frozen cache checksum mismatch")
    if sha256(source / "protocol.json") != receipt["protocol_sha256"]:
        raise ValueError("Frozen protocol checksum mismatch")
    validate_separation(p["windows"])
    arrays, metadata, raw_features = {}, {}, {}
    with h5py.File(source / "features.h5", "r") as f:
        if not f.attrs["complete"]:
            raise ValueError("Incomplete feature cache")
        book, mask = f["codebook"][:], f["mask"][:]
        for split, expected in (("train", 64), ("val", 16), ("test", 16)):
            g = f[split]
            if g["tokens"].shape != (expected, 320, 100) or not np.all(g["subjects"][:] == 1):
                raise ValueError("Expected the full Subject-1 V20 split")
            raw_features[split] = spatial_history(g["tokens"][:], book, mask)
            arrays[f"y_{split}"] = g["targets"][:, -50:].reshape(-1, 60)
            arrays[f"tasks_{split}"] = g["tasks"][:]
            arrays[f"starts_{split}"] = g["starts"][:]
            metadata[split] = dict(windows=expected, pose_frames=expected * 50,
                                   tasks=g["tasks"][:].tolist(), starts=g["starts"][:].tolist())
    # This downstream projection sees only token-derived training inputs, never targets or held-out inputs.
    projection = PCA(n_components=args.components, svd_solver="randomized", random_state=260921)
    arrays["X_train"] = projection.fit_transform(raw_features["train"]).astype(np.float32)
    for split in ("val", "test"):
        arrays[f"X_{split}"] = projection.transform(raw_features[split]).astype(np.float32)
    joblib.dump(projection, root / "input_pca.joblib")
    np.savez_compressed(root / "data.npz", **arrays)
    config = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), status="preparing", subject=1,
        seeds=SEEDS, checkpoint="tabpfn-v3.5-20260909.safetensors", n_estimators=args.n_estimators,
        components=args.components, pca_variance_fraction=float(projection.explained_variance_ratio_.sum()),
        input_features_before_pca=raw_features["train"].shape[1], metadata=metadata,
        source=str(source.resolve()), source_protocol_sha256=sha256(source / "protocol.json"),
        source_cache_sha256=receipt["sha256"], data_sha256=sha256(root / "data.npz"),
        pca_sha256=sha256(root / "input_pca.joblib"), frozen_tokenizer=p["tokenizer"],
        feature=p["feature"], input="All 320 frozen token streams, lookup through frozen 18-D SPAR codebook, "
        "structural mask, per-channel means over 1/5/25/100 available causal frames; training-only PCA.",
        target="All 60 coordinates: wrist-relative landmarks 1..20 in mm; same final 50 frames/window as V20.",
        readout="60 independent scalar TabPFN 3.5 regressors; three fixed ensemble seeds; no gradient fine-tuning.",
        scope="Exploratory Subject-1 same-session chronological split; not unseen-person/movement evidence. "
        "Same data/endpoints as V20, different decoder and summarized temporal context; no test-driven tuning.",
    )
    write(root / "config.json", config)
    # Fixed untuned linear control on precisely the same downstream feature table.
    baseline = Ridge(alpha=1.0).fit(arrays["X_train"], arrays["y_train"])
    controls = {}
    for split in ("val", "test"):
        y = arrays[f"y_{split}"].reshape(-1, 50, 60)
        pred = baseline.predict(arrays[f"X_{split}"]).reshape(y.shape)
        mean = np.broadcast_to(arrays["y_train"].mean(0), y.shape).copy()
        controls[split] = dict(ridge=scores(pred, y, arrays[f"tasks_{split}"]),
                               training_mean=scores(mean, y, arrays[f"tasks_{split}"]))
        np.savez_compressed(root / f"baselines_{split}.npz", ridge=pred, mean=mean, target=y,
                            tasks=arrays[f"tasks_{split}"], starts=arrays[f"starts_{split}"])
    references = {}
    target = arrays["y_test"].reshape(16, 50, 60)
    for folder, label in ((source, "SPAR V20"), (paths.RESULTS / "hdemg_v20_v1/subject_01", "Spectral V20")):
        for arm in (("tokens", "continuous") if label == "SPAR V20" else ("tokens",)):
            for seed in SEEDS:
                file = folder / "fits" / f"{arm}_{seed}" / "test_predictions.npz"
                if not file.exists():
                    continue
                with np.load(file) as z:
                    np.testing.assert_array_equal(z["target"], target)
                    if "tasks" in z:
                        np.testing.assert_array_equal(z["tasks"], arrays["tasks_test"])
                    pred = z["prediction"] if "prediction" in z else z["pred"]
                    references[f"{label} {arm} {seed}"] = dict(scores(pred, target, arrays["tasks_test"]),
                                                               prediction_sha256=sha256(file))
    write(root / "baselines.json", controls)
    write(root / "references.json", references)
    config["status"] = "prepared"
    write(root / "config.json", config)
    print(json.dumps(dict(stage="prepared", pca_variance_fraction=config["pca_variance_fraction"],
                          shapes={k: list(v.shape) for k, v in arrays.items() if k.startswith("X_")}), indent=2), flush=True)


def run(args):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4)
    torch.manual_seed(260911)
    root = args.root
    c = json.loads((root / "config.json").read_text())
    if sha256(root / "data.npz") != c["data_sha256"]:
        raise ValueError("Prepared dataset changed")
    if args.model_path.name != c["checkpoint"]:
        raise ValueError("Expected the pinned TabPFN 3.5 checkpoint")
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA unavailable; refusing to occupy GPU allocation on CPU")
    with np.load(root / "data.npz", allow_pickle=False) as loaded:
        d = dict(loaded)
    c["runtime"] = dict(torch=torch.__version__, tabpfn=importlib.metadata.version("tabpfn"),
                        sklearn=importlib.metadata.version("scikit-learn"), platform=platform.platform(),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        checkpoint_sha256=sha256(args.model_path), code_sha256=sha256(Path(__file__)))
    if args.mode == "fit":
        c["status"] = "running"
        c["started_utc"] = datetime.now(timezone.utc).isoformat()
    write(root / "config.json", c)
    queries = d["X_val"] if args.mode == "benchmark" else np.concatenate([d["X_val"], d["X_test"]])
    with Monitor(root / ("benchmark-usage.jsonl" if args.mode == "benchmark" else "gpu-usage.jsonl")) as monitor:
        for seed in (SEEDS[:1] if args.mode == "benchmark" else SEEDS):
            for coordinate in (range(1) if args.mode == "benchmark" else range(60)):
                output = root / "coordinates" / f"{seed}_{coordinate:02d}.npz"
                if args.mode != "benchmark" and output.exists():
                    continue
                monitor.phase = f"seed {seed} coordinate {coordinate}/60"
                model = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V3_5, model_path=args.model_path, device=args.device,
                    n_estimators=c["n_estimators"], random_state=seed + coordinate,
                    n_preprocessing_jobs=1,
                )
                start = time.monotonic()
                model.fit(d["X_train"], d["y_train"][:, coordinate])
                fit_s = time.monotonic() - start
                start = time.monotonic()
                pred = np.asarray(model.predict(queries), dtype=np.float32)
                predict_s = time.monotonic() - start
                record = dict(seed=seed, coordinate=coordinate, fit_seconds=fit_s, predict_seconds=predict_s,
                              peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None)
                if not np.isfinite(pred).all():
                    raise ValueError("Nonfinite TabPFN predictions")
                if args.mode == "benchmark":
                    write(root / "benchmark.json", record)
                    print(json.dumps(record), flush=True)
                    return
                output.parent.mkdir(exist_ok=True)
                temporary = output.with_suffix(".partial.npz")
                np.savez_compressed(temporary, prediction=pred, seed=seed, coordinate=coordinate,
                                    data_sha256=c["data_sha256"])
                temporary.replace(output)
                write(output.with_suffix(".json"), record)
                write(root / "progress.json", dict(completed_coordinates=len(list(output.parent.glob("*.npz"))),
                                                    total_coordinates=180, last=record))
                print(json.dumps(record), flush=True)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    results = {}
    for seed in SEEDS:
        pred = []
        for coordinate in range(60):
            with np.load(root / "coordinates" / f"{seed}_{coordinate:02d}.npz") as z:
                if str(z["data_sha256"]) != c["data_sha256"]:
                    raise ValueError("Mixed datasets in resumed coordinates")
                pred.append(z["prediction"])
        predictions = np.stack(pred, axis=-1)
        results[str(seed)] = {}
        for split, values in zip(("val", "test"), np.split(predictions, [len(d["X_val"])])):
            target = d[f"y_{split}"].reshape(-1, 50, 60)
            values = values.reshape(target.shape)
            results[str(seed)][split] = scores(values, target, d[f"tasks_{split}"])
            np.savez_compressed(root / f"predictions_{split}_{seed}.npz", prediction=values, target=target,
                                tasks=d[f"tasks_{split}"], starts=d[f"starts_{split}"])
    write(root / "results.json", dict(complete=True, seeds=results,
                                      test_error_mean_mm=float(np.mean([r["test"]["error_mm"] for r in results.values()])),
                                      test_error_seed_sd_mm=float(np.std([r["test"]["error_mm"] for r in results.values()], ddof=1))))
    c["status"] = "complete"
    write(root / "config.json", c)
    print((root / "results.json").read_text(), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "benchmark", "fit"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--components", type=int, default=128)
    p.add_argument("--n-estimators", type=int, default=8)
    p.add_argument("--model-path", type=Path, default=Path.home()/"Library/Caches/tabpfn/tabpfn-v3.5-20260909.safetensors")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if args.mode == "prepare":
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
