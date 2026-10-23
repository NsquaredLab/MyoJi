"""Fixed nine-person extension of the Subject-1 TabPFN pose pilot.

The original pilot runner and all upstream tokenizer/decoder training stay unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from . import run as pilot

SUBJECTS = list(range(1, 10))
ARMS = ["tokens", "continuous"]
TASKS = [2, 4, 6, 8, 10, 15, 18, 19]
SEEDS = pilot.SEEDS
SPANS = [1, 5, 25, 100]


def continuous_history(features, mask, score_start=50, spans=SPANS):
    """The identical causal summaries used for codebook-decoded token inputs."""
    n, channels, frames, width = features.shape
    if mask.shape != (channels, width):
        raise ValueError("Invalid continuous feature/mask geometry")
    x = (features * mask[None, :, None, :]).transpose(0, 2, 1, 3)
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


def protocol(source, checkpoint):
    upstream = json.loads((source / "protocol.json").read_text())
    if upstream["subjects"] != SUBJECTS or upstream["seeds"] != SEEDS:
        raise ValueError("Expected the frozen nine-participant SPAR campaign")
    pilot.validate_separation(upstream["windows"])
    return dict(
        subjects=SUBJECTS, arms=ARMS, seeds=SEEDS, tasks=TASKS, channels=320,
        source_campaign="hdemg_spar_simple_full_v1", source_protocol_sha256=pilot.sha256(source / "protocol.json"),
        checkpoint=checkpoint.name, checkpoint_sha256=pilot.sha256(checkpoint),
        n_estimators=8, components=128, pca_seed=260921, spans=SPANS, score_start=50,
        split_windows=dict(train=64, val=16, test=16), coordinate_regressors=60,
        fit_mode="fit_preprocessors", input_precision="Original frozen float32 SPAR cache; float32 PCA table",
        input="Same 1/5/25/100-frame causal summaries, structural mask and 128-component training-only PCA in both arms. "
              "PCA fitted separately within each participant/arm. Tokens are looked up through that participant's "
              "existing frozen 1024x18 codebook; continuous inputs are the existing standardized descriptors.",
        target="Final 50 frames per original window, 20 wrist-relative landmarks, all 60 coordinates in mm.",
        scope="Personalized same-session chronological evaluation. Frozen settings, no test-based tuning. "
              "Not unseen-participant pose decoding. New PCA/TabPFN preparation differs from V20's learned readout.",
        statistics=dict(unit="participant", seeds="Average three algorithm seeds within participant first",
            primary=["TabPFN tokens minus SPAR V20 tokens", "TabPFN continuous minus SPAR V20 continuous",
                     "TabPFN tokens minus TabPFN continuous"],
            method="Two-sided exact participant sign-flip; Holm across the three declared primary contrasts. "
                   "Main-figure comparisons additionally extend its existing synchronized max-T family.",
            main_figure_additions=[
                "Each TabPFN arm versus each of the three existing CNN references (6)",
                "Each TabPFN arm versus the corresponding SPAR V20 arm (2)",
                "TabPFN tokens versus continuous (1)",
                "TabPFN absolute within-task token-feature error change versus SPAR V20 (1)"],
            main_figure_family_size=76),
        resources=dict(gpus=1, maximum_workers=2, cpu_threads_per_worker=4,
                       resume="Completed coordinate outputs verified against prepared data hashes"),
        source_cache_sha256={str(s):json.loads((source/f"subject_{s:02d}/cache-receipt.json").read_text())["sha256"]
                             for s in SUBJECTS},
    )


def verify_identities(group, windows, subject, split):
    actual = [(int(subject), int(t), int(start)) for t, start in zip(group["tasks"][:], group["starts"][:])]
    expected = [(int(w["subject"]), int(w["task"]), int(w["start"]))
                for w in windows if w["subject"] == subject and w["split"] == split]
    if actual != expected or not np.all(group["subjects"][:] == subject):
        raise ValueError(f"Cache identities do not match frozen split: subject {subject}, {split}")
    if sorted(set(group["tasks"][:])) != TASKS:
        raise ValueError("Missing movement task")


def prepare_subject(root, source, subject, fixed):
    folder = source / f"subject_{subject:02d}"
    receipt = json.loads((folder / "cache-receipt.json").read_text())
    p = json.loads((folder / "protocol.json").read_text())
    digest = pilot.sha256(folder / "protocol.json")
    if not receipt["complete"] or receipt["protocol_sha256"] != digest:
        raise ValueError("Invalid source cache receipt")
    if pilot.sha256(folder / "features.h5") != fixed["source_cache_sha256"][str(subject)]:
        raise ValueError("Frozen feature cache changed")
    pilot.validate_separation(p["windows"])
    with h5py.File(folder / "features.h5", "r") as f:
        if not f.attrs["complete"] or f.attrs["protocol_sha256"] != digest:
            raise ValueError("Incomplete cache or mismatching HDF5 protocol")
        book, mask = f["codebook"][:], f["mask"][:]
        if book.shape != (1024, 18) or mask.shape != (320, 18):
            raise ValueError("Unexpected frozen SPAR geometry")
        for arm in ARMS:
            output = root / f"subject_{subject:02d}" / arm
            if (output / "config.json").exists():
                config = json.loads((output / "config.json").read_text())
                if config["cohort_protocol_sha256"] != pilot.sha256(root / "protocol.json"):
                    raise ValueError("Cannot resume preparation with another protocol")
                if pilot.sha256(output / "data.npz") != config["data_sha256"]:
                    raise ValueError("Prepared table changed")
                continue
            output.mkdir(parents=True, exist_ok=True)
            arrays, metadata, raw = {}, {}, {}
            for split, expected in fixed["split_windows"].items():
                g = f[split]
                verify_identities(g, p["windows"], subject, split)
                if g["tokens"].shape != (expected, 320, 100):
                    raise ValueError("Unexpected source window geometry")
                raw[split] = (pilot.spatial_history(g["tokens"][:], book, mask)
                              if arm == "tokens" else continuous_history(g["continuous"][:], mask))
                arrays[f"y_{split}"] = g["targets"][:, -50:].reshape(-1, 60)
                for field in ["tasks", "starts"]:
                    arrays[f"{field}_{split}"] = g[field][:]
                metadata[split] = dict(windows=expected, pose_frames=expected * 50,
                    tasks=g["tasks"][:].tolist(), starts=g["starts"][:].tolist())
            projection = PCA(n_components=fixed["components"], svd_solver="randomized", random_state=fixed["pca_seed"])
            arrays["X_train"] = projection.fit_transform(raw["train"]).astype(np.float32)
            for split in ["val", "test"]:
                arrays[f"X_{split}"] = projection.transform(raw[split]).astype(np.float32)
            del raw
            joblib.dump(projection, output / "input_pca.joblib")
            np.savez_compressed(output / "data.npz", **arrays)
            controls = {}
            ridge = Ridge(alpha=1.0).fit(arrays["X_train"], arrays["y_train"])
            for split in ["val", "test"]:
                y = arrays[f"y_{split}"].reshape(-1, 50, 60)
                pred = ridge.predict(arrays[f"X_{split}"]).reshape(y.shape)
                mean = np.broadcast_to(arrays["y_train"].mean(0), y.shape).copy()
                controls[split] = dict(ridge=pilot.scores(pred, y, arrays[f"tasks_{split}"]),
                    training_mean=pilot.scores(mean, y, arrays[f"tasks_{split}"]))
                np.savez_compressed(output / f"baselines_{split}.npz", ridge=pred, mean=mean, target=y,
                    tasks=arrays[f"tasks_{split}"], starts=arrays[f"starts_{split}"])
            references = {}
            for reference_arm in ARMS:
                for seed in SEEDS:
                    path = folder / "fits" / f"{reference_arm}_{seed}" / "test_predictions.npz"
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z["target"], arrays["y_test"].reshape(16, 50, 60))
                        np.testing.assert_array_equal(z["tasks"], arrays["tasks_test"])
                        if "starts" in z:
                            np.testing.assert_array_equal(z["starts"], arrays["starts_test"])
                        references[f"{reference_arm}_{seed}"] = dict(
                            pilot.scores(z["prediction"], z["target"], z["tasks"]),
                            prediction_sha256=pilot.sha256(path))
            pilot.write(output / "baselines.json", controls)
            pilot.write(output / "references.json", references)
            pilot.write(output / "config.json", dict(status="prepared", subject=subject, arm=arm,
                seeds=SEEDS, checkpoint=fixed["checkpoint"], n_estimators=fixed["n_estimators"],
                components=fixed["components"], pca_variance_fraction=float(projection.explained_variance_ratio_.sum()),
                input_features_before_pca=21024, metadata=metadata, source=str(folder.resolve()),
                source_cache_sha256=receipt["sha256"], source_protocol_sha256=digest,
                cohort_protocol_sha256=pilot.sha256(root / "protocol.json"),
                data_sha256=pilot.sha256(output / "data.npz"), pca_sha256=pilot.sha256(output / "input_pca.joblib"),
                codebook_array_sha256=hashlib.sha256(book.tobytes()).hexdigest(), input=fixed["input"],
                target=fixed["target"], scope=fixed["scope"]))
            print(json.dumps(dict(stage="prepared", subject=subject, arm=arm,
                retained_pca_variance=float(projection.explained_variance_ratio_.sum()))), flush=True)


def fit_worker(payload):
    root, checkpoint = map(Path, payload)
    if (root / "results.json").exists():
        result = json.loads((root / "results.json").read_text())
        if result.get("complete"):
            return str(root)
    pilot.run(SimpleNamespace(root=root, model_path=checkpoint, device="cuda", mode="fit"))
    return str(root)


def aggregate(root, source):
    fixed = json.loads((root / "protocol.json").read_text())
    scores, references, hashes = [], [], {}
    for subject in SUBJECTS:
        for arm in ARMS:
            folder = root / f"subject_{subject:02d}" / arm
            completed = json.loads((folder / "results.json").read_text())
            if not completed["complete"]:
                raise ValueError("Incomplete TabPFN run")
            for seed in SEEDS:
                file = folder / f"predictions_test_{seed}.npz"
                with np.load(file) as z, np.load(source / f"subject_{subject:02d}/fits/{arm}_{seed}/test_predictions.npz") as ref:
                    np.testing.assert_array_equal(z["target"], ref["target"])
                    np.testing.assert_array_equal(z["tasks"], ref["tasks"])
                    result = pilot.scores(z["prediction"], z["target"], z["tasks"])
                    np.testing.assert_allclose(result["error_mm"], completed["seeds"][str(seed)]["test"]["error_mm"], rtol=0, atol=2e-6)
                    scores.append(dict(subject=subject, arm=arm, seed=seed, **result,
                                       predictions_sha256=pilot.sha256(file)))
                    references.append(dict(subject=subject, arm=arm, seed=seed,
                        **pilot.scores(ref["prediction"], ref["target"], ref["tasks"])))
                hashes[str(file.relative_to(root))] = pilot.sha256(file)
    if len(scores) != 54:
        raise ValueError("Expected 54 complete participant/arm/seed evaluations")
    summary = {}
    for label, collection in [("tabpfn", scores), ("v20", references)]:
        summary[label] = {}
        for arm in ARMS:
            values = [float(np.mean([r["error_mm"] for r in collection if r["arm"] == arm and r["subject"] == s])) for s in SUBJECTS]
            summary[label][arm] = dict(mean_mm=float(np.mean(values)), participant_sd_mm=float(np.std(values, ddof=1)),
                                      participant_means_mm=dict(zip(map(str,SUBJECTS),values)))
    pilot.write(root / "results.json", dict(complete=True, subjects=SUBJECTS, seeds=SEEDS, arms=ARMS,
        fits=54, coordinate_regressions=3240, summary=summary, scores=scores, references=references,
        protocol_sha256=pilot.sha256(root / "protocol.json"), prediction_hashes=hashes, scope=fixed["scope"]))
    print(json.dumps(dict(stage="complete", summary=summary)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "fit", "all", "aggregate"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=[1,2], default=2)
    args = parser.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    fixed = protocol(args.source, args.model_path)
    locked = args.root / "protocol.json"
    if locked.exists():
        if json.loads(locked.read_text()) != fixed:
            raise ValueError("Cohort protocol is immutable")
    else:
        pilot.write(locked, fixed)
    started = time.monotonic()
    try:
        if args.mode in ["prepare", "all"]:
            with threadpool_limits(limits=4):
                for subject in SUBJECTS:
                    prepare_subject(args.root, args.source, subject, fixed)
        if args.mode in ["fit", "all"]:
            tasks = [(str(args.root / f"subject_{s:02d}" / arm), str(args.model_path)) for s in SUBJECTS for arm in ARMS]
            pilot.write(args.root / "progress.json", dict(complete=False, stage="fitting", finished_arms=0, total_arms=18,
                started_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers, gpus=1))
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(fit_worker, payload): payload for payload in tasks}
                finished = 0
                for future in as_completed(futures):
                    result = future.result()
                    finished += 1
                    pilot.write(args.root / "progress.json", dict(complete=False, stage="fitting", finished_arms=finished,
                        total_arms=18, last=result, elapsed_seconds=time.monotonic()-started, workers=args.workers, gpus=1))
            aggregate(args.root, args.source)
            pilot.write(args.root / "progress.json", dict(complete=True, stage="complete", finished_arms=18, total_arms=18,
                elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat()))
        elif args.mode == "aggregate":
            aggregate(args.root, args.source)
    except Exception as exc:
        pilot.write(args.root / "failure.json", dict(error=repr(exc), utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    main()
