"""Frozen, paired electrode-dropout comparison of token V20, continuous V20 and V19.

The added-noise campaign says in its own limitations that "AWGN after clean
startup is not electrode displacement or general deployment robustness". This is
the sibling experiment for the failure a worn array actually has: whole channels
go silent after the clean calibration prefix.

Two kinds, never mixed. Scattered dropout removes a fraction of electrodes at
random, which is connector or contact failure. Grid dropout removes one or two
of the five 64-channel arrays, which is a detached patch and the harder test of
a channel-independent tokenizer.

No network is fitted here. Every model, the tokenizer and the test split are the
ones the noise campaign froze, and the clean condition is re-asserted against the
stored per-seed test predictions, so a drifted checkpoint fails before any
dropout number is produced. Reproducibility comes from the seeded channel draw
rather than a replayed corruption file, and each corrupted segment's hash is
recorded so a rerun can be gated against this one.

    uv run python -m paper.hdemg.run_hdemg_dropout test --root RUN --family v20 --index 0
    uv run python -m paper.hdemg.run_hdemg_dropout report --root RUN
"""

import argparse
import gc
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

from paper.common import paths
from paper.hdemg import run_hdemg_v20_noise_followup as base
from paper.hdemg.hdemg_dropout import (
    DROPOUT_FRACTIONS,
    DROPOUT_SEEDS,
    GRID_DROPS,
    drop_native,
    dropped_channels,
    grid_bounds,
)
from paper.hdemg.hdemg_noise import array_sha, native_bounds, pose_metrics

SUBJECTS, SEEDS, ARMS = base.SUBJECTS, base.SEEDS, base.ARMS
CONDITIONS = (
    [{"name": "clean", "kind": None, "level": None, "seed": None}]
    + [{"name": f"drop{round(f * 100):02d}_seed{s}", "kind": "fraction",
        "level": f, "seed": s} for f in DROPOUT_FRACTIONS for s in DROPOUT_SEEDS]
    + [{"name": f"grid{g}_seed{s}", "kind": "grids", "level": g, "seed": s}
       for g in GRID_DROPS for s in DROPOUT_SEEDS]
)


def freeze(args):
    """Derive this campaign's protocol from the frozen noise campaign's own."""
    parent = json.loads((args.parent / "protocol.json").read_text())
    p = {k: v for k, v in parent.items() if k not in ("conditions", "noise", "candidates",
                                                      "selection", "filter_state", "statistics")}
    p["version"] = "hdemg_dropout_v1"
    p["parent_campaign"] = str(args.parent)
    p["parent_results_sha256"] = base.sha(args.parent / "results.json")
    for name in ("hdemg/hdemg_dropout.py", "hdemg/run_hdemg_dropout.py"):
        p["source_sha256"][name] = base.sha(base.BASE / name)
    p["conditions"] = CONDITIONS
    p["grids"] = grid_bounds()
    p["dropout"] = (
        "Silence whole electrodes from the end of the clean 500-ms calibration prefix "
        "to the end of each native segment. Scattered draws remove a fraction of the "
        "320 channels; grid draws remove one or two of the five 64-channel arrays. "
        "The draw is seeded by (dropout seed, task, window start), so it reproduces "
        "without a stored corruption file; each corrupted segment's hash is recorded. "
        "No new random corruption enters any other campaign.")
    p["new_network_fits"] = 0
    args.root.mkdir(parents=True, exist_ok=True)
    base.write(args.root / "protocol.json", p)
    base.verify(args.root)
    print(f"froze {len(CONDITIONS)} conditions x {len(ARMS)} arms x "
          f"{len(SEEDS)} seeds x {len(SUBJECTS)} subjects")


def perturbed_window(emg, task, start, condition):
    """The clean segment, or the same segment with its dropped channels silenced."""
    origin, onset, stop = native_bounds(int(start))
    raw = emg[str(task)][:, origin:stop].astype(np.float32)
    if condition["kind"] is None:
        return raw, origin, {"dropped_count": 0}
    channels = dropped_channels(
        raw.shape[0], 0, int(task), int(start), condition["seed"],
        **{condition["kind"]: condition["level"]})
    raw, meta = drop_native(raw, onset - origin, channels)
    return raw, origin, meta


def test_subject(args, p):
    grid_bounds()
    if args.family == "v19":
        assert torch.cuda.get_device_name(args.device) == p["native_test_gpu"]
    subject = args.index + 1
    assert subject in SUBJECTS
    arms = ("native_v19",) if args.family == "v19" else ("token_v20", "continuous_v20")
    models = {arm: base.load_models(p, arm, subject, args.device) for arm in arms}

    path = Path(p["data"]) / f"Sub{subject}_emg.pkl"
    assert base.sha(path) == p["source_archives"][path.name]
    emg = joblib.load(path)
    meta = base.arrays(p, subject, "test")
    outdir = args.root / f"test/{args.family}/subject_{subject:02d}"
    if (outdir / "predictions.h5").exists():
        raise FileExistsError(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.family == "v20":
        from paper.hdemg import run_hdemg_pilot as pilot
        assert base.sha(Path(p["tokenizer_checkpoint"])) == p["tokenizer_sha256"]
        tokenizer = pilot.Tokenizer(Path(p["tokenizer_checkpoint"]), device=args.device)
        before = pilot.state_hash(tokenizer.model)

    rows, replays, drops = [], [], []
    with h5py.File(outdir / "predictions.h5", "w") as out:
        out.attrs["complete"] = False
        out["target"], out["tasks"], out["starts"] = (
            meta["targets"], meta["tasks"], meta["starts"])
        for condition in CONDITIONS:
            name = condition["name"]
            predictions = {arm: np.empty((3, 16, 100, 60), np.float32) for arm in arms}
            native = []
            for index, (task, start) in enumerate(
                    zip(meta["tasks"], meta["starts"], strict=True)):
                segment, origin, meta_drop = perturbed_window(emg, task, start, condition)
                drops.append({"condition": name, "window": index,
                              "corrupted_native_sha256": array_sha(segment),
                              **{k: v for k, v in meta_drop.items()
                                 if k != "dropped_channels"}})
                if args.family == "v19":
                    native.append(base.native_frames(segment, int(start), origin))
                else:
                    ids, vectors, _ = base.encode_float32(
                        tokenizer, segment, int(start), origin, args.device)
                    for arm, values in (("token_v20", ids), ("continuous_v20", vectors)):
                        predictions[arm][:, index:index + 1] = base.predict_v20(
                            models[arm], values[None], arm, args.device)
            if args.family == "v19":
                predictions["native_v19"] = base.predict_native(
                    models["native_v19"], np.stack(native), args.device)
                del native
            for arm, pred in predictions.items():
                assert np.isfinite(pred).all()
                if name == "clean":
                    # A drifted checkpoint must fail here, before any dropout number.
                    for k, seed in enumerate(SEEDS):
                        with np.load(base.fit_folder(p, arm, subject, seed)
                                     / "test_predictions.npz") as f:
                            np.testing.assert_array_equal(
                                f["target"], meta["targets"][:, -50:])
                            np.testing.assert_allclose(
                                pred[k, :, -50:], f["prediction"], atol=2e-4, rtol=4e-6)
                            replays.append({"arm": arm, "seed": seed,
                                            "max_abs_difference": float(np.max(
                                                np.abs(pred[k, :, -50:] - f["prediction"])))})
                out.create_dataset(f"{name}/{arm}", data=pred, compression="gzip")
                for k, seed in enumerate(SEEDS):
                    rows.append({"subject": subject, "seed": int(seed), "arm": arm,
                                 "filtering": "unfiltered", "condition": name,
                                 "kind": condition["kind"], "level": condition["level"],
                                 "dropout_seed": condition["seed"],
                                 **pose_metrics(pred[k, :, -50:],
                                                meta["targets"][:, -50:], meta["starts"])})
            gc.collect()
        out.attrs["complete"] = True
    if args.family == "v20":
        assert pilot.state_hash(tokenizer.model) == before, "tokenizer mutated"
    base.write(outdir / "results.json",
               {"complete": True, "subject": subject, "family": args.family,
                "rows": rows, "clean_replay": replays, "dropout": drops,
                "conditions": CONDITIONS, "new_network_fits": 0})


def report(root, p):
    rows = []
    for family in ("v20", "v19"):
        for subject in SUBJECTS:
            f = root / f"test/{family}/subject_{subject:02d}/results.json"
            data = json.loads(f.read_text())
            assert data["complete"]
            rows += data["rows"]
    assert len(rows) == len(SUBJECTS) * len(SEEDS) * len(ARMS) * len(CONDITIONS), len(rows)
    summary = []
    for arm in ARMS:
        clean = np.array([np.mean([r["mean_3d_point_error"] for r in rows
                                   if r["arm"] == arm and r["condition"] == "clean"
                                   and r["subject"] == s]) for s in SUBJECTS])
        for condition in CONDITIONS:
            per_subject = np.array([
                np.mean([r["mean_3d_point_error"] for r in rows if r["arm"] == arm
                         and r["condition"] == condition["name"] and r["subject"] == s])
                for s in SUBJECTS])
            summary.append({"arm": arm, "condition": condition["name"],
                            "kind": condition["kind"], "level": condition["level"],
                            "mean_3d_error": float(per_subject.mean()),
                            "clean_delta": float((per_subject - clean).mean()),
                            "per_subject_delta": (per_subject - clean).tolist()})
    base.write(root / "results.json",
               {"complete": True, "rows": rows, "summary": summary,
                "conditions": CONDITIONS, "subjects": list(SUBJECTS),
                "seeds": list(SEEDS), "arms": list(ARMS), "new_network_fits": 0,
                "grids": grid_bounds(),
                "scope": "Electrode dropout after the clean calibration prefix on the "
                         "frozen noise-campaign models, split and tokenizer.",
                "limitations": "Silenced channels are an idealization of contact loss: "
                               "a real failure adds drift and motion artefact rather "
                               "than zeros. Dropout is applied at inference only; no "
                               "model saw a dropped channel during fitting."})
    print(f"report complete: {len(rows)} rows, {len(summary)} summary cells")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "test", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path,
                        default=paths.RESULTS / "hdemg_v20_noise_followup_v2")
    parser.add_argument("--family", choices=("v20", "v19"), default="v20")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "freeze":
        return freeze(args)
    p = base.verify(args.root)
    test_subject(args, p) if args.stage == "test" else report(args.root, p)


if __name__ == "__main__":
    main()
