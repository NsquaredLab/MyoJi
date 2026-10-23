"""Personalized HD-EMG decoding on chronological, separated recording blocks."""

import argparse
import copy
import gc
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
from paper.common import paths
from paper.hdemg import run_hdemg_pilot as pilot
import torch

ARMS = ("tokens", "continuous", "raw")
SEEDS = (260911, 260912, 260913)
SUBJECTS = tuple(range(1, 10))
pilot.ARMS = ARMS
pilot.SEEDS = SEEDS


def block_windows(source_samples):
    """All 3 s windows within 60/20/20 blocks, with 1 s incoming gaps."""
    count = source_samples * 125 // 128
    b1, b2 = count * 3 // 5, count * 4 // 5
    bounds = {"train": (0, b1), "val": (b1 + 2000, b2), "test": (b2 + 2000, count)}
    rows = []
    for split, (lo, hi) in bounds.items():
        starts = list(range(lo, hi - 6000 + 1, 6000))
        if not starts:
            raise ValueError(f"Recording too short for split {split}: {source_samples}")
        rows.extend({"split": split, "start": start} for start in starts)
    return rows


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError(args.root / "protocol.json")
    previous = json.loads((args.pilot / "protocol.json").read_text())
    inventory = json.loads((args.pilot / "selection.json").read_text())["inventory"]
    p = copy.deepcopy(previous)
    p.update(
        scope="Personalized decoders on later chronological blocks within the same participant, task and session; participants 1–9. Not unseen-person, new-session or new-task generalization.",
        subjects=list(SUBJECTS),
        splits={s: list(SUBJECTS) for s in ("train", "val", "test")},
        arms=list(ARMS),
        seeds=list(SEEDS),
        previous_protocol_sha256=pilot.sha256(args.pilot / "protocol.json"),
        selection="Chronological blocks [0,60%), [60%,80%), [80%,100%) per subject/task at 2000 Hz. Skip first 2000 samples of val/test blocks. Enumerate every complete nonoverlapping 6000-sample window in each block. No cross-window state; prefix calibration uses only the first 1000 samples of that window. One-second gaps exceed 40 ms FIR support; full 3 s readout input stays within its block.",
        models="A separate decoder per subject, arm and seed; 81 fits total. No sharing of decoder parameters or target statistics across subjects. Same readout/training settings as original pilot.",
        evaluation="Only after all 81 fits complete. Mean 3D error and coordinate R2 for each of 9 subjects and each seed. Macro-subject error per seed, mean and sample SD over 3 seeds. Subject-specific constant training-mean pose baseline. Windows/time points are not independent participants.",
        source_sha256={
            name: pilot.sha256(pilot.source_path(name))
            for name in [
                "hdemg/run_hdemg_within.py",
                "hdemg/run_hdemg_pilot.py",
                "src/myoji/decoders.py",
            ]
        },
        files={
            name: digest
            for name, digest in previous["files"].items()
            if int(name.split("_")[0][3:]) in SUBJECTS
        },
    )
    p.pop("max_windows_per_task")
    p["cache"] = (
        "Per-subject cache: 100 frames ends 2039..5999; uint16 IDs, float16 continuous, float32 normalized raw and targets; fixed 32-channel encoder batch. Frozen original release. Exact matching first windows reused from original training cache after validation."
    )
    manifests = []
    for item in inventory:
        subject = item["subject"]
        if subject not in SUBJECTS:
            continue
        for task, info in item["tasks"].items():
            for row in block_windows(info["emg"][1]):
                manifests.append(dict(row, subject=subject, task=task))
    assert {row["subject"] for row in manifests} == set(SUBJECTS)
    p["window_counts"] = {
        s: sum(r["split"] == s for r in manifests) for s in p["splits"]
    }
    pilot.write_json(args.root / "protocol.json", p)
    pilot.write_json(
        args.root / "selection.json",
        {
            "windows": manifests,
            "protocol_sha256": pilot.sha256(args.root / "protocol.json"),
        },
    )
    for subject in SUBJECTS:
        folder = args.root / f"subject_{subject:02d}"
        sub = dict(
            p,
            subject=subject,
            campaign_protocol_sha256=pilot.sha256(args.root / "protocol.json"),
        )
        pilot.write_json(folder / "protocol.json", sub)
    print(
        json.dumps(
            {"subjects": len(SUBJECTS), "fits": 81, "windows": p["window_counts"]}
        ),
        flush=True,
    )


@torch.inference_mode()
def prepare_subject(args, subject):
    folder = args.root / f"subject_{subject:02d}"
    p = json.loads((folder / "protocol.json").read_text())
    out = folder / "features.h5"
    if (folder / "cache_receipt.json").exists():
        receipt = json.loads((folder / "cache_receipt.json").read_text())
        assert receipt["complete"] and receipt["protocol_sha256"] == pilot.sha256(
            folder / "protocol.json"
        )
        with h5py.File(out, "r") as f:
            assert f.attrs["complete"]
        return
    if out.exists():
        raise FileExistsError(f"Incomplete cache requires explicit recovery: {out}")
    paths = [args.data / f"Sub{subject}_{kind}.pkl" for kind in ("emg", "kinematics")]
    for path in paths:
        assert pilot.sha256(path) == p["files"][path.name]
    emg, kin = [joblib.load(path) for path in paths]
    assert set(emg) == set(kin)
    rows = [
        r
        for r in json.loads((args.root / "selection.json").read_text())["windows"]
        if r["subject"] == subject
    ]
    model = pilot.Tokenizer(args.checkpoint, device=args.device)
    before = pilot.state_hash(model.model)
    reused = 0
    with h5py.File(out, "w") as f, h5py.File(args.pilot / "features.h5", "r") as old:
        assert (
            old.attrs["complete"]
            and old.attrs["protocol_sha256"] == p["previous_protocol_sha256"]
        )
        f.attrs["complete"] = False
        f.attrs["protocol_sha256"] = pilot.sha256(folder / "protocol.json")
        f["codebook"] = model.model.quantizer.codebook.cpu().numpy()
        f["descriptors"] = pilot.hd_monopolar_montage().descriptor().numpy()
        assert np.array_equal(f["codebook"][:], old["codebook"][:])
        assert np.array_equal(f["descriptors"][:], old["descriptors"][:])
        old_train = old["train"]
        old_indices = {
            (int(s), int(t), int(start)): i
            for i, (s, t, start) in enumerate(
                zip(
                    old_train["subjects"][:],
                    old_train["tasks"][:],
                    old_train["starts"][:],
                )
            )
        }
        grouped = {
            split: [r for r in rows if r["split"] == split]
            for split in ("train", "val", "test")
        }
        for split, selected in grouped.items():
            g = f.create_group(split)
            n = len(selected)
            for name, shape, dtype in [
                ("tokens", (n, 320, 100), "u2"),
                ("continuous", (n, 320, 100, 256), "f2"),
                ("raw", (n, 320, 6000), "f4"),
                ("targets", (n, 100, 60), "f4"),
            ]:
                g.create_dataset(name, shape=shape, dtype=dtype)
            for key in ("subjects", "tasks", "starts"):
                singular = {"subjects": "subject", "tasks": "task", "starts": "start"}[
                    key
                ]
                g[key] = np.array([int(r[singular]) for r in selected])
        for task in sorted(emg, key=int):
            assert emg[task].shape[0] == 320 and kin[task].shape == (
                21,
                3,
                emg[task].shape[-1],
            )
            expected = [
                dict(r, subject=subject, task=task)
                for r in block_windows(emg[task].shape[-1])
            ]
            assert expected == [r for r in rows if r["task"] == task]
            signal = pilot.resample(emg[task])
            for split, selected in grouped.items():
                for i, row in enumerate(selected):
                    if row["task"] != task:
                        continue
                    g = f[split]
                    start = row["start"]
                    raw = torch.from_numpy(signal[:, start : start + 6000])[None].to(
                        args.device
                    )
                    assert torch.isfinite(raw).all()
                    x, _, _ = pilot.calibration_normalize(raw, 1000)
                    y = pilot.target_at(kin[task], start)
                    assert np.isfinite(y).all()
                    g["raw"][i] = x[0].cpu().numpy()
                    g["targets"][i] = y
                    previous = old_indices.get((subject, int(task), start))
                    if previous is not None:
                        assert np.array_equal(g["raw"][i], old_train["raw"][previous])
                        assert np.array_equal(y, old_train["targets"][previous])
                        for key in ("tokens", "continuous"):
                            g[key][i] = old_train[key][previous]
                        reused += 1
                    else:
                        for channel in range(0, 320, 32):
                            encoded = model.model(
                                x[:, channel : channel + 32],
                                sensor="hd",
                                channel_descriptors=torch.zeros(
                                    32, model.config.geometry_dim, device=args.device
                                ),
                                include_spatial=False,
                            )
                            g["tokens"][i, channel : channel + 32] = (
                                encoded.tokens[0, :, -100:].cpu().numpy()
                            )
                            g["continuous"][i, channel : channel + 32] = (
                                encoded.continuous[0, :, -100:].half().cpu().numpy()
                            )
                    # Replay a fixed channel group for every window, including reused ones.
                    repeat = (
                        model.model(
                            x[:, :32],
                            sensor="hd",
                            channel_descriptors=torch.zeros(
                                32, model.config.geometry_dim, device=args.device
                            ),
                            include_spatial=False,
                        )
                        .tokens[0, :, -100:]
                        .cpu()
                        .numpy()
                    )
                    assert np.array_equal(repeat, g["tokens"][i, :32])
            f.flush()
            print(json.dumps({"prepared_subject": subject, "task": task}), flush=True)
        assert before == pilot.state_hash(model.model)
        f.attrs["complete"] = True
    pilot.write_json(
        folder / "cache_receipt.json",
        {
            "complete": True,
            "subject": subject,
            "split_windows": {s: len(v) for s, v in grouped.items()},
            "reused_windows": reused,
            "replay_checked_windows": len(rows),
            "protocol_sha256": pilot.sha256(folder / "protocol.json"),
            "selection_sha256": pilot.sha256(args.root / "selection.json"),
            "frozen_state_sha256": before,
            "runtime": model.metadata(),
        },
    )
    del model, emg, kin, signal, x, raw
    gc.collect()
    torch.cuda.empty_cache()


def run_subject(args):
    subject = args.index + 1
    assert subject in SUBJECTS
    prepare_subject(args, subject)
    subargs = copy.copy(args)
    subargs.root = args.root / f"subject_{subject:02d}"
    for index in range(9):
        subargs.index = index
        pilot.fit(subargs)
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args):
    # Global gate: no test evaluation until every subject/arm/seed fit is complete.
    for subject in SUBJECTS:
        folder = args.root / f"subject_{subject:02d}"
        for arm in ARMS:
            for seed in SEEDS:
                fit = json.loads(
                    (folder / "fits" / f"{arm}_{seed}" / "fit.json").read_text()
                )
                assert fit["complete"] and fit["protocol_sha256"] == pilot.sha256(
                    folder / "protocol.json"
                )
                assert fit["best_sha256"] == pilot.sha256(
                    folder / "fits" / f"{arm}_{seed}" / "best.pt"
                )
    per_subject = {}
    for subject in SUBJECTS:
        subargs = copy.copy(args)
        subargs.root = args.root / f"subject_{subject:02d}"
        pilot.evaluate(subargs)
        # Retain training targets for independent constant-baseline verification.
        with h5py.File(subargs.root / "features.h5", "r") as f:
            np.savez_compressed(
                subargs.root / "baseline_training_targets.npz",
                target=f["train"]["targets"][:, -50:],
            )
        per_subject[str(subject)] = json.loads(
            (subargs.root / "results.json").read_text()
        )
    summary = {}
    for arm in ARMS:
        seed_means = [
            float(
                np.mean(
                    [
                        next(
                            r["macro_subject_error"]
                            for r in v["results"]
                            if r["arm"] == arm and r["seed"] == seed
                        )
                        for v in per_subject.values()
                    ]
                )
            )
            for seed in SEEDS
        ]
        summary[arm] = {
            "mean": float(np.mean(seed_means)),
            "seed_sd": float(np.std(seed_means, ddof=1)),
            "seed_macro_subject_errors": dict(zip(map(str, SEEDS), seed_means)),
        }
    baseline = {s: v["constant_train_mean_baseline"][s] for s, v in per_subject.items()}
    pilot.write_json(
        args.root / "results.json",
        {
            "complete": True,
            "scope": json.loads((args.root / "protocol.json").read_text())["scope"],
            "summary": summary,
            "constant_subject_train_mean": baseline,
            "constant_macro_subject_error": float(np.mean(list(baseline.values()))),
            "per_subject": per_subject,
            "protocol_sha256": pilot.sha256(args.root / "protocol.json"),
        },
    )
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "subject", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, default=paths.RESULTS / "hdemg_pilot_v1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=paths.HDEMG)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "freeze":
        freeze(args)
    else:
        p = json.loads((args.root / "protocol.json").read_text())
        assert pilot.sha256(args.checkpoint) == p["checkpoint_sha256"]
        for path, digest in p["source_sha256"].items():
            assert pilot.sha256(pilot.source_path(path)) == digest
        if args.mode == "subject":
            run_subject(args)
        else:
            evaluate(args)


if __name__ == "__main__":
    main()
