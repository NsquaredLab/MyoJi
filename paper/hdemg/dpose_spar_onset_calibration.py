"""SPAR campaign with one calibration per recording, the deployable counterpart of the onset replay.

Declared before running (2026-10-10); exploratory. Every window (training, validation and test) is
normalized with the lower median and MAD of its recording's first 500 ms (1,000 resampled samples)
instead of its own first 500 ms; everything downstream is refitted exactly as in the nine-participant
campaign (results/dpose_spar_simple_full_v1): train-only standardization, a 1,024-code Lloyd codebook per
participant (seed 260911), and the same windows, decoder, recipe and seeds. Test windows begin 34.5 to
37.6 s after the calibration interval.

Contrasts (nine participants, seeds averaged; exact sign-flip, Holm over the three; 10,000-draw
participant bootstrap, seed 260919): tokens minus continuous under onset calibration; onset minus
per-window calibration for tokens; the same for continuous SPAR.

    ~/.local/bin/uv run --no-project --with torch --with h5py --with joblib --with scipy --with numpy \
        uv run python -m paper.hdemg.dpose_spar_onset_calibration run|report [--subjects ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
from paper.hdemg import dpose_spar_63_electrodes as E  # noqa: E402  (contrast, exact sign-flip, campaign paths)
from paper.hdemg import dpose_spar_transfer_checks as T  # noqa: E402  (resample, normalize_with, targets_for)
from paper.hdemg import run_dpose_spar_simple as pilot  # noqa: E402
from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize  # noqa: E402
from myoji.spar import grid_layout, spar_simple_dataset_batched  # noqa: E402

OUT = BASE / "results/dpose_spar_onset_calibration_v1"


def prepare(root, subject, device):
    if (root / "cache-receipt.json").exists():
        return json.loads((root / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    p = json.loads((E.CAMPAIGN / f"subject_{subject:02d}/protocol.json").read_text())
    p.update(name="dpose_spar_onset_calibration_v1",
             scope="Exploratory: every window normalized with its recording's first 500 ms; campaign windows, "
                   "recipe and seeds; standardization, codebook and decoders refitted.")
    pilot.write(root / "protocol.json", p)
    layout = grid_layout(p["geometry"]["grids"])
    mask = np.tile(layout.mask, (1, 6))
    emg = joblib.load(E.DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(E.DATA / f"Sub{subject}_kinematics.pkl")
    rows = {split: [r for r in p["windows"] if r["split"] == split] for split in ("train", "val", "test")}
    feats = {s: np.zeros((len(r), 320, 100, 18), np.float32) for s, r in rows.items()}
    targets = {s: np.zeros((len(r), 100, 60), np.float32) for s, r in rows.items()}
    for task in sorted(emg, key=int):
        wanted = [(s, i, r) for s, rs in rows.items() for i, r in enumerate(rs) if int(r["task"]) == int(task)]
        if not wanted:
            continue
        signal = T.resample(emg[task])
        onset = signal[:, :1000]
        for split, i, r in wanted:
            start = int(r["start"])
            raw = T.normalize_with(signal[:, start:start + 6000], onset)
            analysis = raw[:, E.INDICES].transpose(1, 2, 0).copy()
            f = spar_simple_dataset_batched(analysis, layout=layout, channelwise=True, batch_size=16, device=device)
            assert f.shape == (100, 320, 18) and np.isfinite(f).all()
            feats[split][i] = f.transpose(1, 0, 2)
            targets[split][i] = T.targets_for(kin[task], start)
    mean, scale = fit_standardization(feats["train"], mask)
    z = standardize(feats["train"], mean, scale, mask)
    t = p["tokenizer"]
    book, receipt = fit_codebook(z, vocabulary=t["vocabulary"], seed=t["seed"], max_vectors=t["max_vectors"],
                                 iterations=t["iterations"], device=device)
    path = root / "features.h5"
    with h5py.File(path, "w") as h:
        h["mask"], h["codebook"], h["feature_mean"], h["feature_scale"] = mask, book, mean, scale
        for split, rs in rows.items():
            values = standardize(feats[split], mean, scale, mask)
            g = h.create_group(split)
            g["continuous"], g["targets"] = values, targets[split]
            g["tokens"] = tokenize(values, book, device=device).astype(np.int16)
            g["tasks"] = [int(r["task"]) for r in rs]
            g["starts"] = [int(r["start"]) for r in rs]
        h.attrs["complete"] = True
    pilot.write(root / "quantizer-fit.json", receipt)
    pilot.write(root / "cache-receipt.json", dict(complete=True, sha256=pilot.sha(path), protocol_sha256=pilot.sha(root / "protocol.json")))
    return p


def run(subject, device):
    root = OUT / f"subject_{subject:02d}"
    p = prepare(root, subject, device)
    args = SimpleNamespace(root=root, device=device)
    for arm in pilot.ARMS:
        for seed in pilot.SEEDS:
            pilot.fit(args, p, arm, seed)
    if not (root / "results.json").exists():
        pilot.evaluate(args, p)


def report():
    rng = np.random.default_rng(260919)
    campaign = json.loads((E.CAMPAIGN / "results.json").read_text())["scores"]
    onset = {arm: [] for arm in pilot.ARMS}
    window = {arm: [] for arm in pilot.ARMS}
    for subject in range(1, 10):
        scores = json.loads((OUT / f"subject_{subject:02d}/results.json").read_text())["scores"]
        for arm in pilot.ARMS:
            onset[arm].append(float(np.mean([s["error_mm"] for s in scores if s["arm"] == arm])))
            window[arm].append(float(np.mean([s["error_mm"] for s in campaign
                                              if s["subject"] == subject and s["arm"] in (arm, "spar_" + arm)])))
    tests = dict(tokens_minus_continuous_onset=E.contrast(onset["tokens"], onset["continuous"], rng),
                 tokens_onset_minus_window=E.contrast(onset["tokens"], window["tokens"], rng),
                 continuous_onset_minus_window=E.contrast(onset["continuous"], window["continuous"], rng))
    names = sorted(tests, key=lambda k: tests[k]["p_exact"])
    running = 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, tests[k]["p_exact"] * (len(names) - i)))
        tests[k]["p_holm"] = running
    summary = dict(mean_mm={f"{arm}_{n}": float(np.mean(v[arm])) for n, v in (("onset", onset), ("per_window", window))
                            for arm in pilot.ARMS},
                   per_participant=dict(onset=onset, per_window=window), **tests)
    pilot.write(OUT / "summary.json", summary)
    print(json.dumps({k: (v if k == "mean_mm" else {kk: vv for kk, vv in v.items() if kk != "per_participant"})
                      for k, v in summary.items() if k != "per_participant"}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "run":
        for subject in args.subjects:
            run(subject, args.device)
    else:
        report()


if __name__ == "__main__":
    main()
