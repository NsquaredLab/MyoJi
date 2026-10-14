"""SPAR decoding with the 63 electrodes that the optimized Cortex-M55 port processes within 20 ms.

Declared before running (2026-10-10); exploratory. Montage: the first 63 electrodes of the 320-channel
layout, i.e. the first 8x8 forearm array without its last site, which is the board benchmark's 63-channel
configuration. SPAR patches come from grid_layout with that site marked absent, exactly as for the
unrecorded site of the 13x5 arrays (a missing right or lower neighbour is substituted and its coupling
zeroed), and the band weights sum spectral power over the 63 channels, as on the board. Everything else is
the nine-participant campaign (results/dpose_spar_simple_full_v1): the same windows and splits, per-window
calibration, train-only standardization, a 1,024-code Lloyd codebook per participant fitted on 131,072
vectors of the 63 channels' training windows (seed 260911), and the same decoder, recipe and seeds. The
257 absent channels enter the decoder as masked zeros, so its architecture and parameter count are unchanged.

    ~/.local/bin/uv run --no-project --with torch --with h5py --with joblib --with scipy --with numpy \
        python scripts/dpose_spar_63_electrodes.py run|report [--subjects 1 2 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import firwin, upfirdn

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import paths  # noqa: E402
import run_dpose_spar_simple as pilot  # noqa: E402
from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize  # noqa: E402
from myoji.spar import grid_layout, spar_simple_dataset_batched  # noqa: E402

CAMPAIGN = BASE / "results/dpose_spar_simple_full_v1"
DATA = paths.HDEMG
OUT = BASE / "results/dpose_spar_63_electrodes_v1"
N = 63
TAPS = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
INDICES = pilot.ENDS[:, None] - np.arange(319, -1, -1)[None]


def layout63():
    return grid_layout([np.r_[np.arange(N), -1].reshape(8, 8)])


def prepare(root, subject, device):
    if (root / "cache-receipt.json").exists():
        return json.loads((root / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    p = json.loads((CAMPAIGN / f"subject_{subject:02d}/protocol.json").read_text())
    layout = layout63()
    p.update(name="dpose_spar_63_electrodes_v1", geometry=layout.record(),
             scope="Exploratory: first 63 electrodes of the 320-channel layout (first 8x8 forearm array without its "
                   "last site); campaign windows, recipe and seeds; absent channels are masked zeros.")
    p["feature"]["compact_values_per_frame"] = layout.entries_per_band * 6
    pilot.write(root / "protocol.json", p)
    mask = np.zeros((320, 18), dtype=bool)
    mask[:N] = np.tile(layout.mask, (1, 6))
    emg = joblib.load(DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(DATA / f"Sub{subject}_kinematics.pkl")
    rows = {split: [r for r in p["windows"] if r["split"] == split] for split in ("train", "val", "test")}
    feats = {s: np.zeros((len(r), 320, 100, 18), np.float32) for s, r in rows.items()}
    targets = {s: np.zeros((len(r), 100, 60), np.float32) for s, r in rows.items()}
    for task in sorted(emg, key=int):
        wanted = [(s, i, r) for s, rs in rows.items() for i, r in enumerate(rs) if int(r["task"]) == int(task)]
        if not wanted:
            continue
        x = emg[task][:N].astype(np.float32)
        signal = upfirdn(TAPS, x, up=125, down=128, axis=-1)[..., :x.shape[-1] * 125 // 128].astype(np.float32)
        for split, i, r in wanted:
            start = int(r["start"])
            raw = pilot.normalize_raw(signal[:, start:start + 6000])
            analysis = raw[:, INDICES].transpose(1, 2, 0).copy()
            f = spar_simple_dataset_batched(analysis, layout=layout, channelwise=True, batch_size=16, device=device)
            assert f.shape == (100, N, 18) and np.isfinite(f).all()
            feats[split][i, :N] = f.transpose(1, 0, 2)
            points = kin[task][..., ((start + pilot.ENDS) * 128) // 125].transpose(2, 0, 1).astype(np.float32)
            targets[split][i] = (points[:, 1:] - points[:, :1]).reshape(100, 60)
    mean, scale = fit_standardization(feats["train"], mask)
    z = standardize(feats["train"], mean, scale, mask)
    t = p["tokenizer"]
    book, receipt = fit_codebook(z[:, :N], vocabulary=t["vocabulary"], seed=t["seed"], max_vectors=t["max_vectors"],
                                 iterations=t["iterations"], device=device)
    path = root / "features.h5"
    usage = {}
    with h5py.File(path, "w") as h:
        h["mask"], h["codebook"], h["feature_mean"], h["feature_scale"] = mask, book, mean, scale
        for split, rs in rows.items():
            values = standardize(feats[split], mean, scale, mask)
            tokens = np.zeros(values.shape[:3], dtype=np.int16)
            tokens[:, :N] = tokenize(values[:, :N], book, device=device)
            g = h.create_group(split)
            g["continuous"], g["tokens"], g["targets"] = values, tokens, targets[split]
            g["tasks"] = [int(r["task"]) for r in rs]
            g["starts"] = [int(r["start"]) for r in rs]
            usage[split] = int(len(np.unique(tokens[:, :N])))
        h.attrs["complete"] = True
    pilot.write(root / "quantizer-fit.json", receipt)
    pilot.write(root / "cache-receipt.json", dict(complete=True, sha256=pilot.sha(path), protocol_sha256=pilot.sha(root / "protocol.json"),
                                                 used_codes=usage, channels=N, entries_per_band=layout.entries_per_band))
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


def exact_p(d):
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))


def contrast(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return dict(mean=float(d.mean()), per_participant=d.tolist(), positive=int((d > 0).sum()),
                bootstrap95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], p_exact=exact_p(d))


def report():
    rng = np.random.default_rng(260918)
    campaign = json.loads((CAMPAIGN / "results.json").read_text())["scores"]
    e63 = {arm: [] for arm in pilot.ARMS}
    e320 = {arm: [] for arm in pilot.ARMS}
    for subject in range(1, 10):
        scores = json.loads((OUT / f"subject_{subject:02d}/results.json").read_text())["scores"]
        for arm in pilot.ARMS:
            e63[arm].append(float(np.mean([s["error_mm"] for s in scores if s["arm"] == arm])))
            e320[arm].append(float(np.mean([s["error_mm"] for s in campaign
                                            if s["subject"] == subject and s["arm"] in (arm, "spar_" + arm)])))
    summary = dict(
        channels=N, montage="first 63 electrodes (first 8x8 forearm array without its last site)",
        mean_mm={f"{arm}_{n}": float(np.mean(v[arm])) for n, v in (("63", e63), ("320", e320)) for arm in pilot.ARMS},
        per_participant=dict(e63=e63, e320=e320),
        tokens_minus_continuous_63=contrast(e63["tokens"], e63["continuous"], rng),
        tokens_63_minus_320=contrast(e63["tokens"], e320["tokens"], rng),
        continuous_63_minus_320=contrast(e63["continuous"], e320["continuous"], rng))
    pilot.write(OUT / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_participant"}, indent=1))


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
        assert layout63().channels == N
        for subject in args.subjects:
            run(subject, args.device)
    else:
        report()


if __name__ == "__main__":
    main()
