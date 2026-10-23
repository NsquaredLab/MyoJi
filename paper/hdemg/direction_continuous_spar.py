"""Continuous SPAR under the declared opening-versus-closing protocol (the arm the main text reports as untested).

Declared before running (2026-10-10). The procedure is that of hdemg/direction_at_matched_closure.py (same
pose-recovery caches, per-movement closure rescaling, derivative labels, candidates of the same movement in other
cycles with a one-cycle guard, tolerance 0.05, single nearest candidate, movement-balanced accuracy). The only
addition is continuous SPAR: the standardized, masked 5,760-value vectors from which the 1,024-code tokens were
assigned, rebuilt from the recordings with the pose-recovery signal path
(hdemg/pose_recovery/pose_recovery.py, CPU arithmetic as there). Gates: the rebuilt vectors must reproduce
the cached token IDs, poses, movement labels and cycles exactly, and the token, ID, pose and reference accuracies
must reproduce output/figures/direction-at-matched-closure.json to within 0.001
(nearest-neighbour ties in the pose arm can flip with the thread count; deviations are recorded).

New family, declared now: tokens minus continuous SPAR (the direction cost of quantization), exact two-sided
sign-flip over 512 patterns, 10,000-draw participant bootstrap (seed 260921). Continuous SPAR minus pose is
reported descriptively. The original three-contrast family is unchanged. Runs where the recordings are:

    ~/.local/bin/uv run --no-project --with numpy --with scipy --with joblib --with torch \
        uv run python -m paper.hdemg.direction_continuous_spar run|report [--subjects ...]
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
from paper.hdemg import direction_at_matched_closure as dm  # noqa: E402
from paper.hdemg.pose_recovery import pose_recovery as pr  # noqa: E402
from myoji.codebook import fit_standardization  # noqa: E402

OUT = BASE / "results/direction_continuous_spar_v1"


def rebuild(subject):
    """Continuous SPAR for the in-cycle updates, in the cache's order (copy of pose_recovery.run_subject)."""
    info = pr.protocols()[str(subject)]
    tasks = sorted({int(w["task"]) for w in info["protocol"]})
    emg = joblib.load(pr.DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(pr.DATA / f"Sub{subject}_kinematics.pkl")
    signals = {t: pr.resample(np.ascontiguousarray(emg[str(t)] if str(t) in emg else emg[t])) for t in tasks}
    del emg
    train = [w for w in info["protocol"] if w["split"] == "train"]
    blocks = []
    for w in train:
        nz = pr.normalize(signals[int(w["task"])][:, w["start"]:w["start"] + pr.WINDOW])
        win = nz[:, pr.ENDS[:, None] - np.arange(319, -1, -1)].transpose(1, 2, 0).copy()
        blocks.append(pr.spar_simple_dataset_batched(win, channelwise=True))
    stacked = np.stack(blocks).transpose(0, 2, 1, 3)
    mean, scale = fit_standardization(stacked, pr.MASK)
    book = pr.codebook(subject)
    del blocks, stacked
    conts, idss, poses, labels, cycles = [], [], [], [], []
    for i, task in enumerate(tasks):
        signal = signals[task]
        starts = np.arange(0, signal.shape[1] - pr.WINDOW + 1, pr.STRIDE)
        spar, ends = [], []
        for s in starts:
            nz = pr.normalize(signal[:, s:s + pr.WINDOW])
            win = nz[:, pr.ENDS[:, None] - np.arange(319, -1, -1)].transpose(1, 2, 0).copy()
            spar.append(((pr.spar_simple_dataset_batched(win, channelwise=True) - mean) / scale) * pr.MASK[None])
            ends.extend(s + pr.ENDS)
        spar, ends = np.concatenate(spar), np.array(ends)
        frames = len(ends)
        points = np.asarray(kin[str(task)])[..., (ends * 128) // 125].transpose(2, 0, 1).astype(np.float32)
        truth = points[:, 1:] - points[:, :1]
        spread = np.linalg.norm(truth[:, pr.TIPS, :], axis=2).mean(1)
        cyc, _ = pr.full_cycles(pr.detect_cycles(spread))
        phase = pr.cyclic_phase(cyc, frames)
        inside = np.isfinite(phase)
        cycle_of = np.full(frames, -1)
        for c, (a, b) in enumerate(cyc):
            cycle_of[a:b] = c
        ids = pr.nearest(spar.reshape(-1, 18), book).reshape(frames, 320)
        conts.append(spar[inside].reshape(inside.sum(), -1).astype(np.float32))
        idss.append(ids[inside].astype(np.uint16))
        poses.append(truth[inside].reshape(inside.sum(), -1).astype(np.float32))
        labels.append(np.full(inside.sum(), i))
        cycles.append(cycle_of[inside] + 1000 * i)
    return (np.concatenate(conts), np.concatenate(idss), np.concatenate(poses),
            np.concatenate(labels), np.concatenate(cycles))


def direction(subject, cont, tolerance=dm.TOLERANCE):
    """dm.run_subject with continuous SPAR added as a fourth representation (k = 1)."""
    z = np.load(dm.CACHE / f"pose_recovery_sub{subject:02d}.npz")
    label, cycle, closure_mm, pose, ids = z["label"], z["cycle"], z["from_open"], z["pose"], z["ids"]
    book = dm.codebook(subject)
    per_movement = {}
    for m in np.unique(label):
        sel = np.flatnonzero(label == m)
        lo, hi = np.percentile(closure_mm[sel], [1, 99])
        closure = np.clip((closure_mm[sel] - lo) / max(hi - lo, 1e-6), 0, 1)
        direction_label = dm.labels_from_closure(closure, cycle[sel])
        keep = direction_label != 0
        reps = {"tokens": (book[ids[sel]] * dm.MASK[None]).reshape(len(sel), -1).astype(np.float32),
                "ids": ids[sel].astype(np.int16), "pose": pose[sel].astype(np.float32),
                "continuous": cont[sel].astype(np.float32)}
        cyc = cycle[sel]
        allowed = (np.abs(cyc[:, None] - cyc[None]) > dm.GUARD) & (np.abs(closure[:, None] - closure[None]) <= tolerance) & keep[None]
        acc = {}
        for name, x in reps.items():
            dist = dm.pairwise("hamming" if name == "ids" else "euclid", x, x)
            dist[~allowed] = np.inf
            predicted = direction_label[dist.argmin(1)]
            valid = keep & allowed.any(1) & (predicted != 0)
            acc[name] = float((predicted[valid] == direction_label[valid]).mean())
        n_close = (allowed & (direction_label[None] > 0)).sum(1)
        n_open = (allowed & (direction_label[None] < 0)).sum(1)
        valid = keep & ((n_close + n_open) > 0)
        majority = np.where(n_close >= n_open, 1, -1)
        acc["reference"] = float((majority[valid] == direction_label[valid]).mean())
        per_movement[int(m)] = acc
    return {name: float(np.mean([v[name] for v in per_movement.values()])) for name in ("tokens", "ids", "pose", "reference", "continuous")}


def run(subject):
    path = OUT / f"subject_{subject:02d}.json"
    if path.exists():
        return
    t0 = time.monotonic()
    cont, ids, pose, label, cycle = rebuild(subject)
    z = np.load(dm.CACHE / f"pose_recovery_sub{subject:02d}.npz")
    gates = dict(ids=bool(np.array_equal(ids, z["ids"])), pose=bool(np.allclose(pose, z["pose"])),
                 label=bool(np.array_equal(label, z["label"])), cycle=bool(np.array_equal(cycle, z["cycle"])))
    assert all(gates.values()), (subject, gates)
    acc = direction(subject, cont)
    stored = json.loads(dm.OUT.read_text())["per_subject"][subject - 1]["accuracy"]
    reproduced = {k: abs(acc[k] - stored[k]) for k in ("tokens", "ids", "pose", "reference")}  # nearest-neighbour ties can flip with thread count
    assert max(reproduced.values()) < 1e-3, (subject, reproduced)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(subject=subject, accuracy=acc, gates=gates, reproduced=reproduced,
                                    seconds=time.monotonic() - t0), indent=1) + "\n")
    print(json.dumps(dict(subject=subject, accuracy=acc, seconds=round(time.monotonic() - t0))), flush=True)


def exact_p(d):
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))


def report():
    rows = [json.loads((OUT / f"subject_{s:02d}.json").read_text())["accuracy"] for s in range(1, 10)]
    acc = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    rng = np.random.default_rng(260921)
    out = dict(mean={k: float(v.mean()) for k, v in acc.items()}, sd={k: float(v.std(ddof=1)) for k, v in acc.items()},
               per_participant={k: v.tolist() for k, v in acc.items()})
    for a, b, role in (("tokens", "continuous", "tested (new one-contrast family)"), ("continuous", "pose", "descriptive")):
        d = acc[a] - acc[b]
        boot = np.array([d[rng.integers(0, 9, 9)].mean() for _ in range(10000)])
        out[f"{a}_minus_{b}"] = dict(role=role, mean=float(d.mean()), positive=int((d > 0).sum()), per_participant=d.tolist(),
                                     bootstrap95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], p_exact=exact_p(d))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_participant"}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    args = parser.parse_args()
    if args.stage == "run":
        for s in args.subjects:
            run(s)
    else:
        report()


if __name__ == "__main__":
    main()
