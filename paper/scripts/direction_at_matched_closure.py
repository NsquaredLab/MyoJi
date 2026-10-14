"""Opening versus closing at matched hand closure, from the pose-recovery caches.

Declared procedure (fixed before running):
  * Updates: every in-cycle update of the nine participants' eight movements
    (tmp/token-umap/pose-recovery/pose_recovery_subXX.npz, the Fig. 5 analysis).
  * Closure: `from_open` (mean landmark distance from the movement's open-hand
    template, mm), rescaled per movement to [0, 1] between its 1st and 99th
    percentiles. Direction label: sign of the centred five-update moving-average
    derivative of closure within the cycle (closing = closure increasing);
    updates whose smoothed derivative is exactly zero are dropped.
  * Candidates for a query: updates of the same movement in other cycles,
    excluding the query's own cycle and the two adjacent cycles, whose rescaled
    closure lies within +/- TOLERANCE (0.05) of the query's.
  * Prediction: label of the single nearest candidate in the representation:
    dequantized 1,024-code tokens (Euclidean on the masked 5,760 values), raw
    token IDs (Hamming over 320 electrodes), pose (Euclidean on the 60 wrist-
    relative coordinates). Reference: the majority label among the candidates.
  * Accuracy per participant: mean over movements of the per-movement accuracy.
  * Primary contrast: tokens minus pose; secondary: tokens minus reference and
    IDs minus pose. Two-sided exact sign-flip over 2^9 patterns, Holm over the
    three; 10,000-draw participant bootstrap intervals (seed 260913).
  * Sensitivity (reported, not inferential): tolerance 0.025 and 0.10; k = 5
    majority vote.
Descriptive, same-session, one tokenizer; no pose decoder is trained.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import paths  # noqa: E402
from myoji.spar import grid_layout  # noqa: E402

CACHE = BASE / "tmp/token-umap/pose-recovery"
OUT = BASE / "output/figures/direction-at-matched-closure.json"
TOLERANCE, GUARD, SEED = 0.05, 1, 260913
MASK = np.tile(grid_layout().mask, (1, 6)).astype(np.float32)
torch.set_num_threads(8)


def codebook(subject):
    """The participant's 1,024-code SPAR codebook from the nine-participant campaign."""
    with np.load(paths.ASSETS / "spar/codebooks-1024.npz") as books:
        return books[f"subject_{subject:02d}/codebook"]


def labels_from_closure(closure, cycle):
    """Direction by the sign of the smoothed closure derivative within each cycle."""
    direction = np.zeros(len(closure), np.int8)
    for c in np.unique(cycle):
        ix = np.flatnonzero(cycle == c)
        x = closure[ix]
        smooth = np.convolve(np.pad(x, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        d = np.gradient(smooth)
        direction[ix] = np.sign(d).astype(np.int8)
    return direction


def pairwise(metric, a, b):
    """(n, m) distances; metric in euclid, hamming."""
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    if metric == "hamming":
        out = torch.empty(len(a), len(b))
        for s in range(0, len(a), 256):
            out[s:s + 256] = (ta[s:s + 256, None, :] != tb[None]).float().mean(-1)
        return out.numpy()
    d = ta.square().sum(-1, keepdim=True) + tb.square().sum(-1)[None] - 2 * ta @ tb.T
    return d.clamp_min(0).numpy()


def run_subject(subject, tolerance=TOLERANCE, k=1):
    z = np.load(CACHE / f"pose_recovery_sub{subject:02d}.npz")
    label, cycle, closure_mm, pose, ids = z["label"], z["cycle"], z["from_open"], z["pose"], z["ids"]
    book = codebook(subject)
    per_movement = {}
    for m in np.unique(label):
        sel = np.flatnonzero(label == m)
        lo, hi = np.percentile(closure_mm[sel], [1, 99])
        closure = np.clip((closure_mm[sel] - lo) / max(hi - lo, 1e-6), 0, 1)
        direction = labels_from_closure(closure, cycle[sel])
        keep = direction != 0
        reps = {"tokens": (book[ids[sel]] * MASK[None]).reshape(len(sel), -1).astype(np.float32),
                "ids": ids[sel].astype(np.int16), "pose": pose[sel].astype(np.float32)}
        cyc = cycle[sel]
        allowed = (np.abs(cyc[:, None] - cyc[None]) > GUARD) & (np.abs(closure[:, None] - closure[None]) <= tolerance) & keep[None]
        acc = {}
        for name, x in reps.items():
            dist = pairwise("hamming" if name == "ids" else "euclid", x, x)
            dist[~allowed] = np.inf
            if k == 1:
                nn = dist.argmin(1)
                predicted = direction[nn]
            else:
                nn = np.argpartition(dist, k, axis=1)[:, :k]
                votes = direction[nn] * np.isfinite(np.take_along_axis(dist, nn, 1))
                predicted = np.sign(votes.sum(1)).astype(np.int8)
            valid = keep & allowed.any(1) & (predicted != 0)
            acc[name] = float((predicted[valid] == direction[valid]).mean())
        # majority-label reference among each query's candidates
        n_close = (allowed & (direction[None] > 0)).sum(1)
        n_open = (allowed & (direction[None] < 0)).sum(1)
        valid = keep & ((n_close + n_open) > 0)
        acc["reference"] = float((np.maximum(n_close, n_open)[valid] / (n_close + n_open)[valid]).mean())
        per_movement[int(m)] = dict(accuracy=acc, queries=int(valid.sum()), candidates_median=float(np.median(allowed.sum(1)[valid])),
                                    share_closing=float((direction[keep] > 0).mean()), dropped_zero_derivative=int((~keep).sum()))
    summary = {name: float(np.mean([v["accuracy"][name] for v in per_movement.values()])) for name in ("tokens", "ids", "pose", "reference")}
    return dict(subject=subject, tolerance=tolerance, k=k, per_movement=per_movement, accuracy=summary)


def exact_sign_flip(d):
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    null = (signs * d).mean(1)
    return float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12))


def holm(p):
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def aggregate(results):
    acc = {name: np.array([r["accuracy"][name] for r in results]) for name in ("tokens", "ids", "pose", "reference")}
    rng = np.random.default_rng(SEED)
    contrasts = {}
    raw = []
    for a, b in (("tokens", "pose"), ("tokens", "reference"), ("ids", "pose")):
        d = acc[a] - acc[b]
        boot = np.array([d[rng.integers(0, 9, 9)].mean() for _ in range(10000)])
        contrasts[f"{a}_minus_{b}"] = dict(mean=float(d.mean()), per_participant=d.tolist(), positive_participants=int((d > 0).sum()),
                                           bootstrap95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], p_raw=exact_sign_flip(d))
        raw.append(contrasts[f"{a}_minus_{b}"]["p_raw"])
    for key, p in zip(list(contrasts), holm(raw)):
        contrasts[key]["p_holm"] = p
    return dict(mean={k: float(v.mean()) for k, v in acc.items()}, sd={k: float(v.std(ddof=1)) for k, v in acc.items()},
                range={k: [float(v.min()), float(v.max())] for k, v in acc.items()}, per_participant={k: v.tolist() for k, v in acc.items()},
                contrasts=contrasts)


def main():
    primary = [run_subject(s) for s in range(1, 10)]
    out = dict(procedure=__doc__, tolerance=TOLERANCE, k=1, guard_cycles=GUARD, seed=SEED,
               primary=aggregate(primary), per_subject=primary, sensitivity={})
    for tol, k in ((0.025, 1), (0.10, 1), (0.05, 5)):
        out["sensitivity"][f"tolerance_{tol}_k{k}"] = aggregate([run_subject(s, tol, k) for s in range(1, 10)])["mean"]
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out["primary"][k] for k in ("mean", "sd", "range")}, indent=1))
    print(json.dumps(out["primary"]["contrasts"], indent=1))
    print(json.dumps(out["sensitivity"], indent=1))


if __name__ == "__main__":
    main()
