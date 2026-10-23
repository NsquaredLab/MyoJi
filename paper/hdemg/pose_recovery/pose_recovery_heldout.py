"""Pose recovery on held-out data: test-window queries, candidate cycles outside the training windows.

Same features, codebooks, cycles and neighbour rule as pose_recovery.py (imported, unchanged), with two
restrictions. Queries are updates whose 320-sample analysis window lies inside one of the decoding
protocol's test windows. Candidate cycles are repetition cycles in which no update's analysis window
overlaps a training window, i.e. the windows that fitted the SPAR standardization and the codebooks.
Before searching, the recomputed inputs are checked against the saved pose_recovery_sub*.npz.

    cd <repository root>
    ~/.local/bin/uv run --no-project --with numpy --with scipy --with joblib --with torch==2.14.1 \
        python -m paper.hdemg.pose_recovery.pose_recovery_heldout --subjects 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import torch

from paper.hdemg.pose_recovery import pose_recovery as pr
from paper.hdemg.pose_recovery.aggregate import balanced

torch.set_num_threads(16)               # leave cores for the GPU campaign running alongside
LOOKBACK = 319                          # an update ending at sample e analyses samples e-319 .. e


def overlaps(ends, ranges, task):
    """Updates whose analysis window intersects any [a, b) of this task."""
    hit = np.zeros(len(ends), bool)
    for t, a, b in ranges:
        if t == task:
            hit |= (ends >= a) & (ends - LOOKBACK < b)
    return hit


def within(ends, ranges, task):
    """Updates whose analysis window lies inside some [a, b) of this task."""
    hit = np.zeros(len(ends), bool)
    for t, a, b in ranges:
        if t == task:
            hit |= (ends - LOOKBACK >= a) & (ends < b)
    return hit


def run_subject(subject):
    t0 = time.monotonic()
    info = pr.protocols()[str(subject)]
    saved = np.load(pr.RECOVERY / f"pose_recovery_sub{subject:02d}.npz")
    tasks = sorted({int(w["task"]) for w in info["protocol"]})
    emg = joblib.load(pr.DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(pr.DATA / f"Sub{subject}_kinematics.pkl")
    signals = {t: pr.resample(np.ascontiguousarray(emg[str(t)] if str(t) in emg else emg[t])) for t in tasks}
    del emg
    digest = hashlib.sha256()
    for w in info["protocol"]:
        digest.update(pr.normalize(signals[int(w["task"])][:, w["start"]:w["start"] + pr.WINDOW]).tobytes())
    assert digest.hexdigest() == info["normalized_raw_sha256"], f"subject {subject}: local recording is not the manuscript's"
    ranges = {split: [(int(w["task"]), w["start"], w["start"] + pr.WINDOW) for w in info["protocol"] if w["split"] == split]
              for split in ("train", "test")}

    # Standardization and the 64-code book, exactly as in pose_recovery.py.
    blocks = []
    for task, a, _ in ranges["train"]:
        nz = pr.normalize(signals[task][:, a:a + pr.WINDOW])
        blocks.append(pr.spar_simple_dataset_batched(nz[:, pr.ENDS[:, None] - np.arange(319, -1, -1)].transpose(1, 2, 0).copy(),
                                                     channelwise=True))
    stacked = np.stack(blocks).transpose(0, 2, 1, 3)
    mean, scale = pr.fit_standardization(stacked, pr.MASK)
    book = pr.codebook(subject)
    standardized = ((stacked - mean) / scale) * pr.MASK[None, :, None, :]
    small_book, _ = pr.fit_codebook(standardized.reshape(-1, 18), vocabulary=pr.SMALL, seed=pr.SEED, device="cpu")
    del blocks, stacked, standardized

    parts = []
    for task in tasks:
        signal = signals[task]
        spar, ends = [], []
        for s in np.arange(0, signal.shape[1] - pr.WINDOW + 1, pr.STRIDE):
            nz = pr.normalize(signal[:, s:s + pr.WINDOW])
            win = nz[:, pr.ENDS[:, None] - np.arange(319, -1, -1)].transpose(1, 2, 0).copy()
            spar.append(((pr.spar_simple_dataset_batched(win, channelwise=True) - mean) / scale) * pr.MASK[None])
            ends.extend(s + pr.ENDS)
        spar, ends = np.concatenate(spar), np.array(ends)
        frames = len(ends)
        points = np.asarray(kin[str(task)])[..., (ends * 128) // 125].transpose(2, 0, 1).astype(np.float32)
        truth = points[:, 1:] - points[:, :1]
        cycles, _ = pr.full_cycles(pr.detect_cycles(np.linalg.norm(truth[:, pr.TIPS, :], axis=2).mean(1)))
        phase = pr.cyclic_phase(cycles, frames)
        keep = np.isfinite(phase)
        cycle_of = np.full(frames, -1)
        for c, (a, b) in enumerate(cycles):
            cycle_of[a:b] = c
        in_train = np.zeros(frames, bool)            # the published definition: end inside a training window
        for t, a, b in ranges["train"]:
            if t == task:
                in_train |= (ends >= a) & (ends < b)
        parts.append(dict(cont=spar[keep].reshape(keep.sum(), -1).astype(np.float32),
                          ids=pr.nearest(spar.reshape(-1, 18), book).reshape(frames, 320)[keep].astype(np.uint16),
                          small=pr.nearest(spar.reshape(-1, 18), small_book).reshape(frames, 320)[keep].astype(np.uint16),
                          pose=truth[keep].reshape(keep.sum(), -1), phase=phase[keep], cycle=cycle_of[keep],
                          in_train=in_train[keep], query=within(ends, ranges["test"], task)[keep],
                          touches_train=overlaps(ends, ranges["train"], task)[keep]))
    del signals, kin

    cat = lambda key: np.concatenate([p[key] for p in parts])
    cont, ids, small_ids, pose, phase = cat("cont"), cat("ids"), cat("small"), cat("pose").astype(np.float32), cat("phase")
    label = np.concatenate([np.full(len(p["cont"]), i) for i, p in enumerate(parts)])
    gcycle = np.concatenate([p["cycle"] + 1000 * i for i, p in enumerate(parts)])
    query, touches = cat("query"), cat("touches_train")
    # Inputs must be the published ones before any restriction is applied.
    for key, value in (("label", label), ("cycle", gcycle), ("phase", phase), ("pose", pose), ("in_train", cat("in_train"))):
        assert np.array_equal(value, saved[key]), f"subject {subject}: {key} differs from pose_recovery_sub{subject:02d}.npz"
    id_agreement = float((ids == saved["ids"]).mean())
    assert id_agreement > .999, id_agreement
    n = len(label)
    tok = (book[ids] * pr.MASK[None]).reshape(n, -1).astype(np.float32)
    small_tok = (small_book[small_ids] * pr.MASK[None]).reshape(n, -1).astype(np.float32)

    uniq = np.unique(gcycle)
    heldout = np.array([not touches[gcycle == c].any() for c in uniq])
    col = {c: i for i, c in enumerate(uniq)}
    allowed = np.repeat(heldout[None], n, axis=0)
    for i in range(n):
        for g in range(-pr.GUARD, pr.GUARD + 1):
            if gcycle[i] + g in col:
                allowed[i, col[gcycle[i] + g]] = False

    searches = {"continuous": (cont, "euclid"), "tokens": (tok, "euclid"),
                "tokens_hamming": (ids.astype(np.int32), "hamming"), "tokens_64": (small_tok, "euclid"),
                "pose_oracle": (pose, "euclid")}
    nn, err = {}, {}
    for name, (x, metric) in searches.items():
        nn[name], _, cycle_ids = pr.nearest_per_cycle(x, gcycle, allowed, metric=metric)
        assert (cycle_ids == uniq).all() and not touches[nn[name]].any()
        err[name] = pr.pose_err(nn[name], pose, pose)
        print(json.dumps(dict(subject=subject, search=name, query_median_mm=round(float(np.median(err[name][query])), 3))), flush=True)

    rng = np.random.default_rng(pr.SEED + subject)
    members = {c: np.flatnonzero(gcycle == c) for c in uniq}
    shift = []
    for _ in range(pr.REPLICATES):
        s = np.arange(n)
        for idx in members.values():
            s[idx] = idx[(np.arange(len(idx)) + rng.integers(0, len(idx))) % len(idx)]
        shift.append(float(np.median(pr.pose_err(nn["tokens"], pose[s], pose[s])[query])))

    q = query
    summary = dict(subject=subject, raw_hash_matches=True, token_id_agreement_with_saved=id_agreement,
                   queries=int(q.sum()), frames_in_cycle=n, query_share=float(q.mean()),
                   heldout_cycles=int(heldout.sum()), cycles=len(uniq),
                   heldout_cycles_per_movement={pr.NAMES[p_task]: int(heldout[[c // 1000 == i for c in uniq]].sum())
                                               for i, p_task in enumerate(tasks)},
                   rule=f"test-window queries; nearest frame in each held-out cycle (own cycle +/- {pr.GUARD} excluded), "
                        f"{pr.K_CYCLES} closest cycles, mean pose",
                   balanced_mm={k: balanced(v[q], label[q], gcycle[q], phase[q])[0] for k, v in err.items()},
                   median_mm={k: float(np.median(v[q])) for k, v in err.items()},
                   tokens_phase_shift_median_mm=[float(np.median(shift)), float(np.percentile(shift, 2.5)), float(np.percentile(shift, 97.5))],
                   replicates=pr.REPLICATES, seed=pr.SEED, minutes=round((time.monotonic() - t0) / 60, 1))
    np.savez_compressed(pr.RECOVERY / f"pose_recovery_heldout_sub{subject:02d}.npz", query=q, heldout_cycles=uniq[heldout],
                        touches_train=touches, **{f"err_{k}": v for k, v in err.items()})
    (pr.RECOVERY / f"pose_recovery_heldout_sub{subject:02d}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    for s in ap.parse_args().subjects:
        run_subject(s)
