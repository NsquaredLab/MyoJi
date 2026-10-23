"""Do tokens carry the hand's pose? Nearest-repetition pose recovery, all participants.

For every in-cycle frame of a participant's eight movements: find its nearest frame in
each OTHER repetition cycle (own cycle and its two neighbours excluded) in some
representation, keep the five closest cycles, average their measured poses, and
measure the mean 3D landmark distance (mm) to the frame's own pose. Done for the
continuous SPAR features, the 1,024-code tokens the decoder sees (dequantized), the
bare token IDs (Hamming), a 64-code book, and, as the ceiling, pose space itself.
Controls re-pair poses with frames (phase shift within each cycle; global shuffle)
without touching the token-space neighbours, which is exactly what shifting the
tokens would do, so a thousand replicates cost nothing.

Runs where the recordings live (the workstation), sha-gated like cross_subject.py:

    cd <repository root>
    uv run python -m paper.hdemg.pose_recovery.pose_recovery --subjects 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.signal import butter, detrend, find_peaks, firwin, sosfiltfilt, upfirdn

from myoji.codebook import fit_codebook, fit_standardization
from myoji.spar import grid_layout, spar_simple_dataset_batched

HERE = Path(__file__).resolve().parent
from paper.common import paths  # noqa: E402

DATA = paths.HDEMG
RECOVERY = paths.RESULTS / "token_umap/pose_recovery"

TAPS = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
WINDOW, STRIDE, LEAD, HOP = 6000, 4000, 2039, 40
ENDS = LEAD + HOP * np.arange(100)
TIPS = [0, 4, 8, 12, 16]                 # fingertips in the 20-point wrist-relative truth
SMALL, SEED, K_CYCLES, GUARD, REPLICATES = 64, 42, 5, 1, 1000
NAMES = {2: "Thumb", 4: "Index", 6: "Middle", 8: "Ring", 10: "Little", 15: "Fist",
         18: "Two-finger pinch", 19: "Three-finger pinch"}
torch.set_num_threads(32)

MASK = np.tile(grid_layout().mask, (1, 6))


def codebook(subject):
    """The participant's 1,024-code SPAR codebook from the nine-participant campaign."""
    with np.load(paths.ASSETS / "spar/codebooks-1024.npz") as books:
        return books[f"subject_{subject:02d}/codebook"]


def protocols():
    """Per participant: the SPAR campaign's windows and the sha256 of its normalized raw windows."""
    campaign = paths.RESULTS / "hdemg_spar_simple_full_v1"
    out = {}
    for subject in range(1, 10):
        folder = campaign / f"subject_{subject:02d}"
        out[str(subject)] = dict(
            protocol=json.loads((folder / "protocol.json").read_text())["windows"],
            normalized_raw_sha256=json.loads((folder / "cache-receipt.json").read_text())["normalized_raw_sha256"])
    return out


# ---- signal path, identical to cross_subject.py / build_token_umap_cache.py ----------
def resample(raw):
    n = raw.shape[-1]
    return upfirdn(TAPS, raw.astype(np.float32), up=125, down=128, axis=-1)[..., : n * 125 // 128].astype(np.float32)


def normalize(block):
    x = torch.from_numpy(block)[None]
    centre = x[..., :1000].sort(-1).values[..., 499:500]
    scale = (x[..., :1000] - centre).abs().sort(-1).values[..., 499:500] * 1.4826
    return ((x - centre) / scale.clamp_min(1e-4)).clamp(-12, 12)[0].numpy()


def nearest(x, book):
    f = torch.from_numpy(np.ascontiguousarray(x))
    b = torch.from_numpy(np.ascontiguousarray(book))
    return (f.square().sum(-1, keepdim=True) - 2 * f @ b.T + b.square().sum(-1)[None]).argmin(-1).numpy()


# ---- cycles, copied from emg_manifolds.data.cycles and make_token_manifold_figure ----
def detect_cycles(reference, fs=50.0, prominence=0.2):
    signal = detrend(np.asarray(reference, dtype=float), axis=0)
    if len(signal) > 16:
        signal = sosfiltfilt(butter(4, 1.0, btype="low", fs=fs, output="sos"), signal)
    peaks, _ = find_peaks(signal, prominence=prominence)
    troughs, _ = find_peaks(-signal)
    cycles = [(peaks[i], peaks[i + 1]) for i in range(len(peaks) - 1)
              if any(peaks[i] < t < peaks[i + 1] for t in troughs)]
    return np.asarray(cycles, dtype=np.intp).reshape(-1, 2)


def full_cycles(cycles, ratio=1.8):
    lengths = cycles[:, 1] - cycles[:, 0]
    middle = np.median(lengths)
    keep = (lengths >= middle / ratio) & (lengths <= middle * ratio)
    return cycles[keep], int((~keep).sum())


def cyclic_phase(cycles, n):
    phase = np.full(n, np.nan)
    for start, end in cycles:
        for j in range(int(start), min(int(end), n)):
            phase[j] = (j - start) / (end - start)
    return phase


# ---- neighbour rule --------------------------------------------------------------------
def nearest_per_cycle(x, cycle, allowed_cycles, k=K_CYCLES, metric="euclid", chunk=512):
    """For every row: its nearest row in each allowed other cycle, keep the k closest
    cycles. Returns (n, k) row indices and (n, k) distances."""
    n = len(x)
    order = np.argsort(cycle, kind="stable")
    sorted_cycle = cycle[order]
    bounds = np.flatnonzero(np.diff(sorted_cycle)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    cycle_ids = sorted_cycle[starts]
    out_idx = np.empty((n, k), np.int64)
    out_d = np.empty((n, k), np.float32)
    xt = torch.from_numpy(np.ascontiguousarray(x))
    if metric == "euclid":
        sq = xt.square().sum(1)
        xs = xt[order]
    else:
        chunk = 48                              # (chunk, n, 320) bool per step: keep it ~250 MB
    for s in range(0, n, chunk):
        q = xt[s:s + chunk]
        if metric == "euclid":
            d = (sq[s:s + chunk, None] - 2 * q @ xs.T + sq[order][None]).clamp_min(0)
        else:                                   # hamming on uint16 IDs, as a fraction
            d = (q[:, None, :] != xt[order][None]).to(torch.float32).mean(2)
        d = d.numpy()
        per = np.full((len(q), len(cycle_ids)), np.inf, np.float32)
        arg = np.zeros((len(q), len(cycle_ids)), np.int64)
        for c, (a, b) in enumerate(zip(starts, ends)):
            seg = d[:, a:b]
            j = seg.argmin(1)
            per[:, c] = seg[np.arange(len(q)), j]
            arg[:, c] = order[a + j]
        ok = allowed_cycles[s:s + chunk][:, :]            # (chunk, n_cycles)
        per[~ok] = np.inf
        top = np.argpartition(per, k, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        out_idx[s:s + chunk] = arg[rows, top]
        out_d[s:s + chunk] = per[rows, top]
    assert np.isfinite(out_d).all(), "a frame had fewer allowed cycles than k"
    return out_idx, out_d, cycle_ids


def pose_err(nn, pose_of_query, pose_of_ref):
    pred = pose_of_ref[nn].mean(1)
    return np.linalg.norm((pred - pose_of_query).reshape(len(nn), -1, 3), axis=2).mean(1)


# ---- per participant -------------------------------------------------------------------
def run_subject(subject):
    t0 = time.monotonic()
    info = protocols()[str(subject)]
    tasks = sorted({int(w["task"]) for w in info["protocol"]})
    emg = joblib.load(DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(DATA / f"Sub{subject}_kinematics.pkl")
    signals = {t: resample(np.ascontiguousarray(emg[str(t)] if str(t) in emg else emg[t])) for t in tasks}
    del emg

    digest = hashlib.sha256()
    for w in info["protocol"]:
        digest.update(normalize(signals[int(w["task"])][:, w["start"]:w["start"] + WINDOW]).tobytes())
    matched = digest.hexdigest() == info["normalized_raw_sha256"]
    assert matched, f"subject {subject}: local recording is not the manuscript's"

    train = [w for w in info["protocol"] if w["split"] == "train"]
    blocks = []
    for w in train:
        nz = normalize(signals[int(w["task"])][:, w["start"]:w["start"] + WINDOW])
        win = nz[:, ENDS[:, None] - np.arange(319, -1, -1)].transpose(1, 2, 0).copy()
        blocks.append(spar_simple_dataset_batched(win, channelwise=True))
    stacked = np.stack(blocks).transpose(0, 2, 1, 3)
    mean, scale = fit_standardization(stacked, MASK)
    book = codebook(subject)
    standardized = ((stacked - mean) / scale) * MASK[None, :, None, :]
    small_book, _ = fit_codebook(standardized.reshape(-1, 18), vocabulary=SMALL, seed=SEED, device="cpu")
    del blocks, stacked, standardized
    train_ranges = [(int(w["task"]), w["start"], w["start"] + WINDOW) for w in train]

    parts = []
    for task in tasks:
        signal = signals[task]
        starts = np.arange(0, signal.shape[1] - WINDOW + 1, STRIDE)
        spar, ends = [], []
        for s in starts:
            nz = normalize(signal[:, s:s + WINDOW])
            win = nz[:, ENDS[:, None] - np.arange(319, -1, -1)].transpose(1, 2, 0).copy()
            spar.append(((spar_simple_dataset_batched(win, channelwise=True) - mean) / scale) * MASK[None])
            ends.extend(s + ENDS)
        spar, ends = np.concatenate(spar), np.array(ends)          # (frames, 320, 18)
        frames = len(ends)
        # kinematics are sampled at 2048 Hz, the EMG grid at 2000 Hz
        points = np.asarray(kin[str(task)])[..., (ends * 128) // 125].transpose(2, 0, 1).astype(np.float32)
        truth = points[:, 1:] - points[:, :1]                         # (frames, 20, 3) mm, wrist-relative
        spread = np.linalg.norm(truth[:, TIPS, :], axis=2).mean(1)
        cycles, dropped = full_cycles(detect_cycles(spread))
        phase = cyclic_phase(cycles, frames)
        inside = np.isfinite(phase)
        cycle_of = np.full(frames, -1)
        for c, (a, b) in enumerate(cycles):
            cycle_of[a:b] = c
        in_train = np.zeros(frames, bool)
        for t, a, b in train_ranges:
            if t == task:
                in_train |= (ends >= a) & (ends < b)
        ids = nearest(spar.reshape(-1, 18), book).reshape(frames, 320)
        small = nearest(spar.reshape(-1, 18), small_book).reshape(frames, 320)
        keep = inside
        parts.append(dict(task=task, cont=spar[keep].reshape(keep.sum(), -1).astype(np.float32),
                          ids=ids[keep].astype(np.uint16), small=small[keep].astype(np.uint16),
                          pose=truth[keep].reshape(keep.sum(), -1), phase=phase[keep],
                          cycle=cycle_of[keep], n_cycles=len(cycles), dropped=dropped,
                          in_train=in_train[keep], frames=frames, ends=ends[keep]))
        print(json.dumps(dict(subject=subject, task=task, frames=frames, in_cycle=int(keep.sum()),
                              cycles=len(cycles), partial_dropped=dropped)), flush=True)
    del signals, kin

    # ---- pool the eight movements ----
    cont = np.concatenate([p["cont"] for p in parts])
    ids = np.concatenate([p["ids"] for p in parts])
    small_ids = np.concatenate([p["small"] for p in parts])
    pose = np.concatenate([p["pose"] for p in parts]).astype(np.float32)
    phase = np.concatenate([p["phase"] for p in parts])
    label = np.concatenate([np.full(len(p["cont"]), i) for i, p in enumerate(parts)])
    gcycle = np.concatenate([p["cycle"] + 1000 * i for i, p in enumerate(parts)])
    in_train = np.concatenate([p["in_train"] for p in parts])
    n = len(label)
    tok = (book[ids] * MASK[None]).reshape(n, -1).astype(np.float32)
    small_tok = (small_book[small_ids] * MASK[None]).reshape(n, -1).astype(np.float32)

    if PROBE_ONLY:
        return linear_probe(subject, dict(continuous=cont, tokens=tok, tokens_64=small_tok), pose, label, gcycle)

    # allowed cycles per frame: not its own, not the adjacent ones of the same movement
    uniq = np.unique(gcycle)
    col = {c: i for i, c in enumerate(uniq)}
    allowed = np.ones((n, len(uniq)), bool)
    for i in range(n):
        c = gcycle[i]
        for g in range(-GUARD, GUARD + 1):
            if c + g in col:
                allowed[i, col[c + g]] = False

    searches = {"continuous": (cont, "euclid"), "tokens": (tok, "euclid"),
                "tokens_hamming": (ids.astype(np.int32), "hamming"), "tokens_64": (small_tok, "euclid"),
                "pose_oracle": (pose, "euclid")}
    nn, dist, err = {}, {}, {}
    for name, (x, metric) in searches.items():
        t1 = time.monotonic()
        nn[name], dist[name], cycle_ids = nearest_per_cycle(x, gcycle, allowed, metric=metric)
        assert (cycle_ids == uniq).all()
        err[name] = pose_err(nn[name], pose, pose)
        print(json.dumps(dict(subject=subject, search=name, median_mm=round(float(np.median(err[name])), 3),
                              seconds=round(time.monotonic() - t1, 1))), flush=True)

    # ---- controls: re-pair poses, neighbours untouched ----
    rng = np.random.default_rng(SEED + subject)
    starts_c = {c: np.flatnonzero(gcycle == c) for c in uniq}
    shifted = np.empty((REPLICATES, n), np.int64)
    for r in range(REPLICATES):
        s = np.arange(n)
        for c, idx in starts_c.items():
            L = len(idx)
            s[idx] = idx[(np.arange(L) + rng.integers(0, L)) % L]
        shifted[r] = s
    control = {}
    for name in ("tokens", "continuous"):
        e = np.array([np.median(pose_err(nn[name], pose[shifted[r]], pose[shifted[r]])) for r in range(REPLICATES)])
        control[f"{name}_phase_shift_median_mm"] = [float(np.median(e)), float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]
    perm = rng.permutation(n)
    control["tokens_global_shuffle_median_mm"] = float(np.median(pose_err(nn["tokens"], pose[perm], pose[perm])))
    movement_mean = np.stack([pose[label == c].mean(0) for c in range(len(parts))])
    control["movement_mean_pose_median_mm"] = float(np.median(
        np.linalg.norm((movement_mean[label] - pose).reshape(n, -1, 3), axis=2).mean(1)))

    # displacement from the movement's open-hand template (first/last 5 % of each cycle)
    open_tpl = np.zeros_like(movement_mean)
    for c in range(len(parts)):
        m = (label == c) & ((phase < .05) | (phase > .95))
        open_tpl[c] = pose[m].mean(0)
    from_open = np.linalg.norm((pose - open_tpl[label]).reshape(n, -1, 3), axis=2).mean(1)

    out = RECOVERY / f"pose_recovery_sub{subject:02d}.npz"
    np.savez_compressed(out, label=label, task=np.array([p["task"] for p in parts]), phase=phase, cycle=gcycle,
                        in_train=in_train, from_open=from_open, pose=pose, ids=ids,
                        **{f"err_{k}": v for k, v in err.items()}, **{f"nn_{k}": v for k, v in nn.items()},
                        **{f"dist_{k}": v for k, v in dist.items()})
    summary = dict(subject=subject, raw_hash_matches=matched, frames_in_cycle=n,
                   per_task={NAMES[p["task"]]: dict(in_cycle=len(p["cont"]), cycles=p["n_cycles"], partial_dropped=p["dropped"]) for p in parts},
                   rule=f"nearest frame in each other cycle (own cycle +/- {GUARD} excluded), {K_CYCLES} closest cycles, mean pose",
                   median_mm={k: float(np.median(v)) for k, v in err.items()},
                   outside_train_windows_median_mm={k: float(np.median(v[~in_train])) for k, v in err.items()},
                   controls=control, replicates=REPLICATES, seed=SEED, minutes=round((time.monotonic() - t0) / 60, 1))
    (RECOVERY / f"pose_recovery_sub{subject:02d}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


PROBE_ONLY = False


def linear_probe(subject, reps, pose, label, gcycle, folds=5, alphas=(1e1, 1e2, 1e3, 1e4, 1e5)):
    """Ridge regression representation -> 60 landmark coordinates, cross-validated by
    repetition: cycles are dealt into 5 folds by their index within each movement, so
    a test frame never shares a cycle with a training frame. Error = mean 3D distance
    over the 20 landmarks (mm). The same alpha grid and folds for every representation;
    the best alpha per representation is reported (equal optimism for all)."""
    within = np.array([c % 1000 for c in gcycle])
    fold = within % folds
    out = {}
    for name, x in reps.items():
        X = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        Y = torch.from_numpy(pose.astype(np.float32))
        errs = {a: [] for a in alphas}
        for f in range(folds):
            tr, te = torch.from_numpy(fold != f), torch.from_numpy(fold == f)
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ym = Y[tr].mean(0)
            G = Xtr.T @ Xtr; B = Xtr.T @ (Y[tr] - ym)
            for a in alphas:
                W = torch.linalg.solve(G + a * torch.eye(G.shape[0]), B)
                pred = Xte @ W + ym
                errs[a].append(torch.linalg.norm((pred - Y[te]).reshape(-1, 20, 3), dim=2).mean(1).numpy())
        mean_err = {a: float(np.concatenate(e).mean()) for a, e in errs.items()}
        best = min(mean_err, key=mean_err.get)
        out[name] = dict(mm=mean_err[best], alpha=best, all_alphas=mean_err)
        print(json.dumps(dict(subject=subject, probe=name, mm=round(mean_err[best], 3), alpha=best)), flush=True)
    (RECOVERY / f"probe_sub{subject:02d}.json").write_text(json.dumps(dict(subject=subject, folds=folds, probe=out), indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--probe", action="store_true", help="linear probe only, skip the neighbour searches")
    args = ap.parse_args()
    PROBE_ONLY = args.probe
    RECOVERY.mkdir(parents=True, exist_ok=True)
    for s in args.subjects:
        run_subject(s)
