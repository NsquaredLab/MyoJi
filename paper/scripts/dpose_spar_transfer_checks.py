"""Two frozen-pipeline checks of the nine-participant SPAR campaign, plus a task census.

Nothing is fitted: standardization, codebooks and the six decoders per participant are the
frozen ones of results/dpose_spar_simple_full_v1 (checkpoints) and the regenerated cache
of results/stm32n6_pose_validation_v1/prepare (standardization statistics, which reproduce
the campaign's decoder error within 0.003 mm per fit).

census    For every task recording of every participant, the dominant frequency of the mean
          fingertip spread and the per-digit movement amplitude, to identify the 1.5 Hz
          counterparts of the evaluated 0.5 Hz movements without guessing task identifiers.
evaluate  (a) Fixed-startup calibration: the 16 test windows normalized with the median/MAD
          of the recording's first 500 ms instead of each window's own first 500 ms,
          scored by the frozen token and continuous decoders beside the per-window
          baseline recomputed in the same run, with token agreement between the two
          calibrations. (b) Transfer: every task not among the eight evaluated ones,
          windowed like the campaign (6,000 samples, stride 4,000, per-window
          calibration), scored by the frozen decoders of both arms.
report    Participant-level aggregation with paired contrasts (exact sign-flip over 512
          patterns, 10,000-draw participant bootstrap, seed 260917).

    ~/.local/bin/uv run --no-project --with torch --with h5py --with joblib --with scipy --with numpy \
        python scripts/dpose_spar_transfer_checks.py census|evaluate|report --subjects 1 2 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import firwin, upfirdn

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import paths  # noqa: E402
import run_dpose_spar_simple as pilot  # noqa: E402
from myoji.codebook import standardize, tokenize  # noqa: E402
from myoji.spar import grid_layout, spar_simple_dataset_batched  # noqa: E402

CAMPAIGN = BASE / "results/dpose_spar_simple_full_v1"
CACHES = BASE / "results/stm32n6_pose_validation_v1/prepare"
DATA = paths.HDEMG
OUT = BASE / "results/dpose_spar_transfer_checks_v1"
EVALUATED = (2, 4, 6, 8, 10, 15, 18, 19)
SEEDS, ENDS = pilot.SEEDS, pilot.ENDS
TAPS = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
INDICES = ENDS[:, None] - np.arange(319, -1, -1)[None]
TIPS = (0, 4, 8, 12, 16)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1) + "\n")


def resample(raw):
    return upfirdn(TAPS, raw.astype(np.float32), up=125, down=128, axis=-1)[..., :raw.shape[-1] * 125 // 128].astype(np.float32)


def normalize_with(block, reference):
    """Median/MAD of the reference's first 1,000 samples applied to block (same arithmetic as the release)."""
    x = torch.from_numpy(block)[None]
    r = torch.from_numpy(reference)[None]
    centre = r[..., :1000].sort(-1).values[..., 499:500]
    scale = (r[..., :1000] - centre).abs().sort(-1).values[..., 499:500] * 1.4826
    return ((x - centre) / scale.clamp_min(1e-4)).clamp(-12, 12)[0].numpy()


def targets_for(kin, start):
    points = kin[..., ((start + ENDS) * 128) // 125].transpose(2, 0, 1).astype(np.float32)
    return (points[:, 1:] - points[:, :1]).reshape(100, 60)


# ----------------------------------------------------------------------------- census
def census(subjects):
    rows = []
    for subject in subjects:
        kin = joblib.load(DATA / f"Sub{subject}_kinematics.pkl")
        for task in sorted(kin, key=int):
            points = kin[task]                              # (21, 3, T) at 2,048 Hz
            rel = points[1:] - points[:1]                    # 20 wrist-relative landmarks
            dist = np.linalg.norm(rel, axis=1)               # (20, T) distance from the wrist
            spread = dist[list(TIPS)].mean(0)
            x = spread - spread.mean()
            n = len(x)
            spectrum = np.abs(np.fft.rfft(x * np.hanning(n)))
            freqs = np.fft.rfftfreq(n, 1 / 2048)
            band = (freqs >= 0.2) & (freqs <= 3.0)
            peak = float(freqs[band][spectrum[band].argmax()])
            rows.append(dict(subject=subject, task=int(task), seconds=n / 2048, dominant_hz=round(peak, 3),
                             spread_range_mm=float(np.percentile(spread, 99) - np.percentile(spread, 1)),
                             tip_amplitude_mm=[float(np.percentile(dist[t], 99) - np.percentile(dist[t], 1)) for t in TIPS]))
            print(json.dumps(rows[-1]), flush=True)
    write(OUT / "census.json", rows)


# ----------------------------------------------------------------------------- evaluate
def load_participant(subject, device):
    source = CAMPAIGN / f"subject_{subject:02d}"
    with h5py.File(CACHES / f"subject_{subject:02d}/features.h5", "r") as f:
        mean, scale, mask = f["feature_mean"][:], f["feature_scale"][:], f["mask"][:]
        test_rows = [(int(t), int(s)) for t, s in zip(f["test/tasks"][:], f["test/starts"][:])]
    fits = {}
    for arm in pilot.ARMS:
        for seed in SEEDS:
            saved = torch.load(source / f"fits/{arm}_{seed}/best.pt", map_location="cpu", weights_only=True)
            fits[(arm, seed)] = saved
    book = fits[("tokens", SEEDS[0])]["model"]["decoder.codebook"].numpy()
    for key, saved in fits.items():
        assert np.array_equal(saved["model"]["decoder.codebook"].numpy(), book), key
    models = {}
    for (arm, seed), saved in fits.items():
        model = pilot.SparPose(torch.from_numpy(book), torch.from_numpy(mask))
        model.load_state_dict(saved["model"])
        models[(arm, seed)] = model.to(device).eval()
    layout = grid_layout()
    return dict(mean=mean, scale=scale, mask=mask, book=book, fits=fits, models=models, layout=layout, test_rows=test_rows)


def score(part, normalized_windows, targets, device):
    """Errors of both arms (seeds averaged) for a list of normalized (320, 6000) windows."""
    features = []
    for raw in normalized_windows:
        analysis = raw[:, INDICES].transpose(1, 2, 0).copy()
        feats = spar_simple_dataset_batched(analysis, layout=part["layout"], channelwise=True, batch_size=16, device=device)
        features.append(feats.transpose(1, 0, 2))           # (320, 100, 18)
    spar = np.stack(features)                                # (W, 320, 100, 18)
    values = standardize(spar, part["mean"], part["scale"], part["mask"])
    tokens = tokenize(values, part["book"], device=device)
    target = np.stack(targets)[:, -50:]
    errors, per_window = {}, {}
    for arm, inputs in (("tokens", torch.from_numpy(tokens.astype(np.int64))), ("continuous", torch.from_numpy(values))):
        seed_errors, seed_windows = [], []
        for seed in SEEDS:
            saved = part["fits"][(arm, seed)]
            data = dict(features=inputs.to(device))
            pred = pilot.prediction(part["models"][(arm, seed)], data, arm, saved["mean"], saved["scale"])
            seed_errors.append(pilot.error_mm(pred, target))
            seed_windows.append([pilot.error_mm(pred[i:i + 1], target[i:i + 1]) for i in range(len(target))])
        errors[arm] = float(np.mean(seed_errors))
        per_window[arm] = np.mean(seed_windows, axis=0).tolist()
    return errors, per_window, tokens


def evaluate(subject, device):
    t0 = time.monotonic()
    part = load_participant(subject, device)
    emg = joblib.load(DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(DATA / f"Sub{subject}_kinematics.pkl")
    signals = {int(t): resample(emg[t]) for t in emg}
    del emg
    # (a) test windows: per-window versus fixed-startup calibration
    windows_pw, windows_fs, targets = [], [], []
    for task, start in part["test_rows"]:
        block = signals[task][:, start:start + 6000]
        windows_pw.append(normalize_with(block, block))
        windows_fs.append(normalize_with(block, signals[task][:, :1000]))
        targets.append(targets_for(kin[str(task)] if str(task) in kin else kin[task], start))
    err_pw, win_pw, tok_pw = score(part, windows_pw, targets, device)
    err_fs, win_fs, tok_fs = score(part, windows_fs, targets, device)
    calibration = dict(per_window=err_pw, fixed_startup=err_fs, per_window_windows=win_pw, fixed_startup_windows=win_fs,
                       token_agreement=float((tok_pw == tok_fs).mean()),
                       window_gap_seconds=[round(start / 2000, 1) for _, start in part["test_rows"]])
    print(json.dumps(dict(subject=subject, stage="calibration", per_window=err_pw, fixed=err_fs,
                          agreement=calibration["token_agreement"], seconds=time.monotonic() - t0)), flush=True)
    # (b) transfer: all non-evaluated tasks, campaign-like windows, per-window calibration
    transfer = {}
    for task in sorted(signals):
        if task in EVALUATED:
            continue
        sig = signals[task]
        starts = list(range(0, sig.shape[-1] - 6000 + 1, 4000))
        ktask = kin[str(task)] if str(task) in kin else kin[task]
        wins = [normalize_with(sig[:, s:s + 6000], sig[:, s:s + 6000]) for s in starts]
        tg = [targets_for(ktask, s) for s in starts]
        err, win, _ = score(part, wins, tg, device)
        transfer[str(task)] = dict(windows=len(starts), starts=starts, errors=err, per_window=win)
        print(json.dumps(dict(subject=subject, stage="transfer", task=task, windows=len(starts), errors=err,
                              seconds=time.monotonic() - t0)), flush=True)
    write(OUT / f"subject_{subject:02d}.json", dict(subject=subject, device=device, seconds=time.monotonic() - t0,
                                                   calibration=calibration, transfer=transfer))


# ----------------------------------------------------------------------------- report
def exact_sign_flip(d):
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    null = (signs * d).mean(1)
    return float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12))


def contrast(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return dict(mean=float(d.mean()), per_participant=d.tolist(), positive=int((d > 0).sum()),
                bootstrap95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], p_exact=exact_sign_flip(d))


def report():
    rng = np.random.default_rng(260917)
    files = sorted(OUT.glob("subject_*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    subjects = [r["subject"] for r in rows]
    cal = dict(subjects=subjects)
    for arm in ("tokens", "continuous"):
        pw = [r["calibration"]["per_window"][arm] for r in rows]
        fs = [r["calibration"]["fixed_startup"][arm] for r in rows]
        cal[arm] = dict(per_window_mean=float(np.mean(pw)), fixed_startup_mean=float(np.mean(fs)),
                        fixed_minus_per_window=contrast(fs, pw, rng))
    cal["token_agreement_mean"] = float(np.mean([r["calibration"]["token_agreement"] for r in rows]))
    cal["token_agreement_min"] = float(np.min([r["calibration"]["token_agreement"] for r in rows]))
    cal["tokens_minus_continuous_fixed_startup"] = contrast([r["calibration"]["fixed_startup"]["tokens"] for r in rows],
                                                            [r["calibration"]["fixed_startup"]["continuous"] for r in rows], rng)
    cal["tokens_minus_continuous_per_window"] = contrast([r["calibration"]["per_window"]["tokens"] for r in rows],
                                                         [r["calibration"]["per_window"]["continuous"] for r in rows], rng)
    tasks = sorted({t for r in rows for t in r["transfer"]}, key=int)
    transfer = {}
    for task in tasks:
        tok = [r["transfer"][task]["errors"]["tokens"] for r in rows if task in r["transfer"]]
        con = [r["transfer"][task]["errors"]["continuous"] for r in rows if task in r["transfer"]]
        transfer[task] = dict(n=len(tok), tokens_mean=float(np.mean(tok)), continuous_mean=float(np.mean(con)),
                              tokens_minus_continuous=contrast(tok, con, rng) if len(tok) == 9 else None,
                              windows=int(np.mean([r["transfer"][task]["windows"] for r in rows if task in r["transfer"]])))
    # pooled contrasts: the five 1.5 Hz digit tasks (census: tasks 1, 3, 5, 7, 9) and, for reference,
    # the per-window error of the 0.5 Hz digit test windows (tasks 2, 4, 6, 8, 10) in the same replay
    fast = ("1", "3", "5", "7", "9")
    fast_tok = [float(np.mean([r["transfer"][t]["errors"]["tokens"] for t in fast])) for r in rows]
    fast_con = [float(np.mean([r["transfer"][t]["errors"]["continuous"] for t in fast])) for r in rows]
    slow_tok, slow_con = [], []
    for r in rows:
        with h5py.File(CACHES / f"subject_{r['subject']:02d}/features.h5", "r") as f:
            keep = [i for i, t in enumerate(f["test/tasks"][:]) if int(t) in (2, 4, 6, 8, 10)]
        slow_tok.append(float(np.mean([r["calibration"]["per_window_windows"]["tokens"][i] for i in keep])))
        slow_con.append(float(np.mean([r["calibration"]["per_window_windows"]["continuous"][i] for i in keep])))
    fast_contrast = contrast(fast_tok, fast_con, rng)
    pooled = dict(fast_digit_tasks=list(map(int, fast)), tokens_mean=float(np.mean(fast_tok)), continuous_mean=float(np.mean(fast_con)),
                  tokens_per_participant=fast_tok, continuous_per_participant=fast_con, tokens_minus_continuous=fast_contrast,
                  slow_digit_test_windows=dict(tokens_mean=float(np.mean(slow_tok)), continuous_mean=float(np.mean(slow_con)),
                                               tokens_per_participant=slow_tok, continuous_per_participant=slow_con))
    raw = {"fixed_startup": cal["tokens_minus_continuous_fixed_startup"]["p_exact"], "fast_digits": fast_contrast["p_exact"]}
    order = sorted(raw, key=raw.get)
    holm, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, raw[k] * (len(order) - i)))
        holm[k] = running
    pooled["holm_over_two_new_token_contrasts"] = dict(raw=raw, adjusted=holm)
    census_rows = json.loads((OUT / "census.json").read_text()) if (OUT / "census.json").exists() else []
    by_task = {}
    for c in census_rows:
        by_task.setdefault(str(c["task"]), []).append(c)
    census_summary = {t: dict(dominant_hz_median=float(np.median([c["dominant_hz"] for c in cs])),
                              tip_amplitude_mm_median=[float(np.median([c["tip_amplitude_mm"][i] for c in cs])) for i in range(5)])
                      for t, cs in by_task.items()}
    write(OUT / "summary.json", dict(subjects=subjects, calibration=cal, transfer=transfer, fast_digits_pooled=pooled, census=census_summary))
    print(json.dumps(dict(fast_digits_pooled={k: v for k, v in pooled.items() if "per_participant" not in k}), indent=1))
    print(json.dumps(dict(calibration={k: v for k, v in cal.items() if k != "subjects"}), indent=1))
    print(json.dumps({t: dict(hz=census_summary.get(t, {}).get("dominant_hz_median"), tokens=round(v["tokens_mean"], 3),
                              continuous=round(v["continuous_mean"], 3),
                              diff=(round(v["tokens_minus_continuous"]["mean"], 3) if v["tokens_minus_continuous"] else None))
                      for t, v in transfer.items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("census", "evaluate", "report"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "census":
        census(args.subjects)
    elif args.stage == "evaluate":
        for subject in args.subjects:
            evaluate(subject, args.device)
    else:
        report()


if __name__ == "__main__":
    main()
