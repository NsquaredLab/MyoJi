"""Frozen SPAR pipelines under the learned route's noise and electrode-dropout protocol.

Declared before running (2026-10-10); exploratory. Nothing is fitted: each participant's frozen
standardization, codebook and three decoders per arm (dpose_spar_transfer_checks.load_participant).
Corruption uses the frozen helpers unchanged: dpose_noise.native_bounds / unit_noise / perturb_native /
resampled_window (SNR 40, 30, 20, 10 dB; noise seeds 260921-260923; noise added to the native 2,048 Hz
segment after the clean calibration prefix) and dpose_dropout.dropped_channels / drop_native (5, 10, 25 %
scattered electrodes and one or two whole arrays; seeds 260931-260933; channels zeroed after the prefix;
dropped_channels called with subject 0, as run_dpose_dropout.py does). Calibration comes from the clean
first 1,000 resampled samples; resampling is the campaign's causal Kaiser FIR.

Families, as for the learned route (plot_noise_robustness.py): per-participant error increase from clean,
averaged over decoder seeds and corruption draws; noise family = SPAR-token minus continuous-SPAR increase
at the four SNRs (4 contrasts); dropout family = the same at the five dropout conditions (5 contrasts);
exact sign-flip over 512 patterns, Holm within each family; 10,000-draw participant bootstrap, seed 260930.

    ~/.local/bin/uv run --no-project --with torch --with h5py --with joblib --with scipy --with numpy \
        python scripts/dpose_spar_corruption.py run|report|table [--subjects ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import dpose_spar_transfer_checks as T  # noqa: E402
import paths  # noqa: E402
from dpose_dropout import DROPOUT_FRACTIONS, DROPOUT_SEEDS, GRID_DROPS, drop_native, dropped_channels  # noqa: E402
from dpose_noise import NOISE_SEEDS, SNR_DB, native_bounds, perturb_native, resampled_window, unit_noise  # noqa: E402

OUT = BASE / "results/dpose_spar_corruption_v1"
CONDITIONS = ([dict(name="clean", kind=None, level=None, seed=None)]
              + [dict(name=f"snr{s}_seed{x}", kind="noise", level=s, seed=x) for s in SNR_DB for x in NOISE_SEEDS]
              + [dict(name=f"drop{round(f * 100):02d}_seed{x}", kind="fraction", level=f, seed=x)
                 for f in DROPOUT_FRACTIONS for x in DROPOUT_SEEDS]
              + [dict(name=f"grid{g}_seed{x}", kind="grids", level=g, seed=x) for g in GRID_DROPS for x in DROPOUT_SEEDS])


def run(subject, device):
    path = OUT / f"subject_{subject:02d}.json"
    if path.exists():
        return
    t0 = time.monotonic()
    part = T.load_participant(subject, device)
    emg = joblib.load(T.DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(T.DATA / f"Sub{subject}_kinematics.pkl")
    full = {}
    segments, targets = [], []
    for task, start in part["test_rows"]:
        key = next(k for k in emg if int(k) == task)
        if task not in full:
            full[task] = T.resample(emg[key])
        origin, onset, stop = native_bounds(start)
        raw = np.asarray(emg[key][:, origin:stop], dtype=np.float32)
        clean = resampled_window(raw, start, origin, T.resample)
        gate = float(np.abs(clean - full[task][:, start:start + 6000]).max())
        assert gate < 1e-4, ("resampling gate", task, start, gate)
        segments.append((task, start, origin, onset, stop, raw))
        targets.append(T.targets_for(kin[key], start))
    rows, clean_tokens = [], None
    for condition in CONDITIONS:
        windows = []
        for task, start, origin, onset, stop, raw in segments:
            if condition["kind"] is None:
                changed = raw
            elif condition["kind"] == "noise":
                draw = unit_noise((320, stop - onset), subject, task, start, condition["seed"])
                changed, _ = perturb_native(raw, onset - origin, draw, condition["level"])
            else:
                channels = dropped_channels(raw.shape[0], 0, task, start, condition["seed"],
                                            **{condition["kind"]: condition["level"]})
                changed, _ = drop_native(raw, onset - origin, channels)
            block = resampled_window(changed, start, origin, T.resample)
            windows.append(T.normalize_with(block, block))
        if condition["kind"] is not None:
            assert all(np.array_equal(w[:, :1000], c[:, :1000]) for w, c in zip(windows, clean_windows)), "clean prefix"
        else:
            clean_windows = windows
        errors, per_window, tokens = T.score(part, windows, targets, device)
        if clean_tokens is None:
            clean_tokens = tokens
        rows.append(dict(**condition, errors=errors, per_window=per_window,
                         token_agreement=float((tokens == clean_tokens).mean())))
        print(json.dumps(dict(subject=subject, condition=condition["name"], errors=errors,
                              agreement=rows[-1]["token_agreement"], seconds=time.monotonic() - t0)), flush=True)
    T.write(path, dict(subject=subject, rows=rows, seconds=time.monotonic() - t0))


def exact_p(d):
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))


def boot(d):
    idx = np.random.default_rng(260930).integers(0, len(d), (10000, len(d)))
    return [float(v) for v in np.percentile(d[idx].mean(1), [2.5, 97.5])]


def holm(tests):
    raw = np.array([t["p_raw"] for t in tests])
    order = np.argsort(raw)
    adj = np.minimum(np.maximum.accumulate(raw[order] * (len(raw) - np.arange(len(raw)))), 1)
    for rank, i in enumerate(order):
        tests[i]["p_holm"] = float(adj[rank])
    return tests


def report():
    files = sorted(OUT.glob("subject_*.json"))
    data = [json.loads(f.read_text()) for f in files]
    assert len(data) == 9, len(data)

    def level(arm, kind, lev):
        out = []
        for d in data:
            vals = [r["errors"][arm] for r in d["rows"] if r["kind"] == kind and r["level"] == lev]
            out.append(np.mean(vals))
        return np.array(out)

    clean = {arm: level(arm, None, None) for arm in ("tokens", "continuous")}
    summary, families = [], {"noise": [], "dropout": []}
    for kind, levels, family in (("noise", SNR_DB, "noise"), ("fraction", DROPOUT_FRACTIONS, "dropout"),
                                 ("grids", GRID_DROPS, "dropout")):
        for lev in levels:
            inc = {arm: level(arm, kind, lev) - clean[arm] for arm in clean}
            agree = float(np.mean([np.mean([r["token_agreement"] for r in d["rows"] if r["kind"] == kind and r["level"] == lev])
                                   for d in data]))
            summary.append(dict(kind=kind, level=lev, token_agreement=agree,
                                **{f"{arm}_error": float(level(arm, kind, lev).mean()) for arm in clean},
                                **{f"{arm}_increase": float(inc[arm].mean()) for arm in clean},
                                **{f"{arm}_increase_ci95": boot(inc[arm]) for arm in clean}))
            d = inc["tokens"] - inc["continuous"]
            families[family].append(dict(kind=kind, level=lev, mean_gap_mm=float(d.mean()), ci95=boot(d),
                                         participants_token_less=int((d < 0).sum()), p_raw=exact_p(d),
                                         per_participant=d.tolist()))
    for tests in families.values():
        holm(tests)
    result = dict(clean={arm: float(v.mean()) for arm, v in clean.items()}, summary=summary, families=families)
    T.write(OUT / "summary.json", result)
    print(json.dumps(result["clean"]))
    for s in summary:
        print(s["kind"], s["level"], "tok %.3f (+%.3f)" % (s["tokens_error"], s["tokens_increase"]),
              "con %.3f (+%.3f)" % (s["continuous_error"], s["continuous_increase"]), "agree %.3f" % s["token_agreement"])
    for name, tests in families.items():
        for t in tests:
            print(name, t["kind"], t["level"], "gap %+.3f" % t["mean_gap_mm"], [round(x, 3) for x in t["ci95"]],
                  "less", t["participants_token_less"], "p", round(t["p_raw"], 4), "holm", round(t["p_holm"], 4))


NAMES = {("noise", 40): "Noise, 1\\,\\% RMS", ("noise", 30): "Noise, 3.2\\,\\% RMS", ("noise", 20): "Noise, 10\\,\\% RMS",
         ("noise", 10): "Noise, 32\\,\\% RMS", ("fraction", 0.05): "5\\,\\% of electrodes",
         ("fraction", 0.1): "10\\,\\% of electrodes", ("fraction", 0.25): "25\\,\\% of electrodes",
         ("grids", 1): "One array", ("grids", 2): "Two arrays"}
CAPTION = (r"\caption{\textbf{SPAR under noise and electrode dropout.} Mean 3D error (mm) of the frozen SPAR pipelines over "
           r"nine participants, averaging three decoder seeds and three corruption draws, and the paired difference in the "
           r"error increase from clean, SPAR tokens minus continuous SPAR (negative: tokens degrade less), with its 95\,\% "
           r"participant-bootstrap interval and exact sign-flip $p$, Holm-adjusted within the noise and the dropout family. "
           r"Noise amplitude is relative to each channel's EMG RMS (40, 30, 20 and 10\,dB).}")


def table():
    """Write output/tables/spar_corruption.tex (Table S8) from summary.json."""
    s = json.loads((OUT / "summary.json").read_text())
    fam = {(t["kind"], t["level"]): t for tests in s["families"].values() for t in tests}
    f2 = lambda x: "0.00" if f"{x:+.2f}" in ("+0.00", "-0.00") else f"{x:+.2f}"
    p = lambda x: f"{x:.2f}" if x >= 0.05 else f"{x:.3f}"
    rows = ["Clean & %.2f & %.2f & & \\\\" % (s["clean"]["tokens"], s["clean"]["continuous"])]
    for x in s["summary"]:
        k = (x["kind"], x["level"])
        t = fam[k]
        rows.append(f"{NAMES[k]} & {x['tokens_error']:.2f} & {x['continuous_error']:.2f} & "
                    f"${f2(t['mean_gap_mm'])}$ $[{f2(t['ci95'][0])},{f2(t['ci95'][1])}]$ & {p(t['p_holm'])} \\\\")
        if k == ("noise", 10):
            rows.append("\\addlinespace")
    tex = "\n".join([r"\begin{table}[htbp]", r"\centering\small", CAPTION, r"\label{tab:spar-corruption}",
                      r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lrrlr}", r"\toprule",
                      r"Condition & Tokens & Continuous & Difference in increase & $p_{\mathrm{Holm}}$ \\", r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"]) + "\n"
    path = paths.OUTPUT / "tables/spar_corruption.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "report", "table"))
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "run":
        for subject in args.subjects:
            run(subject, args.device)
    elif args.stage == "report":
        report()
    else:
        table()


if __name__ == "__main__":
    main()
