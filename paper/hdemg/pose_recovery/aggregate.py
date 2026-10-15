"""Aggregate pose-recovery results: one table per participant, one across participants.

Balanced: frame errors are averaged within ten phase bins, bins
within cycles, cycles within movements, then the eight movements, so long cycles,
pauses and unequal movement counts cannot dominate. Also reported: the median over
frames (what the run logs print), the away-from-open subset (frames displaced from
the movement's open-hand template by more than half that movement's 95th
percentile), and frames outside the codebook's training windows.

    uv run python -m paper.hdemg.pose_recovery.aggregate         # all pose_recovery_sub*.npz in results/token_umap/pose_recovery
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paper.common import paths  # noqa: E402

RECOVERY = paths.RESULTS / "token_umap/pose_recovery"   # pose_recovery.py's results
REPS = ["continuous", "tokens", "tokens_hamming", "tokens_64", "pose_oracle"]


def balanced(err, label, cycle, phase):
    """Mean over phase bins -> cycles -> movements -> movements' mean."""
    bins = np.minimum((phase * 10).astype(int), 9)
    per_movement = []
    for c in np.unique(label):
        per_cycle = []
        for cy in np.unique(cycle[label == c]):
            m = cycle == cy
            per_bin = [err[m & (bins == b)].mean() for b in range(10) if (m & (bins == b)).any()]
            per_cycle.append(np.mean(per_bin))
        per_movement.append(np.mean(per_cycle))
    return float(np.mean(per_movement)), [float(v) for v in per_movement]


def table(path):
    d = np.load(path)
    label, cycle, phase = d["label"], d["cycle"], d["phase"]
    from_open, in_train = d["from_open"], d["in_train"]
    away = np.zeros(len(label), bool)
    for c in np.unique(label):
        m = label == c
        away[m] = from_open[m] > 0.5 * np.percentile(from_open[m], 95)
    rows = {}
    for r in REPS:
        e = d[f"err_{r}"]
        bal, per_movement = balanced(e, label, cycle, phase)
        rows[r] = dict(balanced_mm=bal, median_mm=float(np.median(e)),
                       away_from_open_median_mm=float(np.median(e[away])),
                       outside_train_median_mm=float(np.median(e[~in_train])),
                       per_movement_balanced_mm=per_movement)
    return dict(file=path.name, frames=int(len(label)), away_share=float(away.mean()),
                outside_train_share=float((~in_train).mean()), rows=rows)


def main():
    files = sorted(RECOVERY.glob("pose_recovery_sub*.npz"))
    if not files:
        sys.exit("no pose_recovery_sub*.npz here yet")
    per = [table(f) for f in files]
    names = None
    for t in per:
        s = json.loads((RECOVERY / t["file"].replace(".npz", ".json")).read_text())
        names = names or list(s["per_task"])
        print(f"\n{t['file']}  frames {t['frames']:,}  away-from-open {t['away_share']:.0%}  "
              f"outside train windows {t['outside_train_share']:.0%}")
        print(f"  {'representation':16s} {'balanced':>9s} {'median':>8s} {'away':>8s} {'outside':>8s}   per movement (balanced)")
        for r, v in t["rows"].items():
            pm = " ".join(f"{x:5.1f}" for x in v["per_movement_balanced_mm"])
            print(f"  {r:16s} {v['balanced_mm']:9.2f} {v['median_mm']:8.2f} {v['away_from_open_median_mm']:8.2f} "
                  f"{v['outside_train_median_mm']:8.2f}   {pm}")
        c = s["controls"]
        print(f"  controls: phase-shift tokens {c['tokens_phase_shift_median_mm'][0]:.2f} "
              f"[{c['tokens_phase_shift_median_mm'][1]:.2f}, {c['tokens_phase_shift_median_mm'][2]:.2f}]  "
              f"shuffle {c['tokens_global_shuffle_median_mm']:.2f}  movement-mean pose {c['movement_mean_pose_median_mm']:.2f}")
    if len(per) > 1:
        print("\nACROSS PARTICIPANTS (balanced mm: mean +/- SD over participants; paired tokens - continuous)")
        for r in REPS:
            v = np.array([t["rows"][r]["balanced_mm"] for t in per])
            print(f"  {r:16s} {v.mean():6.2f} +/- {v.std(ddof=1):4.2f}")
        diff = np.array([t["rows"]["tokens"]["balanced_mm"] - t["rows"]["continuous"]["balanced_mm"] for t in per])
        print(f"  tokens - continuous: {diff.mean():+.2f} mm, range [{diff.min():+.2f}, {diff.max():+.2f}], "
              f"{int((diff > 0).sum())} of {len(diff)} participants positive")
    print("  movements:", names)
    (RECOVERY / "aggregate.json").write_text(json.dumps(per, indent=1) + "\n")


if __name__ == "__main__":
    main()
