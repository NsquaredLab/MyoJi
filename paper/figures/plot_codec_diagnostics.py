"""BioCodec reconstruction at its two saved checkpoints, in decibels.

What the reader should see: in every fit, the 40,000-update model used
downstream reconstructs held-out EMG better than the 20,000-update one.

The receipts also store the objective's two reconstruction terms (Smooth-L1
waveform and multi-resolution STFT; the commitment term is not logged). Neither has a unit a reader
can hold, so the axis is the scale-invariant signal-to-distortion ratio
(SI-SDR) the upstream objective computes on the same windows: the arithmetic
mean over single-channel 2 s windows of per-window dB. That mean is not a
pooled or average residual energy, so no percentage axis is derived from it.
Pearson r is closely tied to SI-SDR per window and is not plotted separately.
Only two checkpoints exist; no curve is drawn, and the learning rate is 0.75x
peak at the first and zero at the second, so the gain is
within one cosine schedule, not evidence about a longer budget.

    uv run python -m paper.figures.plot_codec_diagnostics
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from paper.emg2pose import build_pose_paper_assets as data
from paper.common import switzer_font

from paper.common import figstyle as sf

SEEDS = (260940, 260941, 260942)
STEPS = (20000, 40000)
CODEC = sf.GREYS[0]          # comparison model: house grey
GRID_COLOR, GRID_WIDTH = "#DEDEDE", 0.55


def lr_ratio(step, warmup=10000, total=40000):
    """Upstream WarmupCosineLrScheduler with eta_ratio=0, after warmup."""
    return (1 + math.cos(math.pi * max(step - warmup, 0) / (total - warmup))) / 2


def apply_style():
    plt.style.use(["science", "ieee", "simpetru-large"])
    plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    plt.rcParams.update(switzer_font.rcparams())


def draw():
    # Imported by build_pose_paper_assets.py: keep this style out of later figures.
    with plt.rc_context():
        _draw()


def _draw():
    apply_style()
    rows, sdr = [], []
    for seed in SEEDS:
        r = data.read(f"results/paper_completion_v1/biocodec/seed_{seed}/training_receipt.json")
        entries = sorted(r["unlabeled_validation_diagnostics"], key=lambda v: v["step"])
        assert tuple(v["step"] for v in entries) == STEPS
        rows.append({"seed": seed, "diagnostics": entries})   # shape checked by audit_secondary_figures.py
        sdr.append([v["validation"]["sdr"] for v in entries])
    # Every fit improves on every stored diagnostic, not only the plotted one.
    for row in rows:
        a, b = (v["validation"] for v in row["diagnostics"])
        assert b["sdr"] > a["sdr"] and b["corr"] > a["corr"] and b["l_t"] < a["l_t"] and b["l_f"] < a["l_f"]
    assert lr_ratio(20000) == 0.75 and lr_ratio(40000) == 0.0

    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    fig.subplots_adjust(left=.16, right=.96, bottom=.30, top=.80)
    for y, (early, final) in enumerate(sdr):
        ax.plot([early, final], [y, y], color=CODEC, lw=1.0, zorder=2)
        ax.scatter(early, y, s=26, facecolor="white", edgecolor=CODEC, lw=1, zorder=3)
        ax.scatter(final, y, s=26, facecolor=CODEC, edgecolor=CODEC, lw=1, zorder=3)
    # Direct labels on the top row instead of a legend.
    for x, text, ha in ((sdr[0][0], "20,000 updates", "center"), (sdr[0][1], "40,000 updates", "center")):
        ax.annotate(text, (x, 0), xytext=(0, 7), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=8.5, color="#505050",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_yticks(range(len(SEEDS)), [f"Fit {i + 1}" for i in range(len(SEEDS))])
    ax.set_ylim(len(SEEDS) - .5, -.75)
    ax.set_xlim(9, 13)
    ax.set_xticks([9, 10, 11, 12, 13])
    ax.set_xlabel("Scale-Invariant SDR (dB)", fontsize=9.5, labelpad=5)
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", length=3, labelsize=9, top=False)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=9, right=False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.grid(axis="y", visible=False)
    sf.trim_spines(ax, which=("bottom",), offset=0)
    assert not sf.check_font_sizes(fig)
    data.save(fig, "codec_diagnostics", rows)


if __name__ == "__main__":
    draw()
