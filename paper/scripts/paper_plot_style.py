"""Print-size defaults shared by publication figure builders."""

from __future__ import annotations

import matplotlib

import switzer_font


COLORS = {"tokens": "#0072B2", "continuous": "#009E73", "raw": "#3C4149"}
MARKERS = {"tokens": "o", "continuous": "s", "raw": "D"}
PARTICIPANT_COLOR = "#AFAFAF"
REFERENCE_COLOR = "#6F6F6F"
# The current arXiv manuscript has a 6.5-inch text block.
FIGURE_WIDTH_IN = 0.95 * 6.5


def apply_publication_style() -> None:
    matplotlib.rcParams.update(switzer_font.rcparams())  # Switzer, matching Figure 1
    matplotlib.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
