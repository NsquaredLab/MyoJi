"""Shared print styling for the secondary tokenizer-paper figures."""
import matplotlib.pyplot as plt

import switzer_font

TOKEN = '#d80032'
FEATURE = '#5c618e'
GRAY = '#6d6f72'
POINT = '#939699'
INK = '#2b2e4c'
LIGHT = '#e4e6e9'   # background bands; the house LIGHT lifted for use behind data


def apply():
    # The figstyle palette (figstyle.py) and portable serif stack; sized for
    # the 7.16-inch figures, which are reduced in the preprint text block.
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
        'ytick.labelsize': 9, 'legend.fontsize': 8.5,
        'mathtext.fontset': 'stix', 'text.usetex': False,
        'axes.edgecolor': GRAY, 'axes.labelcolor': '#303030',
        'text.color': '#303030', 'xtick.color': GRAY, 'ytick.color': '#505050',
        'axes.linewidth': .75, 'grid.color': '#e0e0e0', 'grid.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    # Switzer, to match Figure 1; see scripts/switzer_font.py.
    plt.rcParams.update(switzer_font.rcparams())


def panel(ax, letter):
    """Panel identifiers only; explanatory titles belong in the caption."""
    ax.text(-.04, 1.075, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', ha='left', va='bottom')


def finish(ax, *, categorical=False, grid='x'):
    ax.minorticks_off()
    ax.set_axisbelow(True)
    ax.grid(axis=grid)
    if categorical:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=7)
    # Equivalent to despine(trim=True), keeping bounds at displayed ticks.
    for side, ticks, limits in [('bottom', ax.get_xticks(), ax.get_xlim()),
                                 ('left', ax.get_yticks(), ax.get_ylim())]:
        valid = [t for t in ticks if min(limits) <= t <= max(limits)]
        if ax.spines[side].get_visible() and len(valid) > 1:
            ax.spines[side].set_bounds(min(valid), max(valid))
