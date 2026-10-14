"""Paper figure styles, layered on SciencePlots.

The parts of simpetru-figstyle 0.1.0 that the paper's figure scripts use, copied
unchanged. Importing this registers the sheets next to it (`simpetru`,
`simpetru-large`) with matplotlib, the same way importing scienceplots registers
`science` and `ieee`:

    import figstyle as sf
    plt.style.use(["science", "ieee", "simpetru-large"])

Variants: `simpetru` (font_scale 0.75), `simpetru-large` (1.0).

Derived from:

    Sîmpetru, R.C., Cnejevici, V., Farina, D., Del Vecchio, A., 2024.
    Influence of spatio-temporal filtering on hand kinematics estimation from
    high-density EMG signals. J. Neural Eng. 21. doi:10.1088/1741-2552/ad3498
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.style as mstyle
import scienceplots  # noqa: F401  (registers "science" / "ieee")
from matplotlib.ticker import FixedLocator

STYLES = Path(__file__).parent


def _register() -> None:
    """Put the bundled sheets in matplotlib's style library, so they can be
    named in plt.style.use() rather than passed as a file path.

    matplotlib 3.11 moved these out of `matplotlib.style.core` up to
    `matplotlib.style`; SciencePlots carries the same shim.
    """
    try:                                    # matplotlib >= 3.11
        library = mstyle.library
        available = mstyle.available
    except AttributeError:                  # matplotlib < 3.11
        library = mstyle.core.library
        available = mstyle.core.available

    # rc_params_from_file lives on the top-level matplotlib module in every
    # supported version; the style submodule kept moving it (3.10 has neither
    # mstyle.rc_params_from_file nor mstyle.core.rc_params_from_file).
    from matplotlib import rc_params_from_file

    for sheet in sorted(STYLES.glob("*.mplstyle")):
        library[sheet.stem] = rc_params_from_file(sheet, use_default_template=False)
    available[:] = sorted(library)


_register()


# Three hue families rather than seven arbitrary colours -- which is why the
# paper's figures read as ordered rather than as a rainbow. Also the
# axes.prop_cycle, so unstyled plots pick them up without asking.
REDS = ("#570014", "#8b0020", "#d80032")   # conditions under test, dark -> bright
GREYS = ("#6d6f72", "#939699")             # baselines and controls
BLUES = ("#2b2e4c", "#5c618e")             # contrast group, dark -> light

# IEEE Author Center limits: a graphic is published at one column (3.5 in /
# 21 picas) or two columns (7.16 in / 43 picas), and may be at most 8.8 in
# deep. IEEE recommends 8-10 pt for text inside a figure; this style treats 7 pt
# as the hard floor (smaller does not survive printing) and 11 pt as the
# ceiling (larger competes with the body text).
FONT_MIN = 7.0
FONT_MAX = 11.0

COL_1 = 3.5
COL_2 = 7.16
MAX_DEPTH = 8.8


def figsize(cols: int = 2, depth: float = 1 / 4) -> tuple[float, float]:
    """IEEE-legal figure size: `cols` columns wide, `depth` of the max depth.

    figsize(2, 1/5) -> (7.16, 1.76); figsize(1, 1/4) -> (3.5, 2.20).
    """
    if cols not in (1, 2):
        raise ValueError("IEEE publishes graphics at one or two columns")
    if not 0 < depth <= 1:
        raise ValueError("depth is a fraction of the 8.8 in maximum")
    return ((COL_1 if cols == 1 else COL_2), round(MAX_DEPTH * depth, 2))


def trim_spines(ax, which=("left", "bottom"), offset=5) -> None:
    """Clip each spine to the range its ticks span, offset outward.

    ``offset`` (points, seaborn-style) moves the trimmed spines away from
    the plot area; 5 pt is the house default, 0 restores flush spines.

    Stops the axis lines overshooting the data, which looks noticeably tidier
    at small figure sizes. A function rather than a setting because it depends
    on tick locations and limits, which only exist once there is data -- the
    same reason seaborn ships `despine` as a function.

    Why not seaborn's `despine(trim=True)`, which is the same idea and already
    a dependency? Because it misses two things in this style, both measured:

    1. It compares ticks against the limits without tolerance, so an axis with
       `ylim` of exactly 0.95 against a locator emitting 0.9500000000000001
       trims to 0.90 -- a spine a whole interval short.
    2. It resets the major ticks but not the minor ones, and `science` turns
       minor ticks on. They stay behind as dashes floating past the spine end.
    """
    for name in which:
        if name not in ax.spines:
            continue
        vertical = name in ("left", "right")
        axis = ax.yaxis if vertical else ax.xaxis

        ticks = ax.get_yticks() if vertical else ax.get_xticks()
        lo, hi = ax.get_ylim() if vertical else ax.get_xlim()
        lo, hi = min(lo, hi), max(lo, hi)

        # Tolerance matters: a locator happily returns 0.9500000000000001 for a
        # limit of exactly 0.95, and a bare `t <= hi` then drops the top tick and
        # leaves the spine ending one interval short.
        tol = (hi - lo) * 1e-9
        visible = [t for t in ticks if lo - tol <= t <= hi + tol]
        if len(visible) < 2:
            continue

        first = max(visible[0], lo)
        last = min(visible[-1], hi)
        ax.spines[name].set_bounds(first, last)
        if offset:
            ax.spines[name].set_position(("outward", offset))

        kept = [t for t in axis.get_minorticklocs() if first - tol <= t <= last + tol]
        if kept:
            axis.set_minor_locator(FixedLocator(kept))


def check_label_case(fig):
    """Axis labels and titles in `fig` that break the Title Case convention.

    House rule: axis labels, titles, and legend entries capitalize their
    words ("Tracked Unit", "Time (s)", "Decoder Input Channels"). Checked
    per label by its first alphabetic character, so acronyms (EMG) and
    math ($R^2$) pass; "mRMR" is allowed explicitly. Returns the offending
    strings, empty when the figure is legal.
    """
    bad = []
    for ax in fig.get_axes():
        texts = [ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]
        leg = ax.get_legend()
        if leg is not None:
            texts += [t.get_text() for t in leg.get_texts()]
        for label in texts:
            stripped = (label or "").strip()
            for ch in stripped:
                if ch.isalpha():
                    if ch.islower() and not stripped.startswith("mRMR"):
                        bad.append(stripped[:40])
                    break
    return sorted(set(bad))


def check_font_sizes(fig, lo: float = FONT_MIN, hi: float = FONT_MAX):
    """Text objects in `fig` whose size falls outside [lo, hi] points.

    Returns a list of (size, text) pairs, empty when the figure is legal.
    Call it before savefig in any script that sets explicit `fontsize=`.
    """
    bad = []
    for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize")):
        try:
            size = float(t.get_fontsize())
            label = (t.get_text() or "").strip()
        except (TypeError, ValueError, AttributeError):
            continue
        if label and not (lo <= size <= hi):
            bad.append((round(size, 2), label[:40]))
    return sorted(set(bad))
