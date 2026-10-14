"""Switzer for every generated figure, matching the hand-drawn Figure 1.

The shared figstyle sheets (figstyle.py) set a serif stack, and that style is
used by other papers, so this paper overrides the family locally instead of
editing it. Apply the override *after* ``plt.style.use`` or any rcParams update
that touches ``font.*``.

Switzer (Fontshare) cannot be redistributed with this repository. Install its
static faces (Switzer-Regular.ttf, Switzer-Bold.ttf, ...) in a standard user
font folder, or point MYOJI_SWITZER_DIR at the folder that holds them. Without
them, figures fall back to the rest of the sans-serif list (Helvetica Neue,
Helvetica, Arial, DejaVu Sans) after one warning; only the typeface differs.

Only the static weights work. Switzer also ships as a variable font, and the
variable file is what is normally installed; matplotlib cannot select a named
instance from it, so bold silently renders as regular ("Failed to find font
weight bold, now using 400"). Panel letters are bold, so that failure would be
invisible in the code and wrong in the output; without the static faces the
family is therefore left out rather than used.

Math stays on the STIX fontset: mathtext is conventionally serif even inside
sans-serif text, and Switzer has no math glyph coverage.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

from matplotlib import font_manager

# MYOJI_SWITZER_DIR first, then the standard per-user font folders.
SEARCH_DIRS = tuple(Path(p) for p in [os.environ.get("MYOJI_SWITZER_DIR")] if p) + (
    Path.home() / "Library/Fonts",          # macOS
    Path.home() / ".local/share/fonts",     # Linux
    Path.home() / ".fonts",
)
# TrueType first. matplotlib embeds with pdf.fonttype 42, and declaring a
# CFF/OTF face as TrueType makes readers warn "Mismatch between font type and
# embedded font file", which a publisher's preflight would flag.
SUFFIXES = (".ttf", ".otf")
WEIGHTS = ("Regular", "Italic", "Medium", "Semibold", "Bold", "BoldItalic")
FAMILY = "Switzer"
FALLBACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

_registered: set[str] = set()
_warned = False


def register() -> list[str]:
    """Make the static Switzer faces visible to matplotlib. Idempotent.

    Returns the registered weights, or [] (after one warning) when Regular and
    Bold are not both found.
    """
    global _warned
    for weight in WEIGHTS:
        if weight in _registered:
            continue
        for directory in SEARCH_DIRS:
            for suffix in SUFFIXES:
                path = directory / f"Switzer-{weight}{suffix}"
                if path.is_file():
                    font_manager.fontManager.addfont(str(path))
                    _registered.add(weight)
                    break
            if weight in _registered:
                break
    families = {f.name for f in font_manager.fontManager.ttflist}
    missing = [w for w in ("Regular", "Bold") if w not in _registered]
    if missing or FAMILY not in families:
        if not _warned:
            _warned = True
            warnings.warn(
                f"static Switzer {missing or '(family)'} not found in {[str(d) for d in SEARCH_DIRS]} "
                f"(set MYOJI_SWITZER_DIR); figures use {', '.join(FALLBACK)} instead", stacklevel=2)
        return []
    return sorted(_registered)


def rcparams() -> dict[str, object]:
    """Family override to merge into rcParams once the base style is loaded."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ([FAMILY] if register() else []) + FALLBACK,
        # Unchanged on purpose: see the module docstring.
        "mathtext.fontset": "stix",
    }


def apply(rc=None) -> dict[str, object]:
    """Apply the override to ``rc`` (default: the global rcParams)."""
    import matplotlib

    override = rcparams()
    (matplotlib.rcParams if rc is None else rc).update(override)
    return override


def demo() -> None:
    """Fail if the variable font is being used, or if bold is not distinct."""
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import re
    import tempfile

    apply()
    figure = plt.figure(figsize=(2, 1))
    for y, weight in ((0.6, "normal"), (0.2, "bold")):
        figure.text(0.05, y, "Agk 128", fontsize=11, weight=weight)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "switzer_demo.pdf"
        figure.savefig(path)
        plt.close(figure)
        raw = path.read_bytes().decode("latin-1")
    faces = set(re.findall(r"/BaseFont\s*/[A-Z]{6}\+([A-Za-z0-9-]+)", raw))
    assert any("Bold" in f for f in faces), f"no bold face embedded, got {faces}"
    assert not any("Variable" in f for f in faces), f"variable font leaked in: {faces}"
    print("switzer_font ok:", sorted(faces))


if __name__ == "__main__":
    demo()
