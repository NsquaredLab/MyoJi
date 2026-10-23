"""Every paper/ module imports from its folder, so moved files and rewritten imports stay in step."""

import importlib
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="the paper/ modules need `uv sync --group paper`")

PAPER = Path(__file__).resolve().parents[1] / "paper"
# Optional or external: installed separately (paper/README.md) or only inside Blender.
OPTIONAL = {"emg2pose", "bpy", "mathutils", "tabpfn", "threadpoolctl", "biocodec", "seaborn", "umap"}
MODULES = sorted(
    "paper." + ".".join(p.relative_to(PAPER).with_suffix("").parts)
    for p in PAPER.rglob("*.py")
    if p.name != "__init__.py" and "__pycache__" not in p.parts
    and p.relative_to(PAPER).parts[0] not in ("results", "output")  # run outputs, not code
)


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name.split(".")[0] in OPTIONAL:
            pytest.skip(f"needs {error.name}")
        raise
    except FileNotFoundError as error:  # reads a campaign output at import time
        if Path(error.filename).is_relative_to(PAPER / "results"):
            pytest.skip(f"needs {error.filename}")
        raise
