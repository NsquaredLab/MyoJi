"""Pinned V19 architecture and native, window-local input representation."""

import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchaudio.functional as AF

SOURCE = Path(__file__).parent / "vendor/raulnet_v19.py"
SOURCE_SHA256 = "5cb3f1abd32e4729844f4469e488ddd4ba163fc20cb8a4a5318f9ca038b05fb1"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
spec = importlib.util.spec_from_file_location("pinned_raulnet_v19", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RaulNetV19 = module.RaulNetV19
CONFIG = {
    "learning_rate": 1e-4,
    "nr_of_input_channels": 2,
    "input_length__samples": 320,
    "nr_of_outputs": 60,
    "cnn_encoder_channels": (64, 32, 32),
    "mlp_encoder_channels": (128, 128),
    "event_search_kernel_length": 31,
    "event_search_kernel_stride": 8,
    "nr_of_electrode_grids": 5,
    "nr_of_electrodes_per_grid": 64,
}


def source_ends(start):
    """The exact final 50 target endpoints of the frozen token campaign."""
    return ((int(start) + np.arange(4039, 6000, 40)) * 128) // 125


def native_windows(emg, start):
    ends = source_ends(start)
    indices = ends[:, None] - np.arange(319, -1, -1)[None]
    assert indices.min() >= int(start) * 128 // 125
    return emg[:, indices].transpose(1, 0, 2).copy()


def dual_input(raw):
    # Exact MyoVerse Lowpass(20, fs=2048, order=4, Q=.707) operation;
    # retain torchaudio's native clamping and zero state per 320-sample window.
    x = raw.float()
    filtered = x
    for _ in range(4):
        filtered = AF.lowpass_biquad(filtered, 2048, 20, 0.707)
    return torch.stack((x, filtered), dim=1)


def configure_eager(model):
    # Skip only the optional compile optimization; layers/weights are unchanged.
    with patch.object(torch.nn.Module, "compile", lambda self, *a, **kw: None):
        model.configure_model()
    assert sum(p.numel() for p in model.parameters()) == 7489436
    return model
