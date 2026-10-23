"""Preprocessing shared by both tokenizers.

Recordings at 2,048 Hz are resampled causally to 2,000 Hz. Each window is then
calibrated from its own prefix: channel median and MAD (scaled by 1.4826),
clipped at +-12 scale units, so no later sample influences the normalization.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import firwin, upfirdn


def calibration_normalize(
    emg: torch.Tensor,
    calibration_samples: int,
    clip: float = 12.0,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize using only the prefix available before scored tokens.

    The median and MAD are detached calibration state.  This avoids the
    future leakage of whole-window RevIN while still putting recordings with
    different ADC units and gains into a common numerical range.
    """

    if emg.ndim != 3:
        raise ValueError("EMG must have shape (batch, channel, time)")
    if calibration_samples <= 0 or calibration_samples > emg.shape[-1]:
        raise ValueError("calibration_samples must fit inside the EMG window")
    calibration = emg[..., :calibration_samples]
    centre = (
        calibration.sort(dim=-1)
        .values[
            ..., (calibration.shape[-1] - 1) // 2 : (calibration.shape[-1] - 1) // 2 + 1
        ]
        .detach()
    )
    absolute_deviation = (calibration - centre).abs()
    scale = (
        1.4826
        * absolute_deviation.sort(dim=-1).values[
            ..., (calibration.shape[-1] - 1) // 2 : (calibration.shape[-1] - 1) // 2 + 1
        ]
    ).detach()
    scale = scale.clamp_min(epsilon)
    normalized = ((emg - centre) / scale).clamp(-clip, clip)
    return normalized, centre, scale


def normalize_raw(raw):
    """Exact original prefix median/MAD, including the lower even-length median."""
    x = torch.from_numpy(raw)[None]
    centre = x[..., :1000].sort(-1).values[..., 499:500]
    scale = (x[..., :1000] - centre).abs().sort(-1).values[..., 499:500] * 1.4826
    return ((x - centre) / scale.clamp_min(1e-4)).clamp(-12, 12)[0].numpy()


def resample_2048_to_2000(emg):
    """Causal 125/128 polyphase resampling (10,241-tap Kaiser FIR, 800 Hz cutoff, 20 ms delay)."""
    taps = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
    return upfirdn(taps, emg.astype(np.float32), up=125, down=128, axis=-1)[
        ..., : emg.shape[-1] * 125 // 128
    ].astype(np.float32)
