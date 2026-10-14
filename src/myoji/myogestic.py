"""Token features for MyoGestic pipelines (``pip install "myoji[myogestic]"``).

MyoGestic hands each stream's trailing window to ``@pipeline.extract`` as a
channels-first ``(channels, samples)`` array. These functions turn that window
into what a low-bitrate wearable would send, following the paper's per-window
protocol: 2,048 Hz input is resampled causally to 2 kHz, the window is
calibrated on its own first 500 ms (median/MAD, clipped at +-12), and the
newest 160 ms is described. Use a ``Stream`` window of ``WINDOW_MS``.

Nothing here imports MyoGestic, so the same calls work on recorded arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize
from myoji.preprocessing import normalize_raw, resample_2048_to_2000
from myoji.spar import Layout, spar_simple_dataset_batched

RATE_HZ = 2000
CALIBRATION_SAMPLES = 1000  # first 500 ms of every window
ANALYSIS_SAMPLES = 320  # one 160 ms SPAR update
SETTLE_SAMPLES = 100  # resampled samples dropped while the causal FIR fills its history
WINDOW_MS = 1100  # long enough at 2,000 and 2,048 Hz


def at_2khz(window: np.ndarray, fs: float) -> np.ndarray:
    """``(channels, samples)`` at 2,000 or 2,048 Hz -> float32 at 2,000 Hz."""
    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("window must be (channels, samples)")
    if fs == RATE_HZ:
        return x
    if fs == 2048:
        return resample_2048_to_2000(x)[:, SETTLE_SAMPLES:]
    raise ValueError(f"tokenizers expect 2,000 Hz (or 2,048 Hz, resampled here), not {fs} Hz")


def _calibrated(window: np.ndarray, fs: float, minimum: int) -> np.ndarray:
    x = at_2khz(window, fs)
    if x.shape[-1] < minimum:
        raise ValueError(f"need {minimum} samples at 2 kHz after resampling; use Stream(window_ms={WINDOW_MS})")
    return normalize_raw(np.ascontiguousarray(x))


def spar_descriptors(window: np.ndarray, fs: float, layout: Layout) -> np.ndarray:
    """SPAR descriptors of the window's newest 160 ms, ``(channels, 18)``.

    ``layout`` maps the device's electrode grid(s), e.g.
    ``myoji.spar.grid_layout([np.arange(64).reshape(8, 8)])`` for one 8x8 grid.
    """
    x = _calibrated(window, fs, CALIBRATION_SAMPLES + ANALYSIS_SAMPLES)
    analysis = x[:, -ANALYSIS_SAMPLES:].T[None]
    return spar_simple_dataset_batched(analysis, layout=layout, channelwise=True)[0]


@dataclass
class SparCodebook:
    """Train-only standardization plus a k-means codebook, fitted on SPAR descriptors."""

    mean: np.ndarray
    scale: np.ndarray
    mask: np.ndarray
    book: np.ndarray

    @classmethod
    def fit(cls, descriptors: np.ndarray, layout: Layout, *, vocabulary: int = 1024, seed: int = 260911):
        """``descriptors``: ``(windows, channels, 18)`` from training windows only."""
        x = np.asarray(descriptors, dtype=np.float32)[:, :, None]
        mask = np.tile(layout.mask, (1, x.shape[-1] // layout.mask.shape[1]))
        mean, scale = fit_standardization(x, mask)
        book, _ = fit_codebook(standardize(x, mean, scale, mask), vocabulary=vocabulary, seed=seed)
        return cls(mean, scale, mask, book)

    def tokens(self, descriptors: np.ndarray) -> np.ndarray:
        """``(..., channels, 18)`` descriptors -> ``(..., channels)`` token IDs."""
        d = np.asarray(descriptors, dtype=np.float32)
        z = standardize(d.reshape(-1, *d.shape[-2:])[:, :, None], self.mean, self.scale, self.mask)
        return tokenize(z, self.book).reshape(d.shape[:-1])

    def vectors(self, tokens: np.ndarray) -> np.ndarray:
        """What a receiver looks up: the code vectors, absent boundary couplings zeroed."""
        return self.book[np.asarray(tokens, dtype=np.int64)] * self.mask


def learned_tokens(window: np.ndarray, fs: float, tokenizer) -> np.ndarray:
    """Newest learned token per channel, ``(channels,)``.

    ``tokenizer`` is a ``myoji.learned.Tokenizer``; the window needs 980 ms at
    2 kHz (500 ms calibration plus the 480 ms causal context).
    """
    # ponytail: re-encodes the whole window every tick; feed a TokenStream from the stream
    # buffer instead if 50 Hz x channels outruns the CPU.
    x = at_2khz(window, fs)
    if x.shape[-1] < tokenizer.minimum_samples:
        raise ValueError(f"need {tokenizer.minimum_samples} samples at 2 kHz; use Stream(window_ms={WINDOW_MS})")
    return np.asarray(tokenizer.encode(x, sample_rate_hz=RATE_HZ)["tokens"])[:, -1]
