"""Deterministic native-rate EMG perturbations and paired pose measurements."""

import hashlib

import numpy as np

NOISE_SEEDS = (260921, 260922, 260923)
SNR_DB = (40, 30, 20, 10)
ENDS = np.arange(2039, 6000, 40, dtype=np.int64)


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(value.dtype.str.encode())
    h.update(value.tobytes())
    return h.hexdigest()


def native_bounds(start):
    """Retain FIR history and the exact absolute 125/128 resampling phase."""
    if start < 0:
        raise ValueError("start must be nonnegative")
    first = start * 128 // 125
    origin = max(0, (first // 128 - 1) * 128)
    onset = ((start + 1000) * 128 + 124) // 125
    stop = ((start + 6000) * 128 + 124) // 125
    return origin, onset, stop


def unit_noise(shape, subject, task, start, seed):
    """Independent windows/seeds; share each realization across SNRs/models."""
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, subject, task, start]))
    )
    return rng.standard_normal(shape, dtype=np.float32)


def perturb_native(clean, onset, noise, snr_db):
    """SNR refers to demeaned raw EMG power after the clean prefix.

    The signal power sets a benchmark corruption magnitude, not an input
    normalization statistic. No labels enter this construction. Zero-power
    channels receive zero noise. The actual realized SNR is reported separately.
    """
    clean = np.asarray(clean, dtype=np.float32)
    if noise.shape != clean[:, onset:].shape or not (0 <= onset < clean.shape[1]):
        raise ValueError("noise shape/onset must match the native segment")
    if not np.isfinite(clean).all() or not np.isfinite(noise).all():
        raise ValueError("finite signal and noise are required")
    signal = clean[:, onset:].astype(np.float64)
    power = np.mean((signal - signal.mean(-1, keepdims=True)) ** 2, axis=-1)
    scale = (np.sqrt(power) * 10 ** (-float(snr_db) / 20)).astype(np.float32)
    result = clean.copy()
    result[:, onset:] += noise * scale[:, None]
    if not np.isfinite(result).all():
        raise ValueError("corruption overflowed")
    realized = result[:, onset:].astype(np.float64) - signal
    noise_power = np.mean(realized**2, axis=-1)
    valid = (power > 0) & (noise_power > 0)
    measured = 10 * np.log10(power[valid] / noise_power[valid])
    return result, {
        "positive_power_channels": int(valid.sum()),
        "zero_power_channels": int((power == 0).sum()),
        "realized_snr_db_mean": float(measured.mean()) if len(measured) else None,
        "realized_snr_db_min": float(measured.min()) if len(measured) else None,
        "realized_snr_db_max": float(measured.max()) if len(measured) else None,
    }


def resampled_window(segment, start, origin, resample):
    if origin % 128:
        raise ValueError("native origin must preserve the resampling phase")
    offset = start - origin * 125 // 128
    result = resample(segment)[:, offset : offset + 6000]
    if result.shape != (segment.shape[0], 6000):
        raise ValueError("insufficient native signal support")
    return result


def pose_metrics(prediction, target, starts):
    """Variance-weighted R²; differentiate only within each scored window."""
    p, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (50, 60):
        raise ValueError("expected matched (windows,50,60) arrays")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("predictions/targets must be finite")
    starts = np.asarray(starts)
    if starts.shape != (len(p),):
        raise ValueError("one start per window is required")
    delta = p - y
    denom = ((y - y.mean((0, 1))) ** 2).sum()
    source_times = ((starts[:, None] + ENDS[None, -50:]) * 128 // 125) / 2048
    dt = np.diff(source_times, axis=1)[..., None]
    velocity = np.diff(y, axis=1) / dt
    predicted_velocity = np.diff(p, axis=1) / dt
    vdenom = ((velocity - velocity.mean((0, 1))) ** 2).sum()
    return {
        "mean_3d_point_error": float(
            np.linalg.norm(delta.reshape(-1, 20, 3), axis=-1).mean()
        ),
        "coordinate_mae": float(np.abs(delta).mean()),
        "coordinate_r2": float(1 - (delta**2).sum() / denom) if denom > 0 else None,
        "velocity_r2": float(1 - ((predicted_velocity - velocity) ** 2).sum() / vdenom)
        if vdenom > 0
        else None,
    }
