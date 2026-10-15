"""Deterministic electrode dropout, the sibling perturbation to added noise.

Kept out of dpose_noise.py on purpose: that module's sha256 is recorded in the
frozen noise campaign's protocol and re-checked by its verify() gate, so editing
it would invalidate a completed campaign.
"""

import numpy as np

# Electrode dropout. The added-noise study cannot speak to connector or contact
# failure -- its own limitations say "AWGN after clean startup is not electrode
# displacement or general deployment robustness" -- so this is the sibling
# perturbation: whole channels go to zero after the clean prefix, leaving the
# resampling phase, calibration prefix and token grid untouched.

DROPOUT_SEEDS = (260931, 260932, 260933)
DROPOUT_FRACTIONS = (0.05, 0.10, 0.25)
# Five physically separable arrays: three 8x8 forearm grids and two 13x5 wrist
# grids whose last site is unrecorded, so every grid contributes 64 channels.
# This mirrors the default map in grid_layout(); see hd_monopolar_montage.
GRIDS = tuple((64 * g, 64 * (g + 1)) for g in range(5))
GRID_DROPS = (1, 2)


def grid_bounds():
    """Fail loudly if the recorded layout ever stops covering 320 electrodes."""
    spans = [stop - start for start, stop in GRIDS]
    if [g[0] for g in GRIDS[1:]] != [g[1] for g in GRIDS[:-1]]:
        raise ValueError("grid spans must tile without gaps")
    if sum(spans) != 320 or spans != [64] * 5:
        raise ValueError(f"unexpected grid layout: {spans}")
    return spans


def dropped_channels(channels, subject, task, start, seed, *, fraction=None, grids=None):
    """Channels to silence: a scattered fraction, or whole grids. Never both."""
    if (fraction is None) == (grids is None):
        raise ValueError("give exactly one of fraction or grids")
    grid_bounds()
    rng = np.random.default_rng(
        [int(seed), int(subject), int(task), int(start), 0 if grids is None else 1])
    if grids is not None:
        if not 0 < grids < len(GRIDS):
            raise ValueError("grids must silence at least one and leave one live")
        chosen = rng.choice(len(GRIDS), size=int(grids), replace=False)
        return np.sort(np.concatenate([np.arange(*GRIDS[g]) for g in chosen]))
    count = int(round(float(fraction) * channels))
    if not 0 < count < channels:
        raise ValueError(f"fraction {fraction} silences {count} of {channels}")
    return np.sort(rng.choice(channels, size=count, replace=False))


def drop_native(clean, onset, channels):
    """Zero the named electrodes from onset on; the clean prefix is preserved."""
    clean = np.asarray(clean, dtype=np.float32)
    if not 0 <= onset < clean.shape[1]:
        raise ValueError("onset must fall inside the segment")
    channels = np.asarray(channels, dtype=np.int64)
    if channels.size == 0 or channels.min() < 0 or channels.max() >= clean.shape[0]:
        raise ValueError("channel indices out of range")
    if len(set(channels.tolist())) != channels.size:
        raise ValueError("channel indices must be unique")
    result = clean.copy()
    result[channels, onset:] = 0.0
    live = np.setdiff1d(np.arange(clean.shape[0]), channels)
    power = np.mean(clean[:, onset:].astype(np.float64) ** 2, axis=-1)
    return result, {
        "dropped_channels": channels.tolist(),
        "dropped_count": int(channels.size),
        "live_count": int(live.size),
        "dropped_fraction": float(channels.size / clean.shape[0]),
        "dropped_signal_power_fraction": float(
            power[channels].sum() / power.sum()) if power.sum() > 0 else 0.0,
        "silent_before_drop": int((power[channels] == 0).sum()),
    }
