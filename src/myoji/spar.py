"""SPAR: spectral-spatial descriptors of each electrode and its grid neighbours.

For every electrode, a four-electrode patch (centre, auxiliary, horizontal and
vertical neighbour) gives one Hermitian 4x4 cross-spectral matrix per frequency
band. The real parts of three matrix-logarithm entries (centre power D0 and the
couplings O02, O03 to the outgoing horizontal and vertical neighbours) form an
18-value descriptor per electrode and update (6 bands x 3 entries).

Adapted from MyoKit's SPAR_RIE_M / SPAR_SIMPLE by Amin Olamazadeh (commit
e19a2fcf8dbbcf836e586eadc7dd9dec2942d252), see THIRD_PARTY.md. The spectral
weighting, local Hermitian 4x4 matrices, regularization and real matrix-log
entries are retained. Unlike the original two-row ring, rectangular grids have
no wraparound edges; a missing boundary neighbour is substituted from the
opposite side and its coupling entry is masked to zero.

D0 is an entry of log(S), not log(S[0,0]); off-diagonals and neighbouring
channels affect it. Taking real log entries does not confer phase invariance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_FREQUENCY_BANDS = ((20, 40), (40, 60), (60, 100), (100, 160), (160, 250), (250, 500))
DEFAULT_SEGMENT_MS = 64.0
DEFAULT_HOP_MS = 16.0


@dataclass(frozen=True)
class Layout:
    members: np.ndarray  # centre, auxiliary, horizontal, vertical
    right: np.ndarray   # -1 where no outgoing horizontal edge exists
    down: np.ndarray    # -1 where no outgoing vertical edge exists
    grids: tuple

    @property
    def channels(self):
        return len(self.members)

    @property
    def mask(self):
        return np.stack([np.ones(self.channels, dtype=bool), self.right >= 0, self.down >= 0], axis=-1)

    @property
    def entries_per_band(self):
        return int(self.mask.sum())

    def record(self):
        return dict(grids=self.grids, members=self.members.tolist(), right=self.right.tolist(), down=self.down.tolist(),
                    channels=self.channels, entries_per_band=self.entries_per_band,
                    horizontal_edges=int((self.right >= 0).sum()), vertical_edges=int((self.down >= 0).sum()))


def grid_layout(grids=None):
    """Build patches from explicit grid-site -> acquisition-channel maps.

    -1 denotes an unrecorded site. Defaults mirror hd_monopolar_montage:
    three 8x8 grids then two 13x5 grids with the last site unrecorded.
    A different channel map can be supplied without changing the SPD math.
    """
    if grids is None:
        grids = [np.arange(64 * g, 64 * (g + 1)).reshape(8, 8) for g in range(3)]
        grids += [np.r_[np.arange(64 * g, 64 * (g + 1)), -1].reshape(13, 5) for g in (3, 4)]
    grids = [np.asarray(grid, dtype=np.int64) for grid in grids]
    if any(grid.ndim != 2 or min(grid.shape) < 2 or (grid < -1).any() for grid in grids):
        raise ValueError("grids must be two-dimensional site maps; use -1 for an absent electrode")
    all_channels = np.concatenate([g[g >= 0] for g in grids])
    channels = len(all_channels)
    if not np.array_equal(np.sort(all_channels), np.arange(channels)):
        raise ValueError("each acquisition channel 0..C-1 must occur in exactly one grid site")
    members = np.empty((channels, 4), dtype=np.int64)
    right = np.full(channels, -1, dtype=np.int64)
    down = np.full(channels, -1, dtype=np.int64)
    for grid in grids:
        occupied = [(r, c, int(grid[r, c])) for r, c in np.argwhere(grid >= 0)]
        if len(occupied) < 4:
            raise ValueError("each grid needs at least four distinct electrodes for a 4x4 patch")
        for row, col, channel in occupied:
            def site(r, c):
                return int(grid[r, c]) if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] else -1
            right[channel] = site(row, col + 1)
            down[channel] = site(row + 1, col)
            neighbours = [v for _, _, v in sorted(occupied, key=lambda p: (abs(p[0] - row) + abs(p[1] - col), p[0], p[1]))
                          if v != channel]
            horizontal = right[channel] if right[channel] >= 0 else site(row, col - 1)
            if horizontal < 0:
                horizontal = neighbours[0]
            vertical = down[channel] if down[channel] >= 0 else site(row - 1, col)
            if vertical < 0 or vertical == horizontal:
                vertical = next(v for v in neighbours if v != horizontal)
            auxiliary = site(row, col - 1)
            if auxiliary < 0 or auxiliary in (channel, horizontal, vertical):
                auxiliary = next(v for v in neighbours if v not in (horizontal, vertical))
            members[channel] = channel, auxiliary, horizontal, vertical
    return Layout(members, right, down, tuple(g.tolist() for g in grids))


def spectral_layout(sfreq, samples, frequency_bands=None, segment_ms=DEFAULT_SEGMENT_MS, hop_ms=DEFAULT_HOP_MS):
    if not np.isfinite([sfreq, segment_ms, hop_ms]).all() or min(sfreq, segment_ms, hop_ms) <= 0 or samples < 8:
        raise ValueError("positive finite sample rate/times and at least eight samples are required")
    bands = DEFAULT_FREQUENCY_BANDS if frequency_bands is None else tuple(frequency_bands)
    if not bands or any(not np.isfinite([lo, hi]).all() or lo < 0 or lo >= hi for lo, hi in bands):
        raise ValueError("frequency bands must be finite ordered nonempty intervals")
    segment = min(max(8, int(round(segment_ms * sfreq / 1000))), samples)
    hop = max(1, int(round(hop_ms * sfreq / 1000)))
    frequencies = np.fft.rfftfreq(segment, 1 / sfreq)
    valid = [(tuple(band), np.flatnonzero((frequencies >= band[0]) & (frequencies < min(band[1], sfreq / 2)))) for band in bands]
    valid = [(band, indices) for band, indices in valid if len(indices)]
    if not valid:
        raise ValueError("no FFT bins fall in the requested bands")
    return segment, hop, valid


def compute_spar_simple_feature_dim(sfreq=2000., n_channels=320, frequency_bands=None, *, window_samples=320,
                                    segment_ms=DEFAULT_SEGMENT_MS, hop_ms=DEFAULT_HOP_MS, layout=None):
    layout = grid_layout() if layout is None else layout
    if n_channels != layout.channels:
        raise ValueError("channel count must match the explicit grid layout")
    return layout.entries_per_band * len(spectral_layout(sfreq, window_samples, frequency_bands, segment_ms, hop_ms)[2])


def entry_mask(layout, bad_channels=None):
    mask = layout.mask.copy()
    bad = np.asarray([] if bad_channels is None else bad_channels, dtype=np.int64)
    if bad.ndim != 1 or ((bad < 0) | (bad >= layout.channels)).any():
        raise ValueError("bad-channel indices must be in 0..C-1")
    alive = np.ones(layout.channels, dtype=bool)
    alive[bad] = False
    mask &= alive[:, None]
    mask[:, 1] &= alive[np.maximum(layout.right, 0)]
    mask[:, 2] &= alive[np.maximum(layout.down, 0)]
    return mask


@torch.inference_mode()
def spds_tensor(buffers, sfreq=2000., *, layout=None, frequency_bands=None, segment_ms=DEFAULT_SEGMENT_MS,
                hop_ms=DEFAULT_HOP_MS, bad_channels=None, band_channel_scales=None, channel_weights=None, device="cpu"):
    """(batch,samples,channels) -> (batch,bands,channels,4,4), on device.

    Form only the local covariance blocks; avoid the unnecessary CxC global
    covariance. Bin weights still use total spectral power across all channels,
    exactly as in the source. Boundary auxiliary sites are recorded in Layout.
    """
    layout = grid_layout() if layout is None else layout
    x = torch.as_tensor(buffers, dtype=torch.float32, device=device)
    if x.ndim != 3 or x.shape[2] != layout.channels or x.shape[0] < 1:
        raise ValueError(f"buffers must have nonempty (batch,samples,{layout.channels}) shape")
    if not torch.isfinite(x).all():
        raise ValueError("EMG contains non-finite samples")
    entry_mask(layout, bad_channels)  # validate once, including empty lists/arrays
    if bad_channels is not None and len(bad_channels):
        x = x.clone()
        x[..., torch.as_tensor(bad_channels, dtype=torch.long, device=device)] = 0
    length, hop, valid = spectral_layout(sfreq, x.shape[1], frequency_bands, segment_ms, hop_ms)
    hann = torch.hann_window(length, periodic=False, dtype=torch.float32, device=device)
    fft = torch.fft.rfft(x.transpose(1, 2).unfold(-1, length, hop) * hann, dim=-1)
    total_power = fft.abs().square().mean(dim=-2).sum(dim=1)
    members = torch.as_tensor(layout.members, device=device)
    local = fft[:, members]
    scale = torch.ones((len(valid), layout.channels), dtype=torch.float32, device=device)
    if band_channel_scales is not None:
        values = torch.as_tensor(band_channel_scales, dtype=torch.float32, device=device)
        if values.shape != scale.shape or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("band_channel_scales must be finite nonnegative (bands,channels)")
        scale *= values
    if channel_weights is not None:
        values = torch.as_tensor(channel_weights, dtype=torch.float32, device=device)
        if values.shape != (layout.channels,) or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("channel_weights must be finite nonnegative (channels,)")
        scale *= values.sqrt()[None]
    eye = torch.eye(4, device=device, dtype=torch.complex64)
    matrices = []
    for band, (_, indices) in enumerate(valid):
        bins = torch.as_tensor(indices, device=device)
        power = total_power[:, bins]
        weight = power / power.sum(-1, keepdim=True).clamp_min(1e-12)
        weighted = local[..., bins] * weight.sqrt()[:, None, None, None, :]
        weighted = weighted * scale[band, members][None, :, :, None, None]
        flattened = weighted.flatten(-2)
        covariance = flattened @ flattened.mH / fft.shape[-2]
        # Suppress only roundoff asymmetry, before the source's relative ridge.
        covariance = (covariance + covariance.mH) * .5
        largest = covariance.diagonal(dim1=-2, dim2=-1).real.amax(-1).clamp_min(1e-12)
        matrices.append(covariance + (1e-6 * largest)[..., None, None] * eye)
    return torch.stack(matrices, dim=1)


@torch.inference_mode()
def project_tensor(spds, *, layout=None, bad_channels=None, channelwise=False):
    layout = grid_layout() if layout is None else layout
    if spds.ndim != 5 or spds.shape[2:] != (layout.channels, 4, 4) or not torch.isfinite(spds).all():
        raise ValueError("SPDs must be finite (batch,bands,channels,4,4)")
    largest = spds.diagonal(dim1=-2, dim2=-1).real.amax(-1).clamp_min(1e-12)
    spds = spds + (1e-8 * largest)[..., None, None] * torch.eye(4, device=spds.device, dtype=spds.dtype)
    eigenvalues, vectors = torch.linalg.eigh(spds)
    matrix_log = (vectors * eigenvalues.clamp_min(1e-12).log().unsqueeze(-2)) @ vectors.mH
    values = torch.stack([matrix_log[..., 0, 0].real, matrix_log[..., 0, 2].real, matrix_log[..., 0, 3].real], dim=-1)
    mask = torch.as_tensor(entry_mask(layout, bad_channels), device=values.device)
    values = values * mask[None, None]
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite SPAR projection")
    if channelwise:
        return values.permute(0, 2, 1, 3).reshape(len(values), layout.channels, -1).contiguous()
    # Same compact layout as the supplied code: all D0, then all O02, then O03.
    return torch.cat([values[..., 0], values[..., 1][:, :, layout.right >= 0],
                      values[..., 2][:, :, layout.down >= 0]], dim=-1).flatten(1)


def compute_spar_simple_spd_batched(buffers, sfreq=2000., *, batch_size=16, **kwargs):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return np.concatenate([spds_tensor(buffers[start:start + batch_size], sfreq, **kwargs).cpu().numpy()
                           for start in range(0, len(buffers), batch_size)])


def compute_spar_simple_spd_single(window, fs=2000., **kwargs):
    return compute_spar_simple_spd_batched(np.asarray(window)[None], sfreq=fs, **kwargs)[0]


def spar_simple_dataset_batched(buffers, sfreq=2000., *, batch_size=16, aligner=None, channelwise=False, **kwargs):
    if batch_size < 1 or len(buffers) < 1:
        raise ValueError("nonempty buffers and positive batch_size are required")
    out = []
    for start in range(0, len(buffers), batch_size):
        spds = spds_tensor(buffers[start:start + batch_size], sfreq, **kwargs)
        if aligner is not None:
            aligned = aligner(spds.cpu().numpy())
            spds = torch.as_tensor(aligned, dtype=spds.dtype, device=spds.device)
        out.append(project_tensor(spds, layout=kwargs.get("layout"), bad_channels=kwargs.get("bad_channels"),
                                  channelwise=channelwise).cpu().numpy())
    return np.concatenate(out)


def compute_spar_simple_single(window, fs=2000., **kwargs):
    return spar_simple_dataset_batched(np.asarray(window)[None], sfreq=fs, **kwargs)[0]
