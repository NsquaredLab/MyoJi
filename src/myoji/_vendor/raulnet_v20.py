"""Experimental token-input RaulNet: V19-style spatial mixing plus a causal TCN.

V20 consumes categorical channel-by-time IDs from an existing frozen tokenizer.
It does not consume raw EMG, copy V19 weights, or retokenize the signal. Its
spatial convolutions retain V19's grid/electrode kernel, dilation and padding
conventions; temporal convolutions operate on the token-frame sequence instead
of the native-rate waveform. Every normalization is local to one frame.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


class _CausalTokenBlock(nn.Module):
    """Residual, depthwise temporal convolution with per-frame LayerNorm."""

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(
            width, width, kernel_size=2, dilation=dilation, groups=width
        )
        self.mixing = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        past = F.pad(self.norm(frames).transpose(1, 2), (self.dilation, 0))
        return frames + self.mixing(self.temporal(past).transpose(1, 2))


class RaulNetV20(nn.Module):
    """Decode frozen EMG token IDs into a causal sequence of hand poses.

    Parameters
    ----------
    codebook:
        Finite floating-point tensor ``(vocabulary, embedding_dim)`` from the
        exact tokenizer that produced the IDs. Copied into a persistent frozen
        buffer; it is not a trainable embedding table.
    nr_of_electrode_grids, nr_of_electrodes_per_grid:
        Fixed channel layout. Adjacent blocks of ``nr_of_electrodes_per_grid``
        channels form each grid, in the acquisition order used by V19. This
        inherits V19's grid-order assumption, not verified physical geometry or
        invariance to channel permutations. All channels must be supplied.
    context_frames:
        Maximum input length and requested temporal coverage. Dilations are
        powers of two until coverage reaches this length. With the default
        100 frames, seven blocks have a nominal 128-frame receptive field;
        the available input history limits predictions to at most 100 frames.
    use_temporal:
        Set false for a frame-only ablation with the same spatial module and
        pose head. No temporal blocks or temporal parameters are then created.

    Inputs are integer ``(batch, channels, token_frames)`` arrays. Outputs are
    ``(batch, token_frames, nr_of_outputs)``. The default 320-channel/60-output
    configuration matches the DPose token interface. Raw-signal preprocessing
    and tokenizer latency remain upstream; causality is not a runtime benchmark.

    This is an experimental architecture, not a trained or validated model.
    Unlike V19's Lightning wrapper, V20 is a plain ``nn.Module``; the experiment
    owns the optimizer, target normalization, splits, loss and checkpoint choice.
    """

    def __init__(
        self,
        codebook: torch.Tensor,
        *,
        nr_of_electrode_grids: int = 5,
        nr_of_electrodes_per_grid: int = 64,
        nr_of_outputs: int = 60,
        cnn_encoder_channels: tuple[int, int, int] = (64, 32, 32),
        mlp_encoder_channels: tuple[int, int] = (128, 128),
        temporal_width: int = 64,
        context_frames: int = 100,
        dropout: float = 0.1,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        if (
            not isinstance(codebook, torch.Tensor)
            or codebook.ndim != 2
            or not codebook.is_floating_point()
            or codebook.shape[0] < 2
            or codebook.shape[1] < 1
            or not torch.isfinite(codebook).all()
        ):
            raise ValueError(
                "codebook must be a finite floating (vocabulary,dim) tensor"
            )
        values = (
            nr_of_electrode_grids,
            nr_of_electrodes_per_grid,
            nr_of_outputs,
            temporal_width,
            context_frames,
            *cnn_encoder_channels,
            *mlp_encoder_channels,
        )
        if len(cnn_encoder_channels) != 3 or len(mlp_encoder_channels) != 2:
            raise ValueError("expected three CNN widths and two MLP widths")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values):
            raise ValueError("dimensions and context_frames must be positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.nr_of_electrode_grids = nr_of_electrode_grids
        self.nr_of_electrodes_per_grid = nr_of_electrodes_per_grid
        self.nr_of_outputs = nr_of_outputs
        self.channels = nr_of_electrode_grids * nr_of_electrodes_per_grid
        self.context_frames = context_frames
        self.use_temporal = use_temporal
        self.register_buffer(
            "codebook", codebook.detach().clone().to(dtype=torch.float32)
        )
        self.vocabulary_size, self.embedding_dim = self.codebook.shape

        c0, c1, c2 = cnn_encoder_channels
        grids, electrodes = nr_of_electrode_grids, nr_of_electrodes_per_grid
        # Frames are folded into the batch dimension before these modules.
        # InstanceNorm2d therefore reduces only grid/electrode axes, never time.
        self.token_projection = nn.Sequential(
            nn.Conv2d(self.embedding_dim, c0, kernel_size=1),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm2d(c0),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                c0,
                c1,
                kernel_size=(grids, _ceil_div(electrodes, 2)),
                dilation=(1, 2),
                padding=(_ceil_div(grids, 2), _ceil_div(electrodes, 4)),
                padding_mode="circular",
            ),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm2d(c1),
            nn.Conv2d(c1, c2, kernel_size=(grids, _ceil_div(electrodes, 7))),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm2d(c2),
            nn.Flatten(),
        )
        spatial_grids = 2 * _ceil_div(grids, 2) - grids + 2
        spatial_electrodes = (
            electrodes
            + 2 * _ceil_div(electrodes, 4)
            - 2 * _ceil_div(electrodes, 2)
            - _ceil_div(electrodes, 7)
            + 3
        )
        if min(spatial_grids, spatial_electrodes) < 1 or grids * electrodes < 2:
            raise ValueError(
                "grid layout is too small for spatial mixing/normalization"
            )
        spatial_dim = c2 * spatial_grids * spatial_electrodes
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, temporal_width),
            nn.GELU(),
        )
        layers = math.ceil(math.log2(context_frames)) if use_temporal else 0
        self.temporal = nn.Sequential(
            *[_CausalTokenBlock(temporal_width, 2**i, dropout) for i in range(layers)]
        )
        self.receptive_field_frames = 2**layers
        self.pose_head = nn.Sequential(
            nn.Linear(temporal_width, mlp_encoder_channels[0]),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_encoder_channels[0], mlp_encoder_channels[1]),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_encoder_channels[1], nr_of_outputs),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup categorical IDs; never apply convolutions to integer magnitudes."""
        if token_ids.ndim != 3 or token_ids.shape[1] != self.channels:
            raise ValueError(f"tokens must have shape (batch,{self.channels},frames)")
        if token_ids.shape[0] < 1 or not 1 <= token_ids.shape[2] <= self.context_frames:
            raise ValueError("empty batch or token sequence outside context_frames")
        if token_ids.dtype not in (torch.int32, torch.int64, torch.uint16):
            raise TypeError(
                "token IDs must be int32, int64 or uint16, not waveform values"
            )
        indices = token_ids.to(dtype=torch.int64)
        if torch.any(indices < 0) or torch.any(indices >= self.vocabulary_size):
            raise ValueError("token ID outside the supplied codebook")
        if indices.device != self.codebook.device:
            raise ValueError("tokens and model must be on the same device")
        return self.forward_embeddings(F.embedding(indices, self.codebook))

    def forward_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Consume explicit vectors for a matched continuous-feature control.

        The ordinary token API is ``forward``. This separate path allows an
        experiment to isolate quantization while reusing the identical readout.
        """
        if (
            embeddings.ndim != 4
            or embeddings.shape[1] != self.channels
            or embeddings.shape[-1] != self.embedding_dim
            or embeddings.shape[0] < 1
            or not 1 <= embeddings.shape[2] <= self.context_frames
        ):
            raise ValueError("embeddings must have shape (batch,channels,frames,dim)")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must be floating-point vectors")
        batch, _, steps, _ = embeddings.shape
        frames = embeddings.permute(0, 2, 3, 1).reshape(
            batch * steps,
            self.embedding_dim,
            self.nr_of_electrode_grids,
            self.nr_of_electrodes_per_grid,
        )
        frames = self.frame_projection(self.spatial(self.token_projection(frames)))
        sequence = frames.reshape(batch, steps, -1)
        return self.pose_head(self.temporal(sequence))


__all__ = ["RaulNetV20"]
