"""Channel-wise tokenization components for variable-layout biosignals.

The temporal encoder is shared across electrodes. Geometry is introduced only
when channel tokens are pooled into a fixed number of learned latent slots, so
the same module can represent sensors with different channel counts and layouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ChannelTokenizationOutput:
    """Continuous tokens before any task-specific discretization."""

    channel_tokens: torch.Tensor
    spatial_tokens: torch.Tensor
    muscle_tokens: torch.Tensor


class ChannelPatchEncoder(nn.Module):
    """Encode temporal patches independently with weights shared by all channels.

    Inputs have shape ``(batch, channel, time)`` and outputs have shape
    ``(batch, channel, token, embedding)``. With no temporal padding, output
    token ``i`` only observes samples through
    ``patch_size - 1 + i * stride``.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        embedding_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        hidden_dim = hidden_dim or embedding_dim
        self.patch_size = patch_size
        self.stride = stride
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=patch_size, stride=stride),
            nn.GELU(),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=1),
        )

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        if emg.ndim != 3:
            raise ValueError(
                "EMG must have shape (batch, channel, time), "
                f"received {tuple(emg.shape)}"
            )
        if emg.shape[-1] < self.patch_size:
            raise ValueError(
                f"need at least {self.patch_size} time samples, "
                f"received {emg.shape[-1]}"
            )

        batch, channels, time = emg.shape
        channel_signals = emg.reshape(batch * channels, 1, time)
        tokens = self.encoder(channel_signals).transpose(1, 2)
        return tokens.reshape(batch, channels, tokens.shape[1], self.embedding_dim)

    def token_end_samples(
        self, n_tokens: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Return the inclusive input-sample index associated with each token."""
        return self.patch_size - 1 + torch.arange(n_tokens, device=device) * self.stride


class GeometryAwareChannelPool(nn.Module):
    """Pool any number of positioned channel tokens into learned latent slots.

    ``channel_positions`` can be shared across a batch with shape
    ``(channel, geometry)`` or supplied per example as
    ``(batch, channel, geometry)``. ``channel_mask`` uses ``True`` for available
    channels. Jointly permuting tokens, positions, and masks leaves the result
    unchanged.
    """

    def __init__(
        self,
        embedding_dim: int,
        geometry_dim: int,
        num_slots: int = 8,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        query_residual: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or geometry_dim <= 0 or num_slots <= 0:
            raise ValueError("embedding_dim, geometry_dim, and num_slots must be positive")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.geometry_dim = geometry_dim
        self.num_slots = num_slots
        self.query_residual = query_residual
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.slots = nn.Parameter(
            torch.randn(num_slots, embedding_dim) / embedding_dim**0.5
        )
        self.key_norm = nn.LayerNorm(embedding_dim)
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = embedding_dim * mlp_ratio
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _geometry_embeddings(
        self,
        channel_positions: torch.Tensor,
        batch: int,
        channels: int,
        tokens: int,
    ) -> torch.Tensor:
        if channel_positions.ndim == 2:
            if channel_positions.shape != (channels, self.geometry_dim):
                raise ValueError(
                    "shared channel_positions must have shape "
                    f"({channels}, {self.geometry_dim})"
                )
            geometry = self.geometry_encoder(channel_positions)
            geometry = geometry[None, None].expand(batch, tokens, -1, -1)
        elif channel_positions.ndim == 3:
            if channel_positions.shape != (batch, channels, self.geometry_dim):
                raise ValueError(
                    "batched channel_positions must have shape "
                    f"({batch}, {channels}, {self.geometry_dim})"
                )
            geometry = self.geometry_encoder(channel_positions)
            geometry = geometry[:, None].expand(-1, tokens, -1, -1)
        else:
            raise ValueError("channel_positions must have two or three dimensions")
        return geometry.reshape(batch * tokens, channels, self.embedding_dim)

    @staticmethod
    def _attention_mask(
        channel_mask: torch.Tensor | None,
        batch: int,
        channels: int,
        tokens: int,
    ) -> torch.Tensor | None:
        if channel_mask is None:
            return None
        if channel_mask.ndim == 1:
            if channel_mask.shape != (channels,):
                raise ValueError(f"channel_mask must contain {channels} entries")
            channel_mask = channel_mask[None].expand(batch, -1)
        elif channel_mask.shape != (batch, channels):
            raise ValueError(f"channel_mask must have shape ({batch}, {channels})")
        channel_mask = channel_mask.bool()
        if not torch.all(channel_mask.any(dim=1)):
            raise ValueError("each example must contain at least one available channel")
        return (~channel_mask[:, None].expand(-1, tokens, -1)).reshape(
            batch * tokens, channels
        )

    def forward(
        self,
        channel_tokens: torch.Tensor,
        channel_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        spatialized: bool = False,
    ) -> torch.Tensor:
        if channel_tokens.ndim != 4:
            raise ValueError(
                "channel_tokens must have shape (batch, channel, token, embedding)"
            )
        batch, channels, tokens, embedding = channel_tokens.shape
        if embedding != self.embedding_dim:
            raise ValueError(
                f"token embedding dimension must be {self.embedding_dim}, "
                f"received {embedding}"
            )

        values = channel_tokens.permute(0, 2, 1, 3).reshape(
            batch * tokens, channels, embedding
        )
        if not spatialized:
            values = values + self._geometry_embeddings(
                channel_positions, batch, channels, tokens
            )
        key_padding_mask = self._attention_mask(
            channel_mask, batch, channels, tokens
        )
        queries = self.slots[None].expand(batch * tokens, -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.key_norm(values),
            values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        muscle_tokens = attended
        if self.query_residual:
            muscle_tokens = muscle_tokens + queries
        muscle_tokens = muscle_tokens + self.output_mlp(
            self.output_norm(muscle_tokens)
        )
        muscle_tokens = muscle_tokens.reshape(
            batch, tokens, self.num_slots, embedding
        )
        return muscle_tokens.permute(0, 2, 1, 3)

    def spatialize(
        self,
        channel_tokens: torch.Tensor,
        channel_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Attach learned channel-position embeddings to per-channel tokens."""
        if channel_tokens.ndim != 4:
            raise ValueError(
                "channel_tokens must have shape (batch, channel, token, embedding)"
            )
        batch, channels, tokens, embedding = channel_tokens.shape
        if embedding != self.embedding_dim:
            raise ValueError(
                f"token embedding dimension must be {self.embedding_dim}, "
                f"received {embedding}"
            )
        values = channel_tokens.permute(0, 2, 1, 3).reshape(
            batch * tokens, channels, embedding
        )
        geometry = self._geometry_embeddings(
            channel_positions, batch, channels, tokens
        )
        spatial = (values + geometry).reshape(batch, tokens, channels, embedding)
        return spatial.permute(0, 2, 1, 3)


class ChannelTokenizerFrontend(nn.Module):
    """Shared channel patch encoding followed by geometry-aware slot pooling."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        embedding_dim: int,
        geometry_dim: int,
        num_slots: int = 8,
        num_heads: int = 4,
        channel_hidden_dim: int | None = None,
        dropout: float = 0.0,
        query_residual: bool = True,
    ) -> None:
        super().__init__()
        self.channel_encoder = ChannelPatchEncoder(
            patch_size=patch_size,
            stride=stride,
            embedding_dim=embedding_dim,
            hidden_dim=channel_hidden_dim,
        )
        self.channel_pool = GeometryAwareChannelPool(
            embedding_dim=embedding_dim,
            geometry_dim=geometry_dim,
            num_slots=num_slots,
            num_heads=num_heads,
            dropout=dropout,
            query_residual=query_residual,
        )

    def encode_channels(
        self,
        emg: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode each channel independently and optionally mask unavailable ones."""
        channel_tokens = self.channel_encoder(emg)
        if channel_mask is not None:
            if channel_mask.ndim == 1:
                channel_mask = channel_mask[None].expand(emg.shape[0], -1)
            channel_tokens = channel_tokens * channel_mask[:, :, None, None]
        return channel_tokens

    def spatialize_channels(
        self,
        channel_tokens: torch.Tensor,
        channel_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Add learned spatial embeddings without changing the channel axis."""
        return self.channel_pool.spatialize(channel_tokens, channel_positions)

    def pool_channels(
        self,
        channel_tokens: torch.Tensor,
        channel_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        spatialized: bool = False,
    ) -> torch.Tensor:
        """Map positioned channel tokens into a fixed learned slot axis."""
        return self.channel_pool(
            channel_tokens,
            channel_positions,
            channel_mask=channel_mask,
            spatialized=spatialized,
        )

    def forward(
        self,
        emg: torch.Tensor,
        channel_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> ChannelTokenizationOutput:
        channel_tokens = self.encode_channels(emg, channel_mask=channel_mask)
        spatial_tokens = self.spatialize_channels(
            channel_tokens, channel_positions
        )
        muscle_tokens = self.pool_channels(
            spatial_tokens,
            channel_positions,
            channel_mask=channel_mask,
            spatialized=True,
        )
        return ChannelTokenizationOutput(
            channel_tokens=channel_tokens,
            spatial_tokens=spatial_tokens,
            muscle_tokens=muscle_tokens,
        )
