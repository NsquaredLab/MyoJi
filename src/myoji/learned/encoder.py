"""Learned causal EMG tokenizer: a shared per-channel encoder and one EMA codebook.

Every electrode is encoded independently by the same temporal patch encoder and
four causal residual blocks, then mapped to the nearest of 1,024 code vectors.
Montage descriptors are attached only for optional downstream spatial readouts;
they never change the primitive token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from myoji._vendor.channel_tokenization import ChannelTokenizerFrontend
from myoji.learned.config import TokenizerConfig
from myoji.learned.montage import circular_bipolar_montage, hd_monopolar_montage
from myoji.learned.quantizer import EMAVectorQuantizer


class CausalResidualBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, kernel_size: int = 5, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.norm = nn.LayerNorm(embedding_dim)
        self.depthwise = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=kernel_size,
            groups=embedding_dim,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(embedding_dim, 4 * embedding_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(4 * embedding_dim, embedding_dim, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        features = self.norm(tokens).transpose(1, 2)
        features = F.pad(features, (self.left_padding, 0))
        features = self.depthwise(features)
        features = self.pointwise(features).transpose(1, 2)
        return residual + self.dropout(features)


@dataclass
class TokenizerOutput:
    """Tokenizer values with optional spatial and downstream readout views.

    ``tokens`` and ``quantized`` describe the units that were actually vector
    quantized. Channel-wise models additionally attach position embeddings in
    ``spatial_tokens`` and expose an optional fixed-slot ``pooled`` readout.
    """

    continuous: torch.Tensor
    quantized: torch.Tensor
    tokens: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    active_fraction: torch.Tensor
    channel_tokens: torch.Tensor | None = None
    spatial_tokens: torch.Tensor | None = None
    pooled: torch.Tensor | None = None


def _adapt_geometry(positions: torch.Tensor, geometry_dim: int) -> torch.Tensor:
    if positions.shape[1] == geometry_dim:
        return positions
    if positions.shape[1] > geometry_dim:
        return positions[:, :geometry_dim]
    padding = positions.new_zeros(positions.shape[0], geometry_dim - positions.shape[1])
    return torch.cat((positions, padding), dim=1)


def wrist_channel_positions(
    channels: int = 16, geometry_dim: int = 3
) -> torch.Tensor:
    """Return montage-aware circular descriptors in acquisition order."""
    return _adapt_geometry(
        circular_bipolar_montage(channels).descriptor(), geometry_dim
    )


def hd_channel_positions(
    grids: int = 5,
    rows: int = 8,
    columns: int = 8,
    geometry_dim: int = 3,
) -> torch.Tensor:
    """Return descriptors for the native five-grid DPose montage.

    The retained arguments are accepted for API compatibility, but the real
    montage is fixed: three 8x8 grids plus two 13x5 grids with 64 recorded
    sites each.
    """
    if (grids, rows, columns) != (5, 8, 8):
        raise ValueError("the documented DPose montage is fixed at 320 channels")
    return _adapt_geometry(hd_monopolar_montage().descriptor(), geometry_dim)


class ChannelwiseEMGTokenizer(nn.Module):
    """Discretize each electrode independently, then attach its position.

    The temporal patch and causal encoders share weights across every electrode
    and sensor. A shared codebook quantizes channel content before a learned
    position embedding is attached. Geometry-aware slot pooling is retained as
    a downstream readout, not as the unit of discretization.
    """

    def __init__(
        self,
        sensor_positions: dict[str, torch.Tensor],
        config: TokenizerConfig,
    ) -> None:
        super().__init__()
        if not sensor_positions:
            raise ValueError("at least one sensor geometry is required")
        self.config = config
        self._position_buffers: dict[str, str] = {}
        for index, (sensor, positions) in enumerate(sensor_positions.items()):
            if positions.ndim != 2 or positions.shape[1] != config.geometry_dim:
                raise ValueError(
                    f"positions for {sensor!r} must have shape "
                    f"(channel, {config.geometry_dim})"
                )
            buffer_name = f"_sensor_positions_{index}"
            self.register_buffer(buffer_name, positions.float())
            self._position_buffers[sensor] = buffer_name

        self.frontend = ChannelTokenizerFrontend(
            patch_size=config.receptive_samples,
            stride=config.hop_samples,
            embedding_dim=config.embedding_dim,
            geometry_dim=config.geometry_dim,
            num_slots=config.muscle_slots,
            num_heads=config.channel_attention_heads,
            channel_hidden_dim=config.channel_hidden_dim,
            query_residual=False,
        )
        self.temporal_encoder = nn.Sequential(
            *[
                CausalResidualBlock(
                    config.embedding_dim,
                    kernel_size=config.temporal_kernel_size,
                )
                for _ in range(config.temporal_layers)
            ]
        )
        self.pre_quantizer_norm = nn.LayerNorm(config.embedding_dim)
        self.quantizer = EMAVectorQuantizer(
            codebook_size=config.codebook_size,
            embedding_dim=config.embedding_dim,
            commitment_weight=config.commitment_weight,
            decay=config.ema_decay,
            update_sample_size=config.codebook_update_tokens,
        )

    @property
    def sensors(self) -> tuple[str, ...]:
        return tuple(self._position_buffers)

    @property
    def output_slots(self) -> int:
        return self.config.muscle_slots

    def sensor_positions(self, sensor: str) -> torch.Tensor:
        if sensor not in self._position_buffers:
            choices = ", ".join(self.sensors)
            raise KeyError(f"unknown sensor '{sensor}'; choose one of: {choices}")
        return getattr(self, self._position_buffers[sensor])

    def forward(
        self,
        emg: torch.Tensor,
        sensor: str,
        channel_positions: torch.Tensor | None = None,
        channel_descriptors: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        quantize: bool = True,
        include_spatial: bool = True,
    ) -> TokenizerOutput:
        if emg.ndim == 5:
            emg = emg.reshape(emg.shape[0], -1, emg.shape[-1])
        if emg.ndim != 3:
            raise ValueError(
                "EMG must have shape (batch, channel, time) or a flattened "
                "five-dimensional grid layout"
            )

        if channel_positions is not None and channel_descriptors is not None:
            raise ValueError(
                "pass channel_descriptors or legacy channel_positions, not both"
            )
        positions = (
            self.sensor_positions(sensor)
            if channel_positions is None and channel_descriptors is None
            else (
                channel_descriptors
                if channel_descriptors is not None
                else channel_positions
            )
        )
        positions = positions.to(device=emg.device, dtype=emg.dtype)
        if positions.shape[-2] != emg.shape[1]:
            raise ValueError(
                f"received {emg.shape[1]} EMG channels but "
                f"{positions.shape[-2]} channel positions"
            )
        if channel_mask is not None:
            channel_mask = channel_mask.to(device=emg.device)

        channel_tokens = self.frontend.encode_channels(
            emg,
            channel_mask=channel_mask,
        )
        batch, channels, tokens, embedding = channel_tokens.shape
        continuous = self.temporal_encoder(
            channel_tokens.reshape(batch * channels, tokens, embedding)
        ).reshape(batch, channels, tokens, embedding)
        continuous = self.pre_quantizer_norm(continuous)
        if not quantize:
            spatial_tokens = None
            pooled = None
            if include_spatial:
                spatial_tokens = self.frontend.spatialize_channels(
                    continuous, positions
                )
                pooled = self.frontend.pool_channels(
                    spatial_tokens,
                    positions,
                    channel_mask=channel_mask,
                    spatialized=True,
                )
            return TokenizerOutput(
                continuous=continuous,
                quantized=continuous,
                tokens=torch.full(
                    continuous.shape[:-1],
                    -1,
                    dtype=torch.long,
                    device=continuous.device,
                ),
                commitment_loss=continuous.new_zeros(()),
                perplexity=continuous.new_zeros(()),
                active_fraction=continuous.new_zeros(()),
                channel_tokens=channel_tokens,
                spatial_tokens=spatial_tokens,
                pooled=pooled,
            )
        quantized = self.quantizer(continuous)
        spatial_tokens = None
        pooled = None
        if include_spatial:
            spatial_tokens = self.frontend.spatialize_channels(
                quantized.quantized, positions
            )
            pooled = self.frontend.pool_channels(
                spatial_tokens,
                positions,
                channel_mask=channel_mask,
                spatialized=True,
            )
        return TokenizerOutput(
            continuous=continuous,
            quantized=quantized.quantized,
            tokens=quantized.indices,
            commitment_loss=quantized.commitment_loss,
            perplexity=quantized.perplexity,
            active_fraction=quantized.active_fraction,
            channel_tokens=channel_tokens,
            spatial_tokens=spatial_tokens,
            pooled=pooled,
        )

    def token_end_times(self, n_tokens: int) -> torch.Tensor:
        sample_indices = (
            self.config.receptive_samples
            - 1
            + torch.arange(n_tokens) * self.config.hop_samples
        )
        return sample_indices / self.config.sample_rate_hz


def build_channelwise_tokenizer(
    config: TokenizerConfig | None = None,
) -> ChannelwiseEMGTokenizer:
    config = config or TokenizerConfig()
    return ChannelwiseEMGTokenizer(
        sensor_positions={
            "wrist": wrist_channel_positions(geometry_dim=config.geometry_dim),
            "hd": hd_channel_positions(geometry_dim=config.geometry_dim),
        },
        config=config,
    )


def build_default_tokenizer(config: TokenizerConfig | None = None) -> ChannelwiseEMGTokenizer:
    """Build the channel-wise tokenizer used by the main experiments."""
    return build_channelwise_tokenizer(config)
