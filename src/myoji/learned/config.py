"""Configuration shared by the tokenizer models and data pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from myoji.learned.montage import CHANNEL_DESCRIPTOR_DIM


@dataclass(frozen=True)
class TokenizerConfig:
    """Core temporal and representation settings.

    Physical units are kept in the public configuration so that experiments do
    not silently change their receptive field when the sampling rate changes.
    """

    sample_rate_hz: int = 2_000
    token_rate_hz: int = 50
    receptive_field_ms: float = 160.0
    embedding_dim: int = 256
    muscle_slots: int = 8
    geometry_dim: int = CHANNEL_DESCRIPTOR_DIM
    channel_hidden_dim: int = 32
    channel_attention_heads: int = 4
    temporal_layers: int = 4
    temporal_kernel_size: int = 5
    codebook_size: int = 1_024
    commitment_weight: float = 0.25
    ema_decay: float = 0.9
    codebook_update_tokens: int | None = 4_096

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.token_rate_hz <= 0:
            raise ValueError("sample and token rates must be positive")
        if self.sample_rate_hz % self.token_rate_hz:
            raise ValueError("sample_rate_hz must be divisible by token_rate_hz")
        if self.receptive_field_ms <= 0:
            raise ValueError("receptive_field_ms must be positive")
        if self.embedding_dim <= 0 or self.codebook_size <= 1:
            raise ValueError("embedding_dim and codebook_size must be positive")
        if self.muscle_slots <= 0 or self.geometry_dim <= 0:
            raise ValueError("muscle_slots and geometry_dim must be positive")
        if self.channel_hidden_dim <= 0:
            raise ValueError("channel_hidden_dim must be positive")
        if self.embedding_dim % self.channel_attention_heads:
            raise ValueError(
                "embedding_dim must be divisible by channel_attention_heads"
            )
        if self.temporal_layers < 0:
            raise ValueError("temporal_layers cannot be negative")
        if self.temporal_kernel_size < 1:
            raise ValueError("temporal_kernel_size must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.codebook_update_tokens is not None and self.codebook_update_tokens <= 0:
            raise ValueError("codebook_update_tokens must be positive or None")

    @property
    def hop_samples(self) -> int:
        return self.sample_rate_hz // self.token_rate_hz

    @property
    def receptive_samples(self) -> int:
        samples = self.sample_rate_hz * self.receptive_field_ms / 1_000.0
        rounded = round(samples)
        if abs(samples - rounded) > 1e-9:
            raise ValueError("receptive_field_ms must map to an integer sample count")
        return rounded

    @property
    def effective_receptive_samples(self) -> int:
        temporal_context = (
            self.temporal_layers
            * (self.temporal_kernel_size - 1)
            * self.hop_samples
        )
        return self.receptive_samples + temporal_context
