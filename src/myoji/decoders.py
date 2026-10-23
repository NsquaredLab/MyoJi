"""Pose decoders ("receivers") that read tokens or continuous features.

``SparPose`` is the proposed spatial CNN + TCN applied to SPAR descriptors or
SPAR token IDs; the codebook lookup is a frozen buffer and absent boundary
edges are masked. ``CachedPoseDecoder`` reads cached learned-tokenizer channel
representations with geometry-aware channel attention and an optional causal
TCN or Transformer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from myoji._vendor.channel_tokenization import GeometryAwareChannelPool
from myoji._vendor.raulnet_v20 import RaulNetV20


class SparPose(nn.Module):
    """Same decoder in both arms; token lookup is frozen and boundary slots masked."""
    def __init__(self, codebook, mask):
        super().__init__()
        self.decoder = RaulNetV20(codebook)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))

    def forward(self, values, arm):
        if arm == "tokens":
            if values.dtype not in (torch.int32, torch.int64):
                raise TypeError("token IDs must be integer categories")
            vectors = F.embedding(values, self.decoder.codebook)
        elif arm == "continuous":
            if not values.is_floating_point():
                raise TypeError("continuous SPAR descriptors must be floating point")
            vectors = values
        else:
            raise ValueError(arm)
        return self.decoder.forward_embeddings(vectors * self.mask[None, :, None, :])


class CalibrationMetadataStyleConditioner(nn.Module):
    """Encode acquisition style separately from semantic primitive tokens."""

    metadata_fields = ("user", "session", "side")

    def __init__(
        self,
        channels: int,
        cardinalities: Sequence[int],
        style_dim: int,
        metadata_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channels <= 0 or style_dim <= 0:
            raise ValueError("style channels and dimension must be positive")
        if len(cardinalities) != len(self.metadata_fields):
            raise ValueError("style cardinalities must describe user, session, and side")
        if any(cardinality < 1 for cardinality in cardinalities):
            raise ValueError("every style vocabulary needs at least one known value")
        if not 0 <= metadata_dropout <= 1:
            raise ValueError("metadata dropout must be in [0, 1]")
        self.metadata_dropout = metadata_dropout
        self.calibration_projection = nn.Sequential(
            nn.LayerNorm(2 * channels + 1),
            nn.Linear(2 * channels + 1, style_dim),
            nn.GELU(),
            nn.Linear(style_dim, style_dim),
        )
        self.metadata_embeddings = nn.ModuleList(
            nn.Embedding(cardinality + 1, style_dim, padding_idx=0)
            for cardinality in cardinalities
        )
        self.output_norm = nn.LayerNorm(style_dim)

    def forward(
        self,
        calibration_centre: torch.Tensor,
        calibration_scale: torch.Tensor,
        style_ids: torch.Tensor,
    ) -> torch.Tensor:
        if calibration_centre.shape != calibration_scale.shape:
            raise ValueError("calibration centre and scale must have matching shapes")
        if calibration_scale.ndim != 3 or calibration_scale.shape[-1] != 1:
            raise ValueError("calibration statistics must have shape (batch, channel, 1)")
        if style_ids.shape != (calibration_scale.shape[0], len(self.metadata_fields)):
            raise ValueError("style IDs must have shape (batch, 3)")
        log_scale = calibration_scale.squeeze(-1).clamp_min(1e-8).log()
        global_log_scale = log_scale.mean(dim=-1, keepdim=True)
        relative_log_scale = log_scale - global_log_scale
        relative_centre = (
            calibration_centre.squeeze(-1) / calibration_scale.squeeze(-1)
        ).clamp(-12.0, 12.0)
        features = torch.cat(
            (relative_log_scale, global_log_scale, relative_centre),
            dim=-1,
        )
        style = self.calibration_projection(features)
        metadata_style = torch.zeros_like(style)
        for column, embedding in enumerate(self.metadata_embeddings):
            ids = style_ids[:, column]
            if self.metadata_dropout >= 1.0:
                # Identity conditioning fully disabled: the embeddings are
                # never trained, so eval must not read their random rows.
                ids = torch.zeros_like(ids)
            elif self.training and self.metadata_dropout > 0:
                keep = torch.rand(ids.shape, device=ids.device) >= self.metadata_dropout
                ids = torch.where(keep, ids, torch.zeros_like(ids))
            metadata_style = metadata_style + embedding(ids)
        metadata_style = metadata_style / math.sqrt(len(self.metadata_embeddings))
        return self.output_norm(style + metadata_style)


class CausalConvBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 2, dilation=dilation, groups=width)
        self.mixing = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        past = F.pad(self.norm(x).transpose(1, 2), (self.dilation, 0))
        return x + self.mixing(self.depthwise(past).transpose(1, 2))


class CachedPoseDecoder(nn.Module):
    """Shared frame readout, optionally followed by temporal processing.

    Geometry-aware channel attention precedes causal time attention. Calibration
    statistics are the only style inputs; user/session IDs are never supplied.
    The frame arm has the existing landmark readout's architecture (at width
    256, eight slots, style width 32), refit on the common cache and objective.
    Token embeddings remain frozen buffers in the training pipeline.
    """

    def __init__(
        self,
        embedding_dim: int,
        geometry_dim: int,
        channels: int,
        *,
        kind: str = "frame",
        width: int = 256,
        slots: int = 8,
        heads: int = 4,
        layers: int = 2,
        context_frames: int = 100,
        output_dim: int = 48,
        dropout: float = 0.1,
        input_mode: str = "full",
    ) -> None:
        super().__init__()
        if kind not in ("frame", "tcn", "transformer"):
            raise ValueError(f"unknown temporal decoder: {kind}")
        if min(width, slots, heads, layers, context_frames) < 1 or width % heads:
            raise ValueError(
                "positive dimensions and width divisible by heads required"
            )
        self.kind = kind
        if input_mode not in ("full", "features_only", "calibration_only"):
            raise ValueError(f"unknown input mode: {input_mode}")
        self.input_mode = input_mode
        self.context_frames = context_frames
        self.project = (
            nn.Identity() if width == embedding_dim else nn.Linear(embedding_dim, width)
        )
        self.pool = GeometryAwareChannelPool(
            embedding_dim=width,
            geometry_dim=geometry_dim,
            num_slots=slots,
            num_heads=heads,
            query_residual=False,
        )
        self.style = CalibrationMetadataStyleConditioner(
            channels,
            (1, 1, 1),
            style_dim=32,
            metadata_dropout=1.0,
        )
        self.style.metadata_embeddings.requires_grad_(False)
        self.frame = nn.Sequential(
            nn.LayerNorm(slots * width + 32),
            nn.Linear(slots * width + 32, width),
            nn.GELU(),
        )
        if kind == "tcn":
            self.temporal = nn.Sequential(
                *[
                    CausalConvBlock(width, 2**i, dropout)
                    for i in range(math.ceil(math.log2(max(2, context_frames))))
                ]
            )
        elif kind == "transformer":
            self.temporal = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    width,
                    heads,
                    dim_feedforward=2 * width,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                ),
                layers,
                enable_nested_tensor=False,
            )
            # TransformerEncoder clones its initial layer: independently initialize
            # matrices so the two layers do not start with identical weights.
            for layer in self.temporal.layers:
                for parameter in layer.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
        else:
            self.temporal = nn.Identity()
        self.output = nn.Linear(width, output_dim)
        position = torch.arange(context_frames).float().unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, width, 2).float() * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(context_frames, width)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: width // 2])
        self.register_buffer("position", encoding)

    def forward(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        centre: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4 or features.shape[2] > self.context_frames:
            raise ValueError(
                "features must be (batch, channel, time, dim) within context limit"
            )
        if self.input_mode == "calibration_only":
            features = torch.zeros_like(features)
        if self.input_mode == "features_only":
            centre, scale = torch.zeros_like(centre), torch.ones_like(scale)
        features = self.project(features)
        spatial = self.pool.spatialize(features, descriptors)
        pooled = self.pool(spatial, descriptors, spatialized=True)
        batch, slots, steps, width = pooled.shape
        pooled = pooled.permute(0, 2, 1, 3).reshape(batch, steps, slots * width)
        ids = torch.zeros(batch, 3, dtype=torch.long, device=features.device)
        style = self.style(centre, scale, ids).unsqueeze(1).expand(-1, steps, -1)
        frames = self.frame(torch.cat((pooled, style), dim=-1))
        if self.kind == "transformer":
            mask = torch.ones(
                steps, steps, dtype=torch.bool, device=frames.device
            ).triu(1)
            frames = self.temporal(frames + self.position[:steps], mask=mask)
        else:
            frames = self.temporal(frames)
        return self.output(frames)


class RawPoseDecoder(nn.Module):
    """Train the channel encoder and causal pose readout from scratch on raw EMG."""

    def __init__(self, raw_config: dict, **readout_kwargs):
        super().__init__()
        from myoji.learned import TokenizerConfig, build_default_tokenizer

        self.encoder = build_default_tokenizer(TokenizerConfig(**raw_config))
        # This arm uses continuous channel features, never codebook lookup or the
        # encoder's unused spatial readout. It loads no pretrained weights.
        self.encoder.frontend.channel_pool.requires_grad_(False)
        self.readout = CachedPoseDecoder(**readout_kwargs)
        self.context_frames = readout_kwargs["context_frames"]

    def forward(self, raw, descriptors, centre, scale):
        features = self.encoder(
            raw,
            sensor="wrist",
            channel_descriptors=descriptors,
            quantize=False,
            include_spatial=False,
        ).continuous[:, :, -self.context_frames :]
        return self.readout(features, descriptors, centre, scale)
