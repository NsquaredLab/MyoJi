"""Label-free pretraining: reconstruct each causal patch's log-power spectrum.

The same head is applied to every channel. Training alternates sources, first
without quantization, then after a k-means codebook initialization with the EMA
vector quantizer, a commitment loss and a low-code-usage penalty.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from myoji.learned.config import TokenizerConfig
from myoji.learned.encoder import ChannelwiseEMGTokenizer, build_default_tokenizer
from myoji.preprocessing import calibration_normalize

__all__ = [
    "UniversalEMGTokenizerModule",
    "atomic_torch_save",
    "calibration_normalize",
    "last_checkpoint_path",
    "load_universal_checkpoint",
    "restore_training_state",
    "sample_calibration_latents",
    "training_state",
]


class UniversalEMGTokenizerModule(nn.Module):
    """Pretrain on any EMG recording through causal patch spectra.

    The same head is applied to every channel and montage.  The target is a
    log-power spectrum of the causal input patch, which is useful for raw EMG,
    1 kHz implanted recordings, spatial differentials, and bipolar wearables.
    Raw waveform phase is deliberately not the only target.  Quantization can
    remain disabled for continuous pretraining and enabled after codebook
    calibration.
    """

    def __init__(
        self,
        tokenizer: ChannelwiseEMGTokenizer,
        spectral_bins: int = 64,
        calibration_ms: float = 500.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        use_quantization: bool = False,
        polarity_flip_probability: float = 0.5,
        amplitude_jitter: float = 0.2,
        usage_weight: float = 0.1,
        usage_target_fraction: float = 0.25,
        usage_temperature: float = 0.05,
        usage_sample_size: int = 4_096,
    ) -> None:
        super().__init__()
        if (
            spectral_bins <= 0
            or spectral_bins > tokenizer.config.receptive_samples // 2
        ):
            raise ValueError("spectral_bins must fit inside the patch rFFT")
        calibration_samples = round(
            tokenizer.config.sample_rate_hz * calibration_ms / 1_000.0
        )
        if calibration_samples < tokenizer.config.receptive_samples:
            raise ValueError("calibration must span at least one receptive field")
        self.tokenizer = tokenizer
        self.spectral_bins = spectral_bins
        self.calibration_samples = calibration_samples
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_quantization = use_quantization
        self.polarity_flip_probability = polarity_flip_probability
        self.amplitude_jitter = amplitude_jitter
        if usage_weight < 0:
            raise ValueError("usage_weight cannot be negative")
        if not 0 < usage_target_fraction <= 1:
            raise ValueError("usage_target_fraction must be in (0, 1]")
        if usage_temperature <= 0 or usage_sample_size <= 0:
            raise ValueError("usage temperature and sample size must be positive")
        self.usage_weight = usage_weight
        self.usage_target_fraction = usage_target_fraction
        self.usage_temperature = usage_temperature
        self.usage_sample_size = usage_sample_size
        self.spectral_head = nn.Sequential(
            nn.LayerNorm(tokenizer.config.embedding_dim),
            nn.Linear(tokenizer.config.embedding_dim, tokenizer.config.embedding_dim),
            nn.GELU(),
            nn.Linear(tokenizer.config.embedding_dim, spectral_bins),
        )
        self.register_buffer(
            "spectral_window",
            torch.hann_window(tokenizer.config.receptive_samples, periodic=False),
        )

    @staticmethod
    def _sensor_name(sensor: str | Sequence[str]) -> str:
        if isinstance(sensor, str):
            return sensor
        unique = set(sensor)
        if len(unique) != 1:
            raise ValueError("a batch must contain exactly one sensor geometry")
        return next(iter(unique))

    def _augment(self, emg: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return emg
        if self.amplitude_jitter > 0:
            gain = torch.exp(
                self.amplitude_jitter
                * torch.randn(
                    emg.shape[0],
                    emg.shape[1],
                    1,
                    device=emg.device,
                    dtype=emg.dtype,
                )
            )
            emg = emg * gain
        if self.polarity_flip_probability > 0:
            flip = (
                torch.rand(emg.shape[0], emg.shape[1], 1, device=emg.device)
                < self.polarity_flip_probability
            )
            emg = torch.where(flip, -emg, emg)
        return emg

    def _spectral_targets(self, emg: torch.Tensor) -> torch.Tensor:
        patches = emg.unfold(
            dimension=-1,
            size=self.tokenizer.config.receptive_samples,
            step=self.tokenizer.config.hop_samples,
        )
        patches = patches - patches.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(patches * self.spectral_window, dim=-1)
        log_power = torch.log1p(spectrum.real.square() + spectrum.imag.square())
        return log_power[..., 1 : self.spectral_bins + 1]

    def scored_token_mask(self, n_tokens: int, device: torch.device) -> torch.Tensor:
        end_samples = (
            self.tokenizer.config.receptive_samples
            - 1
            + torch.arange(n_tokens, device=device) * self.tokenizer.config.hop_samples
        )
        return end_samples >= self.calibration_samples - 1

    def _scored_token_mask(self, n_tokens: int, device: torch.device) -> torch.Tensor:
        """Compatibility alias for earlier experiment code."""

        return self.scored_token_mask(n_tokens, device)

    def _usage_loss(
        self, continuous: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalize soft-assignment entropy below a bounded vocabulary target."""

        if not self.use_quantization or self.usage_weight == 0:
            zero = continuous.new_zeros(())
            return zero, zero
        flat = continuous.reshape(-1, continuous.shape[-1])
        if len(flat) > self.usage_sample_size:
            indices = torch.randperm(len(flat), device=flat.device)[
                : self.usage_sample_size
            ]
            flat = flat[indices]
        codebook = self.tokenizer.quantizer.codebook.detach().to(flat.dtype)
        distances = self.tokenizer.quantizer._distances(flat.float(), codebook.float())
        probabilities = torch.softmax(-distances / self.usage_temperature, dim=-1)
        average = probabilities.mean(dim=0)
        entropy = -(average * average.clamp_min(1e-8).log()).sum()
        target_codes = max(
            2,
            round(self.tokenizer.config.codebook_size * self.usage_target_fraction),
        )
        shortfall = torch.relu(entropy.new_tensor(math.log(target_codes)) - entropy)
        return shortfall.square(), entropy.exp()

    def compute_loss(
        self, batch: Mapping[str, torch.Tensor | str | Sequence[str]]
    ) -> dict[str, torch.Tensor]:
        emg = batch.get("emg")
        if not isinstance(emg, torch.Tensor):
            raise TypeError("batch requires a tensor 'emg' entry")
        sensor = self._sensor_name(batch["sensor"])
        sample_rate = batch.get("sample_rate_hz")
        if isinstance(sample_rate, torch.Tensor):
            sample_rates = torch.unique(sample_rate)
            if sample_rates.numel() != 1:
                raise ValueError("one batch cannot mix sampling rates")
            sample_rate = int(sample_rates.item())
        if (
            sample_rate is not None
            and int(sample_rate) != self.tokenizer.config.sample_rate_hz
        ):
            raise ValueError(
                "EMG must be resampled before batching: received "
                f"{sample_rate} Hz, expected {self.tokenizer.config.sample_rate_hz} Hz"
            )

        normalized, _, scale = calibration_normalize(
            emg,
            calibration_samples=self.calibration_samples,
        )
        targets = self._spectral_targets(normalized).detach()
        model_input = self._augment(normalized)
        tokenizer_kwargs = {}
        descriptors = batch.get("channel_descriptors")
        channel_mask = batch.get("channel_mask")
        if isinstance(descriptors, torch.Tensor):
            tokenizer_kwargs["channel_descriptors"] = descriptors
        if isinstance(channel_mask, torch.Tensor):
            tokenizer_kwargs["channel_mask"] = channel_mask
        output = self.tokenizer(
            model_input,
            sensor=sensor,
            quantize=self.use_quantization,
            include_spatial=False,
            **tokenizer_kwargs,
        )
        predictions = self.spectral_head(output.quantized)
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"spectral prediction shape {tuple(predictions.shape)} does not "
                f"match targets {tuple(targets.shape)}"
            )
        scored = self.scored_token_mask(predictions.shape[-2], emg.device)
        if not torch.any(scored):
            raise ValueError("EMG window is too short to contain a scored token")
        errors = F.smooth_l1_loss(predictions, targets, reduction="none")
        spectral_loss = errors[..., scored, :].mean()
        usage_loss, soft_perplexity = self._usage_loss(output.continuous)
        loss = spectral_loss + output.commitment_loss + self.usage_weight * usage_loss
        return {
            "loss": loss,
            "spectral_loss": spectral_loss,
            "commitment_loss": output.commitment_loss,
            "usage_loss": usage_loss,
            "soft_perplexity": soft_perplexity,
            "perplexity": output.perplexity,
            "active_fraction": output.active_fraction,
            "median_calibration_scale": scale.median(),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def sample_calibration_latents(
    continuous: torch.Tensor,
    total_tokens: int,
    calibration_steps: int,
) -> torch.Tensor:
    """Sample one step's equal share of a multi-window latent reservoir."""
    if continuous.ndim < 2:
        raise ValueError("continuous latents need a final embedding dimension")
    if total_tokens <= 0 or calibration_steps <= 0:
        raise ValueError("token budget and calibration steps must be positive")

    flat = continuous.reshape(-1, continuous.shape[-1]).detach()
    per_step = max(1, (total_tokens + calibration_steps - 1) // calibration_steps)
    if len(flat) <= per_step:
        return flat
    indices = torch.randperm(len(flat), device=flat.device)[:per_step]
    return flat[indices]


def last_checkpoint_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.last{output.suffix}")


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    """Write a checkpoint atomically so an interrupted save stays resumable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_state(
    module: UniversalEMGTokenizerModule,
    optimizer: torch.optim.Optimizer,
    config: TokenizerConfig,
    sources,
    progress: dict[str, object],
    history: list[dict[str, object]],
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "sources": [
            {
                "name": source.name,
                "channels": source.montage.channels,
                "sample_rate_hz": source.source_rate_hz,
                "seen_recordings": len(source.seen_recordings),
                "available_recordings": source.available_recordings,
            }
            for source in sources
        ],
        "progress": dict(progress),
        "history": history,
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "source_seen_recordings": {
            source.name: sorted(source.seen_recordings) for source in sources
        },
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    return payload


def restore_training_state(
    path: Path,
    module: UniversalEMGTokenizerModule,
    optimizer: torch.optim.Optimizer,
    config: TokenizerConfig,
    sources,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("config") != asdict(config):
        raise ValueError("resume checkpoint tokenizer configuration does not match")
    required = {
        "state_dict",
        "optimizer_state_dict",
        "progress",
        "numpy_rng_state",
        "torch_rng_state",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"resume checkpoint is missing {sorted(missing)}")
    module.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng.bit_generator.state = checkpoint["numpy_rng_state"]
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)
    seen = checkpoint.get("source_seen_recordings", {})
    for source in sources:
        source.seen_recordings.update(seen.get(source.name, []))
    progress = dict(checkpoint["progress"])
    history = list(checkpoint.get("history", []))
    print(
        json.dumps(
            {"phase": "resume", "checkpoint": str(path), "progress": progress},
            sort_keys=True,
        ),
        flush=True,
    )
    return progress, history


def _completed_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("progress", {}).get("phase") != "complete":
        raise ValueError(f"{path} is not a completed tokenizer checkpoint")
    return checkpoint


def load_universal_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[UniversalEMGTokenizerModule, dict[str, object]]:
    """Load a completed tokenizer with its spectral head, frozen, in eval mode.

    Pose-aligned checkpoints are accepted through their ``universal_state_dict``
    view, so every checkpoint family exposes the same envelope head.
    """

    checkpoint = _completed_checkpoint(path, device)
    config = TokenizerConfig(**checkpoint["config"])
    state_dict = checkpoint.get("universal_state_dict", checkpoint["state_dict"])
    spectral_weight = state_dict.get("spectral_head.3.weight")
    if spectral_weight is None:
        raise ValueError(f"{path} is missing the universal spectral head")
    module = UniversalEMGTokenizerModule(
        build_default_tokenizer(config),
        spectral_bins=int(spectral_weight.shape[0]),
    ).to(device)
    module.load_state_dict(state_dict)
    module.eval().requires_grad_(False)
    return module, checkpoint
