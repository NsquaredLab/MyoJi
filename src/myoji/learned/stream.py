"""Frozen learned-tokenizer inference: whole recordings or a bounded-history stream."""

from __future__ import annotations

import hashlib
import platform
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from myoji.learned.config import TokenizerConfig
from myoji.learned.encoder import build_default_tokenizer
from myoji.preprocessing import calibration_normalize

FORMAT = "emg-tokenizer-release-v1"
PREPROCESSING = "prefix-sorted-median-mad-f32-clip12-v2"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_checkpoint(source: Path, output: Path, *, calibration_samples: int):
    """Export a trusted research checkpoint without task heads or optimizer state."""
    from myoji.learned.pretrain import load_universal_checkpoint

    if output.exists():
        raise FileExistsError(output)
    if calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive and explicitly supplied")
    module, checkpoint = load_universal_checkpoint(source, torch.device("cpu"))
    if not bool(module.tokenizer.quantizer.initialized):
        raise ValueError("cannot release an uninitialized codebook")
    payload = {
        "format": FORMAT,
        "config": asdict(module.tokenizer.config),
        "state_dict": module.tokenizer.state_dict(),
        "calibration_samples": calibration_samples,
        "preprocessing": PREPROCESSING,
        "source_sha256": sha256(source),
        "pose_supervised": "universal_state_dict" in checkpoint,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return sha256(output)


class Tokenizer:
    """CPU float32 by default. The checkpoint, not the caller, fixes preprocessing."""

    def __init__(self, bundle: Path | str, *, device: str = "cpu"):
        self.path = Path(bundle)
        self.model_version = sha256(self.path)
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT or saved.get("preprocessing") != PREPROCESSING:
            raise ValueError("unsupported release format or preprocessing")
        self.config = TokenizerConfig(**saved["config"])
        self.calibration_samples = int(saved["calibration_samples"])
        if self.calibration_samples <= 0:
            raise ValueError("invalid calibration sample count")
        self.device = torch.device(device)
        with torch.random.fork_rng():
            self.model = build_default_tokenizer(self.config)
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval().requires_grad_(False)
        if not bool(self.model.quantizer.initialized):
            raise ValueError("release codebook is uninitialized")
        self.source_sha256 = saved["source_sha256"]
        self.pose_supervised = bool(saved["pose_supervised"])
        self._lock = threading.Lock()

    @property
    def inference_configuration(self):
        return f"{self.device.type}-float32-torch-{torch.__version__}"

    @property
    def first_token_index(self):
        # Entire raw receptive field must follow the calibration prefix.
        offset = self.calibration_samples + self.config.effective_receptive_samples
        return (
            offset - self.config.receptive_samples + self.config.hop_samples - 1
        ) // self.config.hop_samples

    @property
    def minimum_samples(self):
        return (
            self.config.receptive_samples
            + self.first_token_index * self.config.hop_samples
        )

    def metadata(self):
        return {
            "format": FORMAT,
            "model_version": self.model_version,
            "source_sha256": self.source_sha256,
            "preprocessing_version": PREPROCESSING,
            "inference_configuration": self.inference_configuration,
            "config": asdict(self.config),
            "calibration_samples": self.calibration_samples,
            "minimum_samples": self.minimum_samples,
            "pose_supervised": self.pose_supervised,
            "input_layout": "channel,sample",
            "output_layout": "channel,token",
            "channel_geometry": "separate downstream metadata; does not affect primitive IDs",
            "runtime": {
                "torch": str(torch.__version__),
                "python": platform.python_version(),
                "device": str(self.device),
                "precision": "float32",
                "threads": torch.get_num_threads(),
                "device_name": torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else platform.processor(),
                "machine": platform.machine(),
                "cuda": torch.version.cuda,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        }

    def _raw(self, emg, sample_rate_hz):
        if sample_rate_hz != self.config.sample_rate_hz:
            raise ValueError(
                f"sample_rate_hz must equal {self.config.sample_rate_hz}; no implicit resampling"
            )
        values = np.asarray(emg)
        if values.ndim != 2 or not 1 <= values.shape[0] <= 320:
            raise ValueError(
                "EMG must be a rectangular channel-by-sample array with 1..320 channels"
            )
        if values.dtype.kind not in "fiu":
            raise ValueError("EMG must contain real numeric samples")
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(values, dtype=np.float32, order="C")
        if not np.isfinite(values).all():
            raise ValueError("EMG contains nonfinite or float32-overflow samples")
        return torch.from_numpy(values.copy()).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def _ids(self, normalized):
        # Primitive IDs do not use geometry. Never fabricate physical positions.
        descriptors = torch.zeros(
            normalized.shape[1], self.config.geometry_dim, device=self.device
        )
        return self.model(
            normalized,
            sensor="wrist",
            channel_descriptors=descriptors,
            include_spatial=False,
        ).tokens[0]

    def encode(self, emg, *, sample_rate_hz: int):
        raw = self._raw(emg, sample_rate_hz)
        if raw.shape[-1] < self.minimum_samples:
            raise ValueError(
                f"need at least {self.minimum_samples} samples including calibration and causal warm-up"
            )
        with self._lock, torch.inference_mode():
            normalized, centre, scale = calibration_normalize(
                raw, self.calibration_samples
            )
            ids = self._ids(normalized)[:, self.first_token_index :]
            ends = (
                self.config.receptive_samples
                - 1
                + torch.arange(
                    self.first_token_index, self.first_token_index + ids.shape[-1]
                )
                * self.config.hop_samples
            )
            return {
                "model_version": self.model_version,
                "preprocessing_version": PREPROCESSING,
                "inference_configuration": self.inference_configuration,
                "sample_rate_hz": self.config.sample_rate_hz,
                "token_rate_hz": self.config.token_rate_hz,
                "tokens": ids.cpu().tolist(),
                "end_sample_indices": ends.tolist(),
                "timestamps_seconds": (
                    ends.double() / self.config.sample_rate_hz
                ).tolist(),
                "calibration": {
                    "samples": self.calibration_samples,
                    "centre": centre[0, :, 0].cpu().tolist(),
                    "scale": scale[0, :, 0].cpu().tolist(),
                },
                "warmup_tokens_omitted": self.first_token_index,
                "trailing_samples": int(raw.shape[-1] - int(ends[-1]) - 1),
            }

    def stream(self, *, sample_rate_hz: int):
        if sample_rate_hz != self.config.sample_rate_hz:
            raise ValueError("stream rate must match checkpoint rate")
        return TokenStream(self, sample_rate_hz)


class TokenStream:
    """Bounded-history local stream; calibration is fixed once the prefix arrives."""

    def __init__(self, tokenizer, rate):
        self.tokenizer, self.rate = tokenizer, rate
        self.buffer = None
        self.start = self.received = 0
        self.next_token = 0
        self.centre = self.scale = None

    def push(self, emg):
        t = self.tokenizer
        raw = t._raw(emg, self.rate)
        if raw.shape[-1] == 0:
            raise ValueError("empty stream chunk")
        if self.buffer is not None and raw.shape[1] != self.buffer.shape[1]:
            raise ValueError("channel count cannot change inside a stream")
        with t._lock, torch.inference_mode():
            self.buffer = (
                raw if self.buffer is None else torch.cat([self.buffer, raw], dim=-1)
            )
            self.received += raw.shape[-1]
            if self.centre is None and self.received >= t.calibration_samples:
                _, self.centre, self.scale = calibration_normalize(
                    self.buffer, t.calibration_samples
                )
            available = max(
                0,
                (self.received - t.config.receptive_samples) // t.config.hop_samples
                + 1,
            )
            if self.received < t.minimum_samples or available <= self.next_token:
                return {
                    "tokens": [[] for _ in range(raw.shape[1])],
                    "end_sample_indices": [],
                    "model_version": t.model_version,
                    "preprocessing_version": PREPROCESSING,
                    "inference_configuration": t.inference_configuration,
                }
            c = t.config
            normalized = ((self.buffer - self.centre) / self.scale).clamp(-12.0, 12.0)
            ids = t._ids(normalized)
            first = self.start // c.hop_samples
            begin = max(self.next_token, t.first_token_index)
            end = first + ids.shape[-1]
            selected = ids[:, begin - first :]
            ends = c.receptive_samples - 1 + torch.arange(begin, end) * c.hop_samples
            self.next_token = end
            history = (
                c.effective_receptive_samples - c.receptive_samples
            ) // c.hop_samples
            new_start = max(0, end - history) * c.hop_samples
            self.buffer = self.buffer[..., new_start - self.start :].clone()
            self.start = new_start
            return {
                "tokens": selected.cpu().tolist(),
                "end_sample_indices": ends.tolist(),
                "model_version": t.model_version,
                "preprocessing_version": PREPROCESSING,
                "inference_configuration": t.inference_configuration,
            }
