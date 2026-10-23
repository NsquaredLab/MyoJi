"""Memory-bounded batch sources for label-free tokenizer pretraining.

Each source yields calibrated-length windows at 2 kHz with montage descriptors.
Pretraining alternates between the HD-EMG recordings and the emg2pose training
subset (live windows or a frozen window cache).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import resample_poly

from myoji.data.emg2pose import EMG2PoseWindowDataset
from myoji.learned.montage import ChannelMontage, circular_bipolar_montage, hd_monopolar_montage


def resampled_window(
    emg: np.ndarray,
    source_rate_hz: int,
    target_rate_hz: int,
    target_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample one recording-local window and anti-alias it to the target rate."""

    emg = np.asarray(emg)
    if emg.ndim != 2:
        raise ValueError("EMG must have shape (channel, time)")
    if source_rate_hz <= 0 or target_rate_hz <= 0 or target_samples <= 0:
        raise ValueError("rates and target_samples must be positive")
    source_samples = math.ceil(target_samples * source_rate_hz / target_rate_hz)
    if emg.shape[-1] < source_samples:
        raise ValueError(
            f"recording has {emg.shape[-1]} samples but {source_samples} are needed"
        )
    start = int(rng.integers(0, emg.shape[-1] - source_samples + 1))
    window = np.asarray(
        emg[:, start : start + source_samples], dtype=np.float32
    )
    if source_rate_hz != target_rate_hz:
        divisor = math.gcd(source_rate_hz, target_rate_hz)
        window = resample_poly(
            window,
            up=target_rate_hz // divisor,
            down=source_rate_hz // divisor,
            axis=-1,
        ).astype(np.float32, copy=False)
    if window.shape[-1] < target_samples:
        raise RuntimeError("polyphase resampling returned an undersized window")
    return torch.from_numpy(np.ascontiguousarray(window[:, :target_samples]))


class EMGBatchSource:
    """Small interface used by the round-robin mixed training script."""

    name: str
    sensor: str
    montage: ChannelMontage
    source_rate_hz: int

    def sample_batch(
        self,
        batch_size: int,
        target_rate_hz: int,
        target_samples: int,
        rng: np.random.Generator,
    ) -> dict[str, torch.Tensor | str | int]:
        raise NotImplementedError

    def _batch(
        self,
        arrays: list[torch.Tensor],
        target_rate_hz: int,
    ) -> dict[str, torch.Tensor | str | int]:
        return {
            "emg": torch.stack(arrays),
            "sensor": self.sensor,
            "sample_rate_hz": target_rate_hz,
            "channel_descriptors": self.montage.descriptor(),
            "source": self.name,
        }


class EMG2PoseBatchSource(EMGBatchSource):
    """Random batches from the native split-preserving HDF5 adapter."""

    def __init__(self, dataset: EMG2PoseWindowDataset, name: str = "emg2pose") -> None:
        self.dataset = dataset
        self.name = name
        self.sensor = "wrist"
        self.source_rate_hz = 2_000
        self.montage = circular_bipolar_montage(16)
        self.available_recordings = len(dataset.recordings)
        self.seen_recordings: set[str] = set()

    def sample_batch(
        self,
        batch_size: int,
        target_rate_hz: int,
        target_samples: int,
        rng: np.random.Generator,
    ) -> dict[str, torch.Tensor | str | int]:
        if target_rate_hz != self.source_rate_hz:
            raise ValueError("EMG2Pose adapter currently expects the native 2 kHz rate")
        if target_samples != self.dataset.window_samples:
            raise ValueError("dataset and training window sizes do not match")
        if self.dataset.remote_cache is None:
            indices = rng.integers(0, len(self.dataset), size=batch_size)
        else:
            anchor = int(rng.integers(0, len(self.dataset)))
            lower, upper = self.dataset.recording_window_bounds(anchor)
            if upper - lower >= batch_size:
                indices = rng.choice(
                    np.arange(lower, upper), size=batch_size, replace=False
                )
            else:
                indices = rng.integers(lower, upper, size=batch_size)
        samples = [self.dataset[int(index)] for index in indices]
        self.seen_recordings.update(str(sample["filename"]) for sample in samples)
        arrays = [sample["emg"] for sample in samples]
        return self._batch(arrays, target_rate_hz)


class HDEMGPickleBatchSource(EMGBatchSource):
    """Lazy one-participant-at-a-time access to the 320-channel HD-EMG pickles."""

    def __init__(
        self,
        paths: list[Path] | tuple[Path, ...],
        source_rate_hz: int = 2_048,
        batches_per_participant: int = 1,
        name: str = "hdemg",
    ) -> None:
        if not paths:
            raise ValueError("at least one HD-EMG pickle is required")
        self.paths = tuple(Path(path) for path in paths)
        self.source_rate_hz = source_rate_hz
        self.batches_per_participant = batches_per_participant
        self.name = name
        self.sensor = "hd"
        self.montage = hd_monopolar_montage()
        self._cache_path: Path | None = None
        self._cache: dict[str, np.ndarray] | None = None
        self._cache_uses = batches_per_participant
        self._path_cycle: list[Path] = []
        self.available_recordings = len(self.paths)
        self.seen_recordings: set[str] = set()

    def _next_path(self, rng: np.random.Generator) -> Path:
        if not self._path_cycle:
            order = rng.permutation(len(self.paths))
            self._path_cycle = [self.paths[int(index)] for index in order]
        return self._path_cycle.pop()

    def _recording(self, rng: np.random.Generator) -> np.ndarray:
        if self._cache is None or self._cache_uses >= self.batches_per_participant:
            self._cache_path = self._next_path(rng)
            self.seen_recordings.add(self._cache_path.name)
            loaded = joblib.load(self._cache_path)
            if not isinstance(loaded, dict):
                raise ValueError(f"{self._cache_path} does not contain task recordings")
            self._cache = loaded
            self._cache_uses = 0
        self._cache_uses += 1
        task = list(self._cache)[int(rng.integers(0, len(self._cache)))]
        emg = np.asarray(self._cache[task])
        if emg.ndim != 2 or emg.shape[0] != 320:
            raise ValueError(f"HD-EMG task {task!r} has unexpected shape {emg.shape}")
        return emg

    def sample_batch(
        self,
        batch_size: int,
        target_rate_hz: int,
        target_samples: int,
        rng: np.random.Generator,
    ) -> dict[str, torch.Tensor | str | int]:
        recording = self._recording(rng)
        arrays = [
            resampled_window(
                recording,
                self.source_rate_hz,
                target_rate_hz,
                target_samples,
                rng,
            )
            for _ in range(batch_size)
        ]
        return self._batch(arrays, target_rate_hz)


class FrozenEMG2PoseSource(EMGBatchSource):
    def __init__(self, path: Path, metadata_path: Path):
        with metadata_path.open("rb") as f:
            metadata_hash = hashlib.file_digest(f, "sha256").hexdigest()
        with metadata_path.open() as f:
            metadata = {Path(r["filename"]).stem: r["split"] for r in csv.DictReader(f)}
        with h5py.File(path) as h5:
            manifest = json.loads(h5.attrs["manifest"])
            if (
                manifest["split"] != "train"
                or manifest["metadata_sha256"] != metadata_hash
            ):
                raise ValueError(
                    "training cache must use the supplied official training metadata"
                )
            names = [item["filename"] for item in manifest["items"]]
            if not names or any(
                metadata.get(Path(name).stem) != "train" for name in names
            ):
                raise ValueError("cache contains a recording outside official training")
            self.windows = h5["emg"][:]
        if self.windows.ndim != 3 or self.windows.shape[:2] != (len(names), 16):
            raise ValueError("cache manifest and raw array shapes differ")
        if manifest["sample_rate_hz"] != 2000:
            raise ValueError("native 2 kHz cache required")
        self.filenames = names
        self.name, self.sensor, self.source_rate_hz = "emg2pose", "wrist", 2000
        self.montage = circular_bipolar_montage(16)
        self.available_recordings = len(set(names))
        self.seen_recordings = set()

    def sample_batch(self, batch_size, target_rate_hz, target_samples, rng):
        if (
            target_rate_hz != self.source_rate_hz
            or target_samples > self.windows.shape[-1]
        ):
            raise ValueError("requested training window does not fit native cache")
        arrays = []
        for index in rng.integers(0, len(self.windows), size=batch_size):
            start = int(rng.integers(0, self.windows.shape[-1] - target_samples + 1))
            window = self.windows[index, :, start : start + target_samples].copy()
            if not np.isfinite(window).all():
                raise ValueError("nonfinite training EMG")
            arrays.append(torch.from_numpy(window))
            self.seen_recordings.add(self.filenames[index])
        return self._batch(arrays, target_rate_hz)
