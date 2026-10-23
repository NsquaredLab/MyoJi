"""Native, split-preserving access to EMG2Pose HDF5 recordings."""

from __future__ import annotations

import csv
import warnings
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from myoji.data.remote_tar import RemoteTarHDF5Cache


_MINI_PREFIX = "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording"
EMG2POSE_MINI_SPLITS = {
    "train": tuple(f"{_MINI_PREFIX}-1_{side}" for side in ("left", "right")),
    "val": tuple(f"{_MINI_PREFIX}-2_{side}" for side in ("left", "right")),
    "test": tuple(f"{_MINI_PREFIX}-3_{side}" for side in ("left", "right")),
}
EMG2POSE_EXPANDED_RECORDING_SPLITS = {
    "train": tuple(
        f"{_MINI_PREFIX}-{recording}_{side}"
        for recording in range(1, 11)
        for side in ("left", "right")
    ),
    "val": tuple(
        f"{_MINI_PREFIX}-{recording}_{side}"
        for recording in range(11, 13)
        for side in ("left", "right")
    ),
    "test": tuple(
        f"{_MINI_PREFIX}-{recording}_{side}"
        for recording in range(13, 16)
        for side in ("left", "right")
    ),
}


_SPLIT_ALIASES = {
    "training": "train",
    "validation": "val",
    "testing": "test",
}

_EMG2POSE_SAMPLE_RATE = 2_000
# Each uncompressed official timeseries row contains 16 float32 EMG values,
# 20 float32 joint angles, and one float64 timestamp. The tar member size is a
# cheap upper bound on row count and protects online loading from metadata
# intervals that are occasionally much longer than the stored recording.
_EMG2POSE_TIMESERIES_ITEM_BYTES = (16 + 20) * 4 + 8
_REMOTE_LENGTH_SAFETY_SAMPLES = 256
_CLAMP_WARNED: set[str] = set()


@dataclass(frozen=True)
class EMG2PoseRecording:
    path: Path
    filename: str
    session: str
    user: str
    stage: str
    side: str
    generalization: str
    metadata_split: str
    moving_hand: str
    held_out_user: bool
    held_out_stage: bool
    start_time: float
    end_time: float
    length: int


def _metadata_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid metadata boolean: {value!r}")


def _metadata_float(value: str) -> float:
    return float(value) if value else float("nan")


class EMG2PoseWindowDataset(Dataset):
    """Window EMG2Pose without copying it into a second archive.

    The metadata CSV is authoritative for split membership. Windows are indexed
    within individual HDF5 files, so no sample can cross a recording boundary.
    Invalid inverse-kinematics labels are returned as a mask instead of causing
    the entire EMG window to be discarded.
    """

    group_name = "emg2pose"
    timeseries_name = "timeseries"

    def __init__(
        self,
        data_root: Path | str,
        metadata_csv: Path | str,
        split: str = "train",
        window_samples: int = 10_000,
        stride_samples: int | None = None,
        generalization: str | None = None,
        filenames: Sequence[str] | None = None,
        jitter: bool = False,
        seed: int = 0,
        remote_cache: RemoteTarHDF5Cache | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_csv = Path(metadata_csv)
        self.split = _SPLIT_ALIASES.get(split, split)
        self.window_samples = window_samples
        self.stride_samples = stride_samples or window_samples
        self.generalization = generalization
        self.filenames = set(filenames) if filenames is not None else None
        self.jitter = jitter
        self.remote_cache = remote_cache
        self._rng = np.random.default_rng(seed)

        if window_samples <= 0 or self.stride_samples <= 0:
            raise ValueError("window_samples and stride_samples must be positive")
        if remote_cache is not None:
            self.data_root.mkdir(parents=True, exist_ok=True)
        elif not self.data_root.is_dir():
            raise FileNotFoundError(f"EMG2Pose data directory not found: {data_root}")
        if not self.metadata_csv.is_file():
            raise FileNotFoundError(f"EMG2Pose metadata not found: {metadata_csv}")

        self.recordings = self._load_recordings()
        self._cumulative_windows = np.cumsum(
            [self._window_count(recording.length) for recording in self.recordings],
            dtype=np.int64,
        )
        if not len(self):
            raise ValueError(
                f"no {self.split!r} recordings contain a full "
                f"{self.window_samples}-sample window"
            )

    def _load_recordings(self) -> list[EMG2PoseRecording]:
        recordings = []
        missing = []
        with self.metadata_csv.open(newline="") as file:
            rows = csv.DictReader(file)
            required = {"filename", "split", "user", "stage", "side"}
            missing_columns = required.difference(rows.fieldnames or [])
            if missing_columns:
                names = ", ".join(sorted(missing_columns))
                raise ValueError(f"metadata is missing required columns: {names}")

            for row in rows:
                if self.filenames is None and row["split"] != self.split:
                    continue
                if self.filenames is not None and row["filename"] not in self.filenames:
                    continue
                row_generalization = row.get("generalization", "")
                if (
                    self.generalization is not None
                    and row_generalization != self.generalization
                ):
                    continue

                filename = row["filename"]
                path = self.data_root / filename
                if not path.suffix:
                    path = path.with_suffix(".hdf5")
                if path.is_file():
                    with h5py.File(path, "r") as hdf5:
                        length = len(hdf5[self.group_name][self.timeseries_name])
                elif self.remote_cache is not None and self.remote_cache.has_member(
                    filename
                ):
                    start_time = _metadata_float(row.get("start", ""))
                    end_time = _metadata_float(row.get("end", ""))
                    duration = end_time - start_time
                    if not np.isfinite(duration) or duration <= 0:
                        raise ValueError(
                            f"remote recording {filename!r} needs finite metadata "
                            "start/end times"
                        )
                    metadata_samples = round(duration * _EMG2POSE_SAMPLE_RATE)
                    size_bound_samples = (
                        self.remote_cache.member_size(filename)
                        // _EMG2POSE_TIMESERIES_ITEM_BYTES
                    )
                    length = max(
                        0,
                        min(metadata_samples, size_bound_samples)
                        - _REMOTE_LENGTH_SAFETY_SAMPLES,
                    )
                else:
                    missing.append(path)
                    continue
                recordings.append(
                    EMG2PoseRecording(
                        path=path,
                        filename=filename,
                        session=row.get("session", ""),
                        user=row["user"],
                        stage=row["stage"],
                        side=row["side"],
                        generalization=row_generalization,
                        metadata_split=row["split"],
                        moving_hand=row.get("moving_hand", ""),
                        held_out_user=_metadata_bool(row.get("held_out_user", "")),
                        held_out_stage=_metadata_bool(row.get("held_out_stage", "")),
                        start_time=_metadata_float(row.get("start", "")),
                        end_time=_metadata_float(row.get("end", "")),
                        length=length,
                    )
                )

        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = " ..." if len(missing) > 3 else ""
            raise FileNotFoundError(
                f"{len(missing)} metadata recordings were not found: {preview}{suffix}"
            )
        return recordings

    def _window_count(self, recording_length: int) -> int:
        return max(
            0,
            (recording_length - self.window_samples) // self.stride_samples + 1,
        )

    def __len__(self) -> int:
        if not len(self._cumulative_windows):
            return 0
        return int(self._cumulative_windows[-1])

    def _window_location(self, idx: int) -> tuple[EMG2PoseRecording, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        recording_idx = bisect_right(self._cumulative_windows, idx)
        previous = (
            0 if recording_idx == 0 else int(self._cumulative_windows[recording_idx - 1])
        )
        local_window = idx - previous
        recording = self.recordings[recording_idx]
        start = local_window * self.stride_samples

        if self.jitter:
            remaining = recording.length - (start + self.window_samples)
            if remaining > 0:
                start += int(
                    self._rng.integers(0, min(self.stride_samples, remaining) + 1)
                )
        return recording, start

    def recording_at(self, idx: int) -> EMG2PoseRecording:
        """Return the recording containing global window ``idx``, RNG-free."""

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.recordings[bisect_right(self._cumulative_windows, idx)]

    def recording_window_bounds(self, idx: int) -> tuple[int, int]:
        """Return the global half-open window range for ``idx``'s recording."""

        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        recording_idx = bisect_right(self._cumulative_windows, idx)
        lower = (
            0
            if recording_idx == 0
            else int(self._cumulative_windows[recording_idx - 1])
        )
        return lower, int(self._cumulative_windows[recording_idx])

    def recording_window_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return non-empty global window ranges in recording order."""

        ranges = []
        lower = 0
        for upper_value in self._cumulative_windows:
            upper = int(upper_value)
            if upper > lower:
                ranges.append((lower, upper))
            lower = upper
        return tuple(ranges)

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor | str | int | float | bool]:
        recording, start = self._window_location(idx)
        end = start + self.window_samples
        path = recording.path
        if not path.is_file():
            assert self.remote_cache is not None
            path = self.remote_cache.materialize(recording.filename)
        with h5py.File(path, "r") as hdf5:
            stored = hdf5[self.group_name][self.timeseries_name]
            actual_length = len(stored)
            if end > actual_length:
                # Remote lengths are estimated from metadata intervals and tar
                # member sizes, which can overrun the stored recording. Clamp
                # to the true tail window instead of slicing past the end.
                if actual_length < self.window_samples:
                    raise RuntimeError(
                        f"{recording.filename} stores only {actual_length} "
                        f"samples, shorter than one {self.window_samples}-"
                        "sample window"
                    )
                if recording.filename not in _CLAMP_WARNED:
                    _CLAMP_WARNED.add(recording.filename)
                    warnings.warn(
                        f"{recording.filename}: estimated length "
                        f"{recording.length} overruns stored length "
                        f"{actual_length}; clamping window to the tail",
                        stacklevel=2,
                    )
                start = actual_length - self.window_samples
                end = actual_length
            timeseries = stored[start:end]

        emg = torch.from_numpy(np.asarray(timeseries["emg"], dtype=np.float32).T)
        joint_angles = torch.from_numpy(
            np.asarray(timeseries["joint_angles"], dtype=np.float32).T
        )
        valid_pose = torch.from_numpy(
            ~np.all(np.isclose(timeseries["joint_angles"], 0.0), axis=-1)
        )

        return {
            "emg": emg,
            "joint_angles": joint_angles,
            "valid_pose": valid_pose,
            "sensor": "wrist",
            "filename": recording.filename,
            "session": recording.session,
            "user": recording.user,
            "stage": recording.stage,
            "side": recording.side,
            "generalization": recording.generalization,
            "dataset_split": self.split,
            "metadata_split": recording.metadata_split,
            "moving_hand": recording.moving_hand,
            "held_out_user": recording.held_out_user,
            "held_out_stage": recording.held_out_stage,
            "recording_start_time": recording.start_time,
            "recording_end_time": recording.end_time,
            "start_sample": start,
        }

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)


def evaluation_view(
    dataset: EMG2PoseWindowDataset,
    max_windows: int,
    max_recordings: int = 0,
) -> "EMG2PoseWindowDataset | Subset":
    """Deterministically cap a dataset's windows for evaluation or extraction.

    With a remote cache and a recording cap, windows are drawn evenly across
    recordings so many distinct HDF5 payloads are covered while each is
    downloaded only once, instead of covering few recordings densely.
    """

    if max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    if dataset.remote_cache is not None and max_recordings > 0:
        indices = recording_stratified_indices(
            dataset,
            max_windows,
            max_recordings,
            seed=0,
            random_recordings=False,
        )
    else:
        indices = [
            int(index)
            for index in np.linspace(
                0, len(dataset) - 1, num=max_windows, dtype=np.int64
            )
        ]
    return Subset(dataset, indices)


def recording_stratified_indices(
    dataset: EMG2PoseWindowDataset,
    max_windows: int,
    max_recordings: int,
    *,
    seed: int,
    random_recordings: bool,
) -> list[int]:
    """Cover many recordings while reusing each downloaded HDF5 payload."""

    ranges = dataset.recording_window_ranges()
    count = min(len(ranges), max_recordings, max_windows)
    if count <= 0:
        raise ValueError("window and recording caps must be positive")
    if random_recordings:
        selected_recordings = sorted(
            int(index)
            for index in np.random.default_rng(seed).choice(
                len(ranges), size=count, replace=False
            )
        )
    else:
        selected_recordings = [
            int(index)
            for index in np.linspace(0, len(ranges) - 1, num=count, dtype=np.int64)
        ]
    per_recording = int(np.ceil(max_windows / count))
    selected = []
    for recording_index in selected_recordings:
        lower, upper = ranges[recording_index]
        windows = min(per_recording, upper - lower)
        selected.extend(
            int(index)
            for index in np.linspace(lower, upper - 1, num=windows, dtype=np.int64)
        )
    return sorted(dict.fromkeys(selected))[:max_windows]
