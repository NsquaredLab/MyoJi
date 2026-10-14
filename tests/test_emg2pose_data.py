import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from myoji.data import (
    EMG2POSE_EXPANDED_RECORDING_SPLITS,
    EMG2PoseWindowDataset,
)


def test_expanded_recording_splits_are_disjoint_and_complete() -> None:
    groups = EMG2POSE_EXPANDED_RECORDING_SPLITS
    assert [len(groups[split]) for split in ("train", "val", "test")] == [20, 4, 6]
    assert not set(groups["train"]) & set(groups["val"])
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["val"]) & set(groups["test"])


def write_recording(path: Path, length: int, value: float) -> None:
    dtype = np.dtype(
        [
            ("emg", np.float32, (16,)),
            ("joint_angles", np.float32, (20,)),
            ("time", np.float64),
        ]
    )
    data = np.zeros(length, dtype=dtype)
    data["emg"] = value
    data["joint_angles"] = value
    data["joint_angles"][3] = 0.0
    data["time"] = np.arange(length) / 2_000
    with h5py.File(path, "w") as hdf5:
        group = hdf5.create_group("emg2pose")
        group.create_dataset("timeseries", data=data)


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "filename",
        "session",
        "user",
        "stage",
        "side",
        "split",
        "generalization",
        "moving_hand",
        "held_out_user",
        "held_out_stage",
        "start",
        "end",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "session": "session-1",
                    "moving_hand": "both",
                    "held_out_user": "False",
                    "held_out_stage": "False",
                    "start": "10.0",
                    "end": "20.0",
                    **row,
                }
            )


def test_official_split_and_recording_boundaries_are_preserved(tmp_path: Path) -> None:
    write_recording(tmp_path / "train_a.hdf5", length=12, value=1.0)
    write_recording(tmp_path / "train_b.hdf5", length=8, value=2.0)
    write_recording(tmp_path / "test.hdf5", length=20, value=3.0)
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {
                "filename": "train_a.hdf5",
                "user": "u1",
                "stage": "one",
                "side": "left",
                "split": "train",
                "generalization": "none",
            },
            {
                "filename": "train_b.hdf5",
                "user": "u2",
                "stage": "two",
                "side": "right",
                "split": "train",
                "generalization": "none",
            },
            {
                "filename": "test.hdf5",
                "user": "u3",
                "stage": "three",
                "side": "left",
                "split": "test",
                "generalization": "user",
            },
        ],
    )

    dataset = EMG2PoseWindowDataset(
        tmp_path,
        metadata,
        split="training",
        window_samples=6,
        stride_samples=4,
    )

    assert len(dataset) == 3
    assert [sample.filename for sample in dataset.recordings] == [
        "train_a.hdf5",
        "train_b.hdf5",
    ]
    assert dataset[1]["filename"] == "train_a.hdf5"
    assert dataset[2]["filename"] == "train_b.hdf5"
    assert dataset[2]["start_sample"] == 0
    assert dataset[0]["emg"].shape == (16, 6)
    assert dataset[0]["joint_angles"].shape == (20, 6)
    assert not dataset[0]["valid_pose"][3]
    assert dataset[0]["session"] == "session-1"
    assert dataset[0]["moving_hand"] == "both"
    assert dataset[0]["dataset_split"] == "train"
    assert dataset[0]["metadata_split"] == "train"
    assert dataset[0]["held_out_user"] is False
    assert dataset[0]["held_out_stage"] is False
    assert dataset[0]["recording_start_time"] == 10.0
    assert dataset[0]["recording_end_time"] == 20.0

    mini_override = EMG2PoseWindowDataset(
        tmp_path,
        metadata,
        split="train",
        filenames=("test.hdf5",),
        window_samples=6,
        stride_samples=4,
    )
    assert {recording.filename for recording in mini_override.recordings} == {
        "test.hdf5"
    }


def test_missing_recording_is_reported(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {
                "filename": "missing.hdf5",
                "user": "u1",
                "stage": "one",
                "side": "left",
                "split": "train",
                "generalization": "none",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing.hdf5"):
        EMG2PoseWindowDataset(tmp_path, metadata)


def test_overestimated_length_clamps_to_stored_tail(tmp_path: Path) -> None:
    write_recording(tmp_path / "short.hdf5", length=9, value=1.0)
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {
                "filename": "short.hdf5",
                "user": "u1",
                "stage": "one",
                "side": "left",
                "split": "train",
                "generalization": "none",
            }
        ],
    )
    dataset = EMG2PoseWindowDataset(
        tmp_path,
        metadata,
        split="train",
        window_samples=6,
        stride_samples=4,
    )
    # Simulate a remote length estimate that overruns the stored recording.
    object.__setattr__(dataset.recordings[0], "length", 14)
    dataset._cumulative_windows = np.cumsum(
        [dataset._window_count(r.length) for r in dataset.recordings],
        dtype=np.int64,
    )

    with pytest.warns(UserWarning, match="clamping window to the tail"):
        sample = dataset[2]
    assert sample["emg"].shape == (16, 6)
    assert sample["start_sample"] == 3

    write_recording(tmp_path / "short.hdf5", length=4, value=1.0)
    with pytest.raises(RuntimeError, match="shorter than one"):
        dataset[0]


def test_evaluation_view_caps_windows_and_covers_recordings(tmp_path: Path) -> None:
    from myoji.data import evaluation_view

    for name in ("a.hdf5", "b.hdf5", "c.hdf5"):
        write_recording(tmp_path / name, length=20, value=1.0)
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {
                "filename": name,
                "user": "u1",
                "stage": "one",
                "side": "left",
                "split": "train",
                "generalization": "none",
            }
            for name in ("a.hdf5", "b.hdf5", "c.hdf5")
        ],
    )
    dataset = EMG2PoseWindowDataset(
        tmp_path, metadata, split="train", window_samples=6, stride_samples=2
    )
    full_length = len(dataset)

    assert evaluation_view(dataset, 0) is dataset
    assert evaluation_view(dataset, full_length + 5) is dataset

    view = evaluation_view(dataset, 6)
    assert len(view) == 6
    touched = {dataset.recording_at(i).filename for i in view.indices}
    assert len(touched) >= 2


def test_recording_stratified_indices_spreads_across_recordings() -> None:
    from myoji.data import recording_stratified_indices

    class _RangesOnly:
        def recording_window_ranges(self):
            return ((0, 50), (50, 55), (55, 200))

    indices = recording_stratified_indices(
        _RangesOnly(), max_windows=9, max_recordings=3, seed=0, random_recordings=False
    )
    assert len(indices) <= 9
    assert any(i < 50 for i in indices)
    assert any(50 <= i < 55 for i in indices)
    assert any(i >= 55 for i in indices)
