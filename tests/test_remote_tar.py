import csv
import io
import tarfile
from pathlib import Path

import h5py
import numpy as np

from myoji.data import (
    EMG2PoseWindowDataset,
    HTTPRangeClient,
    RemoteTarHDF5Cache,
    build_remote_tar_index,
)
from myoji.data.remote_tar import RemoteObjectInfo


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
                    **row,
                }
            )


class BytesRangeClient:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def head(self) -> RemoteObjectInfo:
        return RemoteObjectInfo(size=len(self.data), etag="test-etag")

    def read_range(self, offset: int, size: int) -> bytes:
        return self.data[offset : offset + size]

    def download_range(self, offset: int, size: int, destination: io.BytesIO) -> None:
        destination.write(self.read_range(offset, size))


def make_tar(tmp_path: Path) -> tuple[bytes, dict[str, bytes]]:
    members = {}
    for name, value in (("first.hdf5", 1.0), ("second.hdf5", 2.0)):
        path = tmp_path / name
        write_recording(path, length=2_000, value=value)
        members[name] = path.read_bytes()
    archive = tmp_path / "dataset.tar"
    with tarfile.open(archive, "w") as tar:
        for name in members:
            tar.add(tmp_path / name, arcname=f"emg2pose_dataset/{name}")
    return archive.read_bytes(), members


def test_remote_tar_index_records_exact_member_ranges(tmp_path: Path) -> None:
    archive, expected = make_tar(tmp_path)
    index = tmp_path / "index.json"
    build_remote_tar_index(
        "https://example.test/dataset.tar",
        index,
        client=BytesRangeClient(archive),
        checkpoint_every=1,
    )
    cache = RemoteTarHDF5Cache(
        "https://example.test/dataset.tar",
        index,
        tmp_path / "cache",
        max_cache_bytes=max(len(data) for data in expected.values()),
        client=BytesRangeClient(archive),
    )

    first = cache.materialize("first.hdf5")
    assert cache.member_size("first") == len(expected["first.hdf5"])
    assert first.read_bytes() == expected["first.hdf5"]
    second = cache.materialize("second.hdf5")
    assert second.read_bytes() == expected["second.hdf5"]
    assert not first.exists()
    assert cache.usage_bytes() <= cache.max_cache_bytes


def test_https_range_client_configures_http_connect_proxy(monkeypatch) -> None:
    monkeypatch.setenv("https_proxy", "http://proxy.example.test:8080")
    monkeypatch.setenv("no_proxy", "")
    client = HTTPRangeClient("https://archive.example.test/data.tar")

    connection = client._new_connection()

    assert connection.host == "proxy.example.test"
    assert connection.port == 8080
    assert connection._tunnel_host == "archive.example.test"


def test_window_dataset_materializes_remote_recording_on_demand(tmp_path: Path) -> None:
    archive, expected = make_tar(tmp_path)
    index = tmp_path / "index.json"
    client = BytesRangeClient(archive)
    build_remote_tar_index(
        "https://example.test/dataset.tar",
        index,
        client=client,
    )
    cache_dir = tmp_path / "cache"
    cache = RemoteTarHDF5Cache(
        "https://example.test/dataset.tar",
        index,
        cache_dir,
        max_cache_bytes=sum(len(data) for data in expected.values()),
        client=client,
    )
    metadata = tmp_path / "metadata.csv"
    write_metadata(
        metadata,
        [
            {
                "filename": "first.hdf5",
                "user": "u1",
                "stage": "one",
                "side": "left",
                "split": "train",
                "generalization": "none",
                "start": "0.0",
                "end": "1.0",
            }
        ],
    )
    dataset = EMG2PoseWindowDataset(
        cache_dir,
        metadata,
        split="train",
        window_samples=500,
        stride_samples=500,
        remote_cache=cache,
    )

    assert not (cache_dir / "first.hdf5").exists()
    sample = dataset[0]
    assert (cache_dir / "first.hdf5").is_file()
    assert sample["emg"].shape == (16, 500)
    assert np.allclose(sample["emg"].numpy(), 1.0)
    with h5py.File(cache_dir / "first.hdf5", "r") as hdf5:
        assert len(hdf5["emg2pose"]["timeseries"]) == 2_000


def test_remote_window_count_is_capped_by_tar_member_size(tmp_path: Path) -> None:
    short = tmp_path / "short.hdf5"
    usable = tmp_path / "usable.hdf5"
    write_recording(short, length=300, value=1.0)
    write_recording(usable, length=2_000, value=2.0)
    archive_path = tmp_path / "dataset.tar"
    with tarfile.open(archive_path, "w") as tar:
        tar.add(short, arcname="emg2pose_dataset/short.hdf5")
        tar.add(usable, arcname="emg2pose_dataset/usable.hdf5")
    archive = archive_path.read_bytes()
    client = BytesRangeClient(archive)
    index = tmp_path / "index.json"
    build_remote_tar_index(
        "https://example.test/dataset.tar",
        index,
        client=client,
    )
    cache_dir = tmp_path / "cache"
    cache = RemoteTarHDF5Cache(
        "https://example.test/dataset.tar",
        index,
        cache_dir,
        max_cache_bytes=len(short.read_bytes()) + len(usable.read_bytes()),
        client=client,
    )
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
                "start": "0.0",
                "end": "10.0",
            },
            {
                "filename": "usable.hdf5",
                "user": "u2",
                "stage": "two",
                "side": "right",
                "split": "train",
                "generalization": "none",
                "start": "0.0",
                "end": "10.0",
            },
        ],
    )

    dataset = EMG2PoseWindowDataset(
        cache_dir,
        metadata,
        split="train",
        window_samples=500,
        stride_samples=500,
        remote_cache=cache,
    )

    assert len(dataset) == 3
    assert {dataset[index]["filename"] for index in range(len(dataset))} == {
        "usable.hdf5"
    }
    assert not (cache_dir / "short.hdf5").exists()


class _FakeResponse:
    def __init__(self, payload: bytes, cut_after: int | None = None) -> None:
        self._buffer = io.BytesIO(payload)
        self._cut_after = cut_after
        self._served = 0

    def read(self, size: int) -> bytes:
        if self._cut_after is not None and self._served >= self._cut_after:
            return b""
        if self._cut_after is not None:
            size = min(size, self._cut_after - self._served)
        chunk = self._buffer.read(size)
        self._served += len(chunk)
        return chunk

    def close(self) -> None:
        pass


def test_download_range_retries_throttling_and_resumes_short_reads(
    monkeypatch,
) -> None:
    from myoji.data.remote_tar import HTTPRangeClient, RetryableHTTPError

    payload = bytes(range(256)) * 4
    client = HTTPRangeClient("https://archive.example.test/data.tar", attempts=4)
    calls = []

    def scripted(offset: int, size: int):
        calls.append((offset, size))
        if len(calls) == 1:
            raise RetryableHTTPError("HTTP 503")
        if len(calls) == 2:
            return _FakeResponse(payload[offset : offset + size], cut_after=100)
        return _FakeResponse(payload[offset : offset + size])

    monkeypatch.setattr(client, "_range_response", scripted)
    monkeypatch.setattr(client, "_reset_connection", lambda: None)
    monkeypatch.setattr("myoji.data.remote_tar.time.sleep", lambda _: None)

    assert client.read_range(16, 500) == payload[16:516]
    assert calls == [(16, 500), (16, 500), (116, 400)]


def test_download_range_gives_up_after_consecutive_failures(monkeypatch) -> None:
    import pytest

    from myoji.data.remote_tar import HTTPRangeClient, RetryableHTTPError

    client = HTTPRangeClient("https://archive.example.test/data.tar", attempts=3)

    def always_throttled(offset: int, size: int):
        raise RetryableHTTPError("HTTP 503")

    monkeypatch.setattr(client, "_range_response", always_throttled)
    monkeypatch.setattr("myoji.data.remote_tar.time.sleep", lambda _: None)

    with pytest.raises(RetryableHTTPError):
        client.read_range(0, 10)
