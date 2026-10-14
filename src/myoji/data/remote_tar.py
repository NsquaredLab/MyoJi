"""HTTP-range access to files stored in a large, uncompressed tar archive."""

from __future__ import annotations

import fcntl
import http.client
import io
import json
import math
import os
import random
import ssl
import tarfile
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import BinaryIO, Callable, Self
from urllib.parse import SplitResult, urlsplit
from urllib.request import getproxies, proxy_bypass


EMG2POSE_ARCHIVE_URL = (
    "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
)
EMG2POSE_METADATA_URL = (
    "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv"
)
REMOTE_TAR_INDEX_FORMAT = 1


@dataclass(frozen=True)
class RemoteObjectInfo:
    size: int
    etag: str


@dataclass(frozen=True)
class RemoteTarMember:
    archive_path: str
    offset: int
    size: int


class RetryableHTTPError(OSError):
    """A response that indicates throttling or a transient server fault."""


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class HTTPRangeClient:
    """Small persistent HTTP client with strict byte-range validation."""

    def __init__(self, url: str, timeout: float = 60.0, attempts: int = 6) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"HTTP(S) URL required, got {url!r}")
        if timeout <= 0 or attempts <= 0:
            raise ValueError("timeout and attempts must be positive")
        self.url = url
        self.timeout = timeout
        self.attempts = attempts
        self._parsed = parsed
        proxy_url = getproxies().get(parsed.scheme)
        if proxy_url and not proxy_bypass(parsed.hostname):
            if "://" not in proxy_url:
                proxy_url = f"http://{proxy_url}"
            proxy = urlsplit(proxy_url)
            if proxy.scheme != "http" or not proxy.hostname:
                raise ValueError("only an HTTP CONNECT proxy is supported")
            if proxy.username or proxy.password:
                raise ValueError("authenticated HTTP proxies are not supported")
            self._proxy: SplitResult | None = proxy
        else:
            self._proxy = None
        self._connection: http.client.HTTPConnection | None = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_connection"] = None
        return state

    @property
    def _request_target(self) -> str:
        if self._proxy is not None and self._parsed.scheme == "http":
            return self.url
        target = self._parsed.path or "/"
        if self._parsed.query:
            target += f"?{self._parsed.query}"
        return target

    def _new_connection(self) -> http.client.HTTPConnection:
        parsed: SplitResult = self._parsed
        if self._proxy is not None:
            proxy_port = self._proxy.port or 80
            if parsed.scheme == "https":
                connection = http.client.HTTPSConnection(
                    self._proxy.hostname,
                    proxy_port,
                    timeout=self.timeout,
                    context=ssl.create_default_context(),
                )
                connection.set_tunnel(parsed.hostname, parsed.port or 443)
                return connection
            return http.client.HTTPConnection(
                self._proxy.hostname,
                proxy_port,
                timeout=self.timeout,
            )
        if parsed.scheme == "https":
            return http.client.HTTPSConnection(
                parsed.hostname,
                parsed.port,
                timeout=self.timeout,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(
            parsed.hostname,
            parsed.port,
            timeout=self.timeout,
        )

    def _reset_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def _request(
        self, method: str, headers: dict[str, str] | None = None
    ) -> http.client.HTTPResponse:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                if self._connection is None:
                    self._connection = self._new_connection()
                self._connection.request(method, self._request_target, headers=headers or {})
                return self._connection.getresponse()
            except (OSError, http.client.HTTPException) as error:
                last_error = error
                self._reset_connection()
                if attempt + 1 < self.attempts:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def head(self) -> RemoteObjectInfo:
        response = self._request("HEAD")
        try:
            response.read()
            if response.status != 200:
                raise OSError(f"HEAD {self.url} returned HTTP {response.status}")
            if response.getheader("Accept-Ranges", "").lower() != "bytes":
                raise OSError(f"server does not advertise byte ranges for {self.url}")
            length = response.getheader("Content-Length")
            if length is None:
                raise OSError(f"server did not report Content-Length for {self.url}")
            return RemoteObjectInfo(
                size=int(length),
                etag=response.getheader("ETag", "").strip('"'),
            )
        finally:
            response.close()

    def _range_response(self, offset: int, size: int) -> http.client.HTTPResponse:
        if offset < 0 or size <= 0:
            raise ValueError("range offset must be non-negative and size must be positive")
        end = offset + size - 1
        response = self._request("GET", {"Range": f"bytes={offset}-{end}"})
        if response.status != 206:
            response.close()
            self._reset_connection()
            error_type = (
                RetryableHTTPError
                if response.status in _RETRYABLE_STATUSES
                else OSError
            )
            raise error_type(
                f"range GET {self.url} returned HTTP {response.status}, expected 206"
            )
        content_range = response.getheader("Content-Range", "")
        if not content_range.startswith(f"bytes {offset}-{end}/"):
            response.close()
            self._reset_connection()
            raise OSError(f"unexpected Content-Range: {content_range!r}")
        return response

    def read_range(self, offset: int, size: int) -> bytes:
        buffer = io.BytesIO()
        self.download_range(offset, size, buffer)
        return buffer.getvalue()

    def download_range(
        self,
        offset: int,
        size: int,
        destination: BinaryIO,
        chunk_bytes: int = 4 * 1024**2,
    ) -> None:
        # Long runs must survive S3 throttling (429/5xx) and dropped
        # connections: retry with backoff and resume from the received byte.
        # The failure budget counts consecutive zero-progress attempts only.
        received_total = 0
        failures = 0
        while received_total < size:
            try:
                response = self._range_response(
                    offset + received_total, size - received_total
                )
            except RetryableHTTPError:
                failures += 1
                if failures >= self.attempts:
                    raise
                time.sleep(min(30.0, 2.0**failures) * (0.5 + random.random() / 2))
                continue
            got = 0
            needed = size - received_total
            try:
                while got < needed:
                    try:
                        chunk = response.read(min(chunk_bytes, needed - got))
                    except (OSError, http.client.HTTPException):
                        break
                    if not chunk:
                        break
                    destination.write(chunk)
                    got += len(chunk)
            finally:
                response.close()
            received_total += got
            if received_total >= size:
                return
            self._reset_connection()
            if got:
                failures = 0
                continue
            failures += 1
            if failures >= self.attempts:
                raise OSError(
                    f"short range read: expected {size} bytes, "
                    f"received {received_total}"
                )
            time.sleep(min(30.0, 2.0**failures) * (0.5 + random.random() / 2))


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        json.dump(payload, file, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _index_payload(
    archive_url: str,
    object_info: RemoteObjectInfo,
    members: dict[str, RemoteTarMember],
    next_offset: int,
    complete: bool,
) -> dict[str, object]:
    return {
        "format": REMOTE_TAR_INDEX_FORMAT,
        "archive_url": archive_url,
        "archive_size": object_info.size,
        "archive_etag": object_info.etag,
        "next_offset": next_offset,
        "complete": complete,
        "members": {name: asdict(member) for name, member in members.items()},
    }


def build_remote_tar_index(
    archive_url: str,
    output_path: Path | str,
    *,
    checkpoint_every: int = 250,
    progress: Callable[[int, int], None] | None = None,
    client: HTTPRangeClient | None = None,
) -> Path:
    """Index an uncompressed remote tar without downloading member payloads.

    The operation is resumable. Only 512-byte tar headers are transferred, and
    the partial index is checkpointed next to the requested output path.
    """

    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    output = Path(output_path)
    partial = output.with_suffix(output.suffix + ".partial")
    range_client = client or HTTPRangeClient(archive_url)
    object_info = range_client.head()
    members: dict[str, RemoteTarMember] = {}
    offset = 0

    if partial.is_file():
        payload = json.loads(partial.read_text())
        if (
            payload.get("archive_url") != archive_url
            or payload.get("archive_size") != object_info.size
            or payload.get("archive_etag") != object_info.etag
        ):
            raise ValueError("partial tar index belongs to a different archive version")
        offset = int(payload["next_offset"])
        members = {
            name: RemoteTarMember(**entry)
            for name, entry in dict(payload["members"]).items()
        }

    zero_headers = 0
    headers_since_checkpoint = 0
    while offset + tarfile.BLOCKSIZE <= object_info.size:
        header = range_client.read_range(offset, tarfile.BLOCKSIZE)
        if header == tarfile.NUL * tarfile.BLOCKSIZE:
            zero_headers += 1
            offset += tarfile.BLOCKSIZE
            if zero_headers == 2:
                break
            continue
        zero_headers = 0
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        data_offset = offset + tarfile.BLOCKSIZE
        if info.isfile():
            basename = PurePosixPath(info.name).name
            if basename in members:
                raise ValueError(f"duplicate tar member basename: {basename}")
            members[basename] = RemoteTarMember(info.name, data_offset, info.size)
        offset = data_offset + math.ceil(info.size / tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
        headers_since_checkpoint += 1
        if progress is not None:
            progress(len(members), offset)
        if headers_since_checkpoint >= checkpoint_every:
            _write_json_atomic(
                partial,
                _index_payload(archive_url, object_info, members, offset, False),
            )
            headers_since_checkpoint = 0

    payload = _index_payload(archive_url, object_info, members, offset, True)
    _write_json_atomic(output, payload)
    partial.unlink(missing_ok=True)
    return output


class _CacheLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file: BinaryIO | None = None

    def __enter__(self) -> Self:
        self.file = self.path.open("a+b")
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self.file is not None
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


class RemoteTarHDF5Cache:
    """Materialize remote tar members into a hard-bounded local LRU cache."""

    def __init__(
        self,
        archive_url: str,
        index_path: Path | str,
        cache_dir: Path | str,
        max_cache_bytes: int,
        *,
        client: HTTPRangeClient | None = None,
    ) -> None:
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        payload = json.loads(Path(index_path).read_text())
        if payload.get("format") != REMOTE_TAR_INDEX_FORMAT:
            raise ValueError("unsupported remote tar index format")
        if not payload.get("complete"):
            raise ValueError("remote tar index is incomplete")
        if payload.get("archive_url") != archive_url:
            raise ValueError("remote tar index URL does not match archive URL")
        self.archive_url = archive_url
        self.index_path = Path(index_path)
        self.cache_dir = Path(cache_dir)
        self.max_cache_bytes = max_cache_bytes
        self.archive_size = int(payload["archive_size"])
        self.archive_etag = str(payload["archive_etag"])
        self.members = {
            name: RemoteTarMember(**entry)
            for name, entry in dict(payload["members"]).items()
        }
        self.client = client or HTTPRangeClient(archive_url)
        self._thread_clients = threading.local()
        object_info = self.client.head()
        if (
            object_info.size != self.archive_size
            or object_info.etag != self.archive_etag
        ):
            raise ValueError("remote archive no longer matches the tar index")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.cache_dir / ".remote-tar-cache.lock"

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["client"] = HTTPRangeClient(self.archive_url)
        state.pop("_thread_clients", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._thread_clients = threading.local()

    def _thread_client(self) -> HTTPRangeClient:
        if not isinstance(self.client, HTTPRangeClient):
            # An injected test double is not duplicated per thread.
            return self.client
        client = getattr(self._thread_clients, "client", None)
        if client is None:
            client = HTTPRangeClient(self.archive_url)
            self._thread_clients.client = client
        return client

    def _member(self, filename: str) -> tuple[str, RemoteTarMember]:
        name = PurePosixPath(filename).name
        candidates = (name, f"{name}.hdf5") if not Path(name).suffix else (name,)
        for candidate in candidates:
            if candidate in self.members:
                return candidate, self.members[candidate]
        raise FileNotFoundError(f"{filename!r} is not present in the remote tar index")

    def has_member(self, filename: str) -> bool:
        try:
            self._member(filename)
        except FileNotFoundError:
            return False
        return True

    def member_size(self, filename: str) -> int:
        """Return the indexed byte size without materializing the member."""

        _, member = self._member(filename)
        return member.size

    def _cached_files(self) -> list[Path]:
        return [
            path
            for path in self.cache_dir.iterdir()
            if path.is_file() and path.suffix == ".hdf5"
        ]

    def _evict_for(self, required_bytes: int, target: Path) -> None:
        files = self._cached_files()
        total = sum(path.stat().st_size for path in files)
        if target in files:
            total -= target.stat().st_size
            files.remove(target)
        for path in sorted(files, key=lambda item: item.stat().st_mtime_ns):
            if total + required_bytes <= self.max_cache_bytes:
                break
            size = path.stat().st_size
            path.unlink()
            total -= size
        if total + required_bytes > self.max_cache_bytes:
            raise OSError("cache budget could not be satisfied after eviction")

    def materialize(self, filename: str) -> Path:
        name, member = self._member(filename)
        if member.size > self.max_cache_bytes:
            raise ValueError(
                f"{name} is {member.size} bytes, larger than the "
                f"{self.max_cache_bytes}-byte cache limit"
            )
        target = self.cache_dir / name
        # Short locked sections only: the download itself happens unlocked on
        # a per-thread client so concurrent prefetch workers fetch in
        # parallel. A duplicate fetch of the same member is benign -- the
        # atomic replace means the second writer wins with identical bytes.
        with _CacheLock(self._lock_path):
            if target.is_file() and target.stat().st_size == member.size:
                target.touch()
                return target
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=self.cache_dir,
                prefix=f".{name}.",
                suffix=".partial",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                self._thread_client().download_range(member.offset, member.size, file)
                file.flush()
                os.fsync(file.fileno())
            with _CacheLock(self._lock_path):
                if target.is_file() and target.stat().st_size == member.size:
                    temporary.unlink(missing_ok=True)
                else:
                    if target.exists():
                        target.unlink()
                    self._evict_for(member.size, target)
                    os.replace(temporary, target)
                target.touch()
                return target
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    def usage_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._cached_files())
