"""Prepare the tiny metadata and byte-offset index for online EMG2Pose access."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from paper.common import paths

from myoji.data import (
    EMG2POSE_ARCHIVE_URL,
    EMG2POSE_METADATA_URL,
    HTTPRangeClient,
    build_remote_tar_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-url", default=EMG2POSE_ARCHIVE_URL)
    parser.add_argument("--metadata-url", default=EMG2POSE_METADATA_URL)
    parser.add_argument("--output-dir", type=Path, default=paths.EMG2POSE)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    return parser.parse_args()


def download_metadata(url: str, destination: Path) -> None:
    client = HTTPRangeClient(url)
    info = client.head()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        client.download_range(0, info.size, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint frequency must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = args.output_dir / "emg2pose_metadata.csv"
    index = args.output_dir / "emg2pose_dataset.tar.index.json"
    download_metadata(args.metadata_url, metadata)

    last_report = -1

    def progress(members: int, offset: int) -> None:
        nonlocal last_report
        report = members // args.checkpoint_every
        if report != last_report:
            last_report = report
            print(
                f"indexed {members:,} files; archive position {offset / 1024**3:.2f} GiB",
                flush=True,
            )

    build_remote_tar_index(
        args.archive_url,
        index,
        checkpoint_every=args.checkpoint_every,
        progress=progress,
    )
    print(f"metadata: {metadata}")
    print(f"index: {index}")


if __name__ == "__main__":
    main()
