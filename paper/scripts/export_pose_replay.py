"""Export the first fully valid unseen-user pose window, before inspecting predictions."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import paths

from myoji.data.remote_tar import EMG2POSE_ARCHIVE_URL, RemoteTarHDF5Cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "pose.npz"
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(paths.RESULTS / "clean_temporal_v1/full.h5", "r") as cache:
        group = cache["test"]
        selected = next(
            int(i)
            for i in group["unique_indices"][:]
            if group["held_out_user"][i] and group["valid"][i, -50:].all()
        )
        filename = group["filename"].asstr()[selected]
        start = int(group["start_sample"][selected])
        values = {
            key: group[key][selected]
            for key in ("tokens", "targets", "valid", "centre", "scale")
        }
        values["descriptors"] = cache["descriptors"][:]
        values["end_samples"] = np.arange(2039, 6000, 40)
        manifest = {
            "selection": "first unique test window with held_out_user true and all final 50 pose frames valid; no predictions used",
            "cache_index": selected,
            "filename": filename,
            "start_sample": start,
            "user": group["user"].asstr()[selected],
            "cache_id": json.loads(cache.attrs["manifest"])["cache_id"],
        }
    remote = RemoteTarHDF5Cache(
        EMG2POSE_ARCHIVE_URL,
        paths.EMG2POSE / "emg2pose_dataset.tar.index.json",
        args.output / "recording",
        max_cache_bytes=1024**3,
    )
    path = remote.materialize(filename)
    with h5py.File(path, "r") as raw:
        values["raw"] = (
            raw["emg2pose/timeseries"][start : start + 6000]["emg"].astype(np.float32).T
        )
    np.savez(output, **values)
    (args.output / "pose.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
