import hashlib
import json

import h5py
import numpy as np
import pytest

from myoji.learned.sources import FrozenEMG2PoseSource


def test_cache_checks_actual_official_split_and_samples_without_labels(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("filename,split\ntraining,train\nvalidation,val\n")
    manifest = {
        "split": "train",
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "sample_rate_hz": 2000,
        "items": [{"filename": "training"}],
    }
    cache = tmp_path / "raw.h5"
    with h5py.File(cache, "w") as h:
        h["emg"] = np.zeros((1, 16, 6000), dtype=np.float32)
        h.attrs["manifest"] = json.dumps(manifest)
    source = FrozenEMG2PoseSource(cache, metadata)
    batch = source.sample_batch(2, 2000, 4000, np.random.default_rng(0))
    assert batch["emg"].shape == (2, 16, 4000)
    assert source.seen_recordings == {"training"}
    manifest["items"] = [{"filename": "validation"}]
    with h5py.File(cache, "a") as h:
        h.attrs["manifest"] = json.dumps(manifest)
    with pytest.raises(ValueError, match="outside official training"):
        FrozenEMG2PoseSource(cache, metadata)
