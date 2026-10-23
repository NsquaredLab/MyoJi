import joblib
import numpy as np

from myoji.learned.sources import HDEMGPickleBatchSource, resampled_window


def test_resampled_window_has_exact_target_shape() -> None:
    rng = np.random.default_rng(0)
    emg = rng.normal(size=(8, 2_048)).astype(np.float32)

    window = resampled_window(emg, 2_048, 2_000, 1_000, rng)

    assert window.shape == (8, 1_000)


def test_hdemg_source_loads_one_participant_lazily(tmp_path) -> None:
    path = tmp_path / "Sub1_emg.pkl"
    joblib.dump(
        {"task": np.zeros((320, 2_048), dtype=np.int16)},
        path,
        compress=3,
    )
    source = HDEMGPickleBatchSource([path])

    batch = source.sample_batch(1, 2_000, 1_000, np.random.default_rng(1))

    assert batch["emg"].shape == (1, 320, 1_000)
    assert batch["channel_descriptors"].shape == (320, 14)
    assert batch["sensor"] == "hd"
