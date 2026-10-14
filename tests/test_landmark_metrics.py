import pytest
import torch

from myoji.metrics import (
    normalized_landmark_metrics,
    normalized_landmark_pose_velocity_metrics,
    normalized_landmark_velocity_metrics,
    per_window_landmark_mae,
)


def test_normalized_landmark_metrics_use_euclidean_point_error() -> None:
    targets = torch.zeros(1, 2, 6)
    predictions = targets.clone()
    predictions[..., 0] = 3.0
    predictions[..., 1] = 4.0
    valid = torch.ones(1, 2, dtype=torch.bool)

    metrics = normalized_landmark_metrics(predictions, targets, valid)

    assert metrics["landmark_valid_landmarks"] == 4
    assert metrics["landmark_mae_palm_lengths"] == 2.5


def test_normalized_landmark_velocity_metrics_name_units() -> None:
    targets = torch.randn(1, 3, 6)
    metrics = normalized_landmark_velocity_metrics(
        targets,
        targets,
        torch.ones(1, 3, dtype=torch.bool),
    )

    assert metrics["landmark_velocity_mae_palm_lengths_per_second"] == 0.0


def test_derived_landmark_velocity_metrics_use_adjacent_valid_frames() -> None:
    targets = torch.randn(2, 5, 48)
    valid = torch.tensor([[True, True, False, True, True], [True] * 5])

    metrics = normalized_landmark_pose_velocity_metrics(
        targets,
        targets,
        valid,
        token_rate_hz=50,
    )

    assert metrics["landmark_velocity_mae_palm_lengths_per_second"] == 0.0
    assert metrics["landmark_velocity_r2"] == 1.0


def test_per_window_landmark_mae_matches_pooled_metric_and_marks_empty_windows() -> None:
    generator = torch.Generator().manual_seed(0)
    predictions = torch.randn(3, 5, 48, generator=generator)
    targets = torch.randn(3, 5, 48, generator=generator)
    valid = torch.ones(3, 5, dtype=torch.bool)

    per_window = per_window_landmark_mae(predictions, targets, valid)
    pooled = normalized_landmark_metrics(predictions, targets, valid)
    assert per_window.shape == (3,)
    assert per_window.mean().item() == pytest.approx(pooled["landmark_mae_palm_lengths"], rel=1e-6)

    valid[1] = False
    partial = per_window_landmark_mae(predictions, targets, valid)
    assert torch.isnan(partial[1])
    torch.testing.assert_close(partial[[0, 2]], per_window[[0, 2]])
