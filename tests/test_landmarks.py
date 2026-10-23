import pytest
import torch

from myoji.data import (
    DYNAMIC_LANDMARK_INDICES,
    forward_kinematic_landmarks,
    normalized_dynamic_landmarks,
)
from myoji.data.landmarks import PALM_ANCHOR_INDICES, WRIST_LANDMARK_INDEX

pytest.importorskip("emg2pose")


def test_normalized_landmarks_are_wrist_relative_and_unit_palm_scale() -> None:
    angles = torch.zeros(2, 4, 20)
    raw = forward_kinematic_landmarks(angles)
    wrist = raw[..., WRIST_LANDMARK_INDEX : WRIST_LANDMARK_INDEX + 1, :]
    relative = raw - wrist
    scale = torch.linalg.vector_norm(
        relative[..., PALM_ANCHOR_INDICES, :], dim=-1
    ).mean(dim=-1)
    normalized = normalized_dynamic_landmarks(angles)

    assert raw.shape == (2, 4, 21, 3)
    assert normalized.shape == (2, 4, len(DYNAMIC_LANDMARK_INDICES), 3)
    assert torch.isfinite(normalized).all()
    assert torch.all(scale > 0)


def test_finger_angles_move_dynamic_landmarks() -> None:
    neutral = normalized_dynamic_landmarks(torch.zeros(1, 1, 20))
    flexed_angles = torch.zeros(1, 1, 20)
    flexed_angles[..., 5:8] = 0.7
    flexed = normalized_dynamic_landmarks(flexed_angles)

    assert not torch.allclose(neutral, flexed)
