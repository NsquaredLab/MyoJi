"""Canonical hand landmarks from Meta's pinned emg2pose/UmeTrack model.

emg2pose (CC BY-NC-SA 4.0) is imported only when landmarks are computed, so the
rest of myoji works without it: ``uv sync --group paper`` installs it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from emg2pose.UmeTrack.lib.common.hand import HandModel


WRIST_LANDMARK_INDEX = 5
PALM_ANCHOR_INDICES = (8, 11, 14, 17)
STATIC_LANDMARK_INDICES = frozenset(
    (WRIST_LANDMARK_INDEX, *PALM_ANCHOR_INDICES)
)
DYNAMIC_LANDMARK_INDICES = tuple(
    index for index in range(21) if index not in STATIC_LANDMARK_INDICES
)


@lru_cache(maxsize=1)
def _default_hand_model() -> HandModel:
    import emg2pose
    from emg2pose.UmeTrack.lib.common.hand import HandModel

    model_path = (
        Path(emg2pose.__file__).parent
        / "UmeTrack"
        / "dataset"
        / "generic_hand_model.json"
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            "Meta's generic hand model is missing; install emg2pose with "
            "`uv sync --group paper` (see README.md)"
        )
    values = json.loads(model_path.read_text())
    tensors = {
        name: None if value is None else torch.as_tensor(value)
        for name, value in values.items()
    }
    return HandModel(**tensors)


def _expand_model_field(
    value: torch.Tensor | None,
    leading_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if value is None:
        return None
    value = value.to(device=device, dtype=dtype or value.dtype)
    return value.reshape((1,) * len(leading_shape) + value.shape).expand(
        *leading_shape, *value.shape
    )


def _broadcast_skinning_model(
    leading_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> HandModel:
    from emg2pose.UmeTrack.lib.common.hand import HandModel

    model = _default_hand_model()
    return HandModel(
        joint_rotation_axes=_expand_model_field(
            model.joint_rotation_axes, leading_shape, device, dtype
        ),
        joint_rest_positions=_expand_model_field(
            model.joint_rest_positions, leading_shape, device, dtype
        ),
        joint_frame_index=model.joint_frame_index.to(device),
        joint_parent=model.joint_parent.to(device),
        joint_first_child=model.joint_first_child.to(device),
        joint_next_sibling=model.joint_next_sibling.to(device),
        landmark_rest_positions=_expand_model_field(
            model.landmark_rest_positions, leading_shape, device, dtype
        ),
        landmark_rest_bone_weights=_expand_model_field(
            model.landmark_rest_bone_weights, leading_shape, device, dtype
        ),
        landmark_rest_bone_indices=_expand_model_field(
            model.landmark_rest_bone_indices, leading_shape, device
        ),
        hand_scale=(
            None if model.hand_scale is None else model.hand_scale.to(device, dtype)
        ),
        mesh_vertices=None,
        mesh_triangles=None,
        dense_bone_weights=None,
        joint_limits=None,
    )


def forward_kinematic_landmarks(joint_angles: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 20)`` EMG2Pose angles to ``(..., 21, 3)`` landmarks.

    This is the batching wrapper from Meta's EMG2Pose kinematics path, while
    the actual forward kinematics remain in its bundled UmeTrack implementation.
    The local wrapper avoids an upstream Python 3.13 annotation/import issue.
    """

    from emg2pose.UmeTrack.lib.common.hand_skinning import skin_landmarks

    if joint_angles.ndim < 1 or joint_angles.shape[-1] != 20:
        raise ValueError("EMG2Pose joint angles must have shape (..., 20)")
    leading_shape = tuple(joint_angles.shape[:-1])
    wrist_angles = joint_angles.new_zeros(*leading_shape, 2)
    full_angles = torch.cat((joint_angles, wrist_angles), dim=-1)
    hand_model = _broadcast_skinning_model(
        leading_shape,
        joint_angles.device,
        joint_angles.dtype,
    )
    wrist_transform = torch.eye(
        4,
        device=joint_angles.device,
        dtype=joint_angles.dtype,
    ).reshape((1,) * len(leading_shape) + (4, 4))
    wrist_transform = wrist_transform.expand(*leading_shape, 4, 4)
    return skin_landmarks(hand_model, full_angles, wrist_transform)


def normalized_dynamic_landmarks(joint_angles: torch.Tensor) -> torch.Tensor:
    """Return wrist-relative, palm-scale-normalized moving landmarks.

    The four fixed MCP frames and wrist are excluded from supervision. Palm
    scale is the mean wrist-to-MCP distance, which is pose-stable for the
    generic model and gives a dimensionless target transferable to other hand
    models with different bone lengths.
    """

    landmarks = forward_kinematic_landmarks(joint_angles)
    wrist = landmarks[..., WRIST_LANDMARK_INDEX : WRIST_LANDMARK_INDEX + 1, :]
    relative = landmarks - wrist
    palm_scale = torch.linalg.vector_norm(
        relative[..., PALM_ANCHOR_INDICES, :],
        dim=-1,
    ).mean(dim=-1, keepdim=True)
    normalized = relative / palm_scale.clamp_min(1e-6).unsqueeze(-1)
    return normalized[..., DYNAMIC_LANDMARK_INDICES, :]
