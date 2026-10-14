"""Metrics for wrist-relative, bone-scale-normalized hand landmarks."""

from __future__ import annotations

import torch


def _landmark_error_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    prefix: str,
    suffix: str,
) -> dict[str, float | int]:
    if predictions.shape != targets.shape or valid.shape != targets.shape[:2]:
        raise ValueError("landmark prediction, target, and validity shapes do not match")
    if predictions.shape[-1] % 3:
        raise ValueError("flattened landmark dimension must be divisible by three")
    points = predictions.shape[-1] // 3
    predicted = predictions[valid].double().reshape(-1, points, 3)
    expected = targets[valid].double().reshape(-1, points, 3)
    if not len(expected):
        raise ValueError("landmark metrics require valid targets")
    errors = predicted - expected
    distances = torch.linalg.vector_norm(errors, dim=-1)
    flat_errors = errors.flatten(0, 1)
    flat_expected = expected.flatten(0, 1)
    residual_by_coordinate = flat_errors.square().sum(dim=0)
    centred = flat_expected - flat_expected.mean(dim=0, keepdim=True)
    total_by_coordinate = centred.square().sum(dim=0)
    usable = total_by_coordinate > 0
    return {
        f"{prefix}valid_landmarks": int(distances.numel()),
        f"{prefix}mae{suffix}": float(distances.mean()),
        f"{prefix}rmse{suffix}": float(distances.square().mean().sqrt()),
        f"{prefix}r2": (
            float(
                1.0
                - residual_by_coordinate[usable].sum()
                / total_by_coordinate[usable].sum()
            )
            if torch.any(usable)
            else float("nan")
        ),
    }


def normalized_landmark_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float | int]:
    """Euclidean error in units of mean wrist-to-MCP palm distance."""

    return _landmark_error_metrics(
        predictions,
        targets,
        valid,
        prefix="landmark_",
        suffix="_palm_lengths",
    )


def normalized_landmark_velocity_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float | int]:
    """Euclidean landmark-velocity error in palm lengths per second."""

    return _landmark_error_metrics(
        predictions,
        targets,
        valid,
        prefix="landmark_velocity_",
        suffix="_palm_lengths_per_second",
    )


def normalized_landmark_pose_velocity_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    token_rate_hz: int,
) -> dict[str, float | int]:
    """Evaluate velocity obtained by finite-differencing landmark poses."""

    if token_rate_hz <= 0:
        raise ValueError("token rate must be positive")
    velocity_predictions = (
        predictions[:, 1:] - predictions[:, :-1]
    ) * token_rate_hz
    velocity_targets = (targets[:, 1:] - targets[:, :-1]) * token_rate_hz
    velocity_valid = valid[:, 1:] & valid[:, :-1]
    return normalized_landmark_velocity_metrics(
        velocity_predictions,
        velocity_targets,
        velocity_valid,
    )


def per_window_landmark_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean Euclidean landmark error per window over its valid frames.

    Returns shape ``(batch,)`` in the targets' units; a window without any
    valid frame yields NaN so callers can drop it explicitly.
    """

    if predictions.shape != targets.shape or valid.shape != targets.shape[:2]:
        raise ValueError("landmark prediction, target, and validity shapes do not match")
    if predictions.shape[-1] % 3:
        raise ValueError("flattened landmark dimension must be divisible by three")
    points = predictions.shape[-1] // 3
    errors = (predictions - targets).double().reshape(*predictions.shape[:2], points, 3)
    distances = torch.linalg.vector_norm(errors, dim=-1).mean(dim=-1)
    mask = valid.to(distances.dtype)
    counts = mask.sum(dim=-1)
    means = (distances * mask).sum(dim=-1) / counts.clamp_min(1)
    return torch.where(counts > 0, means, torch.full_like(means, float("nan")))
