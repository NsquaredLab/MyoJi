import pytest
import torch

from myoji.learned import sample_calibration_latents


def test_calibration_sample_is_bounded_and_detached() -> None:
    continuous = torch.randn(2, 16, 9, 8, requires_grad=True)

    sample = sample_calibration_latents(
        continuous,
        total_tokens=100,
        calibration_steps=4,
    )

    assert sample.shape == (25, 8)
    assert not sample.requires_grad


def test_calibration_sample_validates_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_calibration_latents(torch.randn(2, 8), 0, 4)
