import pytest
import torch

from myoji.learned import (
    TokenizerConfig,
    UniversalEMGTokenizerModule,
    build_default_tokenizer,
    calibration_normalize,
)
from myoji.learned.montage import MeasurementType, unknown_montage


def config() -> TokenizerConfig:
    return TokenizerConfig(
        sample_rate_hz=1_000,
        token_rate_hz=50,
        receptive_field_ms=80,
        embedding_dim=16,
        temporal_layers=1,
        codebook_size=32,
    )


def test_calibration_normalization_uses_prefix_only() -> None:
    original = torch.randn(1, 3, 400)
    changed = original.clone()
    changed[..., 200:] = 1_000 * torch.randn_like(changed[..., 200:])

    normalized, centre, scale = calibration_normalize(original, 200)
    changed_normalized, changed_centre, changed_scale = calibration_normalize(
        changed, 200
    )

    torch.testing.assert_close(changed_centre, centre)
    torch.testing.assert_close(changed_scale, scale)
    torch.testing.assert_close(changed_normalized[..., :200], normalized[..., :200])


def test_universal_loss_trains_unknown_bipolar_channels() -> None:
    tokenizer = build_default_tokenizer(config())
    module = UniversalEMGTokenizerModule(
        tokenizer,
        spectral_bins=16,
        calibration_ms=100,
        use_quantization=False,
    ).train()
    descriptors = unknown_montage(8, MeasurementType.INTRAMUSCULAR_BIPOLAR).descriptor()
    batch = {
        "emg": torch.randn(2, 8, 400),
        "sensor": ["wearable", "wearable"],
        "sample_rate_hz": torch.tensor([1_000, 1_000]),
        "channel_descriptors": descriptors.expand(2, -1, -1),
    }

    metrics = module.compute_loss(batch)
    metrics["loss"].backward()

    assert metrics["loss"].isfinite()
    assert metrics["commitment_loss"] == 0
    assert module.spectral_head[-1].weight.grad is not None
    assert tokenizer.frontend.channel_encoder.encoder[0].weight.grad is not None


def test_universal_loss_rejects_unresampled_data() -> None:
    module = UniversalEMGTokenizerModule(
        build_default_tokenizer(config()),
        spectral_bins=16,
        calibration_ms=100,
    )

    with pytest.raises(ValueError, match="resampled"):
        module.compute_loss(
            {
                "emg": torch.randn(1, 8, 400),
                "sensor": "wearable",
                "sample_rate_hz": 2_000,
            }
        )


def test_sorted_calibration_matches_original_lower_median_values():
    for samples in (200, 201):
        generator = torch.Generator().manual_seed(32)
        raw = torch.randn(2, 4, 500, generator=generator)
        raw[:, 0] = 0  # Ties and the scale floor are part of the contract.
        prefix = raw[..., :samples]
        centre = prefix.median(-1, keepdim=True).values
        scale = (
            1.4826 * (prefix - centre).abs().median(-1, keepdim=True).values
        ).clamp_min(1e-4)
        actual, actual_centre, actual_scale = calibration_normalize(raw, samples)
        assert torch.equal(actual_centre, centre)
        assert torch.equal(actual_scale, scale)
        assert torch.equal(actual, ((raw - centre) / scale).clamp(-12, 12))
