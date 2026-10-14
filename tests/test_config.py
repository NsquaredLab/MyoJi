import pytest

from myoji.learned.config import TokenizerConfig


def test_default_temporal_units() -> None:
    config = TokenizerConfig()

    assert config.hop_samples == 40
    assert config.receptive_samples == 320
    assert config.effective_receptive_samples == 960
    assert config.codebook_update_tokens == 4_096


def test_token_rate_must_divide_sample_rate() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TokenizerConfig(sample_rate_hz=2_000, token_rate_hz=60)
