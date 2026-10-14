"""CPU inference contracts; synthetic inputs are not a trained-model audit."""

from dataclasses import asdict

import pytest
import torch

from myoji.learned import (
    TokenizerConfig,
    build_default_tokenizer,
    calibration_normalize,
)


@pytest.fixture
def inference_case():
    with torch.random.fork_rng():
        torch.manual_seed(260905)
        config = TokenizerConfig(
            sample_rate_hz=1_000,
            token_rate_hz=50,
            receptive_field_ms=80,
            embedding_dim=16,
            temporal_layers=2,
            temporal_kernel_size=3,
            codebook_size=32,
        )
        model = build_default_tokenizer(config).eval()
        raw = torch.randn(3, 16, 1_200)
        normalized, centre, scale = calibration_normalize(raw, 200)
        with torch.inference_mode():
            features = model(
                normalized, sensor="wrist", quantize=False, include_spatial=False
            ).continuous
            model.quantizer.initialize_codebook(features.reshape(-1, 16))
        return model, raw, normalized, centre, scale


def token_ids(model, emg):
    with torch.inference_mode():
        return model(emg, sensor="wrist", include_spatial=False).tokens.clone()


def test_repeated_inference_preserves_ids_rng_and_model_state(inference_case):
    model, _, normalized, _, _ = inference_case
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    reference = token_ids(model, normalized)
    assert reference.unique().numel() > 1  # A collapsed codebook is insufficient.
    for _ in range(5):
        assert torch.equal(token_ids(model, normalized), reference)
    assert torch.equal(torch.get_rng_state(), rng_before)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    with torch.random.fork_rng():
        torch.manual_seed(999)
        assert torch.equal(token_ids(model, normalized), reference)


def test_checkpoint_reload_preserves_exact_ids(inference_case, tmp_path):
    model, _, normalized, _, _ = inference_case
    checkpoint = tmp_path / "tokenizer.pt"
    reference = token_ids(model, normalized)
    torch.save(
        {"config": asdict(model.config), "state_dict": model.state_dict()}, checkpoint
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with torch.random.fork_rng():
        torch.manual_seed(321)
        restored = build_default_tokenizer(TokenizerConfig(**saved["config"])).eval()
    restored.load_state_dict(saved["state_dict"])
    assert torch.equal(token_ids(restored, normalized), reference)


def test_request_batching_preserves_exact_ids(inference_case):
    model, raw, normalized, _, _ = inference_case
    reference = token_ids(model, normalized)
    for index in range(raw.shape[0]):
        single, _, _ = calibration_normalize(raw[index : index + 1], 200)
        assert torch.equal(token_ids(model, single), reference[index : index + 1])
    permutation = torch.tensor([2, 0, 1])
    reordered, _, _ = calibration_normalize(raw[permutation], 200)
    assert torch.equal(token_ids(model, reordered), reference[permutation])


def test_overlap_chunks_match_whole_recording_with_fixed_calibration(inference_case):
    model, raw, normalized, centre, scale = inference_case
    reference = token_ids(model, normalized)
    config = model.config
    context_tokens = (
        config.effective_receptive_samples - config.receptive_samples
    ) // config.hop_samples
    next_token = 0
    pieces = []
    # Irregular incoming chunk lengths, all after the calibration prefix arrives.
    # Replay enough past input on the ORIGINAL token grid; never recalibrate.
    for received_samples in [231, 487, 762, 1_200]:
        first_token = max(0, next_token - context_tokens)
        start_sample = first_token * config.hop_samples
        chunk = ((raw[..., start_sample:received_samples] - centre) / scale).clamp(
            -12.0, 12.0
        )
        ids = token_ids(model, chunk)
        pieces.append(ids[..., next_token - first_token :])
        next_token = first_token + ids.shape[-1]
    assert torch.equal(torch.cat(pieces, dim=-1), reference)
