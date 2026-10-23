from dataclasses import asdict

import numpy as np
import pytest
import torch

from myoji.learned import (
    TokenizerConfig,
    UniversalEMGTokenizerModule,
    build_default_tokenizer,
    calibration_normalize,
)
from myoji.learned.stream import FORMAT, PREPROCESSING, Tokenizer, export_checkpoint


@pytest.fixture
def released(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(260905)
        config = TokenizerConfig(
            sample_rate_hz=1000,
            token_rate_hz=50,
            receptive_field_ms=80,
            embedding_dim=16,
            temporal_layers=2,
            temporal_kernel_size=3,
            codebook_size=32,
        )
        module = UniversalEMGTokenizerModule(
            build_default_tokenizer(config), spectral_bins=8, calibration_ms=200
        ).eval()
        raw = torch.randn(1, 4, 1200)
        normalized, _, _ = calibration_normalize(raw, 200)
        with torch.inference_mode():
            output = module.tokenizer(
                normalized,
                sensor="wrist",
                channel_descriptors=torch.zeros(4, 14),
                quantize=False,
            )
            module.tokenizer.quantizer.initialize_codebook(
                output.continuous.reshape(-1, 16)
            )
        source = tmp_path / "research.pt"
        torch.save(
            {
                "config": asdict(config),
                "state_dict": module.state_dict(),
                "progress": {"phase": "complete"},
            },
            source,
        )
        bundle = tmp_path / "release.pt"
        export_checkpoint(source, bundle, calibration_samples=200)
    return bundle, raw[0].numpy()


def test_export_and_fresh_process_match(released):
    bundle, raw = released
    tokenizer = Tokenizer(bundle)
    expected = tokenizer.encode(raw, sample_rate_hz=1000)
    assert expected["end_sample_indices"][0] == tokenizer.minimum_samples - 1
    assert len(set(np.asarray(expected["tokens"]).ravel())) > 1
    saved = torch.load(bundle, weights_only=True)
    assert saved["format"] == FORMAT and saved["preprocessing"] == PREPROCESSING
    assert not any("spectral_head" in k for k in saved["state_dict"])


def test_stream_matches_stateless_and_is_bounded(released):
    bundle, raw = released
    t = Tokenizer(bundle)
    reference = t.encode(raw, sample_rate_hz=1000)
    stream = t.stream(sample_rate_hz=1000)
    chunks, ends, previous = [], [], 0
    for end in [7, 101, 211, 359, 360, 361, 487, 761, 1200]:
        result = stream.push(raw[:, previous:end])
        chunks.append(np.asarray(result["tokens"], dtype=int))
        ends.extend(result["end_sample_indices"])
        previous = end
        assert stream.buffer.shape[-1] <= max(
            t.minimum_samples, t.config.effective_receptive_samples
        )
    assert np.array_equal(np.concatenate(chunks, axis=1), reference["tokens"])
    assert ends == reference["end_sample_indices"]


@pytest.mark.parametrize(
    "case", ["rate", "short", "nan", "ragged", "complex", "overflow"]
)
def test_invalid_inputs_are_rejected(released, case):
    bundle, raw = released
    rate = 1000
    if case == "rate":
        rate = 2000
    if case == "short":
        raw = raw[:, :100]
    if case == "nan":
        raw[0, 0] = np.nan
    if case == "ragged":
        raw = [[1], [2, 3]]
    if case == "complex":
        raw = raw.astype(complex)
    if case == "overflow":
        raw = raw.astype(np.float64) + 1e100
    with pytest.raises(ValueError):
        Tokenizer(bundle).encode(raw, sample_rate_hz=rate)


def test_stream_waits_for_next_patch_without_temporal_layers(released):
    bundle, raw = released
    saved = torch.load(bundle, weights_only=True)
    saved["config"]["temporal_layers"] = 0
    saved["state_dict"] = {
        k: v
        for k, v in saved["state_dict"].items()
        if not k.startswith("temporal_encoder.")
    }
    torch.save(saved, bundle)
    model = Tokenizer(bundle)
    reference = model.encode(raw, sample_rate_hz=1000)
    stream, pieces, previous = model.stream(sample_rate_hz=1000), [], 0
    for end in [200, 280, 281, 282, 299, 300, 1200]:
        pieces.append(
            np.asarray(stream.push(raw[:, previous:end])["tokens"], dtype=int)
        )
        previous = end
    assert np.array_equal(np.concatenate(pieces, axis=1), reference["tokens"])
