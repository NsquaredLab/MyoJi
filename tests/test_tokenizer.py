import torch

from myoji.learned import TokenizerConfig, build_default_tokenizer


def small_config() -> TokenizerConfig:
    return TokenizerConfig(
        sample_rate_hz=1_000,
        token_rate_hz=50,
        receptive_field_ms=80,
        embedding_dim=16,
        temporal_layers=2,
        temporal_kernel_size=3,
        codebook_size=32,
    )


def test_both_sensor_geometries_share_output_contract() -> None:
    model = build_default_tokenizer(small_config()).eval()

    wrist = model(torch.randn(2, 16, 400), sensor="wrist")
    hd = model(torch.randn(2, 5, 8, 8, 400), sensor="hd")

    assert wrist.continuous.shape == (2, 16, 17, 16)
    assert hd.continuous.shape == (2, 320, 17, 16)
    assert wrist.tokens.shape == (2, 16, 17)
    assert hd.tokens.shape == (2, 320, 17)
    assert wrist.spatial_tokens.shape == (2, 16, 17, 16)
    assert hd.spatial_tokens.shape == (2, 320, 17, 16)
    assert wrist.pooled.shape == hd.pooled.shape == (2, 8, 17, 16)
    assert wrist.channel_tokens.shape == (2, 16, 17, 16)
    assert hd.channel_tokens.shape == (2, 320, 17, 16)
    assert wrist.tokens.max() < model.config.codebook_size


def test_causal_path_does_not_look_into_future() -> None:
    torch.manual_seed(0)
    model = build_default_tokenizer(small_config()).eval()
    original = torch.randn(1, 16, 400)
    changed = original.clone()
    changed[..., 260:] = torch.randn_like(changed[..., 260:])

    original_output = model(original, sensor="wrist")
    changed_output = model(changed, sensor="wrist")
    unaffected_tokens = (260 - model.config.receptive_samples) // 20 + 1

    torch.testing.assert_close(
        original_output.continuous[:, :, :unaffected_tokens],
        changed_output.continuous[:, :, :unaffected_tokens],
    )
    torch.testing.assert_close(
        original_output.tokens[:, :, :unaffected_tokens],
        changed_output.tokens[:, :, :unaffected_tokens],
    )


def test_joint_channel_and_geometry_permutation_is_invariant() -> None:
    torch.manual_seed(1)
    model = build_default_tokenizer(small_config()).eval()
    emg = torch.randn(1, 16, 200)
    positions = model.sensor_positions("wrist")
    permutation = torch.randperm(16)

    original = model(emg, sensor="wrist")
    permuted = model(
        emg[:, permutation],
        sensor="wrist",
        channel_positions=positions[permutation],
    )

    torch.testing.assert_close(
        permuted.continuous, original.continuous[:, permutation]
    )
    torch.testing.assert_close(permuted.tokens, original.tokens[:, permutation])
    torch.testing.assert_close(
        permuted.spatial_tokens, original.spatial_tokens[:, permutation]
    )
    torch.testing.assert_close(permuted.pooled, original.pooled)


def test_position_is_attached_without_changing_activity_code() -> None:
    torch.manual_seed(2)
    model = build_default_tokenizer(small_config()).eval()
    emg = torch.randn(1, 16, 200)
    positions = model.sensor_positions("wrist")
    shifted_positions = positions.roll(3, dims=0)

    original = model(emg, sensor="wrist", channel_positions=positions)
    shifted = model(emg, sensor="wrist", channel_positions=shifted_positions)

    torch.testing.assert_close(shifted.continuous, original.continuous)
    torch.testing.assert_close(shifted.tokens, original.tokens)
    assert not torch.equal(shifted.spatial_tokens, original.spatial_tokens)
    assert not torch.equal(shifted.pooled, original.pooled)


def test_token_timestamps_are_end_aligned() -> None:
    model = build_default_tokenizer(small_config())

    torch.testing.assert_close(
        model.token_end_times(3),
        torch.tensor([0.079, 0.099, 0.119]),
    )


def test_continuous_stage_does_not_initialize_or_update_vq() -> None:
    model = build_default_tokenizer(small_config()).train()
    before = model.quantizer.codebook.clone()

    output = model(torch.randn(2, 16, 200), sensor="wrist", quantize=False)

    torch.testing.assert_close(output.quantized, output.continuous)
    assert output.tokens.shape == (2, 16, 7)
    assert output.spatial_tokens.shape == output.continuous.shape
    assert output.pooled.shape == (2, 8, 7, 16)
    torch.testing.assert_close(model.quantizer.codebook, before)
    assert not model.quantizer.initialized
    assert torch.all(output.tokens == -1)
    assert output.commitment_loss == 0


def test_content_only_path_skips_downstream_spatial_readout() -> None:
    model = build_default_tokenizer(small_config()).eval()

    output = model(
        torch.randn(1, 16, 200),
        sensor="wrist",
        include_spatial=False,
    )

    assert output.tokens.shape == (1, 16, 7)
    assert output.spatial_tokens is None
    assert output.pooled is None
