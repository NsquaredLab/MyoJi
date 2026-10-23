import torch

from myoji.learned.quantizer import EMAVectorQuantizer


def test_quantizer_shapes_and_encoder_gradient() -> None:
    quantizer = EMAVectorQuantizer(codebook_size=16, embedding_dim=8)
    inputs = torch.randn(2, 7, 8, requires_grad=True)

    output = quantizer(inputs)
    (output.quantized.square().mean() + output.commitment_loss).backward()

    assert output.quantized.shape == inputs.shape
    assert output.indices.shape == (2, 7)
    assert inputs.grad is not None
    assert 1 <= output.perplexity <= 14.001
    assert 0 < output.active_fraction <= 1


def test_eval_does_not_update_codebook() -> None:
    quantizer = EMAVectorQuantizer(codebook_size=8, embedding_dim=4).eval()
    before = quantizer.codebook.clone()

    quantizer(torch.randn(3, 5, 4))

    torch.testing.assert_close(quantizer.codebook, before)


def test_training_output_uses_pre_update_code_vectors() -> None:
    quantizer = EMAVectorQuantizer(
        codebook_size=8,
        embedding_dim=4,
        kmeans_init=False,
    )
    before = quantizer.codebook.clone()

    output = quantizer(torch.randn(2, 3, 4))

    expected = before[output.indices].reshape_as(output.quantized)
    torch.testing.assert_close(output.quantized, expected)


def test_first_training_batch_initializes_codebook_from_data() -> None:
    torch.manual_seed(4)
    quantizer = EMAVectorQuantizer(
        codebook_size=16,
        embedding_dim=4,
        kmeans_iterations=3,
    )
    inputs = torch.randn(4, 12, 4)

    output = quantizer(inputs)

    assert quantizer.initialized
    assert output.active_fraction >= 0.75
    assert output.perplexity > 10


def test_codebook_can_be_calibrated_before_quantized_training() -> None:
    quantizer = EMAVectorQuantizer(
        codebook_size=8,
        embedding_dim=4,
        kmeans_iterations=2,
    )
    calibration = torch.randn(40, 4)

    quantizer.initialize_codebook(calibration)
    output = quantizer(calibration)

    assert quantizer.initialized
    assert output.active_fraction == 1


def test_ema_update_can_cap_a_high_channel_count_batch() -> None:
    torch.manual_seed(5)
    quantizer = EMAVectorQuantizer(
        codebook_size=2,
        embedding_dim=1,
        decay=0.0,
        kmeans_init=False,
        dead_code_threshold=0.0,
        update_sample_size=10,
    )
    quantizer.codebook.copy_(torch.tensor([[-1.0], [1.0]]))
    inputs = torch.cat((torch.full((100, 1), -1.0), torch.full((100, 1), 1.0)))

    quantizer(inputs)

    assert quantizer.ema_cluster_size.sum() == 10
