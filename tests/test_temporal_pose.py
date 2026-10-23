import pytest
import torch

from myoji.decoders import CachedPoseDecoder


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ["frame", "tcn", "transformer"])
def test_decoder_is_causal_and_temporal_arms_use_history(kind):
    torch.manual_seed(15)
    model = CachedPoseDecoder(
        8, 14, 3, kind=kind, width=8, slots=2, heads=2, context_frames=16, dropout=0
    ).eval()
    features = torch.randn(2, 3, 16, 8)
    descriptors = torch.randn(3, 14)
    centre, scale = torch.zeros(2, 3, 1), torch.ones(2, 3, 1)
    expected = model(features, descriptors, centre, scale)
    future = features.clone()
    future[:, :, 9:] += torch.randn_like(future[:, :, 9:]) * 20
    actual = model(future, descriptors, centre, scale)
    torch.testing.assert_close(actual[:, :9], expected[:, :9], atol=1e-6, rtol=1e-5)
    past = features.clone()
    past[:, :, :7] += torch.randn_like(past[:, :, :7]) * 20
    changed = model(past, descriptors, centre, scale)
    if kind == "frame":
        torch.testing.assert_close(changed[:, 8:], expected[:, 8:])
    else:
        assert not torch.allclose(changed[:, 8], expected[:, 8])
    assert expected.shape == (2, 16, 48)
    with pytest.raises(ValueError, match="context limit"):
        model(torch.randn(2, 3, 17, 8), descriptors, centre, scale)
