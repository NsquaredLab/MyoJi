"""The library reproduces the code that produced the paper's numbers.

``fixtures/parity.npz`` holds outputs of the original research implementation on
small seeded inputs; the inputs are regenerated here from the same seeds. The
learned-tokenizer check needs the authors' trained weights (set MYOJI_ASSETS).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize
from myoji.decoders import CachedPoseDecoder, SparPose
from myoji.learned import TokenizerConfig, UniversalEMGTokenizerModule, build_default_tokenizer
from myoji.preprocessing import calibration_normalize, normalize_raw, resample_2048_to_2000
from myoji.spar import grid_layout, spar_simple_dataset_batched, spds_tensor

REFERENCE = np.load(Path(__file__).parent / "fixtures/parity.npz")


def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_spar_features():
    windows = rng(1).standard_normal((2, 320, 320)).astype(np.float32)
    features = spar_simple_dataset_batched(windows, layout=grid_layout(), channelwise=True, batch_size=16)
    np.testing.assert_allclose(features, REFERENCE["spar_features"], rtol=1e-5, atol=1e-5)
    trace = spds_tensor(windows).diagonal(dim1=-2, dim2=-1).real.sum(-1).numpy()
    np.testing.assert_allclose(trace, REFERENCE["spar_spd_trace"], rtol=1e-5)


def test_spar_codebook_and_tokens():
    features = rng(2).standard_normal((3, 320, 8, 18)).astype(np.float32)
    mask = np.tile(grid_layout().mask, (1, 6))
    mean, scale = fit_standardization(features, mask)
    z = standardize(features, mean, scale, mask)
    book, _ = fit_codebook(z, vocabulary=32, seed=260911, max_vectors=4096, iterations=10)
    np.testing.assert_allclose(mean, REFERENCE["code_mean"], rtol=1e-6)
    np.testing.assert_allclose(scale, REFERENCE["code_scale"], rtol=1e-6)
    np.testing.assert_allclose(book, REFERENCE["code_book"], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(tokenize(z, book), REFERENCE["code_tokens"])


def test_calibration_and_resampling():
    raw = (rng(3).standard_normal((4, 1200)) * 50).astype(np.float32)
    np.testing.assert_array_equal(normalize_raw(raw), REFERENCE["normalize_raw"])
    normalized, _, _ = calibration_normalize(torch.from_numpy(raw)[None], 1000)
    np.testing.assert_array_equal(normalized[0].numpy(), REFERENCE["calibration_normalize"])
    resampled = resample_2048_to_2000(rng(4).standard_normal((4, 2048)).astype(np.float32))
    np.testing.assert_allclose(resampled, REFERENCE["resample"], rtol=1e-5, atol=1e-6)


def test_seeded_tokenizer_initialization():
    torch.manual_seed(0)
    state = build_default_tokenizer(TokenizerConfig()).state_dict()
    assert sorted(state) == REFERENCE["init_keys"].tolist()
    sums = [float(state[key].double().sum()) for key in sorted(state)]
    np.testing.assert_allclose(sums, REFERENCE["init_sums"], rtol=1e-6, atol=1e-6)


def test_pretraining_loss():
    torch.manual_seed(1)
    module = UniversalEMGTokenizerModule(build_default_tokenizer(TokenizerConfig(codebook_size=64)))
    batch = {"emg": torch.from_numpy(rng(5).standard_normal((2, 16, 4000)).astype(np.float32)),
             "sensor": "wrist", "sample_rate_hz": 2000}
    module.train()
    torch.manual_seed(2)
    continuous = [float(v.detach()) for v in module.compute_loss(batch).values()]
    module.use_quantization = True
    torch.manual_seed(3)
    quantized = [float(v.detach()) for v in module.compute_loss(batch).values()]
    np.testing.assert_allclose(continuous, REFERENCE["loss_continuous"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(quantized, REFERENCE["loss_quantized"], rtol=1e-5, atol=1e-6)


def test_decoders():
    mask = np.tile(grid_layout().mask, (1, 6))
    torch.manual_seed(4)
    spar = SparPose(torch.randn(1024, 18), torch.from_numpy(mask)).eval()
    tokens = torch.from_numpy(rng(6).integers(0, 1024, (2, 320, 100)))
    continuous = torch.from_numpy(rng(7).standard_normal((2, 320, 100, 18)).astype(np.float32))
    with torch.no_grad():
        np.testing.assert_allclose(spar(tokens, "tokens").numpy(), REFERENCE["sparpose_tokens"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(spar(continuous, "continuous").numpy(), REFERENCE["sparpose_continuous"],
                                   rtol=1e-4, atol=1e-5)
    for kind in ("frame", "tcn", "transformer"):
        torch.manual_seed(5)
        decoder = CachedPoseDecoder(32, 14, 6, kind=kind, width=32, slots=2, heads=2, context_frames=12,
                                    output_dim=9).eval()
        x = torch.from_numpy(rng(8).standard_normal((2, 6, 12, 32)).astype(np.float32))
        descriptors = torch.from_numpy(rng(9).standard_normal((6, 14)).astype(np.float32))
        centre = torch.from_numpy(rng(10).standard_normal((2, 6, 1)).astype(np.float32))
        scale = torch.from_numpy(rng(11).random((2, 6, 1)).astype(np.float32) + 0.5)
        with torch.no_grad():
            np.testing.assert_allclose(decoder(x, descriptors, centre, scale).numpy(), REFERENCE[f"cached_{kind}"],
                                       rtol=1e-4, atol=1e-5)


ASSETS = Path(os.environ["MYOJI_ASSETS"]) if "MYOJI_ASSETS" in os.environ else None


@pytest.mark.skipif(ASSETS is None, reason="needs the authors' trained weights (MYOJI_ASSETS)")
def test_released_learned_tokenizer():
    from myoji.learned import Tokenizer

    tokenizer = Tokenizer(ASSETS / "learned/tokenizer.pt")
    emg = (rng(12).standard_normal((16, 4000)) * 30).astype(np.float32)
    np.testing.assert_array_equal(tokenizer.encode(emg, sample_rate_hz=2000)["tokens"], REFERENCE["release_tokens"])
    hd = (rng(13).standard_normal((320, 3000)) * 30).astype(np.float32)
    np.testing.assert_array_equal(tokenizer.encode(hd, sample_rate_hz=2000)["tokens"], REFERENCE["release_tokens_hd"])
