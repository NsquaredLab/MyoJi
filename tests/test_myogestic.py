import os
from pathlib import Path

import numpy as np
import pytest

from myoji import myogestic
from myoji.preprocessing import normalize_raw
from myoji.spar import grid_layout, spar_simple_dataset_batched

LAYOUT = grid_layout([np.arange(8).reshape(2, 4)])


def window(samples, channels=8, seed=0):
    return (np.random.default_rng(seed).standard_normal((channels, samples)) * 40).astype(np.float32)


def test_spar_descriptors_follow_the_paper_window_protocol():
    x = window(2200)
    expected = spar_simple_dataset_batched(normalize_raw(x)[:, -320:].T[None], layout=LAYOUT, channelwise=True)[0]
    np.testing.assert_array_equal(myogestic.spar_descriptors(x, 2000, LAYOUT), expected)
    assert myogestic.spar_descriptors(window(2253), 2048, LAYOUT).shape == (8, 18)


def test_rejects_short_windows_and_other_rates():
    with pytest.raises(ValueError, match="window_ms"):
        myogestic.spar_descriptors(window(1200), 2000, LAYOUT)
    with pytest.raises(ValueError, match="2,000 Hz"):
        myogestic.spar_descriptors(window(2400), 2400, LAYOUT)


def test_spar_codebook_round_trip():
    descriptors = np.stack([myogestic.spar_descriptors(window(1400, seed=s), 2000, LAYOUT) for s in range(12)])
    codebook = myogestic.SparCodebook.fit(descriptors, LAYOUT, vocabulary=16)
    ids = codebook.tokens(descriptors)
    assert ids.shape == (12, 8) and ids.max() < 16
    assert np.array_equal(codebook.tokens(descriptors[3]), ids[3])
    vectors = codebook.vectors(ids)
    assert vectors.shape == (12, 8, 18)
    assert not vectors[:, ~codebook.mask].any()


@pytest.mark.skipif("MYOJI_ASSETS" not in os.environ, reason="needs the authors' trained weights (MYOJI_ASSETS)")
def test_learned_tokens_are_the_newest_encoded_tokens():
    from myoji.learned import Tokenizer

    tokenizer = Tokenizer(Path(os.environ["MYOJI_ASSETS"]) / "learned/tokenizer.pt")
    x = window(2000, channels=16)
    expected = np.asarray(tokenizer.encode(x, sample_rate_hz=2000)["tokens"])[:, -1]
    np.testing.assert_array_equal(myogestic.learned_tokens(x, 2000, tokenizer), expected)
