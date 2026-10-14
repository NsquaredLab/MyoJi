"""Mathematical, boundary, causality and tokenizer-interface checks for SPAR."""
import unittest

import numpy as np
import torch
from scipy.linalg import logm

from myoji import spar
from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize
from myoji.decoders import SparPose


def ring_layout_32():
    """Two-row wraparound ring of the original MyoKit SPAR (removed from myoji.spar; test-only)."""
    centre = np.arange(32)
    row, col = centre // 16, centre % 16
    left, right, vertical = row * 16 + (col - 1) % 16, row * 16 + (col + 1) % 16, (1 - row) * 16 + col
    return spar.Layout(np.stack([centre, left, right, vertical], axis=-1), right,
                       np.where(centre < 16, vertical, -1), (np.arange(32).reshape(2, 16).tolist(),))


class SparChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.rng = np.random.default_rng(42)

    def test_graph_has_unique_edges_and_no_wrap_or_cross_grid(self):
        layout = spar.grid_layout()
        self.assertEqual(layout.channels, 320)
        self.assertEqual(layout.entries_per_band, 876)
        self.assertEqual(spar.compute_spar_simple_feature_dim(), 5256)
        self.assertEqual((layout.right >= 0).sum(), 270)
        self.assertEqual((layout.down >= 0).sum(), 286)
        pairs = [tuple(sorted((c, int(n)))) for axis in (layout.right, layout.down) for c, n in enumerate(axis) if n >= 0]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(c // 64 == n // 64 for c, n in pairs))
        self.assertTrue(all(len(set(row)) == 4 for row in layout.members))
        self.assertEqual(layout.right[7], -1)
        self.assertEqual(layout.down[63], -1)
        self.assertEqual(layout.down[250], 256 - 1)
        self.assertEqual(layout.down[251 + 1], -1)  # last recorded row has no outgoing edge

    def test_permuted_acquisition_map_preserves_physical_pairs(self):
        maps = [np.asarray(g) for g in spar.grid_layout().grids]
        permutation = self.rng.permutation(320)
        remapped = [np.where(g < 0, -1, permutation[np.maximum(g, 0)]) for g in maps]
        first, second = spar.grid_layout(), spar.grid_layout(remapped)
        for c in range(320):
            np.testing.assert_array_equal(second.members[permutation[c]], permutation[first.members[c]])

    def test_real_log_projection_matches_independent_matrix_log(self):
        layout = ring_layout_32()
        z = self.rng.normal(size=(2, 1, 32, 4, 5)) + 1j * self.rng.normal(size=(2, 1, 32, 4, 5))
        spds = (z @ z.conj().swapaxes(-1, -2) + np.eye(4) * .5).astype(np.complex64)
        actual = spar.project_tensor(torch.from_numpy(spds), layout=layout).numpy()
        expected = []
        for batch in spds:
            logs = []
            for matrix in batch[0]:
                regularized = matrix.astype(np.complex128) + 1e-8 * matrix.diagonal().real.max() * np.eye(4)
                logs.append(logm(regularized))
            logs = np.stack(logs)
            expected.append(np.r_[logs[:, 0, 0].real, logs[:, 0, 2].real, logs[:16, 0, 3].real])
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_channelwise_compact_parity_and_single_batch_custom_bands(self):
        layout = spar.grid_layout()
        x = self.rng.normal(size=(3, 320, 320)).astype(np.float32)
        options = dict(layout=layout, frequency_bands=[(60, 100), (100, 160)])
        full = spar.spar_simple_dataset_batched(x, batch_size=2, **options)
        single = spar.compute_spar_simple_single(x[0], **options)
        np.testing.assert_allclose(single, full[0], atol=2e-5, rtol=2e-5)
        channelwise = spar.spar_simple_dataset_batched(x, channelwise=True, **options).reshape(3, 320, 2, 3).transpose(0, 2, 1, 3)
        compact = np.concatenate([channelwise[..., 0], channelwise[..., 1][:, :, layout.right >= 0],
                                  channelwise[..., 2][:, :, layout.down >= 0]], axis=-1).reshape(3, -1)
        np.testing.assert_array_equal(compact, full)
        self.assertEqual(len(single), spar.compute_spar_simple_feature_dim(frequency_bands=options["frequency_bands"]))

    def test_bad_edges_and_silent_input_are_finite(self):
        values =spar.compute_spar_simple_single(np.zeros((320, 320), np.float32), bad_channels=[1], channelwise=True)
        self.assertTrue(np.isfinite(values).all())
        values = values.reshape(320, 6, 3)
        np.testing.assert_array_equal(values[1], 0)
        np.testing.assert_array_equal(values[0, :, 1], 0)  # edge 0->1
        np.testing.assert_array_equal(values[7, :, 1], 0)  # absent boundary edge
        with self.assertRaises(ValueError):
            spar.compute_spar_simple_single(np.full((320, 320), np.nan))

    def test_scaling_hooks_change_spds_consistently(self):
        x = self.rng.normal(size=(1, 320, 32)).astype(np.float32)
        options = dict(layout=ring_layout_32(), frequency_bands=[(60, 100)])
        base = spar.spds_tensor(x, **options)
        scaled = spar.spds_tensor(x, band_channel_scales=np.full((1, 32), 2), **options)
        weighted = spar.spds_tensor(x, channel_weights=np.full(32, 4), **options)
        torch.testing.assert_close(scaled, 4 * base)
        torch.testing.assert_close(weighted, scaled)

    def test_no_future_sample_in_feature_window(self):
        x = self.rng.normal(size=(320, 6000)).astype(np.float32)
        changed = x.copy()
        changed[:, 4040:] += 1000
        endpoint = 4039
        before = spar.compute_spar_simple_single(x[:, endpoint - 319:endpoint + 1].T)
        after = spar.compute_spar_simple_single(changed[:, endpoint - 319:endpoint + 1].T)
        np.testing.assert_array_equal(before, after)

    def test_quantizer_fit_and_lookup_and_missing_entries(self):
        x = self.rng.normal(size=(3, 4, 10, 6)).astype(np.float32)
        mask = np.ones((4, 6), bool)
        mask[0, 1] = False
        x[:, 0, :, 1] = 1e5  # invalid slots must not bias train statistics
        mean, scale = fit_standardization(x, mask)
        self.assertLess(abs(mean[1]), 1)
        z = standardize(x, mean, scale, mask)
        np.testing.assert_array_equal(z[:, 0, :, 1], 0)
        book, receipt = fit_codebook(z, vocabulary=8, max_vectors=100, iterations=10)
        ids = tokenize(z, book)
        self.assertEqual(ids.shape, (3, 4, 10))
        self.assertEqual(ids.dtype, np.uint16)
        manual = ((z[..., None, :] - book) ** 2).sum(-1).argmin(-1)
        np.testing.assert_array_equal(ids, manual)
        self.assertEqual(receipt["fitted_vectors"], 100)
        self.assertFalse(torch.as_tensor(book).requires_grad)

    def test_decoder_ids_and_continuous_lookup_are_identical_and_causal(self):
        torch.manual_seed(19)
        book = torch.randn(32, 18)
        mask = np.tile(spar.grid_layout().mask, (1, 6))
        model = SparPose(book, mask).eval()
        ids = torch.randint(32, (1, 320, 8))
        with torch.no_grad():
            a = model(ids, "tokens")
            b = model(book[ids], "continuous")
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            changed = ids.clone()
            changed[..., 4:] = (changed[..., 4:] + 1) % 32
            torch.testing.assert_close(model(changed, "tokens")[:, :4], a[:, :4], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 689468)
        self.assertNotIn("decoder.codebook", dict(model.named_parameters()))
        with self.assertRaises(TypeError):
            model(ids.float(), "tokens")
        model.train()
        prediction = model(ids, "tokens")
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
