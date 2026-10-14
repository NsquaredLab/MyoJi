# Portable C tokenizers

Plain C11, FP32 ports of the paper's two frozen tokenizers, from raw EMG
samples to 10-bit token IDs. They contain no pose decoder, no fitting and no
hardware code. Both STM32N6 benchmarks link them
([`../stm32n6_tokenizer`](../stm32n6_tokenizer),
[`../stm32n6_optimized`](../stm32n6_optimized)), and they are the numerical
reference those ports are checked against.

| File | Contents |
|---|---|
| `common.h`, `common.c` | Shapes (320 channels, 320-sample window, 40-sample hop, 1,024 codes). Prefix calibration: lower median and 1.4826 × lower MAD of the first 1,000 samples, scale floor 1e-4. Normalization clipped to ±12. Exhaustive squared-distance code search with first-index tie breaking. |
| `learned.c` | Learned channel encoder: 320-sample patch projection to 32, GELU, projection to 256, four causal residual blocks (LayerNorm, depthwise convolution over a rolling four-step history, 256 → 1,024 → 256 pointwise layers with GELU), final LayerNorm. Codebook 1,024 × 256. |
| `spar.c` | SPAR: seven 128-sample Hann-windowed radix-2 FFT segments per window, six bands weighted by their power over all channels, a local 4 × 4 complex-Hermitian covariance per electrode and band, regularization, Jacobi matrix logarithm, three retained entries per band (18 features), masking and the frozen standardization. Participant 1's 1,024 × 18 codebook. |

The input is finite, channel-major float32 EMG at 2 kHz. The recordings'
2,048 → 2,000 Hz resampling happens before this interface. A token is emitted
per channel every 40 samples (20 ms) from the last 320 samples (160 ms).
The ports are compiled with `-O3 -std=c11 -fno-fast-math -ffp-contract=off`,
so there is no fast-math and no fused multiply-add contraction.

## Frozen fixtures

The model constants, a recorded test window and this port's outputs on it are
generated rather than committed, because they derive from participant
recordings. Generate them from the repository root:

```sh
.venv/bin/python paper/scripts/raw_tokenizer_benchmark/prepare.py
.venv/bin/python paper/scripts/raw_tokenizer_benchmark/host_check.py
```

| Input | Location |
|---|---|
| Learned tokenizer checkpoint (SHA-256 pinned in `prepare.py`) | `$MYOJI_ASSETS/learned/tokenizer.pt` (`paths.LEARNED_TOKENIZER`) |
| Participant 1's SPAR codebook, standardization and montage mask | `$MYOJI_ASSETS/spar/codebooks-1024.npz`, entries `subject_01/{codebook,feature_mean,feature_scale,mask}` |
| Recorded raw HD-EMG window and its saved learned token IDs (training split, task 2) | `paper/output/token-explorer/full-channel-index.json` and the raw and token files it lists |

The token-explorer files contain participant EMG. Generate them from your own
copy of the recordings with `paper/scripts/export_token_explorer_context.py`.
That script reads three inputs:

- `$MYOJI_HDEMG/Sub1_emg.pkl`;
- `--root`, the folder holding the token cache the explorer was built from
  (`movement-inputs.h5`);
- the explorer it extends (`manifest.json`, `data-audit.json` and `tokens/` in
  `paper/output/token-explorer/`).

These files stay in the gitignored `paper/output/` and must not be
redistributed.

`prepare.py` writes the following to `paper/results/raw_tokenizers_cortex_m4_v1/`:

- `learned_weights.f32`: the packed encoder weights and codebook (9,567,360 bytes).
- `weight_offsets.h`: offsets into those weights.
- `constants.c`: the SPAR codebook, mean and scale, plus the Hann window, FFT twiddles and montage.
- `raw_320x2120.f32`: 320 channels × 2,120 samples of raw EMG.
- `reference.npz`: an independent PyTorch reference. It holds the calibration, the normalized samples, and the features and IDs at endpoints 2039, 2079 and 2119.
- `fixture_receipt.json`: the hashes of every input and output.

`prepare.py` asserts that its PyTorch reference reproduces all 960 saved
learned token IDs.

`host_check.py` compiles these sources with the host C compiler (`clang` or
`cc`; no CMSIS) and runs them on the fixture.

- **Asserted:** the calibration and all 678,400 normalized samples equal the
  PyTorch reference.
- **Compared:** the matrix logarithm, against NumPy on 100 random Hermitian
  matrices with condition numbers up to 1e6.
- **Recorded:** the SPAR and learned feature deviations and token mismatches,
  written to `host_receipt.json`.

It writes `host_outputs.npz`, which the STM32N6 host verifications check
against. It holds this port's `calibration`, `spar_features`, `spar_ids`,
`learned_features` and `learned_ids` at the three endpoints and 320 channels.

On this fixture, the C port reproduces all 960 SPAR and 960 learned token IDs.
The largest feature deviations are 0.023 for SPAR and 2.3e-5 for the learned
encoder.
