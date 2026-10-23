# MyoJi

Emoji for muscles: discrete EMG tokens that keep 3D hand-pose decoding while sending a fraction
of the bitrate.

This is the code for *EMG Tokenization Preserves Movement Information at a Fraction of the
Bitrate* (Sîmpetru, Olamazadeh, Rohlf, Cnejevici and Del Vecchio). Every electrode becomes one
token every 20 ms (10 bits with 1,024 codes), from one of two causal tokenizers fitted without
pose labels:

- **SPAR** describes each electrode's spectrum and its coupling to the neighbouring electrodes
  (18 values per update), and a k-means codebook turns the description into a token.
- **The learned tokenizer** is a per-channel encoder (160 ms patches, 480 ms causal context),
  pretrained to reconstruct each patch's log-power spectrum. Each output is mapped to the
  nearest of 1,024 code vectors.

The library is `myoji`. Project page: <https://nsquaredlab.github.io/MyoJi/>, built from
[`website/`](website/) by [`.github/workflows/website.yml`](.github/workflows/website.yml).

```bash
uv sync                  # the library
uv run pytest            # no data needed
```

## The 60-second version

```python
import numpy as np
from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize
from myoji.preprocessing import normalize_raw
from myoji.spar import grid_layout, spar_simple_dataset_batched

# emg: (channels, samples) at 2 kHz; 320 channels in the paper's montage
x = normalize_raw(emg)                                   # calibrate on the first 500 ms, clip at ±12
ends = np.arange(2039, x.shape[1], 40)                   # one update every 20 ms
windows = x[:, ends[:, None] - np.arange(319, -1, -1)].transpose(1, 2, 0)   # 160 ms each

layout = grid_layout()        # three 8x8 and two 13x5 grids; pass your own grid maps otherwise
features = spar_simple_dataset_batched(windows, layout=layout, channelwise=True)  # (updates, channels, 18)

d = features.transpose(1, 0, 2)[None]                    # (windows, channels, updates, 18)
mask = np.tile(layout.mask, (1, 6))                      # absent boundary couplings
mean, scale = fit_standardization(d, mask)               # training windows only
book, _ = fit_codebook(standardize(d, mean, scale, mask), vocabulary=1024)
tokens = tokenize(standardize(d, mean, scale, mask), book)   # uint16, (windows, channels, updates)
```

The learned tokenizer needs trained weights, which are not distributed (see below):

```python
from myoji.learned import Tokenizer

tokenizer = Tokenizer("tokenizer.pt")
tokens = tokenizer.encode(emg, sample_rate_hz=2000)["tokens"]   # per channel, from 980 ms on
stream = tokenizer.stream(sample_rate_hz=2000)                    # or chunk by chunk: stream.push(chunk)
```

The pose decoders that read tokens or continuous features are in `myoji.decoders`
(`SparPose`: the proposed spatial CNN + TCN; `CachedPoseDecoder`: geometry-aware channel
attention with an optional causal TCN or Transformer).

## With MyoGestic

[MyoGestic](https://github.com/NsquaredLab/MyoGestic) pipelines are plain functions, so tokens drop
in as features:

```python
from myoji import myogestic as tokens

@pipeline.extract
def extract(windows):
    return tokens.spar_descriptors(windows["emg"], fs=2048, layout=LAYOUT)  # Stream(window_ms=tokens.WINDOW_MS)
```

`tokens.SparCodebook` fits the codebook inside `@pipeline.train`, and `tokens.learned_tokens` gives
learned tokens. `examples/myogestic_tokens.py` is a complete app with simulated EMG and the Virtual
Hand: `uv sync --extra myogestic && uv run python examples/myogestic_tokens.py`.

## Reproducing the paper

`paper/` reruns every number and figure in the article and its supplement:
- `paper/hdemg`, `paper/receivers`, `paper/emg2pose`, `paper/pretraining`: the campaigns and statistics;
- `paper/figures`: the figure panels;
- `paper/embedded`: the Cortex-M55 SPAR firmware;
- `paper/data`: the frozen window selections and Supplementary Data 1–9.

Start at [`paper/README.md`](paper/README.md). The HD-EMG recordings are available from the
authors on reasonable request; emg2pose is public.

## Trained weights

The trained tokenizers and SPAR codebooks used in the paper are not distributed.
`python -m paper.pretraining.train_mixed_sources` pretrains a learned tokenizer, and
`myoji.learned.export_checkpoint(..., calibration_samples=1000)` turns its checkpoint into the
file `Tokenizer` loads. The SPAR campaigns in `paper/` fit the codebooks.

## Licence and citation

AGPL-3.0-or-later; vendored third-party code keeps its licence ([THIRD_PARTY.md](THIRD_PARTY.md)).
If you use myoji, please cite the paper ([CITATION.cff](CITATION.cff)).
