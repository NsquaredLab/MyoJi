# Project page

Static site for *EMG Tokenization Preserves Movement Information at a Fraction of the Bitrate*, published at
<https://nsquaredlab.github.io/MyoJi/> by the Website workflow.

## Build and preview

The pages are generated. `scripts/build.mjs` writes `index.html`, `tokenizer.html`, `umap.html` and
`static/images/tokenizer.svg` from `index.template.html`, `static/js`, `static/css` and `static/data`; it needs
Node and no packages.

Switzer is not redistributed in this repository (ITF Free Font License). Download it from
[Fontshare](https://www.fontshare.com/fonts/switzer) and place `Switzer-Variable.woff2` and
`Switzer-VariableItalic.woff2` in `static/fonts/`; the workflow does the same. Then, from the repository root:

```sh
node website/scripts/build.mjs
python3 -m http.server 8766 --directory website
```

## Layout

Seven slides. On a desktop each fills the window below the header, page scrolling is off, and the buttons at the
bottom right (or the arrow, page and space keys) move one slide; phones scroll normally.

1. Title and a growing grid of recorded SPAR tokens beside the bit counts for one 20 ms update and the payload of
   raw EMG and tokens against an assumed 1.4 Mbit/s Bluetooth LE link (one electrode slider drives all three).
2. The animated tokenizer: one recorded update through the learned encoder (token 93) or SPAR (token 239).
3. The SPAR patch: each electrode with its right and lower neighbours.
4. Decoded hand pose: participant 6's fist from SPAR tokens and from continuous SPAR.
5. What changes the error: tokenization against calibration, electrode coverage, neighbour information and
   unseen recordings, beside the emg2pose results.
6. Movement through token space: the UMAP of participant 1's token histories.
7. Paper and materials.

The paper links point to the arXiv record, arXiv:2610.32674; the supplement link opens page 12 of the same PDF.

## Data

`static/data` holds recorded values, not simulations: tokens, features and UMAP coordinates of participant 1,
the hand poses of participant 6, and the codebook comparisons shown by the tokenizer figure. The
`scripts/export_*.py` files that wrote them read the authors' result archives, which are not part of this
repository; they document how each file was made.

## Licences

Code: AGPL-3.0-or-later ([LICENSE](../LICENSE)). IBM Plex Mono: SIL Open Font License 1.1
(`static/fonts/OFL-IBMPlexMono.txt`). Switzer: ITF Free Font License, fetched at build time.
