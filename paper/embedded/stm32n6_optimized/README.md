# STM32N6 tokenizer benchmark, optimized implementation

This port runs the same benchmark as
[`../stm32n6_tokenizer`](../stm32n6_tokenizer) on the same board. The
differences are faster kernels (`fast_kernels.c`) and a build-time rewrite of
the SPAR runtime in `build.py`. It produces Fig. 7 of the paper: the number of
SPAR channels tokenized within the 20 ms update rises from 23 in the original
port to 63 in this one. The setup is identical:

- **Hardware:** NUCLEO-N657X0-Q, Cortex-M55 at 600 MHz, FP32, caches on, no NPU.
- **Signal:** synthetic 2 kHz input, 160 ms windows, 20 ms updates.
- **Each update:** window shift, normalization, frontend, exhaustive 1,024-code
  search and 10-bit packing.
- **Protocol:** 16 warm-up updates, then 16 measured updates. The token IDs
  and checksums must match the host build.

## Two builds

| Build | SPAR transform and Jacobi `hypotf` | Relation to the released port |
|---|---|---|
| default | CMSIS-DSP Helium real FFT (`arm_rfft_fast_f32`) and `sqrtf(x*x+y*y)` | Not bit-exact. This is the firmware in Fig. 7. |
| `STM32N6_EXACT=1` | The frozen radix-2 FFT and libm `hypotf` (on the board, newlib's `__ieee754_hypotf` core, which returns the same value) | Bit-exact with the original implementation |

The remaining changes give identical results in both builds:

- **Codebook search:** the codebook is rearranged once into 16-code groups in
  RAM, and Helium computes 16 distances at a time. Each distance keeps the
  scalar FP32 reduction order and the earliest-index tie rule. Code groups form
  the outer loop over chunks of 64 channels, so a group stays in cache.
- **Learned encoder:** up to 16 electrodes pass through each dense layer as
  independent lanes, so each weight is loaded once per batch. There is no
  batching across time. A single channel uses the scalar encoder.
- **Radix-2 FFT (bit-exact build):** a bit-reversal table, and four butterflies
  per Helium instruction where a stage has at least four per block. The final
  stage stops at bin 32, because nothing reads the higher bins.
- **SPAR features:** each electrode's member spectra are read and band-weighted
  once for all six bands. Covariances accumulate with the member index
  innermost, one accumulator per lane in the reference order.
- **Jacobi solver:** it stops after a sweep without rotation, and it skips the
  never-read eigenvector row 1.

No build uses fast-math, fused multiply-add contraction (`-ffp-contract=off`),
approximate activations, pruning, quantization or the NPU.

The learned weights are read from external NOR (50 MHz OPI DTR). The active
route's codebook is copied to internal RAM before timing starts; this one-off
setup is measured separately as `setup_cycles`.

The default build compiles CMSIS-DSP from the pinned STM32CubeN6 checkout.
`vendor/cmsis_patch/arm_vec_fft.h` is a modified copy of one CMSIS-DSP header
(Apache-2.0). It adds explicit MVE casts on 14 lines of bit-reversal index code
so that GCC accepts them, and it changes no floating-point operation. The board
build includes it ahead of the upstream header; the host build uses CMSIS-DSP's
scalar path, which does not need it. `source_receipt.json` records the
upstream and modified hashes and the changed lines.

## Results

Medians over 16 measured updates on one board:

| Configuration | Original implementation | Bit-exact build | Default build |
|---|---:|---:|---:|
| SPAR, largest channel count meeting 20 ms with host agreement | 23 | 46 | **63** |
| SPAR, 320 channels | 276.7 ms | 135.7 ms | 102.3 ms |
| Learned encoder, 1 channel | 104.7 ms | – | 95.7 ms |
| Learned encoder, 16 channels | 1,703.6 ms | – | 294.5 ms |
| Learned encoder, 112 channels | 11,918.8 ms | – | 2,053.7 ms |

The bit-exact build was measured on the SPAR route only, because the learned
route does not use the replaced transform.

At 63 channels the slowest update of the default build took 19.95 ms;
64 channels reached 20.28 ms. The bit-exact build meets the timing at
47 channels, but its tokens differ from the host there, so its capacity is 46.

These capacities are observed over 16 updates with synthetic input, not
guaranteed bounds. Because the boundary sits within 0.3 % of the interval, a
different toolchain or code layout can move it by one channel. The learned
encoder misses the interval even at one channel.

In the default build, board and host tokens agree in every update from 55 to
64 channels and at every whole-grid count (multiples of 64 electrodes). On some
other partial montages, a near-boundary code flips in a minority of updates. The
board (newlib libm, Helium CMSIS) and the host (Apple libm, scalar CMSIS) round
differently, and nearly singular local matrices amplify that difference. Such
configurations are still timed and plotted, and a capacity requires agreement
in every update.

### Pose validation of the default build

`paper/firmware/stm32n6_pose_validation/validate.py` replays the nine
participants' 320-channel test windows, 4,608,000 channel-updates in total,
through host builds of both variants. It scores the frozen SPAR-token decoders
on the resulting tokens:

- **Token agreement:** the default build reproduces 99.92 % of the released
  tokenizer's IDs (lowest participant 99.68 %). The bit-exact build agrees to
  the same extent.
- **Decoding error:** the decoders' mean error changes from 10.8125 to
  10.8120 mm, a difference of −0.0005 mm (95 % participant bootstrap −0.0010
  to −0.0001; exact sign-flip p = 0.055).
- **Frozen fixture:** both builds reproduce all 960 released SPAR IDs, and the
  bit-exact build reproduces the original implementation's features exactly.

`replay63.py` repeats the replay on the first 63 electrodes, using the refitted
63-electrode codebooks and decoders:

- **Accuracy-study montage:** with that study's neighbour rule, the default
  build reproduces 99.96 % of tokens, and the error changes by +0.0004 mm.
- **Benchmark montage:** with this benchmark's tiling rule, which substitutes
  the electrode itself for an absent neighbour (2 of 63 rows differ), agreement
  is 96.8 % and the error changes by +0.66 mm. This montage is a timing
  workload, not the deployed montage.

## Requirements

The base requirements are the same as for [`../stm32n6_tokenizer`](../stm32n6_tokenizer):

- the `paper/results/raw_tokenizers_cortex_m4_v1/` fixtures, including
  `host_outputs.npz`, written by `raw_tokenizer_benchmark/prepare.py` and
  `host_check.py` (see [`../raw_tokenizers`](../raw_tokenizers));
- STM32CubeN6 at commit `80a3a665fccaf9a429343a7ab96a97c1b8a6467a` with five
  submodules, located through `STM32CUBEN6_ROOT`;
- `arm-none-eabi-gcc` 14.3.1 (GNU Tools for STM32 14.3.rel1, STM32CubeIDE
  2.2.0) or `STM32N6_GCC`;
- the ST-LINK GDB server and STM32CubeProgrammer, or `STM32N6_GDB_SERVER` and
  `STM32N6_PROGRAMMER`;
- the learned weights programmed into NOR at `0x71000000`, for the learned
  route.

This port has three further requirements:

- **CubeN6 for host builds:** the default host build also needs
  `STM32CUBEN6_ROOT` (it uses CMSIS-DSP). A host build with `STM32N6_EXACT=1`
  does not.
- **arm64 host:** the default host build needs an arm64 host, because the
  CubeN6 CMSIS headers refuse x86. It was tested on arm64 macOS with Apple
  clang. `verify_fast.py` and `replay63.py` expect macOS `.dylib` names.
- **Original library:** `verify_fast.py` compares against
  `paper/results/stm32n6_tokenizer_v1/host.dylib`, so run the original implementation's
  `verify_host.py` first.

`STM32N6_SRC` and `STM32N6_OUT` redirect the sources and outputs for throwaway
builds. By default, outputs go to `paper/results/stm32n6_optimized_v1/`.

## Verify on the host

Run from the repository root:

```sh
uv run python -m paper.firmware.stm32n6_tokenizer.verify_host
uv run python -m paper.firmware.stm32n6_optimized.verify_host
uv run python -m paper.firmware.stm32n6_optimized.verify_fast
```

`verify_host.py` runs the original implementation's checks. In the default build,
however, it bounds the SPAR features (below 5e-2) instead of requiring
equality. All 960 recorded SPAR IDs must still match.

`verify_fast.py` checks the optimizations:

- **Structural changes, asserted bit-exact:**
  - 1,920 recorded codebook assignments, through both the per-vector and the
    batched search;
  - the tie cases;
  - 96 batched learned feature vectors;
  - 1,482 reordered SPAR feature vectors.
- **Replaced transform (default build), bounded:**
  - relative spectrum deviation below 2e-3; 3.7e-4 was measured;
  - feature deviation below 5e-2; 1.2e-2 was measured;
  - all 960 released SPAR IDs must be reproduced.
- **Synthetic driver runs:** compared with the original implementation's host library.
  They must be identical in the bit-exact build and on the learned route.

With `STM32N6_EXACT=1`, both scripts assert equality everywhere.

## Build and run on the board

The board setup is described in [`../stm32n6_tokenizer`](../stm32n6_tokenizer).
The collector loads SRAM firmware; it never writes NOR, OTP or option bytes. It
needs a current `host_verification.json` and `fast_verification.json` for the
host library.

Run the configurations one at a time, because there is only one board. The
channel lists reproduce the frozen Fig. 7 manifests:

```sh
uv run python -m paper.firmware.stm32n6_optimized.build
uv run python -m paper.firmware.stm32n6_optimized.verify_host
uv run python -m paper.firmware.stm32n6_optimized.verify_fast
uv run python -m paper.firmware.stm32n6_optimized.collect --route learned --frames 16 --warmup 16 --timeout 300 \
  --sweep-channels 1 2 3 4 6 8 9 10 11 12 13 16 32 64 112
uv run python -m paper.firmware.stm32n6_optimized.collect --route spar --frames 16 --warmup 16 --timeout 180 \
  --sweep-channels 1 2 3 4 5 6 8 10 11 12 14 16 20 23 24 28 30 31 32 33 34 35 36 37 38 40 43 45 46 \
  50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 66 68 70 76 82 88 91 94 100 106 112 118 119 120 121 \
  122 123 124 125 126 127 128 140 145 150 155 160 180 181 200 220 240 249 250 251 252 256 270 290 \
  320 350 362 400 450 498 499 500 501 512 550 600 640 700 724 800 900 1024
uv run python -m paper.firmware.stm32n6_optimized.freeze_runs
```

The bit-exact curve is collected the same way. Set `STM32N6_EXACT=1` for the
build and the two verifications only, then freeze that curve to its own
manifest:

```sh
STM32N6_EXACT=1 uv run python -m paper.firmware.stm32n6_optimized.build
STM32N6_EXACT=1 uv run python -m paper.firmware.stm32n6_optimized.verify_host
STM32N6_EXACT=1 uv run python -m paper.firmware.stm32n6_optimized.verify_fast
uv run python -m paper.firmware.stm32n6_optimized.collect --route spar --frames 16 --warmup 16 --timeout 180 \
  --sweep-channels 1 4 8 16 24 30 32 33 38 43 44 45 46 47 48 49 56 320
uv run python -m paper.firmware.stm32n6_optimized.freeze_runs --elf-sha256 <bit-exact ELF SHA-256> \
  --output paper/results/stm32n6_optimized_v1/measurement_manifest_exact.json
```

Afterwards, rebuild the default firmware and rerun both verifications, so that
the build receipt, the host library and the default manifest agree again.
Each MCU build is archived under `builds/<ELF SHA-256>/`, together with its
map, receipt and sources.

`collect.py --recorded` replays the first 64 or fewer channels of
`raw_320x2120.f32` instead of synthetic input. Use it with
`--frames 3 --warmup 17`; the learned route also checks the IDs against
`host_outputs.npz`.

Draw Fig. 7 with:

```sh
uv run python -m paper.firmware.stm32n6_optimized.plot
```

`plot.py` reads three frozen manifests and the run folders they list:

- `paper/results/stm32n6_tokenizer_v1/measurement_manifest.json`
- `paper/results/stm32n6_optimized_v1/measurement_manifest.json`
- `paper/results/stm32n6_optimized_v1/measurement_manifest_exact.json`

The paper's measurements also ship in `paper/data/firmware/`, laid out like
`results/`. That copy holds the three manifests, plus each listed run's
`receipt.json`, `frames.csv` and `output_validation.json`, with the ST-LINK
serial and build paths removed from the receipts. A file in `results/` takes
precedence; otherwise the shipped copy is read. Fig. 7 can therefore be redrawn
without the board.

It writes `stm32n6-optimized.{pdf,svg,png}`, the caption (`caption.txt`,
`caption.tex`, `include.tex`) and `summary.{json,csv}` to
`paper/output/figures/stm32n6_optimized/`.

The figure has two panels:

- **A:** compute time per update against the 20 ms interval, for both ports.
  It shows a geometric subset of the measured channel counts.
- **B:** the default build's per-channel time, split by stage, with a
  proportional fit.

The plot uses the paper's figure style (`paper/common/figstyle.py` and `simpetru.mplstyle`)
and the Switzer font if it is installed (`paper/common/switzer_font.py`).

## Pose validation

Run from the repository root:

```sh
uv run python -m paper.firmware.stm32n6_pose_validation.validate prepare --subjects 1 2 3 4 5 6 7 8 9 --device cuda
uv run python -m paper.firmware.stm32n6_pose_validation.validate build
uv run python -m paper.firmware.stm32n6_pose_validation.validate fixture
uv run python -m paper.firmware.stm32n6_pose_validation.validate subject --subjects 1 2 3 4 5 6 7 8 9 --device cpu
uv run python -m paper.firmware.stm32n6_pose_validation.validate report
uv run python -m paper.firmware.stm32n6_pose_validation.replay63 --data paper/results/stm32n6_pose_validation_v1/prepare
```

| Stage | Inputs |
|---|---|
| `prepare` (GPU host) | HD-EMG recordings (`MYOJI_HDEMG`); `paper/results/dpose_spar_simple_full_v1/subject_XX/{protocol.json,cache-receipt.json}` from `run_dpose_spar_full.py`. Writes `windows.npy` (about 0.7 GB per participant), `cache.npz` and `prepare.json` to `--work` (default `paper/results/stm32n6_pose_validation_v1/prepare/`). |
| `build` (arm64 host) | The raw-tokenizer fixtures, and `STM32CUBEN6_ROOT`. Builds both variants with zeroed standardization, so that they return raw matrix-log entries. |
| `fixture` | `reference.npz`, `host_outputs.npz`, `raw_320x2120.f32`, and participant 1's standardization and mask from `$MYOJI_ASSETS/spar/codebooks-1024.npz` |
| `subject` | The `prepare` outputs (`--data`, same default), and the campaign's frozen decoders `dpose_spar_simple_full_v1/subject_XX/fits/tokens_{260911,260912,260913}/{best.pt,test_predictions.npz}` |
| `replay63.py` | The `prepare` outputs; `paper/results/dpose_spar_63_electrodes_v1/subject_XX/{features.h5,fits/tokens_*/best.pt}` from `dpose_spar_63_electrodes.py`; and the `build` stage's default library |

The results are written to `paper/results/stm32n6_pose_validation_v1/`. No
model is fitted.
