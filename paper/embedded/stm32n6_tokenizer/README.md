# STM32N6 tokenizer benchmark, original port

This firmware runs both frozen tokenizers end to end on the Cortex-M55 CPU of
a NUCLEO-N657X0-Q board at 600 MHz, in FP32 with instruction and data caches on
and without the NPU. Each update shifts in new samples, normalizes them, runs
the frontend, searches all 1,024 codes and packs 10-bit IDs. It reuses the
[`../raw_tokenizers`](../raw_tokenizers) kernels unchanged. `build.py`
generates `spar_runtime.c` from `spar.c`, changing only the channel count and
montage into arguments and adding a cycle-counter poll per channel. This port
is the baseline of Fig. 6 in the paper. The optimized port is in
[`../stm32n6_optimized`](../stm32n6_optimized).

| File | Contents |
|---|---|
| `benchmark.c` | Workload driver: synthetic input, montage, per-stage cycle counts, 10-bit packing, checksums |
| `benchmark.h` | Debugger interface (`bench_config`, `bench_result`, `bench_frames`) |
| `platform.c` | HAL and clock setup, AXISRAM3–6 power-up and test, DWT cycle counter, caches, read-only XSPI NOR mapping for the learned weights |

## What is measured

The input is deterministic, amplitude-modulated broadband noise at 2 kHz. It
is a reproducible compute load, not physiological data. The window is 160 ms
and each update adds 20 ms of samples.

- **Timed per update:** the window shift, normalization, frontend, nearest-code
  search and packing, each counted separately.
- **Timed separately:** calibration.
- **Not timed:** sample generation, ADC, radio and pose decoding.

Each configuration runs 16 warm-up updates followed by 16 measured updates. The
first and last token IDs and every per-update checksum must match a host build
of the same code; a run that disagrees is kept and marked as failed.

The two routes support different channel ranges:

- **SPAR:** 1–1,024 channels. The 320-electrode montage repeats for larger
  counts, and a partial tile masks absent neighbours. At 320 channels it is the
  paper's montage, and band weights are recomputed over the selected channels.
- **Learned encoder:** 1–112 channels, limited by 1,792 KiB of causal-cache
  SRAM. It needs at least 16 warm-up updates.

Results:

- SPAR meets the 20 ms interval, with host agreement, up to 23 channels
  (slowest update 19.22 ms).
- 320 SPAR channels take 276.7 ms per update.
- The learned encoder takes 104.7 ms per update for one channel.

Some partial-montage SPAR counts (4, 8, 24 and 32 channels) flip a
near-boundary code in some updates. The board's newlib `hypotf`/`logf` and the
host libm round differently, and that difference matters on nearly singular
local covariance matrices.

## Requirements

- **Fixtures:** `learned_weights.f32`, `weight_offsets.h` and `constants.c`
  from `paper/results/raw_tokenizers_cortex_m4_v1/`, for every build.
  `raw_320x2120.f32`, `reference.npz` and `host_outputs.npz` from the same
  folder, for host verification. All of them are written by
  `paper/scripts/raw_tokenizer_benchmark/prepare.py` and `host_check.py`; see
  [`../raw_tokenizers`](../raw_tokenizers).
- **Host:** a C compiler (`clang` or `cc`) and NumPy. This port does not use
  CMSIS, so its host build is not tied to arm64.
- **Board build:**
  - STM32CubeN6 at commit `80a3a665fccaf9a429343a7ab96a97c1b8a6467a`, located
    through `STM32CUBEN6_ROOT` (see below).
  - `arm-none-eabi-gcc` 14.3.1 20250623 (GNU Tools for STM32 14.3.rel1, as
    bundled with STM32CubeIDE 2.2.0). The build finds it on `PATH`, in
    `/Applications/STM32CubeIDE.app`, or through `STM32N6_GCC`.
- **Board run:**
  - A NUCLEO-N657X0-Q.
  - The ST-LINK GDB server and STM32CubeProgrammer CLI from STM32CubeIDE, or
    set `STM32N6_GDB_SERVER` and `STM32N6_PROGRAMMER`.
  - `arm-none-eabi-gdb`.

STM32CubeN6 is not shipped. The build takes the GPIO_IOToggle FSBL example's
startup, system file, HAL configuration (with XSPI enabled) and its unchanged
600 MHz `SystemClock_Config`. It adapts the example's linker script (32 KiB
stack, two NOLOAD SRAM regions, the NOR weight symbol) and checks the commit:

```sh
git clone https://github.com/STMicroelectronics/STM32CubeN6.git
git -C STM32CubeN6 checkout 80a3a665fccaf9a429343a7ab96a97c1b8a6467a
git -C STM32CubeN6 submodule update --init Drivers/STM32N6xx_HAL_Driver \
    Drivers/CMSIS/Device/ST/STM32N6xx Drivers/BSP/STM32N6xx_Nucleo \
    Drivers/BSP/Components/Common Drivers/BSP/Components/mx25um51245g
export STM32CUBEN6_ROOT=$PWD/STM32CubeN6
```

## Verify on the host

Run from the repository root:

```sh
.venv/bin/python paper/scripts/stm32n6_tokenizer/verify_host.py
```

This builds `paper/results/stm32n6_tokenizer_v1/host.dylib` (`host.so` on
Linux) and checks it against the raw C port:

- all 678,400 normalized samples and calibrations, bit for bit;
- SPAR features and all 960 SPAR IDs at the three fixture endpoints;
- learned features and IDs of channels 0 and 319 after full causal warm-up;
- 10-bit packing against an independent oracle;
- rejection of invalid configurations, and reproducible synthetic runs.

It writes `host_verification.json` and `host_golden_ids.json`. Host timings
are not microcontroller timings.

## Build and run on the board

1. Set the board up:
   - Jumpers: CN9 pins 1–2 (5V_STLK), JP1 pins 1–2, and JP2 pins 2–3
     (development boot).
   - USB: connect CN10 (ST-LINK), not CN8.
   - The GDB server attaches to access port 1, the Cortex-M55 (ST AN6265).
2. The SPAR route needs nothing else. For the learned route, program
   `learned_weights.f32` (9,567,360 bytes) into the external NOR at
   `0x71000000` (for example with STM32CubeProgrammer) and verify it by
   readback.
   - The firmware maps the NOR read-only at 50 MHz OPI DTR.
   - It refuses the learned route if the region's FNV-1a digest differs from
     the build's.
   - It never erases or programs NOR.

```sh
.venv/bin/python paper/scripts/stm32n6_tokenizer/build.py
.venv/bin/python paper/scripts/stm32n6_tokenizer/collect.py --route spar \
  --sweep-channels 1 4 8 16 20 22 23 24 32 64 128 320 640 1024 --frames 16 --warmup 16 --timeout 180
.venv/bin/python paper/scripts/stm32n6_tokenizer/collect.py --route learned \
  --sweep-channels 1 2 4 8 16 32 64 112 --frames 16 --warmup 16 --timeout 600
.venv/bin/python paper/scripts/stm32n6_tokenizer/freeze_runs.py
.venv/bin/python paper/scripts/stm32n6_tokenizer/plot.py
```

`build.py` writes `benchmark.elf`, the map, sizes, symbols and `build.json`
(the compiler, commands and the hash of every source) to
`paper/results/stm32n6_tokenizer_v1/`.

`collect.py` loads the ELF into SRAM through the ST-LINK GDB server; it does
not touch NOR or option bytes. At the `bench_ready` breakpoint it sets
`bench_config` and then reads the results back. It refuses to start if the
ELF or any source differs from `build.json`, and it does not validate a run
without a current host verification. Each run is kept under `runs/` with its
per-update CSV, raw dumps, logs and receipt.

- A sweep runs one configuration per channel count, keeps every outcome, and
  exits nonzero if any configuration fails validation.
- `--plan-only` writes the GDB script without connecting.

`freeze_runs.py` writes `measurement_manifest.json`. It takes the latest
attempt per route and channel count for the current ELF, whatever its
validation outcome, and pins each run's receipt and CSV by SHA-256.

`plot.py` draws the validated runs to
`paper/output/figures/stm32n6_tokenizer/`. Fig. 6 is drawn from this manifest
by [`../stm32n6_optimized`](../stm32n6_optimized).

The paper's measurements also ship in
`paper/data/firmware/stm32n6_tokenizer_v1/`, laid out like `results/`: the
manifest, plus each listed run's `receipt.json`, `frames.csv` and
`output_validation.json`. The ST-LINK serial and build paths are removed from
those receipts. Both plot scripts read a file from `results/` when it is there,
and the shipped copy otherwise, so the figures can be redrawn without the
board.
