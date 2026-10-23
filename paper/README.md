# Reproducing the paper

The experiment and figure scripts for *EMG Tokenization Preserves Movement Information at a
Fraction of the Bitrate* and its supplement. The reusable methods are in the `myoji` library
(`../src/myoji`).

```
paper/
  common/      shared helpers: data paths, statistics, figure style
  pretraining/ learned-tokenizer pretraining and checkpoint checks
  hdemg/       HD-EMG campaigns: SPAR and learned tokens, robustness, statistics, latency,
               and what tokens carry (Fig. 6)
  receivers/   the other HD-EMG receivers: matched TCN/Transformer/CNN, adapted and TBME CNNs,
               Qwen3, TabPFN
  emg2pose/    the emg2pose campaigns and their tables
  figures/     figure panels and their assembly
  firmware/    host checks and plots for the STM32N6 firmware and the C tokenizers
  vendor/      third-party model code the receivers pin by hash
  embedded/    the Cortex-M55 (STM32N6) SPAR firmware and its portable C tokenizers
  data/        shipped inputs: frozen window selections (reference/), Supplementary Data 1-9
               (supplementary/), and the board measurements behind Fig. 7 (firmware/)
  results/     run folders written by the campaigns (not tracked)
  output/      figures and tables (not tracked)
```

Scripts named "hdemg" work on the paper's 320-channel HD-EMG dataset (nine decoding
participants, 13 for tokenizer fitting). "V20" is RaulNet V20, the paper's spatial CNN + TCN pose
decoder (`SparPose` in `myoji.decoders`); "RaulNet V19" is the raw-EMG CNN the paper calls the
adapted CNN. A receiver is any pose decoder that reads the transmitted tokens or features.

Run the scripts as modules from the repository root, for example
`uv run python -m paper.hdemg.run_hdemg_pilot --help`. Below, `hdemg/run_hdemg_pilot` stands for
that module.

## Figures from the shipped files

A fresh clone redraws Fig. 7 and Supplementary Figs. S1–S3 from the shipped files:

```bash
uv sync --group paper
uv run python -m paper.firmware.stm32n6_optimized.plot   # Fig. 7, to paper/output/figures/stm32n6_optimized/
uv run python -m paper.figures.plot_supp_figures         # Figs. S1–S3, to paper/output/figures/pose_paper/
```

Supplementary Data 1-9 (`data/supplementary/`) hold every plotted value and summary result. The
other figures redraw from the campaign outputs.

## Setup

```bash
uv sync --group paper                      # from the repository root
```

| input | where it comes from | set |
|---|---|---|
| HD-EMG recordings `Sub{n}_emg.pkl`, `Sub{n}_kinematics.pkl` | the authors, on reasonable request | `MYOJI_HDEMG=<folder>` |
| emg2pose | public; `uv run python -m paper.emg2pose.prepare_emg2pose_online` builds the metadata and a byte-range index so recordings stream on demand | `MYOJI_EMG2POSE=<folder>` |
| emg2pose code (hand model) | `git clone https://github.com/facebookresearch/emg2pose` (commit `5f6f62b`), then `uv pip install -e emg2pose` | |
| trained tokenizers and SPAR codebooks | train your own ([README](../README.md#training-your-own-models)); scripts read `learned/tokenizer.pt`, `learned/tokenizer-full.pt` and `spar/codebooks-1024.npz` | `MYOJI_ASSETS=<folder>` |
| BioCodec (codec comparison only) | `git clone https://github.com/klean2050/BioCodec-release` (commit `575bf6e`) | `MYOJI_BIOCODEC=<clone>` |
| Qwen3-0.6B-Base (one receiver) | Hugging Face, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, into the LLM pilot run folder's `model/` | |
| TabPFN v3.5 weights (one receiver) | downloaded by `tabpfn==9.0.0` on first use | `--model-path` |
| commercial hand model `limbKit_02.blend` (Fig. 1 hand renders only, inside Blender) | purchase separately | `MYOJI_HAND_MODEL=<file>` |
| Switzer font (figures only) | Fontshare; optional, figures fall back to another sans-serif | `MYOJI_SWITZER_DIR` |

The HD-EMG scripts check each recording's SHA-256 against the export used for the paper and stop
with an error on a mismatch.

## How a campaign runs

Each campaign script has stages (`freeze`, `prepare`, `fit`, `evaluate`, or `all`), and
`--help` lists them with their inputs. `freeze` writes `protocol.json` to the run folder. It
records the windows, settings, seeds, and the SHA-256 of the data and of every source file the
campaign executes. Later stages stop if any of these change. Finished fits are skipped, so a
campaign can be resumed. Some campaigns build on earlier run folders and check their hashes too.

A few campaigns require the original GPU model or software versions, and some figure and firmware
steps check their inputs against the paper's files by hash. Their error messages name the
requirement.

## Order

Each step reads the run folders of the steps before it, in `paper/results/<name>`.

**HD-EMG, learned tokenizer**

1. `hdemg/run_hdemg_pilot`
2. `hdemg/run_hdemg_within`, which writes `hdemg_within_v1`, the split every HD-EMG decoder uses
3. `hdemg/run_hdemg_v20`, then `hdemg/run_hdemg_v20_continuous` and `hdemg/run_hdemg_v20_one_euro`
4. The other decoders, in any order:
   - `receivers/run_hdemg_matched`: TCN, Transformer and waveform CNN
   - `receivers/run_hdemg_transformer`
   - `receivers/run_hdemg_raulnet`: adapted CNN
   - `receivers/run_hdemg_tbme2024`: retrained TBME CNN
   - `receivers/run_hdemg_llm_pilot`, then `receivers/run_hdemg_llm_full`: Qwen3
   - `receivers/tabpfn_pose/`: TabPFN

**HD-EMG, SPAR** (starts from the window selection and protocol in `data/reference/`)

1. `hdemg/run_hdemg_spar_simple`: participant 1
2. `hdemg/run_hdemg_spar_full`: nine participants; it compares with the learned tokenizer, so it runs
   after learned-tokenizer step 3
3. `hdemg/run_hdemg_spar_v64` (64 codes) and `hdemg/run_hdemg_spar_ablation`

**HD-EMG, robustness and transfer**

- After SPAR step 2: `hdemg/hdemg_spar_63_electrodes`, `hdemg/hdemg_spar_transfer_checks`,
  `hdemg/hdemg_spar_onset_calibration` and `hdemg/hdemg_spar_corruption` (noise and electrode loss)
- After `hdemg/hdemg_spar_63_electrodes` and `receivers/run_hdemg_raulnet`: `hdemg/hdemg_raw_link_budget`
- Noise and electrode loss for the learned tokenizer, after learned-tokenizer steps 3 and 4:
  `hdemg/run_hdemg_noise`, then `hdemg/run_hdemg_v20_noise_followup`, then `hdemg/run_hdemg_dropout`
- After SPAR step 2 and emg2pose step 2: `hdemg/run_hdemg_tokenizer_reuse` (donor-only tokenizers;
  `--recipe paper/data/reference/tokenizer_recipe.json`)

**emg2pose**

1. `emg2pose/prepare_emg2pose_online`
2. `pretraining/audit_tokenizer_release prepare`: the frozen pretraining windows (`results/clean_base_v1`)
3. `pretraining/train_mixed_sources`: tokenizer pretraining
4. `emg2pose/compare_temporal_pose cache`, which writes `results/clean_temporal_v1/full.h5`
5. `emg2pose/run_meta_sessions`
6. `emg2pose/run_tokenizer_seeds`
7. `emg2pose/compare_temporal_pose fit`, then `evaluate`
8. `emg2pose/run_paper_completion`, with `emg2pose/paper_completion_smoke`, `emg2pose/export_pose_replay`,
   `emg2pose/paper_completion_biocodec`, `emg2pose/paper_completion_runtime` and
   `emg2pose/report_paper_completion`
9. `emg2pose/prepare_reviewer_features`, then `emg2pose/run_reviewer_campaign`: finite scalar
   quantization (FSQ), quantized PCA and k-means

**Firmware**: see `embedded/stm32n6_optimized/README.md`; the fixtures come from
`firmware/raw_tokenizer_benchmark/prepare.py` and `host_check.py`.

## Where each result comes from

| result | scripts |
|---|---|
| Fig. 4: errors, payload, latency; tokens minus continuous | the HD-EMG campaigns above, `hdemg/benchmark_hdemg_workstation`, `hdemg/benchmark_hdemg_all_variants`, `hdemg/hdemg_movement_analysis` and `hdemg/hdemg_absolute_change_analysis` (their frozen definitions are in `data/figure_inputs/`), then `figures/build_hdemg_combined` and `figures/plot_fig3_signed_panel` |
| Supplementary Data 2 and 3, paired statistics | `hdemg/hdemg_claim_statistics` (uses `common/hdemg_figure_statistics`) |
| Fig. 5: decoded examples and per-movement errors | `hdemg/run_p1_continuous`, `hdemg/run_hdemg_v20_one_euro`, `figures/plot_decoding_examples` |
| Fig. 6: token-history UMAP, pose retrieval, held-out retrieval | `hdemg/select_hdemg_movement_cycles`, `hdemg/replay_hdemg_spar_figure_video`, `hdemg/build_token_umap_cache`, `hdemg/pose_recovery/`, `hdemg/pose_recovery_heldout_statistics`, `hdemg/direction_*`, `figures/make_token_pose_recovery_figure` |
| Fig. 7: Cortex-M55 timing | `firmware/stm32n6_optimized/plot.py` and `firmware/stm32n6_tokenizer/plot.py`; they redraw from `data/firmware/` without a board |
| Figs. 1 and 2: panels (assembled in a vector editor) | `figures/make_teaser`, `figures/plot_encoder_examples`, `figures/plot_residual_traces`, `figures/plot_residual_featuremap`, `figures/plot_residual_example`, `figures/build_tokenizer_mechanism` |
| Fig. 3: the SPAR patch | drawn in the manuscript's LaTeX source, no script |
| emg2pose cohorts, alternatives, Supplementary Data 4-7 | the emg2pose campaigns, then `emg2pose/build_pose_paper_assets` and `figures/plot_meta_overview`, `figures/plot_meta_transfer`, `figures/plot_compact_controls`, `figures/plot_codec_diagnostics`, `figures/plot_pose_trajectories`, `figures/plot_streaming_cost`, `figures/plot_noise_robustness` |
| SPAR noise and electrode loss, Supplementary Data 8 | `hdemg/hdemg_spar_corruption`, `hdemg/n9_exact_intervals` |
| SPAR ablation and held-out retrieval, Supplementary Data 9 | `hdemg/run_hdemg_spar_ablation`, `hdemg/spar_ablation_statistics`, `hdemg/pose_recovery_heldout_statistics` |
| Supplementary Figs. S1–S3 | `figures/plot_supp_figures` (reads `data/supplementary/` only) |
| Supplementary Fig. S4 | `figures/make_token_history_supplement`, `hdemg/token_history_shuffle_roundness` |
| Supplementary Fig. S5 | `figures/plot_robustness_supplement` |

The figures are regenerated from the campaigns' outputs.

## Scripts by folder

Names in *italics* are helpers imported by other scripts, not run on their own. "emg2pose" is the
public dataset; older docstrings call it "Meta".

### `common/`

| Module | What it does |
|---|---|
| *`paths`* | Where inputs are read and outputs written; set `MYOJI_HDEMG`, `MYOJI_EMG2POSE`, `MYOJI_ASSETS` |
| *`hdemg_figure_statistics`* | Exact paired sign-flip tests, intervals and Holm adjustment for the HD-EMG analyses |
| *`figstyle`*, `simpetru*.mplstyle` | The paper's figure style |
| *`paper_plot_style`*, *`tokenizer_diagnostic_style`* | Print sizes and colours for the Fig. 4 and supplementary builders |
| `switzer_font` | Uses Switzer for the figures when it is installed; running it checks the regular and bold fonts |
| *`reporting_support`* | Checks the participant labels in the emg2pose reports |

### `pretraining/`

| Module | What it does |
|---|---|
| `train_mixed_sources` | Pretrains the learned tokenizer on HD-EMG and emg2pose training windows |
| `audit_tokenizer_release` | Freezes validation inputs and checks that the exported tokenizer is deterministic, also after a reload |

### `hdemg/`: HD-EMG campaigns

**Learned tokenizer**

| Module | What it does |
|---|---|
| `run_hdemg_pilot` | Participant-1 pilot that fixes the windows, preprocessing and splits the later campaigns reuse |
| `run_hdemg_within` | Personalized decoding on chronologically separated recording blocks (the shared split) |
| `run_hdemg_v20`, `run_hdemg_v20_continuous` | Learned tokens, and the matched continuous features, decoded by the spatial CNN + TCN |
| *`report_hdemg_v20`* | Rechecks the V20 predictions against the frozen decoder |
| `run_hdemg_v20_one_euro` | One Euro smoothing of the V20 outputs, chosen on validation (Fig. 5); filter in *`hdemg_one_euro`* |
| `run_p1_continuous` | Continuous test-block predictions for participant 1 (Fig. 5 traces) |

**SPAR**

| Module | What it does |
|---|---|
| `run_hdemg_spar_simple` | SPAR pilot on participant 1 (EMG → SPAR → codebook → decoder) |
| `run_hdemg_spar_full`, `run_hdemg_spar_v64` | SPAR on nine participants, with 1,024 and 64 codes |
| `run_hdemg_spar_ablation` | SPAR with parts of the descriptor removed (Supplementary Data 9) |

**Robustness and transfer**

| Module | What it does |
|---|---|
| `run_hdemg_noise` | Paired added-noise test of the learned tokenizer; helpers *`hdemg_noise`*, *`report_hdemg_noise`* |
| `run_hdemg_v20_noise_followup` | The same noise test for token V20, continuous V20 and the adapted CNN |
| `collect_hdemg_v20_noise_followup` | Independent audit of the follow-up |
| `run_hdemg_dropout` | Paired electrode-dropout test; helper *`hdemg_dropout`* |
| `hdemg_spar_corruption` | The noise and dropout tests for SPAR (Supplementary Data 8) |
| `hdemg_spar_63_electrodes` | SPAR with the 63 electrodes the Cortex-M55 firmware handles in 20 ms |
| `hdemg_raw_link_budget` | Raw EMG from as many electrodes as the assumed Bluetooth LE throughput carries, decoded by the adapted CNN |
| `hdemg_spar_transfer_checks` | Frozen SPAR pipelines on transfer checks |
| `hdemg_spar_onset_calibration` | SPAR with one calibration per recording |
| `run_hdemg_tokenizer_reuse` | Tokenizers trained only on donor participants, then personalized decoders |

**Statistics, costs and what tokens carry**

| Module | What it does |
|---|---|
| `hdemg_claim_statistics` | The paired statistics behind the main claims (Supplementary Data 2 and 3) |
| `n9_exact_intervals` | Exact intervals for the nine-participant contrasts outside the four headline claims |
| `hdemg_movement_analysis`, `hdemg_absolute_change_analysis` | Per-movement errors and token-minus-continuous changes (Fig. 4) |
| `benchmark_hdemg_workstation`, `benchmark_hdemg_all_variants`, `benchmark_hdemg_tbme2024_inference` | Decoder latency and resource measurements (Fig. 4) |
| `select_hdemg_movement_cycles` | Picks the movement cycles shown in the figures before any prediction is read |
| `replay_hdemg_spar_figure_video` | Replays the two nine-participant SPAR decoders on the selected cycles and measures their latency (Fig. 4) |
| `build_token_umap_cache` | Encodes one task into both token types, aligned to the hand kinematics (Fig. 6) |
| `pose_recovery/` | Nearest-repetition pose retrieval, also on held-out data (`pose_recovery`, `pose_recovery_heldout`, `aggregate`) |
| `pose_recovery_heldout_statistics`, `spar_ablation_statistics` | Statistics for held-out retrieval and the SPAR ablation (Supplementary Data 9) |
| `direction_at_matched_closure`, `direction_continuous_spar` | Opening versus closing at matched hand closure |
| `token_history_shuffle_roundness` | The shuffled-history control of the token-history figure (Supplementary Fig. S4) |

### `receivers/`: the other HD-EMG decoders

| Module | What it does |
|---|---|
| `run_hdemg_matched` | TCN, Transformer and waveform CNN with matched capacity and training; helpers *`hdemg_matched_core`*, `report_hdemg_matched` |
| `run_hdemg_transformer`, `report_hdemg_transformer` | The small token Transformer |
| `run_hdemg_raulnet` | The adapted raw-EMG CNN (RaulNet V19); helpers *`hdemg_raulnet_model`*, `report_hdemg_raulnet` |
| `run_hdemg_tbme2024` | The TBME 2024 CNN retrained on the same windows; helper *`hdemg_tbme2024`* |
| `run_hdemg_llm_pilot`, `run_hdemg_llm_full` | The frozen Qwen3 decoder, pilot then nine participants; helpers *`hdemg_llm_core`*, `hpc/hdemg_llm_worker` (runs it on a cluster GPU) |
| `tabpfn_pose/` | The TabPFN decoder: `run` (participant 1), `cohort` (nine), `analyze_cohort`, `benchmark_cohort` (latency), `finish_cohort` (starts the latency stage), *`_support`* |

### `emg2pose/`

| Module | What it does |
|---|---|
| `prepare_emg2pose_online` | Metadata and a byte-range index, so recordings stream on demand |
| `compare_temporal_pose` | Caches token features, fits the causal pose decoders and evaluates them |
| `run_meta_sessions` | Transfer to unseen sessions |
| `run_tokenizer_seeds` | Independent tokenizer seeds |
| `run_paper_completion` | The emg2pose replication and the quantizer comparisons; helper *`paper_completion_core`* |
| `paper_completion_smoke`, `paper_completion_runtime`, `paper_completion_biocodec` | Checks before the costly jobs, runtime measurements, the BioCodec adaptation |
| `export_pose_replay`, `report_paper_completion` | A replay window, and the final report |
| `prepare_reviewer_features`, `run_reviewer_campaign` | The FSQ, quantized PCA and k-means alternatives |
| `build_pose_paper_assets` | The emg2pose figures and tables (Supplementary Data 4–7) |
| `build_completion_results` | Rechecks the paired statistics of the replication and draws its overview figures |

### `figures/`

| Module | Figure |
|---|---|
| `make_teaser`, `render_teaser_hands` (inside Blender), `export_token_explorer_context` | Fig. 1 panels |
| `plot_encoder_examples`, `plot_residual_traces`, `plot_residual_featuremap`, `plot_residual_example`, `build_tokenizer_mechanism` | Fig. 2 panels |
| `make_encoder_replacement_draft`, `export_encoder_examples_affinity` | Editable drawings of the encoder for the vector editor |
| `build_hdemg_combined` | Fig. 4, errors and decoder costs; helpers *`build_hdemg_all_variants`*, *`hdemg_spar_figure_extension`*, *`hdemg_tabpfn_figure_extension`*, *`hdemg_tbme_figure_data`* |
| `plot_fig3_signed_panel` | Fig. 4C, the signed effect of quantization per decoder |
| `build_hdemg_resource_overview` | Payload budgets next to measured decoder costs, as a separate overview |
| `plot_decoding_examples` | Fig. 5, decoded movements of one participant |
| `make_token_pose_recovery_figure` | Fig. 6 panels |
| `plot_supp_figures` | Supplementary Figs. S1–S3 |
| `make_token_history_supplement` | Supplementary Fig. S4 |
| `plot_robustness_supplement` | Supplementary Fig. S5 |
| `plot_meta_overview`, `plot_meta_transfer`, `plot_compact_controls`, `plot_codec_diagnostics`, `plot_pose_trajectories`, `plot_streaming_cost`, `plot_noise_robustness` | emg2pose and robustness panels, drawn by `build_pose_paper_assets` |

### `firmware/`

The full commands are in `embedded/*/README.md`.

| Module | What it does |
|---|---|
| `raw_tokenizer_benchmark/` | `prepare` exports fixtures and PyTorch reference outputs; `host_check` checks the C tokenizers against them |
| `stm32n6_tokenizer/`, `stm32n6_optimized/` | `build` (never flashes), `verify_host` and `verify_fast` (host checks), `collect` (one board run), `freeze_runs` (records the latest run per configuration, failed validations included), `plot` (Fig. 7, from `data/firmware/`) |
| `stm32n6_pose_validation/` | `validate` and `replay63` replay recordings through the C SPAR implementation and score the frozen decoders |

### `vendor/`

Third-party code, pinned by hash (see `THIRD_PARTY.md`): RaulNet V19, the One Euro filter and the TBME 2024 CNN.
