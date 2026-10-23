# Reproducing the paper

Everything that produced the numbers and figures in *EMG Tokenization Preserves Movement
Information at a Fraction of the Bitrate* and its supplement. The reusable methods live in the
`myoji` library (`../src/myoji`); this folder holds the experiments that use it.

```
paper/
  common/      shared helpers: data paths, statistics, figure style
  pretraining/ learned-tokenizer pretraining and its release audit
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

"dpose" in script names is the 320-channel HD-EMG dataset of the paper (nine decoding
participants, 13 for tokenizer fitting).

Run every script as a module from the repository root, e.g.
`uv run python -m paper.hdemg.run_dpose_pilot --help`; below, `hdemg/run_dpose_pilot` stands for
that module.

## Setup

```bash
uv sync --group paper                      # from the repository root
```

| input | where it comes from | set |
|---|---|---|
| HD-EMG recordings `Sub{n}_emg.pkl`, `Sub{n}_kinematics.pkl` | the authors, on reasonable request | `MYOJI_HDEMG=<folder>` |
| emg2pose | public; `uv run python -m paper.emg2pose.prepare_emg2pose_online` builds the metadata and a byte-range index so recordings stream on demand | `MYOJI_EMG2POSE=<folder>` |
| emg2pose code (hand model) | `git clone https://github.com/facebookresearch/emg2pose` (commit `5f6f62b`), then `uv pip install -e emg2pose` | |
| trained tokenizers and SPAR codebooks | not distributed; train your own (see the end of this page) | `MYOJI_ASSETS=<folder>` |
| BioCodec (codec comparison only) | `git clone https://github.com/klean2050/BioCodec-release` (commit `575bf6e`) | `MYOJI_BIOCODEC=<clone>` |
| Qwen3-0.6B-Base (one receiver) | Hugging Face, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, into the LLM pilot run folder's `model/` | |
| TabPFN v3.5 weights (one receiver) | downloaded by `tabpfn==9.0.0` on first use | `--model-path` |
| Switzer font (figures only) | Fontshare; optional, figures fall back to another sans-serif | `MYOJI_SWITZER_DIR` |

The HD-EMG scripts check the SHA-256 of every recording against the export used for the paper,
so a different export stops with a clear error instead of silently changing the numbers.

## How a campaign runs

Each campaign script has stages (`freeze`, `prepare`, `fit`, `evaluate`, or `all`), and
`--help` lists them with their inputs. `freeze` writes `protocol.json` into the run folder with
the windows, settings, seeds, and the SHA-256 of the data and of every source file the campaign
executes; later stages refuse to continue if any of them changed, and finished fits are skipped,
so a campaign can be resumed. Several campaigns build on earlier run folders and check their
hashes in the same way.

A few campaigns pin the original hardware (GPU model) or software versions because they
replicate a timing or a run exactly; their error message names the requirement.

## Order

**HD-EMG, learned tokenizer** (`paper/results/<name>`):
`hdemg/run_dpose_pilot` → `hdemg/run_dpose_within` → `hdemg/run_dpose_v20` → `hdemg/run_dpose_v20_continuous`.
The receivers compared with it build on `dpose_within_v1`: `receivers/run_dpose_matched`
(TCN, Transformer, waveform CNN), `receivers/run_dpose_transformer`, `receivers/run_dpose_raulnet` (adapted CNN),
`receivers/run_dpose_tbme2024` (retrained TBME CNN), `receivers/run_dpose_llm_pilot` → `receivers/run_dpose_llm_full` (Qwen3),
`receivers/tabpfn_pose/` (TabPFN).

**HD-EMG, SPAR**: `hdemg/run_dpose_spar_simple` (participant 1) → `hdemg/run_dpose_spar_full`
(nine participants; compares with the learned route, so it runs after `hdemg/run_dpose_v20` and
`hdemg/run_dpose_v20_continuous`) → `hdemg/run_dpose_spar_v64` (64 codes), `hdemg/run_dpose_spar_ablation`.
The window selection and protocol these start from are shipped in `data/reference/`.

**HD-EMG, robustness and transfer**: `hdemg/dpose_spar_63_electrodes`, `hdemg/dpose_raw_link_budget`,
`hdemg/dpose_spar_transfer_checks`, `hdemg/dpose_spar_onset_calibration`, `hdemg/run_dpose_noise` →
`hdemg/run_dpose_v20_noise_followup` → `hdemg/run_dpose_dropout`, `hdemg/dpose_spar_corruption`,
`hdemg/run_dpose_tokenizer_reuse` (donor-only tokenizers; `--recipe data/reference/tokenizer_recipe.json`).

**emg2pose**: `emg2pose/prepare_emg2pose_online` → `pretraining/audit_tokenizer_release prepare` (frozen training
windows) → `pretraining/train_mixed_sources` (tokenizer pretraining) → `emg2pose/compare_temporal_pose cache`
(`results/clean_temporal_v1/full.h5`) → `emg2pose/run_meta_sessions` → `emg2pose/run_tokenizer_seeds` →
`emg2pose/compare_temporal_pose fit` and `evaluate` → `emg2pose/run_paper_completion`
(with `emg2pose/paper_completion_smoke`, `emg2pose/export_pose_replay`, `emg2pose/paper_completion_biocodec`,
`emg2pose/paper_completion_runtime`, `emg2pose/report_paper_completion`) and `emg2pose/run_reviewer_campaign`
(`emg2pose/prepare_reviewer_features`: FSQ, quantized PCA, k-means).

**Firmware**: see `embedded/stm32n6_optimized/README.md`; the fixtures come from
`firmware/raw_tokenizer_benchmark/prepare.py` and `host_check.py`.

## Where each result comes from

| result | scripts |
|---|---|
| Fig. 4: errors, payload, latency; tokens minus continuous | the HD-EMG campaigns above, `hdemg/benchmark_dpose_workstation`, `hdemg/benchmark_dpose_all_variants`, `hdemg/dpose_movement_analysis` and `hdemg/dpose_absolute_change_analysis` (their frozen definitions are in `data/figure_inputs/`), then `figures/build_dpose_combined` and `figures/plot_fig3_signed_panel` |
| Supplementary Data 2 and 3, paired statistics | `hdemg/dpose_claim_statistics` (uses `common/dpose_figure_statistics`) |
| Fig. 5: decoded examples and per-movement errors | `hdemg/run_p1_continuous`, `hdemg/run_dpose_v20_one_euro`, `figures/plot_decoding_examples` |
| Fig. 6: token-history UMAP, pose retrieval, held-out retrieval | `hdemg/select_dpose_movement_cycles`, `hdemg/replay_dpose_spar_figure_video`, `hdemg/build_token_umap_cache`, `hdemg/pose_recovery/`, `hdemg/pose_recovery_heldout_statistics`, `hdemg/direction_*`, `figures/make_token_pose_recovery_figure` |
| Fig. 7: Cortex-M55 timing | `firmware/stm32n6_optimized/plot.py` and `firmware/stm32n6_tokenizer/plot.py`; they redraw from `data/firmware/` without a board |
| Figs. 1 and 2: panels (assembled in a vector editor) | `figures/make_teaser`, `figures/plot_encoder_examples`, `figures/plot_residual_traces`, `figures/plot_residual_featuremap`, `figures/plot_residual_example`, `figures/build_tokenizer_mechanism` |
| Fig. 3: the SPAR patch | drawn in the manuscript's LaTeX source, no script |
| emg2pose cohorts, alternatives, Supplementary Data 4-7 | the emg2pose campaigns, then `emg2pose/build_pose_paper_assets` and `figures/plot_meta_overview`, `figures/plot_meta_transfer`, `figures/plot_compact_controls`, `figures/plot_codec_diagnostics`, `figures/plot_pose_trajectories`, `figures/plot_streaming_cost`, `figures/plot_noise_robustness` |
| SPAR noise and electrode loss, Supplementary Data 8 | `hdemg/dpose_spar_corruption`, `hdemg/n9_exact_intervals` |
| SPAR ablation and held-out retrieval, Supplementary Data 9 | `hdemg/run_dpose_spar_ablation`, `hdemg/spar_ablation_statistics`, `hdemg/pose_recovery_heldout_statistics` |
| Supplementary Figs. S1–S3 | `figures/plot_supp_figures` (reads `data/supplementary/` only) |
| Supplementary Fig. S4 | `figures/make_token_history_supplement`, `hdemg/token_history_shuffle_roundness` |
| Supplementary Fig. S5 | `figures/plot_robustness_supplement` |

Supplementary Data 1-9 (`data/supplementary/`) hold the plotted values and summary results.
The figures are regenerated from the campaigns' outputs; checking a number does not require
rerunning anything.

## Scripts by folder

Every module, with what it does. Names in *italics* are imported by other scripts and not run
on their own. "emg2pose" in descriptions is the public dataset; older docstrings call it "Meta".

**`common/`**
- *`paths`*: where inputs are read and outputs written; set `MYOJI_HDEMG`, `MYOJI_EMG2POSE`, `MYOJI_ASSETS`.
- *`dpose_figure_statistics`*: exact paired sign-flip tests, intervals and Holm adjustment used across the HD-EMG analyses.
- *`figstyle`*, `simpetru*.mplstyle`: the paper's figure style.
- *`paper_plot_style`*, *`tokenizer_diagnostic_style`*: print sizes and colours for the Fig. 4 and supplementary builders.
- `switzer_font`: uses Switzer for the figures when it is installed; running it checks that the static regular and bold fonts are picked up.
- *`reporting_support`*: label checks for the emg2pose reports, so no participant is dropped silently.

**`pretraining/`**
- `train_mixed_sources`: pretrains the learned tokenizer on HD-EMG and emg2pose training windows.
- `audit_tokenizer_release`: freezes validation inputs and checks that the released tokenizer is deterministic, including after a reload.

**`hdemg/`**: HD-EMG campaigns
- `run_dpose_pilot`: participant-1 pilot that fixes the windows, preprocessing and splits the later campaigns reuse.
- `run_dpose_within`: personalized decoding on chronologically separated recording blocks (the shared split).
- `run_dpose_v20`, `run_dpose_v20_continuous`: learned tokens, and the matched continuous features, decoded by the spatial CNN + TCN.
- *`report_dpose_v20`*: rechecks the V20 predictions against the frozen decoder.
- `run_dpose_v20_one_euro`: One Euro smoothing of the V20 outputs, chosen on validation (Fig. 5).
- `run_dpose_spar_simple`: SPAR pilot on participant 1 (EMG → SPAR → codebook → decoder).
- `run_dpose_spar_full`, `run_dpose_spar_v64`: SPAR on nine participants with 1,024 and 64 codes.
- `run_dpose_spar_ablation`: SPAR with parts of the descriptor removed (Supplementary Data 9).
- `run_p1_continuous`: continuous test-block predictions for participant 1 (Fig. 5 traces).
- `run_dpose_tokenizer_reuse`: tokenizers trained only on donor participants, then personalized decoders.
- `run_dpose_noise` (with *`dpose_noise`*, *`report_dpose_noise`*): paired added-noise test of the learned route.
- `run_dpose_v20_noise_followup`, `collect_dpose_v20_noise_followup`: the same noise test for token V20, continuous V20 and the adapted CNN, and its independent audit.
- `run_dpose_dropout` (with *`dpose_dropout`*): paired electrode-dropout test.
- `dpose_spar_corruption`: the noise and dropout tests on the SPAR route (Supplementary Data 8).
- `dpose_spar_63_electrodes`: SPAR with the 63 electrodes the Cortex-M55 firmware handles in 20 ms.
- `dpose_raw_link_budget`: raw EMG from only as many electrodes as the assumed Bluetooth LE throughput carries, decoded by the adapted CNN.
- `dpose_spar_transfer_checks`, `dpose_spar_onset_calibration`: frozen SPAR pipelines on transfer checks, and with one calibration per recording.
- *`dpose_one_euro`*: the causal One Euro filter.
- `dpose_claim_statistics`: the paired statistics behind the main claims (Supplementary Data 2 and 3).
- `n9_exact_intervals`: exact intervals for the nine-participant contrasts outside the main family.
- `dpose_movement_analysis`, `dpose_absolute_change_analysis`: per-movement errors and token-minus-continuous changes (Fig. 4).
- `benchmark_dpose_workstation`, `benchmark_dpose_all_variants`, `benchmark_dpose_tbme2024_inference`: decoder latency and resource measurements (Fig. 4).
- `select_dpose_movement_cycles`: picks the movement cycles shown in the figures before any prediction is read.
- `replay_dpose_spar_figure_video`: replays the two nine-participant SPAR decoders on the selected movement cycles and measures their latency (Fig. 4).
- `build_token_umap_cache`: encodes one task into both token types, aligned to the hand kinematics (Fig. 6).
- `pose_recovery/` (`pose_recovery`, `pose_recovery_heldout`, `aggregate`): do tokens carry the hand's pose? Nearest-repetition retrieval, also on held-out data.
- `pose_recovery_heldout_statistics`, `spar_ablation_statistics`: statistics for held-out retrieval and the SPAR ablation (Supplementary Data 9).
- `direction_at_matched_closure`, `direction_continuous_spar`: opening versus closing at matched hand closure.
- `token_history_shuffle_roundness`: the shuffled-history control for the token-history figure (Supplementary Fig. S4).

**`receivers/`**: the other HD-EMG decoders
- `run_dpose_matched` (with *`dpose_matched_core`*, `report_dpose_matched`): TCN, Transformer and waveform CNN with matched capacity and training.
- `run_dpose_transformer`, `report_dpose_transformer`: the small token Transformer.
- `run_dpose_raulnet` (with *`dpose_raulnet_model`*, `report_dpose_raulnet`): the adapted raw-EMG CNN (RaulNet V19).
- `run_dpose_tbme2024` (with *`dpose_tbme2024`*): the TBME 2024 CNN retrained on the same windows.
- `run_dpose_llm_pilot`, `run_dpose_llm_full` (with *`dpose_llm_core`*, `hpc/dpose_llm_worker`): the frozen Qwen3 receiver, pilot then nine participants; the worker runs it on a cluster GPU.
- `tabpfn_pose/`: the TabPFN receiver; `run` (participant 1), `cohort` (nine), `analyze_cohort`, `benchmark_cohort` (latency), `finish_cohort` (starts the latency stage), *`_support`*.

**`emg2pose/`**
- `prepare_emg2pose_online`: metadata and byte-range index so recordings stream on demand.
- `compare_temporal_pose`: caches token features, fits the causal pose readouts, evaluates them.
- `run_meta_sessions`, `run_tokenizer_seeds`: transfer to unseen sessions, and independent tokenizer seeds.
- `run_paper_completion` (with *`paper_completion_core`*, `paper_completion_smoke`, `paper_completion_runtime`, `paper_completion_biocodec`, `export_pose_replay`, `report_paper_completion`): the emg2pose replication and the comparison of quantizers, with checks before costly jobs, runtime costs, the BioCodec adaptation, a replay window and the final report.
- `run_reviewer_campaign` (with `prepare_reviewer_features`): FSQ, quantized PCA and k-means controls.
- `build_pose_paper_assets`: builds the emg2pose figures and tables (Supplementary Data 4–7).
- `build_completion_results`: rechecks the paired statistics of the replication and draws its overview figures.

**`figures/`**
- `build_dpose_combined` (with *`build_dpose_all_variants`*, *`dpose_spar_figure_extension`*, *`dpose_tabpfn_figure_extension`*, *`dpose_tbme_figure_data`*): Fig. 4, errors and decoder costs on shared rows.
- `build_dpose_resource_overview`: payload budgets next to measured decoder costs, as a separate overview.
- `plot_fig3_signed_panel`: Fig. 4C, the signed effect of quantization per decoder.
- `plot_decoding_examples`: Fig. 5, decoded movements of one participant.
- `make_token_pose_recovery_figure`: Fig. 6 panels.
- `make_teaser`, `render_teaser_hands` (inside Blender), `export_token_explorer_context`: Fig. 1 panels.
- `plot_encoder_examples`, `plot_residual_traces`, `plot_residual_featuremap`, `plot_residual_example`, `build_tokenizer_mechanism`, `make_encoder_replacement_draft`, `export_encoder_examples_affinity`: Fig. 2 panels, and editable drawings of the encoder for the vector editor.
- `plot_supp_figures`: Supplementary Figs. S1–S3.
- `make_token_history_supplement`: Supplementary Fig. S4.
- `plot_robustness_supplement`: Supplementary Fig. S5.
- `plot_meta_overview`, `plot_meta_transfer`, `plot_compact_controls`, `plot_codec_diagnostics`, `plot_pose_trajectories`, `plot_streaming_cost`, `plot_noise_robustness`: emg2pose and robustness panels drawn by `build_pose_paper_assets`.

**`firmware/`**: see `embedded/*/README.md` for the full commands
- `raw_tokenizer_benchmark/`: `prepare` exports fixtures and PyTorch reference outputs; `host_check` checks the C tokenizers against them.
- `stm32n6_tokenizer/`, `stm32n6_optimized/`: `build` (never flashes), `verify_host` and `verify_fast` (host checks), `collect` (one board run), `freeze_runs` (picks the validated runs), `plot` (Fig. 7, from `data/firmware/`).
- `stm32n6_pose_validation/`: `validate` and `replay63` replay recordings through the C SPAR port and score the frozen decoders.

**`vendor/`**: third-party code, pinned by hash (see `THIRD_PARTY.md`): RaulNet V19, the One Euro filter, the TBME 2024 CNN.

## Steps that need the authors' intermediate outputs

A few steps check or read files that are not distributed:
- The trained weights. Steps that load them from `MYOJI_ASSETS` need weights you train:
  `pretraining/train_mixed_sources`, then `myoji.learned.export_checkpoint` with `calibration_samples=1000`
  for `learned/tokenizer.pt` (the pretraining checkpoint itself is `learned/tokenizer-full.pt`).
  The SPAR campaigns fit the codebooks; `spar/codebooks-1024.npz` holds them per participant as
  `subject_XX/{codebook,feature_mean,feature_scale,mask}`. The firmware fixtures
  (`firmware/raw_tokenizer_benchmark/prepare.py`) accept only the authors' tokenizer, by hash.
- The participant-level test predictions are not distributed. `hdemg/dpose_movement_analysis` checks the
  authors' prediction files by hash (`data/figure_inputs/dpose_movement_source_receipt.json`).
  `hdemg/dpose_absolute_change_analysis` checks the movement analysis computed from them. Run on the
  authors' run folders, both reproduce the published records.
- The decoded movement cycles of an earlier inference run are read by:
  - the SPAR latency benchmark used in Fig. 4;
  - the movement replay (`hdemg/replay_dpose_spar_figure_video`);
  - the EMG strip in Fig. 1 (`figures/make_teaser`, through `figures/export_token_explorer_context`).
  The same holds for `figures/build_tokenizer_mechanism` and `figures/dpose_spar_figure_extension`, which read
  participant 1's excerpt `results/dpose_spar_simple_full_v1/subject_01/figure-video-inputs.npz`.
- `emg2pose/build_pose_paper_assets` checks that the emg2pose millimetre conversion
  (`data/reference/meta_millimetre_conversion.json`) belongs to the authors'
  `paper_completion_v1/results.json`. It also reads a paired-participant audit of the matched
  receivers (`results/dpose_matched_v1/paired_participant_audit.json`) that no script writes.
- The hand renders in Fig. 1 (`figures/render_teaser_hands.py`) run inside Blender with a commercial hand
  model.
- `hdemg/run_dpose_v20_noise_followup` records the GPU model of the adapted-CNN evaluation from its job
  log. Locally, write that log yourself:
  `uv run python -m paper.receivers.run_dpose_raulnet evaluate --root paper/results/dpose_raulnet_v1 2>&1 | tee paper/results/dpose_raulnet_v1/evaluate-local.out`,
  then append `nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader`.
