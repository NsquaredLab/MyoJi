# Reproducing the paper

Everything that produced the numbers and figures in *EMG Tokenization Preserves 3D Hand-Pose
Decoding at a Fraction of the Bitrate* and its supplement. The reusable methods live in the
`myoji` library (`../src/myoji`); this folder holds the experiments that use it.

```
paper/
  common/      shared helpers: data paths, statistics, figure style
  pretraining/ learned-tokenizer pretraining and its release audit
  hdemg/       HD-EMG campaigns: SPAR and learned tokens, robustness, statistics, latency,
               and what tokens carry (Fig. 5)
  receivers/   the other HD-EMG receivers: matched TCN/Transformer/CNN, adapted and TBME CNNs,
               Qwen3, TabPFN
  emg2pose/    the emg2pose campaigns and their tables
  figures/     figure panels and their assembly
  firmware/    host checks and plots for the STM32N6 firmware and the C tokenizers
  vendor/      third-party model code the receivers pin by hash
  embedded/    the Cortex-M55 (STM32N6) SPAR firmware and its portable C tokenizers
  data/        shipped inputs: frozen window selections (reference/), Supplementary Data 1-9
               (supplementary/), and the board measurements behind Fig. 6 (firmware/)
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
| Fig. 3: errors, payload, latency; tokens minus continuous | the HD-EMG campaigns above, `hdemg/benchmark_dpose_workstation`, `hdemg/benchmark_dpose_all_variants`, `hdemg/dpose_movement_analysis` and `hdemg/dpose_absolute_change_analysis` (their frozen definitions are in `data/figure_inputs/`), then `figures/build_dpose_combined` and `figures/plot_fig3_signed_panel` |
| Supplementary Data 2 and 3, paired statistics | `hdemg/dpose_claim_statistics` (uses `common/dpose_figure_statistics`) |
| Fig. 4: decoded examples and per-movement errors | `hdemg/run_p1_continuous`, `hdemg/run_dpose_v20_one_euro`, `figures/plot_decoding_examples` |
| Fig. 5: token-history UMAP, pose recovery, held-out retrieval | `hdemg/select_dpose_movement_cycles`, `hdemg/replay_dpose_spar_figure_video`, `hdemg/build_token_umap_cache`, `hdemg/pose_recovery/`, `hdemg/pose_recovery_heldout_statistics`, `hdemg/direction_*`, `figures/make_token_pose_recovery_figure` |
| Fig. 6: Cortex-M55 timing | `firmware/stm32n6_optimized/plot.py` and `firmware/stm32n6_tokenizer/plot.py`; they redraw from `data/firmware/` without a board |
| Figs. 1 and 2: panels (assembled in a vector editor) | `figures/make_teaser`, `figures/plot_encoder_examples`, `figures/plot_residual_traces`, `figures/plot_residual_featuremap`, `figures/plot_residual_example`, `figures/build_tokenizer_mechanism` |
| emg2pose cohorts, alternatives, Supplementary Data 4-7 | the emg2pose campaigns, then `emg2pose/build_pose_paper_assets` and `figures/plot_meta_overview`, `figures/plot_meta_transfer`, `figures/plot_compact_controls`, `figures/plot_codec_diagnostics`, `figures/plot_pose_trajectories`, `figures/plot_streaming_cost`, `figures/plot_noise_robustness` |
| SPAR corruption, Supplementary Data 8 | `hdemg/dpose_spar_corruption`, `hdemg/n9_exact_intervals` |
| SPAR ablation and held-out retrieval, Supplementary Data 9 | `hdemg/run_dpose_spar_ablation`, `hdemg/spar_ablation_statistics`, `hdemg/pose_recovery_heldout_statistics` |
| Supplementary Figs. S1, S3, S4 | `figures/plot_supp_figures` (reads `data/supplementary/` only) |
| Supplementary Fig. S5 | `figures/make_token_history_supplement`, `hdemg/token_history_shuffle_roundness` |
| Supplementary Fig. S6 | `figures/plot_robustness_supplement` |

Supplementary Data 1-9 (`data/supplementary/`) hold the plotted values and summary results.
The figures are regenerated from the campaigns' outputs; checking a number does not require
rerunning anything.

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
  - the SPAR latency benchmark used in Fig. 3;
  - the movement replay behind the Fig. 5 video (`hdemg/replay_dpose_spar_figure_video`);
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
