# Reproducing the paper

Everything that produced the numbers and figures in *EMG Tokenization Preserves 3D Hand-Pose
Decoding at a Fraction of the Bitrate* and its supplement. The reusable methods live in the
`myoji` library (`../src/myoji`); this folder holds the experiments that use it.

```
paper/
  scripts/     one script per campaign, analysis or figure panel (flat; original names)
  embedded/    the Cortex-M55 (STM32N6) SPAR firmware and its portable C tokenizers
  data/        shipped inputs: frozen window selections (reference/), Supplementary Data 1-9
               (supplementary/), and the board measurements behind Fig. 6 (firmware/)
  results/     run folders written by the campaigns (not tracked)
  output/      figures and tables (not tracked)
```

"dpose" in script names is the 320-channel HD-EMG dataset of the paper (nine decoding
participants, 13 for tokenizer fitting).

## Setup

```bash
uv sync --group paper                      # from the repository root
```

| input | where it comes from | set |
|---|---|---|
| HD-EMG recordings `Sub{n}_emg.pkl`, `Sub{n}_kinematics.pkl` | the authors, on reasonable request | `MYOJI_HDEMG=<folder>` |
| emg2pose | public; `uv run python paper/scripts/prepare_emg2pose_online.py` builds the metadata and a byte-range index so recordings stream on demand | `MYOJI_EMG2POSE=<folder>` |
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
`run_dpose_pilot` → `run_dpose_within` → `run_dpose_v20` → `run_dpose_v20_continuous`.
The receivers compared with it build on `dpose_within_v1`: `run_dpose_matched`
(TCN, Transformer, waveform CNN), `run_dpose_transformer`, `run_dpose_raulnet` (adapted CNN),
`run_dpose_tbme2024` (retrained TBME CNN), `run_dpose_llm_pilot` → `run_dpose_llm_full` (Qwen3),
`tabpfn_pose/` (TabPFN).

**HD-EMG, SPAR**: `run_dpose_spar_simple` (participant 1) → `run_dpose_spar_full`
(nine participants; compares with the learned route, so it runs after `run_dpose_v20` and
`run_dpose_v20_continuous`) → `run_dpose_spar_v64` (64 codes), `run_dpose_spar_ablation`.
The window selection and protocol these start from are shipped in `data/reference/`.

**HD-EMG, robustness and transfer**: `dpose_spar_63_electrodes`, `dpose_raw_link_budget`,
`dpose_spar_transfer_checks`, `dpose_spar_onset_calibration`, `run_dpose_noise` →
`run_dpose_v20_noise_followup` → `run_dpose_dropout`, `dpose_spar_corruption`,
`run_dpose_tokenizer_reuse` (donor-only tokenizers; `--recipe data/reference/tokenizer_recipe.json`).

**emg2pose**: `prepare_emg2pose_online` → `audit_tokenizer_release prepare` (frozen training
windows) → `train_mixed_sources` (tokenizer pretraining) → `compare_temporal_pose cache`
(`results/clean_temporal_v1/full.h5`) → `run_meta_sessions` → `run_tokenizer_seeds` →
`compare_temporal_pose fit` and `evaluate` → `run_paper_completion`
(with `paper_completion_smoke`, `export_pose_replay`, `paper_completion_biocodec`,
`paper_completion_runtime`, `report_paper_completion`) and `run_reviewer_campaign`
(`prepare_reviewer_features`: FSQ, quantized PCA, k-means).

**Firmware**: see `embedded/stm32n6_optimized/README.md`; the fixtures come from
`scripts/raw_tokenizer_benchmark/prepare.py` and `host_check.py`.

## Where each result comes from

| result | scripts |
|---|---|
| Fig. 3: errors, payload, latency; tokens minus continuous | the HD-EMG campaigns above, `benchmark_dpose_workstation`, `benchmark_dpose_all_variants`, `dpose_movement_analysis` and `dpose_absolute_change_analysis` (their frozen definitions are in `data/figure_inputs/`), then `build_dpose_combined` and `plot_fig3_signed_panel` |
| Supplementary Data 2 and 3, paired statistics | `dpose_claim_statistics` (uses `dpose_figure_statistics`) |
| Fig. 4: decoded examples and per-movement errors | `run_p1_continuous`, `run_dpose_v20_one_euro`, `plot_decoding_examples` |
| Fig. 5: token-history UMAP, pose recovery, held-out retrieval | `select_dpose_movement_cycles`, `replay_dpose_spar_figure_video`, `build_token_umap_cache`, `pose_recovery/`, `pose_recovery_heldout_statistics`, `direction_*`, `make_token_pose_recovery_figure` |
| Fig. 6: Cortex-M55 timing | `stm32n6_optimized/plot.py` and `stm32n6_tokenizer/plot.py`; they redraw from `data/firmware/` without a board |
| Figs. 1 and 2: panels (assembled in a vector editor) | `make_teaser`, `plot_encoder_examples`, `plot_residual_traces`, `plot_residual_featuremap`, `plot_residual_example`, `build_tokenizer_mechanism` |
| emg2pose cohorts, alternatives, Supplementary Data 4-7 | the emg2pose campaigns, then `build_pose_paper_assets` and `plot_meta_overview`, `plot_meta_transfer`, `plot_compact_controls`, `plot_codec_diagnostics`, `plot_pose_trajectories`, `plot_streaming_cost`, `plot_noise_robustness` |
| SPAR corruption, Supplementary Data 8 | `dpose_spar_corruption`, `n9_exact_intervals` |
| SPAR ablation and held-out retrieval, Supplementary Data 9 | `run_dpose_spar_ablation`, `spar_ablation_statistics`, `pose_recovery_heldout_statistics` |
| Supplementary Figs. S1, S3, S4 | `plot_supp_figures` (reads `data/supplementary/` only) |
| Supplementary Fig. S5 | `make_token_history_supplement`, `token_history_shuffle_roundness` |
| Supplementary Fig. S6 | `plot_robustness_supplement` |

Supplementary Data 1-9 (`data/supplementary/`) hold the plotted values and summary results.
The figures are regenerated from the campaigns' outputs; checking a number does not require
rerunning anything.

## Steps that need the authors' intermediate outputs

A few steps check or read files that are not distributed:
- The trained weights. Steps that load them from `MYOJI_ASSETS` need weights you train:
  `train_mixed_sources`, then `myoji.learned.export_checkpoint` with `calibration_samples=1000`
  for `learned/tokenizer.pt` (the pretraining checkpoint itself is `learned/tokenizer-full.pt`).
  The SPAR campaigns fit the codebooks; `spar/codebooks-1024.npz` holds them per participant as
  `subject_XX/{codebook,feature_mean,feature_scale,mask}`. The firmware fixtures
  (`raw_tokenizer_benchmark/prepare.py`) accept only the authors' tokenizer, by hash.
- The participant-level test predictions are not distributed. `dpose_movement_analysis` checks the
  authors' prediction files by hash (`data/figure_inputs/dpose_movement_source_receipt.json`).
  `dpose_absolute_change_analysis` checks the movement analysis computed from them. Run on the
  authors' run folders, both reproduce the published records.
- The decoded movement cycles of an earlier inference run are read by:
  - the SPAR latency benchmark used in Fig. 3;
  - the movement replay behind the Fig. 5 video (`replay_dpose_spar_figure_video`);
  - the EMG strip in Fig. 1 (`make_teaser`, through `export_token_explorer_context`).
  The same holds for `build_tokenizer_mechanism` and `dpose_spar_figure_extension`, which read
  participant 1's excerpt `results/dpose_spar_simple_full_v1/subject_01/figure-video-inputs.npz`.
- `build_pose_paper_assets` checks that the emg2pose millimetre conversion
  (`data/reference/meta_millimetre_conversion.json`) belongs to the authors'
  `paper_completion_v1/results.json`. It also reads a paired-participant audit of the matched
  receivers (`results/dpose_matched_v1/paired_participant_audit.json`) that no script writes.
- The hand renders in Fig. 1 (`render_teaser_hands.py`) run inside Blender with a commercial hand
  model.
- `run_dpose_v20_noise_followup` records the GPU model of the adapted-CNN evaluation from its job
  log. Locally, write that log yourself:
  `python run_dpose_raulnet.py evaluate --root ../results/dpose_raulnet_v1 2>&1 | tee ../results/dpose_raulnet_v1/evaluate-local.out`,
  then append `nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader`.
