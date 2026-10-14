"""Combine audited DPose pose errors and decoder costs on shared model rows.

Run with NumPy and Matplotlib. The two source figure builders are imported for
their data validation only; their existing figures are not regenerated.
"""

from __future__ import annotations

import csv
import copy
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
import numpy as np

import build_dpose_all_variants as accuracy
import build_dpose_resource_overview as resources
import dpose_figure_statistics as stats
import dpose_tbme_figure_data as tbme
import figstyle as sf
from paper_plot_style import MARKERS
import paths
import switzer_font


ROOT = resources.ROOT
OUT = resources.OUT
PDFS = resources.PDFS
STEM = "dpose-error-and-resources"
SUPPLEMENT_STEM = "dpose-early-baselines"
IEEE_SIZE_SOURCE = "https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/"
INPUTS = paths.DATA / "figure_inputs"  # frozen analysis records the figure builds on

FIGURE_WIDTH_IN, _ = sf.figsize(cols=2, depth=5 / 8)
FIGURE_HEIGHT_IN = 6.2
SUPPLEMENT_HEIGHT_IN = 2.4
# The early-baseline supplement copies the accepted (Affinity-finished) main
# figure: the same row pitch, header offsets, error axis and black labels.
# Offsets are inches from the top edge, measured on paper/ieee/figures/dpose.pdf;
# the row pitch is inches per data unit.
SUPPLEMENT_TOP_IN = {"legend": 0.075, "key": 0.29, "header": 0.433, "axes": 0.465}
SUPPLEMENT_ROW_IN = 0.165
SUPPLEMENT_ERROR_LIMITS = (4.5, 25.5)
SUPPLEMENT_ERROR_TICKS = (5, 10, 15, 20, 25)
SUPPLEMENT_LATENCY_LIMITS = (0.3, 300)
# Any tick below 1 ms collides with the parameter label "1000"; the 0.86 ms
# marker therefore overlaps the left spine end by less than its own radius.
SUPPLEMENT_LATENCY_TICKS = [1, 10, 100]
SUPPLEMENT_INK = "black"
LITERATURE_RECORD = INPUTS / "dpose_model_literature.json"
LITERATURE_BIB = ROOT / "paper/dpose-model-references.bib"
STATISTICS_BIB = ROOT / "paper/dpose-statistical-references.bib"
LITERATURE = json.loads(LITERATURE_RECORD.read_text())
CNN_COMPARISON_RECORD = INPUTS / "dpose_cnn_maxT_analysis_protocol.json"
CNN_COMPARISON_PROTOCOL = json.loads(CNN_COMPARISON_RECORD.read_text())
MOVEMENT_RECORD = paths.RESULTS / "dpose_movement_analysis.json"  # dpose_movement_analysis.py
MOVEMENT_ANALYSIS = json.loads(MOVEMENT_RECORD.read_text())
assert MOVEMENT_ANALYSIS["complete"] and not MOVEMENT_ANALYSIS["joint_adjusted_p_changes"]
ABSOLUTE_RECORD = paths.RESULTS / "dpose_absolute_change_analysis.json"  # dpose_absolute_change_analysis.py
ABSOLUTE_ANALYSIS = json.loads(ABSOLUTE_RECORD.read_text())
assert ABSOLUTE_ANALYSIS["complete"] and ABSOLUTE_ANALYSIS["family_size"] == 54
STATISTICAL_FAMILY_LABELS = {
    "common": "Matched CNN, TCN and Transformer comparison",
    "native": "CNN and proposed decoder comparison",
    "early": "Smaller decoder baselines",
    "qwen": LITERATURE["figure_labels"]["qwen_group"],
}
QWEN_FULL = "dpose_llm_full_v1"
QWEN_PILOT = "dpose_llm_pilot_v1"
STATISTICAL_FAMILIES = [dict(g, label=STATISTICAL_FAMILY_LABELS[g["id"]], variants=[
    (QWEN_FULL if campaign == QWEN_PILOT else campaign, arm)
    for campaign, arm in g["variants"]])
    for g in resources.EXPERIMENT_GROUPS]
FAMILY_BY_ID = {group["id"]: group for group in STATISTICAL_FAMILIES}
# The historical training-group definitions supply model membership only.
# Current inference uses three CNN references and the declared direct comparisons globally.
ADAPTED_CNN_LABEL = "CNN (Sîmpetru 2023 and 2024, adapted)"
MATCHED_CNN_KEY = ("dpose_matched_v1", "v19")
L1_CNN_KEY = ("dpose_raulnet_v1", "raulnet_v19_q32")
GROUPS = [
    {"id": "cnn_reference", "label": ADAPTED_CNN_LABEL, "variants": [L1_CNN_KEY]},
    {"id": "matched_decoders", "label": "Matched custom decoders",
     "variants": FAMILY_BY_ID["common"]["variants"][1:]},
    dict(FAMILY_BY_ID["qwen"]),
    {"id": "proposed", "label": "Proposed spatial CNN + TCN",
     "variants": FAMILY_BY_ID["native"]["variants"][1:]},
]
# The adapted CNN row is the reference every early-baseline branch tests against.
EARLY_GROUPS = [GROUPS[0], dict(FAMILY_BY_ID["early"])]
UNPLOTTED_GROUPS = [
    {"id": "additional_cnn_references", "label": "Additional CNN references (full statistical table)",
     "variants": [MATCHED_CNN_KEY, tbme.KEY]},
]
DISPLAY_GROUP_BY_KEY = {key: group for group in [*GROUPS, *EARLY_GROUPS]
                        for key in group["variants"]}
DATA_GROUP_BY_KEY = DISPLAY_GROUP_BY_KEY | {
    key: group for group in UNPLOTTED_GROUPS for key in group["variants"]}
DISPLAY_REFERENCES = [L1_CNN_KEY]
REPRESENTATION_NAMES = {"tokens": "EMG tokens", "continuous": "encoder features", "raw": "raw EMG"}
UNIT_CONFIRMATION_RECORD = INPUTS / "dpose_published_reference_2024.json"
UNIT_CONFIRMATION = json.loads(UNIT_CONFIRMATION_RECORD.read_text())["unit_confirmation"]
BASELINE_RECORD = ROOT / "results" / tbme.KEY[0] / "results.json"
BASELINE_REFERENCE = {
    "campaign": tbme.KEY[0], "arm": tbme.KEY[1],
    "label": "Sîmpetru et al. (2024, TBME); Retrained",
    "short_label": "Retrained TBME CNN",
    "value_mm": json.loads(BASELINE_RECORD.read_text())["summary"]["mean"],
    "source": str(BASELINE_RECORD.relative_to(ROOT)),
    "relationship": "Retrained TBME baseline shown only as a dashed error reference: nine participant test means, each averaging three seeds, on the current DPose split and targets.",
}
REFERENCE_COLOR = "#707070"
REFERENCE_DASHES = (0, (4, 3))
# Two budgets the measured values are read against, drawn as dashed rules.
# The radio figure is the Bluetooth SIG approximate maximum application rate
# for LE 2M already cited in the text; the deadline is one tokenizer hop.
BLE_LIMIT_MB_S = 1.4e6 / 8 / 1e6  # 1.4 Mbit/s -> 0.175 decimal MB/s
HOP_DEADLINE_MS = 20.0
GRID_COLOR = "#DEDEDE"
GRID_WIDTH = 0.55
COLORS = {"tokens": sf.REDS[2], "continuous": sf.BLUES[1], "raw": sf.GREYS[0]}
PARTICIPANT_COLOR = sf.GREYS[1]
METRIC_COLUMNS = {"error": (0.285, 0.29), "size": (0.675, 0.17),
                  "latency": (0.865, 0.12)}
METRIC_LABELS = {"error": "3D Error (mm)", "size": "Parameters (M)",
                 "latency": "Latency (ms)"}
MODEL_PANEL_RECT = (0.36, 0.50)
# Three decades only: Switzer is wider than the STIX stack this figure used
# before, and five tick labels collided on this narrow column.
SIZE_TICKS = [0.1, 10, 1000]
LATENCY_SCALE = 1.0  # Stored timing values are always milliseconds.
LATENCY_LIMITS = (0.3, 110)
LATENCY_TICKS = [1, 3, 10, 100]
ERROR_LIMITS = (7.5, 26)
ERROR_TICKS = (10, 15, 20, 25)
PAYLOAD_FORMATS = {"tokens": "uint16", "raw": "int16", "continuous": "FP32"}
PAYLOAD_TICKS = (0.01, 0.1, 1, 10, 100)
BOTTOM_PANELS = {"payload": (0.17, 0.09, 0.27, 0.168),
                 "paired_difference": (0.67, 0.09, 0.27, 0.168)}
PAIRED_LIMITS = (0, 4)
PAIRED_TICKS = (0, 1, 2, 3, 4)
REPRESENTATION_PAIRS = [
    ("TCN", ("dpose_matched_v1", "tcn_tokens"), ("dpose_matched_v1", "tcn_continuous")),
    ("Transformer", ("dpose_matched_v1", "transformer_tokens"),
     ("dpose_matched_v1", "transformer_continuous")),
    ("Qwen3", (QWEN_FULL, "tokens_pretrained"), (QWEN_FULL, "continuous_pretrained")),
    ("Proposed", ("dpose_v20_v1", "tokens"), ("dpose_v20_continuous_v1", "continuous")),
]
REPRESENTATION_TABLE_LABELS = {"TCN": "TCN (matched)", "Transformer": "Transformer (matched)",
                             "Qwen3": "Qwen3 (pretrained)", "Proposed": "Proposed"}
LABEL_RIGHT = 0.265
MARK_Y_OFFSET = 0.14
BRACKET_X, BRACKET_STEP, BRACKET_CAP = 0.624, 0.072, 0.006


def load_data():
    for path, expected in ABSOLUTE_ANALYSIS["sources"].items():
        assert resources.digest(ROOT / path) == expected, f"Absolute-change source changed: {path}"
    for path, expected in MOVEMENT_ANALYSIS["sources"].items():
        assert resources.digest(ROOT / path) == expected, f"Movement source changed: {path}"
    resource_rows, formats, benchmark = resources.load_data()
    formats = [row for row in formats if row["encoding"] == PAYLOAD_FORMATS[row["representation"]]]
    assert len(formats) == 3 and {row["representation"] for row in formats} == set(PAYLOAD_FORMATS)
    panels = accuracy.build_panels(full_qwen=True)
    error_map = {
        (row["campaign"], row["arm"]): dict(row, panel=panel["id"])
        for panel in panels for row in panel["rows"]
    }
    assert set(error_map) == {
        (QWEN_FULL if r["campaign"] == QWEN_PILOT else r["campaign"], r["arm"])
        for r in resource_rows}
    tbme_error, tbme_error_sources = tbme.load_accuracy(ROOT)
    assert BASELINE_REFERENCE["value_mm"] == tbme_error["mean"]
    resource_rows, benchmark, tbme_resource_sources = tbme.load_resources(ROOT, benchmark, resource_rows)
    error_map[tbme.KEY] = tbme_error
    accuracy.SOURCES.update(tbme_error_sources | tbme_resource_sources)
    # Keep the actual cost-checkpoint provenance. The full campaign reused the
    # exact model construction and external model revision benchmarked earlier.
    protocols = {}
    for campaign in (QWEN_FULL, QWEN_PILOT):
        path = ROOT / "results" / campaign / "protocol.json"
        accuracy.SOURCES[str(path.relative_to(ROOT))] = resources.digest(path)
        protocols[campaign] = json.loads(path.read_text())
    full_protocol, pilot_protocol = protocols[QWEN_FULL], protocols[QWEN_PILOT]
    for field in ("model", "revision", "model_sha256"):
        assert full_protocol[field] == pilot_protocol[field]
    assert full_protocol["steps"] == 15000 and full_protocol["validation_opportunities"] == 150
    rows = []
    for resource in resource_rows:
        campaign = QWEN_FULL if resource["campaign"] == QWEN_PILOT else resource["campaign"]
        error = error_map[(campaign, resource["arm"])]
        assert error["representation"] == resource["representation"]
        assert error["panel"] == resource["panel"]
        row = dict(
            resource,
            campaign=campaign,
            resource_campaign=resource["campaign"],
            error_mean=error["mean"],
            error_evaluation=error["evaluation"],
            error_n_participants=error["n_participants"],
            error_n_seeds=error["n_seeds"],
            error_participant_means=error["person_means"],
            error_participant_seed_values=error["person_seed_errors"],
            error_constant_pose=error["constant_pose_error"],
            error_scope=error["source_scope"],
        )
        if campaign == QWEN_FULL:
            assert resource["model_source_sha256"] == full_protocol["source_sha256"]["scripts/dpose_llm_core.py"]
            fit_path = ROOT / "results" / campaign / f"subject_01/fits/{row['arm']}_260911/fit.json"
            fit = json.loads(fit_path.read_text())
            assert fit["trainable_parameters"] + fit["frozen_parameters"] == resource["parameter_count"]
            assert resource["receiver_input_shape"][1:3] == [320, 100]
            row["resource_architecture_matches_full_campaign"] = True
            row["resource_checkpoint_scope"] = "Earlier pilot checkpoint of the identical architecture; standardized FP32 cost benchmark, not a timing of the final BF16 cohort."
            row["error_precision"] = full_protocol["precision"]
        row["source_label"] = row["label"]
        row["source_experiment_group"] = row["experiment_group"]
        row["source_experiment_group_id"] = row["experiment_group_id"]
        row["training_comparison_id"] = row["experiment_group_id"]
        row["statistical_family_id"] = "cnn_references"
        row["statistical_family"] = "Each CNN reference versus all alternatives"
        display_group = DATA_GROUP_BY_KEY[(row["campaign"], row["arm"])]
        row["experiment_group_id"] = display_group["id"]
        row["experiment_group"] = display_group["label"]
        row["label"] = model_label(row)
        row["plot_label"] = plot_model_label(row)
        row["error_units"] = "mm"
        assert isinstance(row["parameter_count"], int) and row["parameter_count"] > 0
        row["model_parameters_million"] = row["parameter_count"] / 1e6
        row["model_literature"] = (dict(citation="Sîmpetru et al. (2024, TBME)",
            relationship="Archived architecture retrained on the current DPose split and targets.",
            protocol="paper/DPOSE_TBME2024_PROTOCOL.md") if row["family"] == "TBME" else
            LITERATURE["relationships"][row["family"]])
        if row["family"] == "TBME":
            row["error_precision"] = "FP16 autocast; FP32 target denormalization"
        rows.append(row)
    assert sum(r["error_n_participants"] * r["error_n_seeds"]
               for r in rows if r["error_evaluation"] == "chronological_test") == 459
    assert all(r["error_evaluation"] == "chronological_test" for r in rows)
    return rows, formats, benchmark


def model_label(row):
    if row["family"] == "TBME":
        return "CNN (Sîmpetru 2024, TBME; retrained)"
    if row["family"] == "V19":
        return f"{ADAPTED_CNN_LABEL}, {plot_model_label(row)}; raw EMG and 20-Hz LP"
    if row["family"] == "V20":
        return f"Proposed spatial CNN + TCN + {REPRESENTATION_NAMES[row['representation']]}"
    return f"{plot_model_label(row)} + {REPRESENTATION_NAMES[row['representation']]}"


def comparison_label(row):
    """Identify models unambiguously outside the grouped figure layout."""
    if row["family"] == "V19":
        training = "matched" if row["campaign"] == "dpose_matched_v1" else "L1 + SWA"
        return f"CNN (Sîmpetru 2023 and 2024, adapted; {training})"
    label = model_label(row)
    if row["experiment_group_id"] == "matched_decoders":
        return f"{label} (matched)"
    if row["experiment_group_id"] == "early":
        return f"{label} (smaller baseline)"
    return label


def plot_model_label(row):
    """Short row labels; colors and shapes identify the input representation."""
    if row["family"] == "TBME":
        return "Retrained"
    if row["family"] == "V19":
        return "Matched training" if row["campaign"] == "dpose_matched_v1" else "L1 + weight averaging"
    if row["family"] == "TabPFN":
        return "SPAR tokens" if row["representation"] == "tokens" else "SPAR features"
    if row["family"] in {"V20", "SPAR"}:
        prefix = "SPAR" if row["family"] == "SPAR" else "Learned"
        return prefix + (" tokens" if row["representation"] == "tokens" else " features")
    if row["experiment_group_id"] == "early":
        return "Causal EMG Transformer" if row["campaign"] == "dpose_transformer_v1" else (
            "End-to-end CNN + TCN" if row["arm"] == "raw" else "Causal TCN")
    if row["arm"] == "tokens_bypass":
        return "Direct pose readout"
    if row["experiment_group_id"] == "qwen":
        return "Qwen3 (random weights)" if row["arm"] == "tokens_random" else "Qwen3 (pretrained)"
    return {"TCN": "Causal TCN", "Transformer": "Causal EMG Transformer"}[row["family"]]


def size_label(row):
    return f"{row['model_parameters_million']:.3f}"


def selected_rows(rows, groups):
    by_key = {(r["campaign"], r["arm"]): r for r in rows}
    return [by_key[key] for group in groups for key in group["variants"]]


def reference_statistics(rows):
    """Use the reviewed 54-test correction while preserving its frozen 51-test input."""
    # Normalize integer participant keys to the serialized source-record schema.
    analysis = json.loads(json.dumps(stats.analyze_references(rows, CNN_COMPARISON_PROTOCOL)))
    old_pairs = analysis["groups"][0]["comparisons"]
    pairs = copy.deepcopy(ABSOLUTE_ANALYSIS["comparisons"])
    assert len(old_pairs) == 51 and len(pairs) == 54
    for old, current in zip(old_pairs, pairs[:51]):
        for field in ["alternative", "reference", "participant_differences", "p_raw",
                      "effect_alternative_minus_reference_mm", "effect_ci95_pointwise_mm"]:
            assert old[field] == current[field], (field, old["alternative"])
        assert old["p_adjusted"] == current["p_adjusted_previous_51"]
    group = analysis["groups"][0]
    group.update(family_size=54, comparisons=pairs,
                 label="CNN references, signed representation gaps and absolute error changes")
    analysis.update(family_size=54, comparison_protocol=ABSOLUTE_ANALYSIS["protocol"],
        retained_51_protocol=CNN_COMPARISON_PROTOCOL,
        minimum_observed_adjusted_p=min(p["p_adjusted"] for p in pairs),
        significant_comparisons=ABSOLUTE_ANALYSIS["significant_comparisons"],
        maxT_reference_distribution=ABSOLUTE_ANALYSIS["maxT_reference_distribution"],
        previous_adjusted_p_changes=ABSOLUTE_ANALYSIS["previous_adjusted_p_changes"],
        previous_significance_changes=ABSOLUTE_ANALYSIS["previous_significance_changes"],
        absolute_change_source=str(ABSOLUTE_RECORD.relative_to(ROOT)))
    analysis["method"] += (
        " The current family retains all 51 previous contrasts and adds three absolute-error-change "
        "contrasts. Within each participant and task, average the two-window errors over three seeds, "
        "take the absolute token-minus-feature difference, then average the eight task changes. "
        "Other-minus-proposed participant contrasts test whether proposed changes error less. "
        "Both improvements and degradations count as changes; this is not a variance or overall "
        "accuracy test. The endpoint was selected after inspecting signed results. The 54-test "
        "correction does not account for the entire adaptive endpoint-selection history.")
    analysis["excluded"]["alternative_vs_alternative"] = (
        "Four within-decoder representation tests, three signed-gap comparisons and three "
        "absolute-error-change comparisons are included; other alternative comparisons are excluded.")
    return analysis


def decoder_layout(rows, groups):
    keys = {(r["campaign"], r["arm"]) for r in rows}
    positions, sections, cursor = {}, [], 0.0
    for group in groups:
        sections.append({"id": group["id"], "label": group["label"], "y": -cursor})
        cursor += 1.05
        for key in group["variants"]:
            assert key in keys
            positions[key] = -cursor
            cursor += 1
        cursor += 0.45
    assert set(positions) == keys
    return positions, sections


def apply_style():
    plt.style.use(["science", "ieee", "simpetru-large"])
    # Tight cropping changes the physical IEEE width. Keep the full artboard;
    # all text and marks are checked to fit inside it before publication.
    plt.rcParams.update({"savefig.bbox": None, "savefig.pad_inches": 0,
                         "legend.fontsize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9})
    # Switzer, to match the hand-drawn Figure 1. Applied after the shared
    # style, which is used by other papers and sets a serif stack.
    plt.rcParams.update(switzer_font.rcparams())


def significance_brackets(rows, groups, analysis):
    """CNN comparison stems; direct representation tests are displayed in C."""
    positions, _ = decoder_layout(rows, groups)
    by_key = {(row["campaign"], row["arm"]): row for row in rows}
    annotations = []
    references = DISPLAY_REFERENCES
    for family in analysis["groups"]:
        for pair in family["comparisons"]:
            if tuple(pair["alternative"]) not in positions:
                continue
            reference = tuple(pair["reference"])
            if "contrast_id" in pair:
                continue
            if reference not in references:
                continue
            column = references.index(reference)
            annotations.append({
                "id": f"significance-{family['id']}-{pair['i']}-{pair['j']}",
                "group": family["id"], "statistical_family_id": family["id"],
                "a": pair["a"], "b": pair["b"], "lane": column, "kind": "cnn_reference",
                "p_adjusted": pair["p_adjusted"], "significant": pair["significant"],
                "symbol": significance_symbol(pair["p_adjusted"]) or "n.s.",
                "x_figure": BRACKET_X + BRACKET_STEP * column,
                "y_a": positions[tuple(pair["alternative"])] + MARK_Y_OFFSET,
                "y_b": positions[reference] + MARK_Y_OFFSET if reference in positions else 0.4,
                "reference_in_view": reference in positions,
            })
    return annotations


def significance_symbol(adjusted_p):
    """Three conventional tiers, always using the multiplicity-adjusted p-value."""
    if not np.isfinite(adjusted_p) or not 0 <= adjusted_p <= 1:
        raise ValueError("Expected a finite adjusted p-value between zero and one")
    return "***" if adjusted_p < 0.001 else "**" if adjusted_p < 0.01 else "*" if adjusted_p < 0.05 else ""


def draw_significance_label(ax, x, y, bracket, transform, color):
    """Center the visible asterisk glyphs on the bracket, rather than the font box."""
    significant = bracket["significant"]
    label = ax.text(x, y, bracket["symbol"], transform=transform,
                    ha="left", va="center", clip_on=False,
                    fontsize=11 if significant else 9, color=color,
                    gid=bracket["id"] + ("-star" if significant else "-ns"))
    if significant:
        ink = TextPath((0, 0), bracket["symbol"],
                       prop=label.get_fontproperties(), usetex=False).get_extents()
        # TextPath coordinates are points relative to the baseline. Applying the
        # offset in physical units keeps the same centering in PNG, SVG and PDF.
        offset = ScaledTranslation(0, -(ink.y0 + ink.y1) / (2 * 72),
                                   ax.figure.dpi_scale_trans)
        label.set_verticalalignment("baseline")
        label.set_transform(transform + offset)
    return label


def draw_models(fig, rows, groups, analysis, *, supplement=False):
    positions, sections = decoder_layout(rows, groups)
    ymin = min(positions.values()) - 0.9
    if supplement:
        figure_height = fig.get_figheight()
        height = (0.8 - ymin) * SUPPLEMENT_ROW_IN / figure_height
        bottom = 1 - SUPPLEMENT_TOP_IN["axes"] / figure_height - height
    else:
        bottom, height = MODEL_PANEL_RECT
    error_limits, error_ticks = ((SUPPLEMENT_ERROR_LIMITS, SUPPLEMENT_ERROR_TICKS)
                                 if supplement else (ERROR_LIMITS, ERROR_TICKS))
    latency_limits, latency_ticks = ((SUPPLEMENT_LATENCY_LIMITS, SUPPLEMENT_LATENCY_TICKS)
                                     if supplement else (LATENCY_LIMITS, LATENCY_TICKS))
    reference_ink = SUPPLEMENT_INK if supplement else REFERENCE_COLOR
    axes = [fig.add_axes([left, bottom, width, height], label=metric)
            for metric, (left, width) in METRIC_COLUMNS.items()]
    error_ax, size_ax, latency_ax = axes
    for ax in axes:
        ax.set_xlabel(METRIC_LABELS[ax.get_label()], labelpad=5, fontsize=9.5)
    for ax in axes:
        ax.set_ylim(ymin, 0.8)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=9, length=3)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)
    error_ax.set_xlim(error_limits)
    error_ax.set_xticks(error_ticks)
    error_ax.xaxis.set_minor_locator(NullLocator())
    error_ax.axvline(BASELINE_REFERENCE["value_mm"], color=reference_ink,
                    linestyle=REFERENCE_DASHES, linewidth=0.75, zorder=0,
                    gid="tbme-reference")
    # Named at the rule rather than in the legend, like the payload and hop rules.
    # The supplement places both names beside their rules, as in the accepted main figure.
    beside = dict(xytext=(3, 0), ha="left", va="top", rotation=0) if supplement else None
    error_ax.annotate(BASELINE_REFERENCE["short_label"], (BASELINE_REFERENCE["value_mm"], 0.8),
                      textcoords="offset points", fontsize=8.5, color=reference_ink,
                      annotation_clip=False,
                      **(beside or dict(xytext=(0, 3), ha="center", va="bottom")))
    deadline = HOP_DEADLINE_MS * LATENCY_SCALE
    if latency_limits[0] < deadline < latency_limits[1]:
        latency_ax.axvline(deadline, color=reference_ink, linestyle=REFERENCE_DASHES,
                           linewidth=0.75, zorder=0, gid="hop-deadline")
        latency_ax.annotate("20 ms", (deadline, 0.8), textcoords="offset points",
                            fontsize=8.5, color=reference_ink, annotation_clip=False,
                            **(beside or dict(xytext=(2, 0), rotation=90, ha="left", va="top")))
    for ax, limits, ticks in [(size_ax, (0.1, 1000), SIZE_TICKS),
                              (latency_ax, latency_limits, latency_ticks)]:
        ax.set_xscale("log")
        ax.set_xlim(limits)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    transform = blended_transform_factory(fig.transFigure, error_ax.transData)
    for group, section in zip(groups, sections):
        y = section["y"]
        fig.text(0.025, y, group["label"], transform=transform, va="center",
                 ha="left", color=SUPPLEMENT_INK if supplement else "#505050",
                 fontsize=9.5, weight="bold",
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0})

    for row in rows:
        y = positions[(row["campaign"], row["arm"])]
        representation = row["representation"]
        pilot = row["error_evaluation"] == "best_validation_only"
        fig.text(LABEL_RIGHT, y + MARK_Y_OFFSET, plot_model_label(row), transform=transform,
                 ha="right", va="center", fontsize=9.5)
        mark_y = y + MARK_Y_OFFSET
        if not pilot:
            values = list(row["error_participant_means"].values())
            assert len(values) == 9 and all(error_limits[0] <= value <= error_limits[1] for value in values)
            error_ax.scatter(values, mark_y + np.linspace(-0.08, 0.08, len(values)),
                             s=15, color=PARTICIPANT_COLOR, edgecolors="white",
                             linewidths=0.35, zorder=2)
        assert error_limits[0] <= row["error_mean"] <= error_limits[1]
        error_ax.scatter(row["error_mean"], mark_y, s=48, marker=MARKERS[representation],
                         facecolor="white" if pilot else COLORS[representation],
                         edgecolor=COLORS[representation] if pilot else "white",
                         linewidth=1.1 if pilot else 0.6, zorder=3)
        for ax, field in [(size_ax, "model_parameters_million"), (latency_ax, "latency_mean_ms")]:
            value = row[field] * (LATENCY_SCALE if field == "latency_mean_ms" else 1)
            ax.scatter(value, mark_y, s=48, marker=MARKERS[representation],
                       facecolor=COLORS[representation], edgecolor="white", linewidth=0.6, zorder=3)
        latency_ax.errorbar(row["latency_mean_ms"] * LATENCY_SCALE, mark_y,
                            xerr=row["latency_sd_ms"] * LATENCY_SCALE,
                            fmt="none", ecolor=COLORS[representation], elinewidth=1.0,
                            capsize=2.5, capthick=1.0, zorder=2)
    brackets = significance_brackets(rows, groups, analysis)
    for lane in range(len(DISPLAY_REFERENCES)):
        selected = [b for b in brackets if b["lane"] == lane]
        if not selected:
            continue
        x, y_ref = selected[0]["x_figure"], selected[0]["y_b"]
        y_low = min(b["y_a"] for b in selected)
        y_high = max(y_ref, max(b["y_a"] for b in selected))
        error_ax.add_line(Line2D([x, x], [y_low, y_high],
            transform=transform, clip_on=False, color="#707070", linewidth=0.7,
            gid=f"comparison-stem-{lane}"))
        error_ax.add_line(Line2D([x - BRACKET_CAP, x], [y_ref, y_ref],
            transform=transform, clip_on=False, color="#707070", linewidth=0.7,
            gid=f"comparison-reference-cap-{lane}"))
    for bracket in brackets:
        x, y = bracket["x_figure"], bracket["y_a"]
        if bracket["kind"] == "direct":
            error_ax.add_line(Line2D([x - 0.006, x, x, x - 0.006],
                [y, y, bracket["y_b"], bracket["y_b"]], transform=transform,
                clip_on=False, color="#505050", linewidth=0.8, gid=bracket["id"]))
            draw_significance_label(error_ax, x + 0.005, (y + bracket["y_b"]) / 2,
                                    bracket, transform, "#505050")
            continue
        error_ax.add_line(Line2D([x, x + BRACKET_CAP], [y, y], transform=transform,
            clip_on=False, color="#707070", linewidth=0.7, gid=bracket["id"]))
        draw_significance_label(error_ax, x + BRACKET_CAP + 0.004, y, bracket,
                                transform, "#303030" if bracket["significant"] else "#707070")
    for ax in axes:
        sf.trim_spines(ax, which=("bottom",), offset=0)
    return tuple(axes)


def draw_payload(ax, formats):
    for row, y in zip(formats, reversed(range(len(formats)))):
        representation = row["representation"]
        assert row["encoding"] == PAYLOAD_FORMATS[representation]
        megabytes_s = row["kbit_s"] / 8000
        ax.scatter(megabytes_s, y, marker=MARKERS[representation], s=40,
                   facecolor=COLORS[representation], edgecolor=COLORS[representation],
                   linewidth=1.1, zorder=3, gid=f"payload-{representation}")
        ax.annotate(row["encoding"], (megabytes_s, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="center", fontsize=9)
    ax.axvline(BLE_LIMIT_MB_S, color=REFERENCE_COLOR, linestyle=REFERENCE_DASHES,
               linewidth=0.75, zorder=0, gid="ble-limit")
    ax.annotate("BLE LE 2M", (BLE_LIMIT_MB_S, -0.72), xytext=(3, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=8.5, color=REFERENCE_COLOR)
    ax.set_xscale("log")
    ax.set_xlim(0.008, 120)
    ax.set_ylim(-1.0, len(formats)-.2)
    ax.set_yticks(list(reversed(range(len(formats)))), [r['label'] for r in formats])
    ax.tick_params(axis="y", length=0, pad=8, labelsize=9.5)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", length=3, labelsize=9)
    ax.xaxis.set_major_locator(FixedLocator(PAYLOAD_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("Payload (MB/s)", labelpad=5, fontsize=9.5)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)
    sf.trim_spines(ax, which=("bottom",), offset=0)


def paired_representation_data(rows):
    """Nine participant means of absolute within-task error changes per decoder."""
    by_key = {(r["campaign"], r["arm"]): r for r in rows}
    summaries = {p["decoder"]: p for p in ABSOLUTE_ANALYSIS["summaries"]}
    paired = []
    for label, token_key, feature_key in REPRESENTATION_PAIRS:
        assert token_key in by_key and feature_key in by_key
        row = copy.deepcopy(summaries[label])
        row.update(label=label, display_label=REPRESENTATION_TABLE_LABELS[label], token_key=list(token_key), feature_key=list(feature_key))
        values = list(row["participant_means_mm"].values())
        assert len(values) == row["n_participants"] == 9
        assert set(row["participant_means_mm"]) == {str(s) for s in range(1, 10)}
        assert all(PAIRED_LIMITS[0] <= v <= PAIRED_LIMITS[1] for v in values)
        np.testing.assert_allclose(np.mean(values), row["mean_absolute_change_mm"], rtol=0, atol=1e-12)
        paired.append(row)
    return paired


def paired_significance(rows, analysis):
    """Display all other-minus-proposed absolute-change comparisons."""
    gaps = {p["display_label"]: p for group in analysis["groups"] for p in group["comparisons"]
            if p["contrast_kind"] == "absolute_error_change_spar"}
    paired = paired_representation_data(rows)
    proposed, = [row for row in paired if row["label"] == "SPAR"]
    annotations = []
    for row, y in zip(paired, reversed(range(len(paired)))):
        if row["label"] == "SPAR":
            continue
        pair = gaps[row["label"]]
        np.testing.assert_allclose(pair["effect_alternative_minus_reference_mm"],
            row["mean_absolute_change_mm"] - proposed["mean_absolute_change_mm"], rtol=0, atol=1e-12)
        annotations.append({"id": f"paired-absolute-significance-{pair['contrast_id']}",
            "contrast_id": pair["contrast_id"], "label": row["label"], "y": y,
            "reference_label": "SPAR", "y_reference": 0,
            "alternative": pair["alternative"], "reference": pair["reference"],
            "p_adjusted": pair["p_adjusted"], "significant": pair["significant"],
            "symbol": significance_symbol(pair["p_adjusted"]) or "n.s.",
            "observation_unit": "participant mean of eight absolute task changes", "n_points": 9,
            "independent_unit": "participant", "n_participants": 9,
            "x_axes": 1.10})
    return annotations


def draw_paired_differences(ax, rows, analysis):
    paired = paired_representation_data(rows)
    for row, y in zip(paired, reversed(range(len(paired)))):
        values = np.asarray([row["participant_means_mm"][str(s)] for s in range(1, 10)])
        # Fixed jitter is shared across decoder rows; it does not encode an outcome.
        offsets = np.linspace(-0.08, 0.08, 9)
        ax.scatter(values, y + offsets, s=15, marker="o", facecolors=PARTICIPANT_COLOR,
                   edgecolors="white", linewidths=0.35,
                   zorder=2, gid=f"paired-{row['label']}-participant-means")
        ax.scatter(row["mean_absolute_change_mm"], y, s=40, marker=MARKERS["tokens"],
                   facecolor=COLORS["tokens"], edgecolor="white", linewidth=0.6,
                   zorder=3, gid=f"paired-{row['label']}-mean")
    ax.set_xlim(PAIRED_LIMITS)
    ax.set_ylim(-0.8, len(paired)-.2)
    labels = [row["display_label"] if row["label"] in {'Proposed','SPAR','TabPFN'} else row["label"] for row in paired]
    ax.set_yticks(list(reversed(range(len(paired)))), labels)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=9.5)
    ax.tick_params(axis="x", length=3, labelsize=9)
    ax.set_xticks(PAIRED_TICKS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace("-", "−")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Absolute Error Change (mm)", labelpad=5, fontsize=9.5)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    stem_x, cap = 1.06, 0.018
    annotations = paired_significance(rows, analysis)
    if annotations:
        ax.add_line(Line2D([stem_x, stem_x], [0, max(a["y"] for a in annotations)],
                          transform=transform, clip_on=False,
                          color="#707070", linewidth=0.7, gid="gap-comparison-stem"))
        ax.add_line(Line2D([stem_x - cap, stem_x], [0, 0], transform=transform, clip_on=False,
                          color="#707070", linewidth=0.7, gid="gap-comparison-proposed-reference"))
    for annotation in annotations:
        ax.add_line(Line2D([stem_x, stem_x + cap], [annotation["y"], annotation["y"]],
                          transform=transform, clip_on=False, color="#707070", linewidth=0.7,
                          gid=annotation["id"] + "-branch"))
        draw_significance_label(ax, annotation["x_axes"], annotation["y"], annotation,
                                transform, "#303030" if annotation["significant"] else "#707070")
    sf.trim_spines(ax, which=("bottom",), offset=0)


def make_figure(rows, formats, analysis, *, supplement=False):
    apply_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, SUPPLEMENT_HEIGHT_IN if supplement else FIGURE_HEIGHT_IN))
    groups = EARLY_GROUPS if supplement else GROUPS
    draw_models(fig, rows, groups, analysis, supplement=supplement)
    if not supplement:
        formats_ax = fig.add_axes(BOTTOM_PANELS["payload"], label="payload")
        draw_payload(formats_ax, formats)
        paired_ax = fig.add_axes(BOTTOM_PANELS["paired_difference"], label="paired_difference")
        draw_paired_differences(paired_ax, rows, analysis)
        fig.text(0.015, 0.975, "A", fontsize=11, weight="bold")
        fig.text(0.015, 0.26, "B", fontsize=11, weight="bold")
        fig.text(0.515, 0.26, "C", fontsize=11, weight="bold")
    height = fig.get_figheight()
    top = {key: 1 - inches / height for key, inches in SUPPLEMENT_TOP_IN.items()}
    fig.text(0.025, top["header"] if supplement else 0.885, "Decoder Variant",
             ha="left", va="center", fontsize=9.5)
    handles = [
        Line2D([], [], marker=MARKERS[key], color=COLORS[key], markersize=5,
               linestyle="none", label=label)
        for key, label in [("tokens", "Tokens"), ("continuous", "Continuous Features"), ("raw", "Raw EMG")]
    ]
    handles += [
        Line2D([], [], color=PARTICIPANT_COLOR, marker="o", markersize=3,
               linestyle="none", label="Participant Mean"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, top["legend"] if supplement else 0.98),
               ncol=4, frameon=False, handletextpad=0.5, columnspacing=1.5, fontsize=9, borderpad=0)
    significance_key = (
        "Paired Comparisons, Max-T Adjusted: * p < 0.05; ** p < 0.01; "
        "*** p < 0.001; n.s. = Not Significant")  # roman p: see note below
    fig.text(0.52, top["key"] if supplement else 0.92, significance_key,
             ha="center", va="center", fontsize=9,
             color=SUPPLEMENT_INK if supplement else "#505050")
    assert all(not ax.get_title() for ax in fig.axes) and fig._suptitle is None
    assert not sf.check_font_sizes(fig)
    assert not sf.check_label_case(fig)
    return fig


def latex_caption(caption):
    replacements = {
        "(Sîmpetru et al., 2023, TNSRE)": r"\cite{simpetru2023proportional}",
        "(Sîmpetru et al., 2024, TBME)": r"\cite{simpetru2024handmodel}",
        "(Sîmpetru et al., 2025)": r"\cite{simpetru2025myogestic}",
        "(Qwen Team, 2025)": r"\cite{qwen3technicalreport}",
        "(Bai et al., 2018; Vaswani et al., 2017)": r"\cite{bai2018tcn,vaswani2017attention}",
        "(Romano and Wolf, 2005; Winkler et al., 2014)": r"\cite{romano2005exact,winkler2014permutation}",
    }
    result = caption.replace("%", r"\%")
    for phrase, citation in replacements.items():
        result = result.replace(phrase, citation)
    return result


def main():
    cohort = ROOT / "results/tabpfn_pose_all9_20260921"
    if (cohort / "results.json").exists() and (cohort / "resource-benchmark.json").exists():
        from dpose_tabpfn_figure_extension import run as run_tabpfn
        run_tabpfn(sys.modules[__name__], cohort)
        return
    # The extension retains the frozen 54-test inputs, adds SPAR, and exports
    # the complete 66-test analysis together with both figures.
    from dpose_spar_figure_extension import run
    run(sys.modules[__name__])


if __name__ == "__main__":
    main()
