"""Show representation budgets separately from the 16 measured decoder costs.

Regenerate with numpy and matplotlib. Detailed per-variant measurements remain
in dpose-resource-comparison; this figure avoids repeating representation costs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np

from paper.common.paper_plot_style import COLORS, FIGURE_WIDTH_IN, MARKERS, apply_publication_style


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures"
PDFS = ROOT / "output/pdf"
STEM = "dpose-resource-overview"
RESULTS = ROOT / "results/dpose_resource_benchmark_v1/results.json"
REFERENCE = OUT / "dpose-all-variants.json"
MB_PER_MINUTE_PER_KBIT_S = 60 * 1000 / 8 / 1e6
FIGURE_HEIGHT_IN = 6.65
FAMILIES = {
    "Small readouts": [
        ("dpose_within_v1", "tokens"), ("dpose_within_v1", "continuous"),
        ("dpose_transformer_v1", "tokens"), ("dpose_within_v1", "raw"),
    ],
    "TCN": [("dpose_matched_v1", "tcn_tokens"), ("dpose_matched_v1", "tcn_continuous")],
    "Transformer": [("dpose_matched_v1", "transformer_tokens"),
                    ("dpose_matched_v1", "transformer_continuous")],
    "V19": [("dpose_matched_v1", "v19"), ("dpose_raulnet_v1", "raulnet_v19_q32")],
    "V20": [("dpose_v20_v1", "tokens"), ("dpose_v20_continuous_v1", "continuous")],
    "Frozen Qwen": [("dpose_llm_pilot_v1", "tokens_pretrained"),
                    ("dpose_llm_pilot_v1", "tokens_random"),
                    ("dpose_llm_pilot_v1", "continuous_pretrained")],
    "Adapter + head": [("dpose_llm_pilot_v1", "tokens_bypass")],
}
EXPERIMENT_GROUPS = [
    {"id": "early", "label": "Early baselines", "source_panel": "A", "variants": [
        ("dpose_within_v1", "raw"),
        ("dpose_within_v1", "tokens"),
        ("dpose_within_v1", "continuous"),
        ("dpose_transformer_v1", "tokens"),
    ]},
    {"id": "common", "label": "Common training recipe", "source_panel": "B", "variants": [
        ("dpose_matched_v1", "v19"),
        ("dpose_matched_v1", "tcn_tokens"),
        ("dpose_matched_v1", "tcn_continuous"),
        ("dpose_matched_v1", "transformer_tokens"),
        ("dpose_matched_v1", "transformer_continuous"),
    ]},
    {"id": "native", "label": "Native V19 / V20", "source_panel": "C", "variants": [
        ("dpose_raulnet_v1", "raulnet_v19_q32"),
        ("dpose_v20_v1", "tokens"),
        ("dpose_v20_continuous_v1", "continuous"),
    ]},
    {"id": "qwen", "label": "Qwen pilot", "source_panel": "D", "variants": [
        ("dpose_llm_pilot_v1", "tokens_bypass"),
        ("dpose_llm_pilot_v1", "tokens_random"),
        ("dpose_llm_pilot_v1", "tokens_pretrained"),
        ("dpose_llm_pilot_v1", "continuous_pretrained"),
    ]},
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    benchmark = json.loads(RESULTS.read_text())
    assert benchmark["complete"] and not benchmark["test_evaluated"]
    assert not benchmark["training_performed"]
    labels = {(r["campaign"], r["arm"]): r["label"]
              for p in json.loads(REFERENCE.read_text())["panels"] for r in p["rows"]}
    reference_panels = {(r["campaign"], r["arm"]): p["id"]
                        for p in json.loads(REFERENCE.read_text())["panels"] for r in p["rows"]}
    by_key = {(r["campaign"], r["arm"]): r for r in benchmark["results"]}
    requested = [key for keys in FAMILIES.values() for key in keys]
    assert len(requested) == len(set(requested)) == len(by_key) == 16
    assert set(requested) == set(by_key) == set(labels)
    grouped_keys = [key for group in EXPERIMENT_GROUPS for key in group["variants"]]
    assert len(grouped_keys) == len(set(grouped_keys)) == 16
    assert set(grouped_keys) == set(by_key)
    groups_by_key = {key: group for group in EXPERIMENT_GROUPS for key in group["variants"]}
    rows = []
    for family, keys in FAMILIES.items():
        for key in keys:
            row = dict(by_key[key], family=family, label=labels[key])
            group = groups_by_key[key]
            assert row["panel"] == reference_panels[key] == group["source_panel"]
            row["experiment_group"] = group["label"]
            row["experiment_group_id"] = group["id"]
            if row["arm"] == "tokens_bypass":
                row["label"] = "Tokens + adapter and pose head"
            assert row["subject"] == 1 and row["seed"] == 260911
            assert row["loaded_state_unchanged"] and row["measured_calls"] == 48
            np.testing.assert_allclose(row["updates_per_second"],
                                       1000 / np.median(row["samples_ms"]), rtol=0, atol=1e-10)
            assert row["model_tensor_bytes"] == sum(row["component_tensor_bytes"].values())
            samples = np.asarray(row["samples_ms"], dtype=float)
            assert len(samples) == 48 and np.isfinite(samples).all() and (samples > 0).all()
            row["latency_mean_ms"] = float(np.mean(samples))
            row["latency_sd_ms"] = float(np.std(samples, ddof=1))
            row["timing_n_calls"] = len(samples)
            assert row["latency_mean_ms"] - row["latency_sd_ms"] > 0
            rows.append(row)
    order = {key: index for index, key in enumerate(grouped_keys)}
    rows.sort(key=lambda row: order[(row["campaign"], row["arm"])])
    token = by_key[("dpose_v20_v1", "tokens")]
    raw = by_key[("dpose_raulnet_v1", "raulnet_v19_q32")]
    continuous16 = by_key[("dpose_within_v1", "continuous")]
    continuous32 = by_key[("dpose_v20_continuous_v1", "continuous")]
    # Cross-check the nominal rate arithmetic against the saved model input/
    # output-cache shapes, rather than treating the budget constants as data.
    arrays = benchmark["inputs"]["reference"]["arrays"]
    channels, context_frames = arrays["tokens"]["shape"][1:]
    vocabulary, feature_dim = arrays["book"]["shape"]
    assert channels == token["channels"] == raw["channels"]
    assert token["payload_bits_per_item"] == int(np.ceil(np.log2(vocabulary)))
    assert arrays["continuous"]["shape"][1:] == [channels, context_frames, feature_dim]
    assert continuous16["items_per_second"] == token["items_per_second"] * feature_dim
    assert continuous32["items_per_second"] == continuous16["items_per_second"]
    assert benchmark["inputs"]["native_raw"]["dtype"] == "torch.int16"
    native_raw_bits = np.dtype(benchmark["inputs"]["native_raw"]["dtype"].split(".")[-1]).itemsize * 8
    specs = [
        ("tokens", "10-bit packed", token["items_per_second"], token["payload_bits_per_item"]),
        ("tokens", "uint16", token["items_per_second"], token["native_array_bits_per_item"]),
        ("raw", "int16", raw["items_per_second"], native_raw_bits),
        ("raw", "FP32", raw["items_per_second"], raw["native_array_bits_per_item"]),
        ("continuous", "FP16", continuous16["items_per_second"], continuous16["native_array_bits_per_item"]),
        ("continuous", "FP32", continuous32["items_per_second"], continuous32["native_array_bits_per_item"]),
    ]
    formats = []
    for representation, encoding, items, bits in specs:
        kbit_s = items * bits / 1000
        formats.append({"representation": representation, "encoding": encoding,
                        "items_per_second": items, "bits_per_item": int(bits),
                        "kbit_s": kbit_s, "mb_min": kbit_s * MB_PER_MINUTE_PER_KBIT_S})
    return rows, formats, benchmark


def caption_text():
    return (
        "DPose representation and decoder costs. Colors and shapes identify input "
        "representation. (a) Calculated payload for 320 "
        "channels, 50 feature frames/s and 256 continuous values/channel/frame; raw "
        "EMG uses 2,048 Hz. Endpoints identify formats: packed 10-bit tokens require "
        "160 kbit/s; uint16 arrays use 256 kbit/s. Raw int16 is the native cache "
        "format; FP32 is a payload budget. Overhead and the derived low-pass plane "
        "are excluded. (b) One row per measured variant, grouped by experiment: "
        "early baselines, common training recipe, native V19/V20 and the Qwen pilot. "
        "Raw baselines lead each applicable group; token/continuous pairs are "
        "adjacent. The Qwen group proceeds from the bypass control through random "
        "and pretrained backbones. Model-size dots count loaded "
        "FP32 tensors, including frozen weights and receiver metadata. Printed "
        "latencies are arithmetic mean plus/minus sample SD over all 48 timed calls, "
        "rounded to 0.001 ms. SD describes call-to-call variation within the "
        "benchmark, not participant variation or a confidence interval. Measurements "
        "use one A100-SXM4-40GB, FP32, batch 1, participant 01, one seed, four "
        "validation inputs and 12 warm-ups. Adapter + head bypasses Qwen; all Qwen "
        "backbones are frozen. Native contexts differ and are fully recomputed; the "
        "two V19 timings do not establish a training-recipe effect on speed. Both "
        "decoder metrics exclude the shared upstream EMG tokenizer; timing also "
        "excludes preprocessing, transfers and I/O. Payload and model size are "
        "log-scaled. This figure does not compare accuracy."
    )


def draw_formats(ax, formats):
    positions = {"tokens": 2, "raw": 1, "continuous": 0}
    for representation, y in positions.items():
        pair = [row for row in formats if row["representation"] == representation]
        xs = [row["kbit_s"] for row in pair]
        ax.plot(xs, [y, y], color=COLORS[representation], linewidth=1.1, zorder=1)
        for index, row in enumerate(pair):
            ax.scatter(row["kbit_s"], y, marker=MARKERS[representation], s=37,
                       facecolor=COLORS[representation] if index == 0 else "white",
                       edgecolor=COLORS[representation], linewidth=1.0, zorder=3)
            # Stagger the datatype labels; token endpoints are only 1.6x apart.
            dx, dy = ((-3, 12) if index == 0 else (3, -15))
            if representation == "continuous":
                dx = -2
            ax.annotate(row["encoding"], (row["kbit_s"], y), xytext=(dx, dy),
                        textcoords="offset points", ha="center", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(85, 260000)
    ax.set_ylim(-1.0, 2.8)
    ax.set_yticks([2, 1, 0], ["Tokens", "Raw EMG", "Features"])
    ax.tick_params(axis="y", length=0, pad=7, labelsize=9.5)
    ax.tick_params(axis="x", length=3, labelsize=9)
    ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000, 100000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"$10^{{{int(np.log10(x))}}}$"))
    ax.set_xlabel("Payload (kbit/s)", labelpad=7)
    ax.spines[["left", "right", "top"]].set_visible(False)


def decoder_layout(rows):
    by_key = {(row["campaign"], row["arm"]): row for row in rows}
    positions, sections, cursor = {}, [], 0.0
    for group in EXPERIMENT_GROUPS:
        sections.append({"id": group["id"], "label": group["label"], "y": -cursor})
        cursor += 1.05
        for key in group["variants"]:
            assert key in by_key
            positions[key] = -cursor
            cursor += 1
        cursor += 0.45
    assert len(positions) == len(rows) == 16
    return positions, sections


def decoder_positions(rows):
    return decoder_layout(rows)[0]


def decoder_label(row):
    if row["family"] == "V19":
        return "V19, common recipe" if row["campaign"] == "dpose_matched_v1" else "V19, native recipe"
    return row["label"]


def latency_label(row):
    return f"{row['latency_mean_ms']:.3f} ± {row['latency_sd_ms']:.3f}"


def draw_decoders(size_ax, timing_ax, rows):
    positions, sections = decoder_layout(rows)
    ordered = sorted(rows, key=lambda r: positions[(r["campaign"], r["arm"])], reverse=True)
    y_ticks = [positions[(r["campaign"], r["arm"])] for r in ordered]
    for ax in [size_ax, timing_ax]:
        ax.set_ylim(min(y_ticks) - 0.7, 0.8)
    size_ax.set_yticks(y_ticks, [decoder_label(row) for row in ordered])
    size_ax.tick_params(axis="y", length=0, pad=12, labelsize=9.5)
    size_ax.tick_params(axis="x", length=3, labelsize=9)
    size_ax.spines[["left", "right", "top"]].set_visible(False)
    timing_ax.set_xlim(0, 1)
    timing_ax.set_axis_off()
    group_transform = blended_transform_factory(size_ax.figure.transFigure, size_ax.transData)
    for section in sections:
        y = section["y"]
        size_ax.figure.add_artist(Line2D([0.075, 0.955], [y + 0.45, y + 0.45],
            transform=group_transform, color="#BBBBBB", linewidth=0.65, clip_on=False))
        size_ax.figure.text(0.075, y, section["label"], transform=group_transform,
                            fontsize=9.5, weight="bold", color="#505050", ha="left", va="center")
    for row in ordered:
        y = positions[(row["campaign"], row["arm"])]
        representation = row["representation"]
        size_ax.axhline(y, color="#E8E8E8", linewidth=0.4, zorder=0)
        size_ax.scatter(row["model_tensor_mb"], y, s=22, marker=MARKERS[representation],
                        color=COLORS[representation], edgecolor="white", linewidth=0.5, zorder=3)
        timing_ax.text(0.96, y, latency_label(row), ha="right", va="center",
                       fontsize=10.5, color="#202020")
    size_ax.set_xscale("log")
    size_ax.xaxis.set_minor_locator(NullLocator())
    size_ax.set_xlim(0.55, 4200)
    size_ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1k"])


def main():
    rows, formats, benchmark = load_data()
    apply_publication_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    formats_ax = fig.add_axes([0.16, 0.79, 0.795, 0.165])
    size_ax = fig.add_axes([0.465, 0.055, 0.185, 0.575])
    timing_ax = fig.add_axes([0.705, 0.055, 0.25, 0.575])
    draw_formats(formats_ax, formats)
    draw_decoders(size_ax, timing_ax, rows)
    fig.text(0.024, 0.964, "(a)", fontsize=10, weight="bold")
    fig.text(0.024, 0.69, "(b)", fontsize=10, weight="bold")
    fig.text(0.075, 0.68, "Decoder variant", ha="left", va="center", fontsize=10)
    fig.text(0.5575, 0.68, "Model size\n(MB, FP32)", ha="center", va="center", fontsize=10)
    fig.text(0.83, 0.68, "Latency (ms)\nmean $\\pm$ SD", ha="center", va="center", fontsize=10)
    handles = [Line2D([], [], marker=MARKERS[key], color=COLORS[key], markersize=5,
                      linestyle="none", label=label)
               for key, label in [("tokens", "Tokens"), ("continuous", "Continuous"), ("raw", "Raw EMG")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=3,
               frameon=False, handletextpad=0.5, columnspacing=1.5)
    assert all(not ax.get_title() for ax in fig.axes) and fig._suptitle is None
    OUT.mkdir(parents=True, exist_ok=True)
    PDFS.mkdir(parents=True, exist_ok=True)
    outputs = [PDFS / f"{STEM}.pdf", OUT / f"{STEM}.png", OUT / f"{STEM}.svg"]
    fig.savefig(outputs[0], metadata={"Title": "DPose representation and decoder resource trade-offs"})
    fig.savefig(outputs[1], dpi=300)
    fig.savefig(outputs[2])
    plt.close(fig)
    caption = caption_text()
    (OUT / f"{STEM}.tex").write_text(
        "\\begin{figure}[t]\n  \\centering\n"
        f"  \\includegraphics[width=0.95\\textwidth]{{output/pdf/{STEM}.pdf}}\n"
        "  \\caption{" + caption.replace("%", "\\%") + "}\n"
        f"  \\label{{fig:{STEM}}}\n\\end{{figure}}\n")
    (OUT / f"{STEM}.md").write_text(
        "# Figure caption\n\n" + caption + "\n\n# Design\n\n"
        "Stream costs appear once per encoding format, not once per decoder. Panel "
        "(a) shows bitrate alone; the redundant storage-per-minute axis was removed. "
        "Panel (b) gives each model its own labelled row, grouped by the same "
        "four experiment campaigns as the accuracy figure. Group membership is "
        "checked against that figure's source panels. Raw baselines come first, "
        "with token and continuous counterparts adjacent. The Qwen pilot starts "
        "with its adapter/head control, then random-token, pretrained-token and "
        "pretrained-continuous Qwen. Thin dividers and bold group labels identify "
        "the boundaries without using color for a second meaning. Model size remains a "
        "position on a logarithmic axis; latency is printed directly as mean +/- "
        "sample SD in milliseconds, to three decimal places, in 10.5-point text. "
        "No error bars are enlarged, and no family averages or accuracy ranking "
        "are displayed. All 48 recorded calls enter each model's summary. Full "
        "values remain in the detailed resource comparison CSV and benchmark JSON. "
        "Adapter + head (previously labelled Head only) is the tokens_bypass control: "
        "EMG token-codebook lookup -> learned projection and geometry-aware channel "
        "pooling -> input bridge -> pose predictor. The Qwen backbone is absent. "
        "The adapter and head have the same structure, initialization and training "
        "budget as in the Qwen pilot arms; each arm is fitted separately.\n\n"
        "# How panel (a) is calculated\n\n"
        "These inputs come from the saved model/cache configuration: token arrays "
        "have 320 channels and 100 time frames; continuous arrays add a 256-value "
        "feature dimension; the saved codebook has 1,024 rows and 256 columns. "
        "The builder cross-checks these shapes against the benchmark's rate arithmetic.\n\n"
        "Rate (kbit/s) = channels x frames/s x values/channel/frame x bits/value / 1,000.\n\n"
        "Storage (decimal MB/min) = rate (kbit/s) x 1,000 x 60 / 8 / 1,000,000 "
        "= rate x 0.0075. This is a nonoverlapping steady-stream payload budget, "
        "not measured HDF5 sizes or network traffic.\n\n"
        "- Tokens: 320 channels x 50 frames/s x 1 ID/channel/frame x 10 bits/ID "
        "= 160 kbit/s = 1.20 MB/min. The 1,024-code vocabulary needs 10 packed bits. "
        "Current uint16 arrays store 16 bits/ID, giving 256 kbit/s and 1.92 MB/min.\n"
        "- Raw: 320 channels x 2,048 samples/s x 16 or 32 bits/sample. "
        "The initial raw decoder's 2,000-Hz stream is 2.3% smaller.\n"
        "- Continuous: 320 channels x 50 frames/s x 256 values/channel/frame "
        "x 16 or 32 bits/value.\n\n"
        "The 50-Hz feature rate corresponds to a 40-sample hop after resampling "
        "to 2,000 Hz. The shared tokenizer and codebook are provisioned separately; "
        "their weights are not repeatedly added to each minute of signal.\n\n"
        "# How panel (b) is measured\n\n"
        "Model size sums unique loaded decoder tensor-storage bytes / 1,000,000, "
        "including frozen weights and receiver codebook/metadata. It is not peak GPU RAM. "
        "Latency mean is sum(t_i)/48; sample SD is sqrt(sum((t_i-mean)^2)/47), "
        "using the saved 48 synchronized latency measurements in milliseconds. "
        "These are repeated calls cycling through four validation inputs from "
        "participant 01, seed 260911, after 12 untimed warm-ups. They do not estimate "
        "variation across participants, training seeds or independent benchmark sessions. "
        "No samples are trimmed: for example, the small token-transformer retains "
        "its recorded 8.390-ms call. Model size is a fixed loaded tensor count and "
        "has no sampling SD here. The earlier inverse-median throughput stays in "
        "the detailed data export for traceability.\n\n"
        "# Benchmark details\n\n"
        + benchmark["precision"] + "\n\n" + benchmark["timing"] + "\n\n" + benchmark["space"] + "\n")
    with (OUT / f"{STEM}-formats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(formats[0]))
        writer.writeheader()
        writer.writerows(formats)
    with (OUT / f"{STEM}-decoders.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment_group", "experiment_group_id",
            "family", "campaign", "arm", "label",
            "representation", "model_tensor_mb", "latency_mean_ms", "latency_sd_ms",
            "timing_n_calls", "updates_per_second"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    # Also supply a native LaTeX table with the same reported statistics.
    table = ["\\begin{tabular}{lrr}\\toprule",
             "Decoder & Mean latency (ms) & SD (ms)\\\\\\midrule"]
    previous_group = None
    for row in rows:
        if row["experiment_group"] != previous_group:
            if previous_group is not None:
                table.append("\\midrule")
            table.append("\\multicolumn{3}{l}{\\textbf{" + row["experiment_group"] + "}}\\\\")
            previous_group = row["experiment_group"]
        label = row["label"]
        if row["family"] == "V19":
            label += " (common recipe)" if row["campaign"] == "dpose_matched_v1" else " (native recipe)"
        table.append(f"{label} & {row['latency_mean_ms']:.3f} & {row['latency_sd_ms']:.3f}\\\\")
    table += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"TABLE_{STEM}-timing.tex").write_text("\n".join(table) + "\n")
    audit = {"complete": True, "variant_count": len(rows), "storage_formats": formats,
             "experiment_groups": EXPERIMENT_GROUPS,
             "decoder_rows": rows, "sources": {str(p.relative_to(ROOT)): digest(p) for p in [RESULTS, REFERENCE]},
             "outputs": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
             "builder_sha256": digest(Path(__file__)),
             "style_sha256": digest(Path(__file__).with_name("paper_plot_style.py")),
             "figure_width_in": FIGURE_WIDTH_IN, "figure_height_in": FIGURE_HEIGHT_IN, "minimum_font_pt": 9,
             "timing_summary": {"center": "arithmetic mean latency in ms",
                                "spread": "sample SD in ms, ddof=1", "n_calls_per_variant": 48,
                                "unit_of_variation": "timed call within one benchmark session",
                                "excluded_timed_calls": 0},
             "latency_display": {"kind": "direct mean plus/minus SD labels", "font_size_pt": 10.5,
                                 "decimal_places": 3, "rows": 16},
             "notes": "One labelled row per variant in source-verified experiment groups; raw baselines first, paired inputs adjacent, Qwen control/random/pretrained sequence. Fixed model sizes and all48call mean/SD values unchanged; no family averages or accuracy encoding."}
    (OUT / f"{STEM}.json").write_text(json.dumps(audit, indent=2) + "\n")
    for filename in [Path(__file__).name, "paper_plot_style.py"]:
        (OUT / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    print(json.dumps({"outputs": [str(p) for p in outputs], "decoder_variants": len(rows),
                      "storage_formats": len(formats)}, indent=2))


if __name__ == "__main__":
    main()
