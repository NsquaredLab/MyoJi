"""Build a source-audited overview of the 16 clean HD-EMG decoder variants.

Run from any directory with numpy and matplotlib installed. Test cohorts and
the one-person validation pilot stay in separately labelled panels.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from paper.common.paper_plot_style import (
    COLORS,
    FIGURE_WIDTH_IN,
    MARKERS,
    PARTICIPANT_COLOR,
    REFERENCE_COLOR,
    apply_publication_style,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "output/figures"
PDFS = ROOT / "output/pdf"
STEM = "hdemg-all-variants"
STYLE = Path(__file__).parents[1] / "common/paper_plot_style.py"
SUBJECTS = set(range(1, 10))
SEEDS = {260911, 260912, 260913}
SOURCES: dict[str, str] = {}


def read_source(campaign: str) -> dict:
    path = ROOT / "results" / campaign / "results.json"
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = json.loads(path.read_text())
    assert result["complete"], campaign
    return result


def cohort(campaign: str, arm: str, label: str, representation: str) -> dict:
    result = read_source(campaign)
    cells: dict[int, dict[int, float]] = defaultdict(dict)
    if "per_subject" in result:
        for subject, person in result["per_subject"].items():
            assert person["complete"]
            for row in person["results"]:
                if row["arm"] != arm:
                    continue
                sid, seed = int(subject), int(row["seed"])
                assert seed not in cells[sid], (campaign, arm, sid, seed)
                cells[sid][seed] = float(row["macro_subject_error"])
        baseline = float(result["constant_macro_subject_error"])
    else:
        for row in result["per_subject_seed"]:
            if row.get("arm", row.get("model")) != arm:
                continue
            sid, seed = int(row["subject"]), int(row["seed"])
            assert seed not in cells[sid], (campaign, arm, sid, seed)
            cells[sid][seed] = float(row["mean_3d_error"])
        baseline = float(result["constant_baseline"])
    assert set(cells) == SUBJECTS, (campaign, arm, set(cells))
    assert all(set(values) == SEEDS for values in cells.values())
    person_means = {s: float(np.mean(list(v.values()))) for s, v in sorted(cells.items())}
    mean = float(np.mean(list(person_means.values())))
    np.testing.assert_allclose(mean, result["summary"][arm]["mean"], rtol=0, atol=1e-6)
    assert np.isfinite(list(person_means.values())).all()
    seed_means = [np.mean([cells[s][seed] for s in sorted(SUBJECTS)]) for seed in sorted(SEEDS)]
    np.testing.assert_allclose(
        np.std(seed_means, ddof=1), result["summary"][arm]["seed_sd"], rtol=0, atol=1e-6
    )
    return {
        "campaign": campaign,
        "arm": arm,
        "label": label,
        "representation": representation,
        "evaluation": "chronological_test",
        "mean": mean,
        "constant_pose_error": baseline,
        "n_participants": 9,
        "n_seeds": 3,
        "person_means": person_means,
        "person_seed_errors": cells,
        "source_scope": result["scope"],
    }


def pilot_rows() -> list[dict]:
    campaign = "hdemg_llm_pilot_v1"
    result = read_source(campaign)
    assert result["test_evaluated"] is False
    rows = {row["arm"]: row for row in result["results"]}
    specs = [
        ("tokens_pretrained", "Tokens + pretrained Qwen", "tokens"),
        ("tokens_random", "Tokens + random Qwen", "tokens"),
        ("tokens_bypass", "Tokens + small head", "tokens"),
        ("continuous_pretrained", "Continuous + pretrained Qwen", "continuous"),
    ]
    assert set(rows) == {arm for arm, _, _ in specs}
    output = []
    for arm, label, representation in specs:
        row = rows[arm]
        assert row["subject"] == 1 and row["seed"] == 260911
        assert row["steps"] == 2000 and row["test_evaluated"] is False
        assert len(row["history"]) == 20
        error = float(row["best_validation_error"])
        np.testing.assert_allclose(error, min(h["val_error"] for h in row["history"]), rtol=0, atol=1e-9)
        output.append({
            "campaign": campaign,
            "arm": arm,
            "label": label,
            "representation": representation,
            "evaluation": "best_validation_only",
            "mean": error,
            "constant_pose_error": float(result["constant_validation_error"]),
            "n_participants": 1,
            "n_seeds": 1,
            "person_means": {1: error},
            "person_seed_errors": {1: {260911: error}},
            "best_step": row["best_step"],
            "source_scope": "Participant 01, one seed, 2000 updates, best of 20 validation evaluations; no test evaluation.",
        })
    return output


def full_qwen_rows() -> list[dict]:
    """Load the completed, independently audited nine-person Qwen test cohort."""
    campaign = "hdemg_llm_full_v1"
    result = read_source(campaign)
    audit_path = ROOT / "results" / campaign / "completion_audit_20260911.json"
    SOURCES[str(audit_path.relative_to(ROOT))] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    audit = json.loads(audit_path.read_text())
    assert audit["passed"] and audit["all_locked_fit_hashes_match"]
    assert audit["fits_verified"] == audit["tests_verified"] == 108
    assert audit["results_sha256"] == SOURCES[f"results/{campaign}/results.json"]
    assert audit["protocol_sha256"] == result["protocol_sha256"]
    cells = defaultdict(lambda: defaultdict(dict))
    baselines = defaultdict(list)
    for row in result["results"]:
        assert row["complete"] and row["test_evaluated"]
        assert row["protocol_sha256"] == result["protocol_sha256"]
        arm, sid, seed = row["arm"], int(row["subject"]), int(row["seed"])
        assert seed not in cells[arm][sid]
        cells[arm][sid][seed] = float(row["mean_3d_error"])
        baselines[sid].append(float(row["constant_pose_error"]))
    specs = [
        ("tokens_pretrained", "Tokens + pretrained Qwen", "tokens"),
        ("tokens_random", "Tokens + random Qwen", "tokens"),
        ("tokens_bypass", "Tokens + direct pose readout", "tokens"),
        ("continuous_pretrained", "Continuous + pretrained Qwen", "continuous"),
    ]
    assert len(result["results"]) == 108 and set(cells) == {a for a, _, _ in specs}
    for values in baselines.values():
        np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-10)
    baseline = float(np.mean([baselines[s][0] for s in sorted(SUBJECTS)]))
    output = []
    for arm, label, representation in specs:
        people = cells[arm]
        assert set(people) == SUBJECTS and all(set(v) == SEEDS for v in people.values())
        means = {s: float(np.mean(list(v.values()))) for s, v in sorted(people.items())}
        mean = float(np.mean(list(means.values())))
        summary = result["summaries"][arm]
        np.testing.assert_allclose(mean, summary["mean"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.std(list(means.values()), ddof=1),
                                   summary["participant_sd"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(list(means.values()),
            [summary["participant_means"][str(s)] for s in means], rtol=0, atol=1e-10)
        output.append({
            "campaign": campaign, "arm": arm, "label": label,
            "representation": representation, "evaluation": "chronological_test",
            "mean": mean, "constant_pose_error": baseline,
            "n_participants": 9, "n_seeds": 3,
            "person_means": means, "person_seed_errors": dict(people),
            "source_scope": result["scope"],
        })
    return output


def build_panels(*, full_qwen: bool = False) -> list[dict]:
    return [
        {
            "id": "A", "title": "Earlier small readouts",
            "subtitle": "TEST | 9 participants x 3 seeds",
            "note": "Original small decoders and initial raw-input model.",
            "rows": [
                cohort("hdemg_within_v1", "tokens", "Small TCN + tokens", "tokens"),
                cohort("hdemg_within_v1", "continuous", "Small TCN + continuous", "continuous"),
                cohort("hdemg_transformer_v1", "tokens", "Small transformer + tokens", "tokens"),
                cohort("hdemg_within_v1", "raw", "Raw EMG, initial model", "raw"),
            ],
        },
        {
            "id": "B", "title": "Common training recipe",
            "subtitle": "TEST | 9 participants x 3 seeds",
            "note": "~7.5M trainable parameters; 15,000 updates per fit.\nInput history, pretraining and total compute still differ.",
            "rows": [
                cohort("hdemg_matched_v1", "tcn_tokens", "TCN + tokens", "tokens"),
                cohort("hdemg_matched_v1", "tcn_continuous", "TCN + continuous", "continuous"),
                cohort("hdemg_matched_v1", "transformer_tokens", "Transformer + tokens", "tokens"),
                cohort("hdemg_matched_v1", "transformer_continuous", "Transformer + continuous", "continuous"),
                cohort("hdemg_matched_v1", "v19", "Raw EMG RaulNet V19", "raw"),
            ],
        },
        {
            "id": "C", "title": "Native V19 and V20 recipes",
            "subtitle": "TEST | 9 participants x 3 seeds",
            "note": "V20 token/continuous use the same recipe.\nNative V19 uses a different input and training recipe.",
            "rows": [
                cohort("hdemg_raulnet_v1", "raulnet_v19_q32", "Raw EMG RaulNet V19", "raw"),
                cohort("hdemg_v20_v1", "tokens", "V20 + tokens", "tokens"),
                cohort("hdemg_v20_continuous_v1", "continuous", "V20 + continuous", "continuous"),
            ],
        },
        {
            "id": "D", "title": "Frozen language-model comparison" if full_qwen else "Existing language-model pilot",
            "subtitle": "TEST | 9 participants x 3 seeds" if full_qwen else "VALIDATION ONLY | Participant 01; 1 seed",
            "note": ("Qwen3-0.6B-Base backbone frozen; adapter and pose head trained.\n"
                     + ("15,000 updates; 150 validation evaluations per fit." if full_qwen
                        else "2,000 updates; best of 20 validation evaluations per arm.")),
            "rows": full_qwen_rows() if full_qwen else pilot_rows(),
        },
    ]


def draw_panel(fig: plt.Figure, panel: dict, top_in: float) -> float:
    """One stacked panel; only panel IDs and evaluation metadata are inside."""
    rows = panel["rows"]
    height_in = len(rows) * 0.235
    figure_height = fig.get_figheight()
    bottom_in = top_in - height_in
    ax = fig.add_axes([0.355, bottom_in / figure_height, 0.540, height_in / figure_height])
    positions = np.arange(len(rows))[::-1]
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(positions, [item["label"] for item in rows])
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", length=2.8, pad=3, color="#555555")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    baseline = rows[0]["constant_pose_error"]
    np.testing.assert_allclose([r["constant_pose_error"] for r in rows], baseline, atol=1e-5, rtol=0)
    ax.axvline(baseline, color=REFERENCE_COLOR, linewidth=0.8, dashes=(3, 3), zorder=1)
    header_y = (top_in + 0.055) / figure_height
    fig.text(0.015, header_y, f"({panel['id'].lower()})", fontsize=10, weight="bold")
    cohort_label = (
        "Test: 9 participants, 3 seeds"
        if rows[0]["evaluation"] == "chronological_test"
        else "Validation only: participant 01, 1 seed"
    )
    fig.text(0.355, header_y, cohort_label, fontsize=9)
    fig.text(0.983, header_y, "Error", ha="right", fontsize=9)
    for y, item in zip(positions, rows):
        values = list(item["person_means"].values())
        assert all(0 <= value <= 30 for value in values)
        if item["n_participants"] > 1:
            ax.scatter(values, y + np.linspace(-0.10, 0.10, len(values)), s=11,
                       color=PARTICIPANT_COLOR, edgecolors="white", linewidths=0.25, zorder=2)
        representation = item["representation"]
        ax.scatter(item["mean"], y, s=37, color=COLORS[representation],
                   marker=MARKERS[representation], edgecolors="white", linewidths=0.6, zorder=3)
        ax.text((0.983 - 0.355) / 0.540, y, f"{item['mean']:.2f}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10)
    if panel["id"] == "D":
        ax.set_xlabel("Mean 3D landmark error (stored-coordinate units)", labelpad=6)
    return bottom_in - 0.43


def main() -> None:
    panels = build_panels()
    rows = [dict(item, panel=panel["id"]) for panel in panels for item in panel["rows"]]
    assert len(rows) == 16
    assert len({(row["campaign"], row["arm"]) for row in rows}) == 16
    apply_publication_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 6.1))
    top_in = 5.47
    for panel in panels:
        top_in = draw_panel(fig, panel, top_in)

    mean_handles = [
        Line2D([], [], color=COLORS[key], marker=MARKERS[key], markersize=5.5,
               linestyle="none", label=label)
        for key, label in [("tokens", "Tokens"), ("continuous", "Continuous"), ("raw", "Raw EMG")]
    ]
    fig.legend(handles=mean_handles, loc="upper center", bbox_to_anchor=(0.51, 1.0),
               ncol=3, frameon=False, handletextpad=0.5, columnspacing=1.6)
    reference_handles = [
        Line2D([], [], color=PARTICIPANT_COLOR, marker="o", markersize=3,
               linestyle="none", label="Participant mean (a-c)"),
        Line2D([], [], color=REFERENCE_COLOR, linewidth=0.8, dashes=(3, 3), label="Constant pose"),
    ]
    fig.legend(handles=reference_handles, loc="upper center", bbox_to_anchor=(0.51, 0.961),
               ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.6)
    assert not any(ax.get_title() for ax in fig.axes)
    assert fig._suptitle is None

    FIGURES.mkdir(parents=True, exist_ok=True)
    PDFS.mkdir(parents=True, exist_ok=True)
    outputs = [PDFS / f"{STEM}.pdf", FIGURES / f"{STEM}.png", FIGURES / f"{STEM}.svg"]
    fig.savefig(outputs[0], metadata={"Title": "HD-EMG: all decoder versions", "Subject": "16 variants; test and validation cohorts separated"})
    fig.savefig(outputs[1], dpi=300)
    fig.savefig(outputs[2])
    plt.close(fig)

    fields = ["panel", "campaign", "arm", "label", "representation", "evaluation", "mean", "constant_pose_error", "n_participants", "n_seeds"]
    with (FIGURES / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "complete": True,
        "variant_count": len(rows),
        "test_participant_seed_cells_verified": 12 * 9 * 3,
        "validation_pilot_cells_verified": 4,
        "units": "millimetres",
        "aggregation": "A-C: participant means over 3 seeds, then equal participant mean; D: best of 20 validation scores, 1 participant and 1 seed.",
        "uncertainty": "Small grey points are actual participant means, not confidence intervals. No population uncertainty inferred for D.",
        "scope": "Clean decoder variants only. Noise, post-filtering and the earlier cross-person diagnostic use different questions and are omitted.",
        "publication_style": {
            "figure_width_in": FIGURE_WIDTH_IN,
            "figure_height_in": 6.1,
            "manuscript_textwidth_in": 6.5,
            "include_width": "0.95\\textwidth",
            "base_font_pt": 10,
            "minimum_font_pt": 9,
            "font_family": "Times New Roman, Times, DejaVu Serif",
            "representation_markers": MARKERS,
            "titles_inside_figure": False,
            "raster_preview_dpi": 300,
            "plot_type_rationale": "Dot plots retain actual participant observations and do not imply confidence intervals; protocol groups use separate axes.",
        },
        "sources": SOURCES,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(STYLE.read_bytes()).hexdigest(),
        "output_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
        "panels": panels,
    }
    (FIGURES / f"{STEM}.json").write_text(json.dumps(audit, indent=2) + "\n")
    test_baseline = panels[0]["rows"][0]["constant_pose_error"]
    validation_baseline = panels[3]["rows"][0]["constant_pose_error"]
    caption = (
        "HD-EMG decoder comparison. "
        "(a) Earlier small readouts; (b) common-recipe models with approximately 7.5 million "
        "trainable parameters (within 1%), 15,000 optimizer updates and matched target exposure; "
        "(c) native RaulNet V19 and V20 recipes; (d) the external Qwen3-0.6B-Base pilot. "
        "Large circles, squares and diamonds denote aggregate errors for discrete tokens, "
        "continuous encoder features and raw EMG, respectively. In (a-c), small gray points "
        "are nine participant means over three seeds; large markers weight participants equally. "
        "They show observations, not confidence intervals. Decoders are fitted separately "
        "per participant and tested on later chronological blocks of the same person, session and task. "
        "Training recipes differ between panels; "
        "within (c), only token and continuous V20 share a recipe. Input history, pretraining "
        "and total compute are not matched in (b). Panel (d) uses participant 01, one seed, "
        "2,000 updates and 16 validation windows: scores are minima over 20 validation "
        "evaluations, with no test evaluation or population uncertainty estimate. Both "
        "pretrained and random Qwen backbones are frozen; all four arms train identical "
        "adapter/head architectures from paired initial weights, and the small-head arm "
        "bypasses Qwen. Panel (d) cannot be ranked against (a-c). Dashed lines show the "
        "macro-participant error of participant-specific training-mean poses in (a-c) "
        f"({test_baseline:.2f}), and participant 01's training-mean pose in (d) ({validation_baseline:.2f}). "
        "Errors average Euclidean distances for stored points 1-20 relative to point 0, "
        "in millimetres; lower is better. Tokenizer pretraining excluded the validation and test windows."
    )
    (FIGURES / f"{STEM}.md").write_text(
        "# Figure caption\n\n" + caption + "\n\n"
        "# Reproduction and scope\n\n"
        "The figure is 6.175 inches wide, matching 0.95 of the current manuscript's "
        "6.5-inch text block. All labels remain 9-10 pt at that inclusion width. "
        "Representation shapes supplement color for grayscale reading. The figure "
        "has no internal title; use the companion LaTeX caption.\n\n"
        "Only the 16 clean decoder variants are included. Noise robustness, post-filter "
        "variants and earlier cross-person diagnostics answer different questions. "
        "The source JSON hashes, all 324 test participant/seed cells, four selected "
        "validation values and aggregation checks are recorded in the companion JSON. "
        "The CSV contains aggregate values.\n\n"
        "Run `uv run python -m paper.figures.build_hdemg_all_variants` from the repository "
        "with NumPy and Matplotlib installed. The standalone builder and shared style "
        "are saved alongside the figure; source results stay in `results/`.\n"
    )
    latex = (
        "% Include from the repository root; adjust only the path for another layout.\n"
        "\\begin{figure}[p]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.95\\textwidth]{output/pdf/hdemg-all-variants.pdf}\n"
        "  \\caption{" + caption.replace("%", "\\%") + "}\n"
        "  \\label{fig:hdemg-all-variants}\n"
        "\\end{figure}\n"
    )
    (FIGURES / f"{STEM}.tex").write_text(latex)
    include_path = FIGURES / "latex_includes.tex"
    include_line = f"\\input{{output/figures/{STEM}.tex}}"
    existing_includes = include_path.read_text() if include_path.exists() else ""
    if include_line not in existing_includes:
        include_path.write_text(existing_includes + "\n% HD-EMG decoder comparison\n" + include_line + "\n")
    for source in [Path(__file__), STYLE]:
        (FIGURES / source.name).write_bytes(source.read_bytes())
    print(json.dumps({"outputs": [str(path) for path in outputs], "variants": len(rows), "source_files": len(SOURCES)}, indent=2))


if __name__ == "__main__":
    main()
