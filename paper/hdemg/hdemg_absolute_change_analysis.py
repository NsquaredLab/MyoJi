"""Absolute token-feature error changes; retain all previous tests in one 54-test family."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from paper.common import hdemg_figure_statistics as stats
from paper.common import paths

ROOT = Path(__file__).resolve().parents[1]
INPUTS = paths.DATA / "figure_inputs"  # frozen protocol and the retained 51-test analysis
PROTOCOL = INPUTS / "hdemg_absolute_change_protocol.json"
RESULT = ROOT / "results/hdemg_absolute_change_analysis.json"
OUT = ROOT / "output/figures"
ENDPOINT_LABELS = {"TCN": "TCN (matched)", "Transformer": "Transformer (matched)",
                   "Qwen3": "Qwen3 (pretrained)", "Proposed": "Proposed"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def qualified_label(key, label):
    """Preserve original labels separately while distinguishing experimental arms."""
    campaign, _ = key
    if campaign == "hdemg_matched_v1":
        return label + " (matched)"
    if campaign in {"hdemg_within_v1", "hdemg_transformer_v1"}:
        return label + " (smaller baseline)"
    return label


def main():
    protocol = json.loads(PROTOCOL.read_text())
    movement_path = ROOT / protocol["source_movement_analysis"]  # hdemg_movement_analysis.py
    previous_path = INPUTS / Path(protocol["previous_analysis"]).name
    assert digest(movement_path) == protocol["source_movement_analysis_sha256"]
    assert digest(previous_path) == protocol["previous_analysis_sha256"]
    movement = json.loads(movement_path.read_text())
    previous = json.loads(previous_path.read_text())
    for name, expected in movement["sources"].items():
        assert digest(ROOT / name) == expected, name
    people, tasks = protocol["subjects"], protocol["tasks"]
    expected_ids = {(s, t) for s in people for t in tasks}
    summaries, cells, subject_rows, subject_values = [], [], [], {}
    for row in movement["plot_rows"]:
        assert {(c["subject"], c["task"]) for c in row["cells"]} == expected_ids
        assert len(row["cells"]) == 72
        lookup = {(c["subject"], c["task"]): c["difference_mm"] for c in row["cells"]}
        signed = np.array([[lookup[s, t] for t in tasks] for s in people])
        absolute = np.abs(signed)
        assert np.isfinite(absolute).all()
        values = absolute.mean(axis=1)
        subject_values[row["label"]] = values
        summaries.append(dict(decoder=row["label"], table_label=ENDPOINT_LABELS[row["label"]], mean_absolute_change_mm=float(values.mean()),
            sd_across_participants_mm=float(values.std(ddof=1)), n_participants=9,
            participant_means_mm={str(s): float(v) for s, v in zip(people, values)}))
        for i, s in enumerate(people):
            subject_rows.append(dict(decoder=row["label"], subject=s, absolute_change_mm=float(values[i])))
            for j, task in enumerate(tasks):
                cells.append(dict(decoder=row["label"], subject=s, task=task,
                                  signed_change_mm=float(signed[i, j]), absolute_change_mm=float(absolute[i, j])))
    old_tests = [p for g in previous["groups"] for p in g["comparisons"]]
    assert len(old_tests) == 51
    comparisons = copy.deepcopy(old_tests)
    differences = [[p["participant_differences"][str(s)] for s in people] for p in comparisons]
    for definition in protocol["comparisons"]:
        label = definition["other"]
        d = subject_values[label] - subject_values["Proposed"]
        differences.append(d.tolist())
        lo, hi = stats.exact_location_interval(d)
        loo = [float(np.delete(d, i).mean()) for i in range(9)]
        comparisons.append(dict(contrast_kind="absolute_error_change", contrast_id=definition["id"],
            display_label=label, table_label=ENDPOINT_LABELS[label],
            label_a=f"{ENDPOINT_LABELS[label]}: absolute token-feature error change",
            label_b="Proposed: absolute token-feature error change",
            alternative=["absolute_error_change", label], reference=["absolute_error_change", "Proposed"],
            effect_alternative_minus_reference_mm=float(d.mean()),
            participant_differences={str(s): float(v) for s, v in zip(people, d)},
            effect_ci95_pointwise_mm=[lo, hi],
            sd_participant_difference_mm=float(d.std(ddof=1)),
            n_participants=9, n_seeds_per_participant=3, n_tasks_per_participant=8,
            permutations=512, participants_favor_proposed=int((d > 0).sum()),
            participants_favor_other=int((d < 0).sum()), participants_tied=int((d == 0).sum()),
            leave_one_participant_out_effects_mm={str(s): v for s, v in zip(people, loo)},
            leave_one_participant_out_effect_range_mm=[min(loo), max(loo)],
            effect_direction="Positive means a smaller absolute change in error for proposed, averaged over movements within each participant."))
    matrix = np.asarray(differences).T
    assert matrix.shape == (9, 54)
    observed, raw, adjusted, maximum = stats.exact_max_t(matrix)
    changes = []
    for i, p in enumerate(comparisons):
        p["qualified_label_a"] = qualified_label(p["alternative"], p["label_a"])
        p["qualified_label_b"] = qualified_label(p["reference"], p["label_b"])
        if i < 51:
            assert raw[i] == p["p_raw"]
            np.testing.assert_allclose(observed[i], p["t_observed"], rtol=1e-12, atol=1e-12)
            assert adjusted[i] >= p["p_adjusted"]
            p["p_adjusted_previous_51"] = p["p_adjusted"]
            p["significant_previous_51"] = p["significant"]
            if adjusted[i] != p["p_adjusted"]:
                changes.append(dict(index=i, label_a=p["qualified_label_a"], label_b=p["qualified_label_b"],
                    alternative=p["alternative"], reference=p["reference"],
                    previous_p=p["p_adjusted"], current_p=float(adjusted[i]),
                    significance_changed=bool((adjusted[i] < .05) != p["significant"])))
        p.update(t_observed=float(observed[i]), p_raw=float(raw[i]),
                 p_maxT=float(adjusted[i]), p_adjusted=float(adjusted[i]), significant=bool(adjusted[i] < .05))
    assert np.all(maximum + 1e-12 >= np.asarray(previous["maxT_reference_distribution"]))
    result = dict(complete=True, protocol=protocol, family_size=54, replication_unit="participant",
        independent_participants=9, tasks=8, seeds=3, summaries=summaries,
        comparisons=comparisons, absolute_change_comparisons=comparisons[-3:],
        previous_adjusted_p_changes=changes,
        previous_significance_changes=[c for c in changes if c["significance_changed"]],
        significant_comparisons=int(sum(p["significant"] for p in comparisons)),
        maxT_reference_distribution=maximum.tolist(),
        sources={str(p.relative_to(ROOT)): digest(p) for p in
                 [PROTOCOL, movement_path, previous_path, Path(__file__), Path(stats.__file__)]},
        original_51_artifacts_preserved=True,
        interpretation="Absolute change in task-averaged error, with nine paired independent participants; not a direct test of variance, overall pose accuracy, or equivalence.")
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    OUT.mkdir(exist_ok=True)
    (OUT / "hdemg-absolute-change-analysis.json").write_bytes(RESULT.read_bytes())
    (OUT / "hdemg_absolute_change_analysis.py").write_bytes(Path(__file__).read_bytes())
    csv_write(OUT / "hdemg-absolute-change-cells.csv", cells)
    csv_write(OUT / "hdemg-absolute-change-participants.csv", subject_rows)
    export = []
    for i, p in enumerate(comparisons):
        low, high = p["effect_ci95_pointwise_mm"]
        export.append(dict(index=i + 1, comparison_id=p.get("contrast_id", f"retained_{i + 1:02d}"),
            kind=p["contrast_kind"], alternative=p["qualified_label_a"], reference=p["qualified_label_b"],
            effect_mm=p["effect_alternative_minus_reference_mm"], pointwise_ci95_low_mm=low,
            pointwise_ci95_high_mm=high, t=p["t_observed"], p_raw=p["p_raw"],
            p_adjusted_54=p["p_adjusted"], p_adjusted_previous_51=p.get("p_adjusted_previous_51", ""),
            significant=p["significant"], independent_participants=9))
    csv_write(OUT / "hdemg-absolute-change-all-54-tests.csv", export)
    lines = ["# Absolute change in decoding error", "",
        "The calculation definition and 54-test family were saved before the p-values were computed.", "",
        "For each participant, movement task and decoder, first average token and feature Euclidean errors separately over the two test windows and three seeds. Take the absolute difference within each cell, then average the eight absolute changes within each participant. Thus A[s,d] = mean_m |E_token[s,m,d] − E_feature[s,m,d]|. The order matters: this is neither |mean_m gap| nor mean_seed |gap_seed|.", "",
        "## Descriptive values", "", "| Decoder | Mean absolute change (mm) | Participant SD (mm) |",
        "|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['table_label']} | {row['mean_absolute_change_mm']:.4f} | {row['sd_across_participants_mm']:.4f} |")
    lines += ["", "## Proposed versus the other decoders", "",
        "Each contrast is other minus proposed absolute change, paired within participant. Positive favors a smaller absolute change for proposed. Nine participants remain independent units; all eight movements, three seeds and all windows are retained. There are no movement-wise tests.", "",
        "| Other decoder | Other − proposed (mm) | 95% pointwise interval (mm) | Raw p | Joint 54 max-T p | Participants favor proposed |",
        "|---|---:|---:|---:|---:|---:|"]
    for p in comparisons[-3:]:
        low, high = p["effect_ci95_pointwise_mm"]
        lines.append(f"| {p['table_label']} | {p['effect_alternative_minus_reference_mm']:.4f} | [{low:.4f}, {high:.4f}] | {p['p_raw']:.6f} | {p['p_adjusted']:.6f} | {p['participants_favor_proposed']}/9 |")
    lines += ["", "## Inference and multiplicity", "",
        "All 51 prior participant contrast vectors are retained exactly and these three vectors are appended. For each of 512 whole-participant sign patterns, use the same participant signs across all 54 columns, recalculate each mean divided by its sample standard error, and take the largest absolute statistic. Adjusted p-values are inclusive tail proportions from this common maximum distribution. Tests remain two-sided. The finite-sample interpretation requires independent participants and joint central sign symmetry of the true-null participant contrast vectors. Absolute-valued outcomes do not automatically establish that symmetry, which cannot be verified from nine participants. Familywise control for the specified procedure does not account for the entire adaptive endpoint-selection history.", "",
        "Intervals invert the existing univariate sign-flip location test; they are pointwise, assume independent participant differences symmetric around a common location, and do not assign stars. Every leave-one-participant-out effect is exported in JSON.", "",
        f"Extending 51 to 54 changed {len(changes)} existing adjusted p-values; {sum(c['significance_changed'] for c in changes)} existing significance decisions changed. No previous raw p-value, effect, interval, or participant vector changed. All 54 updated tests appear in `output/figures/hdemg-absolute-change-all-54-tests.csv`.", "",
        "| Existing comparison | Previous 51 p | Updated 54 p | Decision changed? |",
        "|---|---:|---:|---|"]
    lines.extend(f"| {c['label_a']} versus {c['label_b']} | {c['previous_p']:.6f} | {c['current_p']:.6f} | {'Yes' if c['significance_changed'] else 'No'} |" for c in changes)
    lines += ["", "## Interpretation and artifact scope", "",
        "This measures how closely the token and feature versions preserve each movement's mean error. Both improvements and degradations count as changes. It is not a direct variance test or evidence that one decoder has lower overall pose error; identical average errors can also conceal differences in individual predictions. The inference concerns these eight fixed movement tasks and the current same-session split.", "",
        "The 51-contrast analysis and its figure are unchanged; their stars are not 54-family results. The expanded-family CSV/JSON is the source for this extension.", "",
        "Reproduce from the repository with `uv run python -m paper.hdemg.hdemg_absolute_change_analysis` using the existing figure environment. Source hashes, the frozen protocol, all 288 absolute cell values, 36 participant summaries, all 54 tests, and all 512 maximum statistics are saved. No training or model inference was run.", "",
        "Sources: [Romano and Wolf](https://doi.org/10.1198/016214504000000539), [PALM whole-participant sign flips](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/palm/user_guide.html), [ASA guidance on interpretation](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf)."]
    report = "\n".join(lines) + "\n"
    (paths.RESULTS / "HDEMG_ABSOLUTE_CHANGE_ANALYSIS.md").write_text(report)
    (OUT / "hdemg-absolute-change-analysis.md").write_text(report)
    print(json.dumps(dict(summaries=summaries,
        new_tests=[{k: p[k] for k in ["display_label", "effect_alternative_minus_reference_mm", "p_raw", "p_adjusted", "significant", "participants_favor_proposed"]} for p in comparisons[-3:]],
        previous_p_changes=len(changes), previous_significance_changes=result["previous_significance_changes"]), indent=2))


if __name__ == "__main__":
    main()
