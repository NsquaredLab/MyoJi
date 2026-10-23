"""Reconstruct DPose subject-by-task errors and fit the declared repeated-measures model.

Only saved predictions are read. No training, new inference, or GPU is needed.
Run before build_dpose_combined.py; the frozen protocol specifies the estimand.
"""

from __future__ import annotations

import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from paper.common import dpose_figure_statistics as stats
from paper.common import paths

ROOT = Path(__file__).resolve().parents[1]
INPUTS = paths.DATA / "figure_inputs"  # frozen protocol, retrieval receipt and prior-figure snapshot
PROTOCOL = INPUTS / "dpose_movement_analysis_protocol.json"
RESULT = ROOT / "results/dpose_movement_analysis.json"
OUT = ROOT / "output/figures"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_repeated(values):
    """Saturated task means with participant-cluster covariance, including batches.

    Input axes are (..., participant, task). Only the equal-task marginal contrast
    is tested. No normal/Wald approximation or cell-wise sign flips are used.
    """
    values = np.asarray(values, dtype=np.float64)
    n, tasks = values.shape[-2:]
    if (n, tasks) != (9, 8) or not np.isfinite(values).all():
        raise ValueError("Expected complete finite 9-participant by 8-task data")
    beta = values.mean(axis=-2)
    residual = values - beta[..., None, :]
    covariance = np.einsum("...st,...su->...tu", residual, residual) / (n * (n - 1))
    mean = beta.mean(axis=-1)
    variance = covariance.sum(axis=(-2, -1)) / tasks**2
    se = np.sqrt(np.maximum(variance, 0))
    t = np.divide(mean, se, out=np.where(mean == 0, 0., np.copysign(np.inf, mean)), where=se > 0)
    return beta, covariance, mean, se, t


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run():
    protocol = json.loads(PROTOCOL.read_text())
    subjects, tasks, seeds = (protocol[k] for k in ("subjects", "task_ids", "seeds"))
    archive = INPUTS / Path(protocol["previous_figure_archive"]).name
    previous = json.loads((archive / "results/dpose_cnn_maxT_analysis.json").read_text())
    old_figure = json.loads((archive / "output/figures/dpose-error-and-resources.json").read_text())
    old_rows = {(r["campaign"], r["arm"]): r for r in old_figure["decoder_rows"]}
    sources = {}

    def source(path):
        sources[str(path.relative_to(ROOT))] = digest(path)

    source(PROTOCOL)
    source(Path(__file__))
    receipt_path = INPUTS / "dpose_movement_source_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    source(receipt_path)
    source(archive / "results/dpose_cnn_maxT_analysis.json")
    source(archive / "output/figures/dpose-error-and-resources.json")
    manifest = ROOT / "results/dpose_within_v1/selection.json"
    source(manifest)
    selection = json.loads(manifest.read_text())["windows"]
    records = [r for r in selection if r["split"] == "test"]
    assert len(records) == 144
    canonical = {}
    for subject in subjects:
        chosen = [r for r in records if r["subject"] == subject]
        task = np.array([int(r["task"]) for r in chosen])
        start = np.array([r["start"] for r in chosen])
        assert len(chosen) == 16 and set(task) == set(tasks)
        assert all(np.sum(task == t) == 2 for t in tasks)
        cache = receipt["canonical_test_caches"][str(subject)]
        np.testing.assert_array_equal(cache["tasks"], task)
        np.testing.assert_array_equal(cache["starts"], start)
        np.testing.assert_array_equal(cache["subjects"], np.repeat(subject, 16))
        canonical[subject] = task, start, cache["targets_float64_sha256"]

    definitions = protocol["gap_contrasts"]
    pairs = [(d["display_label"], d["alternative_models"]) for d in definitions]
    pairs.append(("Proposed", definitions[0]["reference_models"]))
    model_keys = sorted({tuple(v) for _, pair in pairs for v in pair.values()})
    arrays, error_cells, seed_cells = {}, [], []
    discrepancies = []
    for campaign, arm in model_keys:
        base = ROOT / "results" / campaign
        path = base / "selection.json"
        if path.exists():
            source(path)
            assert json.loads(path.read_text())["windows"] == selection
        cell = np.zeros((9, 8, 3))
        for si, subject in enumerate(subjects):
            task, start, target_hash = canonical[subject]
            for ki, seed in enumerate(seeds):
                path = base / f"subject_{subject:02d}/fits/{arm}_{seed}/test_predictions.npz"
                source(path)
                if campaign == "dpose_v20_continuous_v1":
                    assert digest(path) == receipt["continuous_prediction_sha256"][str(path.relative_to(base))]
                with np.load(path, allow_pickle=False) as saved:
                    pred = saved["predictions" if "predictions" in saved else "prediction"].astype(np.float64)
                    target = saved["targets" if "targets" in saved else "target"].astype("<f8")
                    sid = saved["subjects" if "subjects" in saved else "subject"]
                    np.testing.assert_array_equal(sid, np.repeat(subject, 16))
                    assert pred.shape == target.shape == (16, 50, 60)
                    assert hashlib.sha256(target.tobytes()).hexdigest() == target_hash
                    if "starts" in saved:
                        np.testing.assert_array_equal(saved["starts"], start)
                distance = np.linalg.norm((pred - target).reshape(16, 50, 20, 3), axis=-1)
                assert np.isfinite(distance).all()
                for ti, movement in enumerate(tasks):
                    cell[si, ti, ki] = distance[task == movement].mean()
                    seed_cells.append(dict(campaign=campaign, arm=arm, subject=subject,
                        task=movement, seed=seed, error_mm=float(cell[si, ti, ki]),
                        windows=2, endpoints_per_window=50, landmarks=20))
                original = old_rows[(campaign, arm)]["error_participant_seed_values"][str(subject)][str(seed)]
                difference = float(cell[si, :, ki].mean() - original)
                discrepancies.append(dict(campaign=campaign, arm=arm, subject=subject,
                                          seed=seed, rescored_minus_published_mm=difference))
                np.testing.assert_allclose(cell[si, :, ki].mean(), original, rtol=0, atol=5e-6)
        arrays[(campaign, arm)] = cell.mean(axis=-1)
        for si, subject in enumerate(subjects):
            for ti, task in enumerate(tasks):
                error_cells.append(dict(campaign=campaign, arm=arm, subject=subject, task=task,
                    error_mm=float(cell[si, ti].mean()), n_seeds=3, n_windows=2))

    gap_arrays, plot_rows, gap_cells = {}, [], []
    for label, pair in pairs:
        gap = arrays[tuple(pair["tokens"])] - arrays[tuple(pair["features"])]
        gap_arrays[label] = gap
        cells = [dict(subject=s, task=t, difference_mm=float(gap[si, ti]))
                 for si, s in enumerate(subjects) for ti, t in enumerate(tasks)]
        gap_cells.extend(dict(decoder=label, **cell) for cell in cells)
        beta, covariance, mean, se, _ = fit_repeated(gap)
        plot_rows.append(dict(label=label, token_key=pair["tokens"], feature_key=pair["features"],
            cells=cells, mean_difference_mm=float(mean), n_participants=9, n_tasks=8,
            n_subject_task_cells=72, n_seeds_per_cell=3,
            task_means_mm={str(t): float(b) for t, b in zip(tasks, beta)},
            participant_means_mm={str(s): float(v) for s, v in zip(subjects, gap.mean(axis=1))},
            task_mean_covariance_mm2=covariance.tolist(), cluster_standard_error_mm=float(se)))

    old_tests = [p for group in previous["groups"] for p in group["comparisons"]]
    marginal = np.array([[p["participant_differences"][str(s)] for p in old_tests] for s in subjects])
    signs = np.array(list(product((-1., 1.), repeat=9)))
    observed = stats.studentized_mean(marginal)
    null_t = stats.studentized_mean(signs[:, :, None] * marginal[None], axis=1)
    results, contrast_cells, max_reduction_error = [], [], 0.
    for definition in definitions:
        label = definition["display_label"]
        values = gap_arrays[label] - gap_arrays["Proposed"]
        beta, covariance, effect, se, t = fit_repeated(values)
        _, _, _, _, permuted_t = fit_repeated(signs[:, :, None] * values[None])
        person = values.mean(axis=1)
        reduced = stats.studentized_mean(signs * person[None], axis=1)
        max_reduction_error = max(max_reduction_error, float(np.max(np.abs(permuted_t - reduced))))
        np.testing.assert_allclose(permuted_t, reduced, rtol=1e-12, atol=1e-12)
        column = next(i for i, p in enumerate(old_tests) if p.get("contrast_id") == definition["id"])
        old = old_tests[column]
        observed[column], null_t[:, column] = t, permuted_t
        contrast_cells.extend(dict(contrast_id=definition["id"], subject=s, task=m,
            other_minus_proposed_gap_mm=float(values[si, ti]))
            for si, s in enumerate(subjects) for ti, m in enumerate(tasks))
        results.append(dict(contrast_id=definition["id"], label=label, column=column,
            effect_mm=float(effect), cluster_standard_error_mm=float(se), t_observed=float(t),
            n_participants=9, n_subject_task_cells=72, n_tasks=8, n_seeds_per_cell=3,
            task_means_mm={str(m): float(b) for m, b in zip(tasks, beta)},
            task_mean_covariance_mm2=covariance.tolist(),
            participant_differences_mm={str(s): float(v) for s, v in zip(subjects, person)},
            effect_ci95_pointwise_mm=stats.exact_location_interval(person),
            prior_effect_mm=old["effect_alternative_minus_reference_mm"],
            prior_t_observed=old["t_observed"],
            difference_from_prior_effect_mm=float(effect - old["effect_alternative_minus_reference_mm"])))
    null_max = np.max(np.abs(null_t), axis=1)
    raw = stats.inclusive_tail(np.abs(null_t), observed)
    adjusted = stats.inclusive_tail(null_max[:, None], observed)
    for result in results:
        column = result.pop("column")
        result.update(p_raw=float(raw[column]), p_adjusted=float(adjusted[column]),
                      significant=bool(adjusted[column] < .05), correction_family_size=51)
    changed = [dict(index=i, contrast_id=p.get("contrast_id"),
                    before=p["p_adjusted"], after=float(adjusted[i]))
               for i, p in enumerate(old_tests) if adjusted[i] != p["p_adjusted"]]
    output = dict(complete=True, protocol=protocol, sources=sources,
        n_prediction_files=216, n_model_seed_task_cells=len(seed_cells),
        n_model_subject_task_cells=len(error_cells), n_gap_cells=len(gap_cells),
        n_contrast_cells=len(contrast_cells), prediction_target_hashes_verified=True,
        canonical_subject_task_order_verified=True, plot_rows=plot_rows, comparisons=results,
        joint_family_size=51, maxT_reference_distribution=null_max.tolist(),
        joint_observed_t=observed.tolist(), joint_p_raw=raw.tolist(), joint_p_adjusted=adjusted.tolist(),
        joint_adjusted_p_changes=changed,
        numerical_precision_audit=dict(rescoring="Float64 Euclidean distances and all reductions",
            max_absolute_seed_error_difference_mm=max(abs(d["rescored_minus_published_mm"]) for d in discrepancies),
            seed_discrepancies=discrepancies,
            max_repeated_vs_marginal_t_difference_all_512_patterns=max_reduction_error),
        interpretation="72 correlated subject-by-task cells per decoder, nine independent participants; fixed eight-task estimand. Overall inference is equivalent to the balanced participant-marginal test, up to rescoring precision. No movement-specific tests.")
    RESULT.write_text(json.dumps(output, indent=2) + "\n")
    OUT.mkdir(exist_ok=True)
    (OUT / "dpose-movement-analysis.json").write_bytes(RESULT.read_bytes())
    (OUT / "dpose_movement_analysis.py").write_bytes(Path(__file__).read_bytes())
    for name, cells in [("errors", error_cells), ("seed-errors", seed_cells),
                        ("gaps", gap_cells), ("contrasts", contrast_cells)]:
        write_csv(OUT / f"dpose-movement-{name}.csv", cells)
    write_csv(OUT / "dpose-movement-tests.csv", [{k: c[k] for k in
        ("contrast_id", "label", "effect_mm", "cluster_standard_error_mm", "t_observed",
         "p_raw", "p_adjusted", "significant", "n_participants", "n_subject_task_cells")}
        for c in results])
    report = ["# Exploratory subject-by-movement analysis", "",
        "C contains 72 subject-by-task observations for each decoder: nine participants, eight shared movement-task types, two test windows per task and three training seeds. The three seeds are averaged within each cell. Distances are recalculated from saved predictions in float64; predictions are never averaged into an ensemble.", "",
        "## Repeated-measures model", "",
        "For each other-minus-proposed tokenization-gap contrast, fit D[s,t] = beta[t] + epsilon[s,t]. Task effects are fixed; within-participant covariance is unrestricted. The eight OLS task means have covariance sum_s (D_s-beta)(D_s-beta)' / (9×8). The estimated contrast is the equal-weight average of these eight task means, with SE sqrt(c'Vc), c=(1/8,…,1/8).", "",
        "The tested null is c'beta=0, allowing nonzero individual task effects. For this balanced complete design, the marginal estimate and its cluster SE reduce exactly to the mean and SE of the nine participant marginals. All eight cells for a participant receive the same sign in each of 512 patterns. Joint max-|t| adjustment retains the previous 51 hypotheses; no new task-wise hypotheses are tested. Exactness requires joint central symmetry of true-null participant-marginal contrast vectors and independent participants. This is not a generic nuisance-adjusted permutation regression or an asymptotic Wald test. It is a repeated-measures model, not a mixed-effects model.", "",
        "| Other decoder | Other − proposed gap (mm) | Cluster SE (mm) | Raw p | Joint max-T p |",
        "|---|---:|---:|---:|---:|"]
    report.extend(f"| {c['label']} | {c['effect_mm']:.4f} | {c['cluster_standard_error_mm']:.4f} | {c['p_raw']:.6f} | {c['p_adjusted']:.6f} |" for c in results)
    report += ["", "Positive estimates favor a smaller signed token−feature gap for proposed. This does not test overall accuracy, absolute gap size, or equivalence. All three comparisons remain nonsignificant." if not any(c["significant"] for c in results) else "See the adjusted p-values above; positive estimates favor a smaller signed gap for proposed.", "",
        "## Audit and scope", "",
        f"All 216 prediction files were verified against subject/task/start metadata and exact target-array hashes from the original test caches. All 27 retrieved proposed-feature files match the remote saved-file hashes. Reconstruction covers {len(seed_cells)} seed-specific task errors, {len(error_cells)} model-by-subject-by-task errors, {len(gap_cells)} signed gaps and {len(contrast_cells)} comparison cells.", "",
        f"Maximum seed-score difference from previous exports: {output['numerical_precision_audit']['max_absolute_seed_error_difference_mm']:.3g} mm, due to float32 versus float64 rescoring. Maximum t discrepancy between the repeated-measures and participant-marginal calculations over every sign pattern: {max_reduction_error:.3g}. Changed adjusted p-values in the full family: {len(changed)}.", "",
        "Nine participants remain the independent replication units. The task-level reformulation was requested after aggregate results had been inspected and is exploratory. It adds detail about these eight movement tasks, not independent participants or evidence for unseen movements. No cells, tasks, participants, seeds, or outliers are excluded. Windows are not assumed to be independent movement repetitions.", "",
        f"Frozen protocol: `{PROTOCOL.relative_to(ROOT)}`. Previous figure: `{protocol['previous_figure_archive']}`.", "",
        "Method references: [Winkler et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/), [FSL repeated-measures designs](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/glm.html), [Romano and Wolf](https://doi.org/10.1198/016214504000000539), [Lazic](https://d-nb.info/1093020903/34)."]
    (paths.RESULTS / "DPOSE_MOVEMENT_ANALYSIS.md").write_text("\n".join(report) + "\n")
    (OUT / "dpose-movement-analysis.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"complete": True, "files_verified": 216,
        "comparisons": [{k: c[k] for k in ("label", "effect_mm", "p_raw", "p_adjusted")} for c in results],
        "joint_p_changes": changed,
        "ranges": {r["label"]: [min(c["difference_mm"] for c in r["cells"]), max(c["difference_mm"] for c in r["cells"])] for r in plot_rows}}, indent=2))


if __name__ == "__main__":
    run()
