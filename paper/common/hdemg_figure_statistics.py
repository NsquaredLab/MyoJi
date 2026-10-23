"""Participant-paired CNN-reference comparisons and archived tests."""

from itertools import combinations, product

import numpy as np

ALPHA = 0.05
METHOD_SOURCE = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html"
CORRECTION_SOURCE = "https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html"
LEGACY_METHOD = (
    "Two-sided exact paired permutation test of the mean participant "
    "error difference. Average the three decoder seeds within each participant; "
    "enumerate all 512 within-participant label swaps for nine independent "
    "participants. The null assumes exchangeable model labels within each pair "
    "(sign symmetry of paired differences). Holm controls familywise error at "
    "0.05 separately over every pair in each retained statistical family: matched CNN, "
    "TCN and Transformer (10), second CNN and proposed decoder variants (3), frozen Qwen (6), "
    "and smaller decoder baselines (6). This is not a single global 25-test family. The CNN/proposed-decoder and smaller-baseline comparisons concern fitted systems "
    "with different recipes; they do not isolate architecture effects. Display-section headings and row "
    "order do not redefine correction families."
)
HOLM_METHOD = (
    "Two-sided exact paired sign-flip tests of mean 3D error differences. "
    "Each of the two CNN references is compared with every one of the 14 other models "
    "in the main figure and supplement: one Holm family of 28 contrasts. Average "
    "the three decoder seeds within each participant, then enumerate all 512 sign "
    "patterns for the nine independent participants. The null assumes exchangeable "
    "within-participant model labels (symmetric paired differences). Effects are "
    "alternative minus CNN-reference error in millimetres; negative values favor "
    "the alternative. Model histories, pretraining and some training protocols differ, "
    "so these are comparisons of fitted systems. The minimum two-sided raw p is 2/512; "
    "with 28 Holm comparisons the minimum achievable adjusted p is 0.109375. "
    "Thus this sample/test/correction combination cannot reject at alpha 0.05, "
    "irrespective of effect magnitude. Non-rejection is not equivalence."
)
MAXT_SOURCE = "https://doi.org/10.1198/016214504000000539"
SIGN_FLIP_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/"
METHOD = (
    "Participant-paired single-step max-T analysis of all declared "
    "CNN-reference, within-decoder token-versus-feature, and between-decoder tokenization-gap contrasts in one family. Average three decoder seeds within each "
    "of nine participants. Standardize each mean paired difference by its sample "
    "standard error, then enumerate all 512 participant sign vectors, applying each "
    "participant sign synchronously to all contrasts. Recompute each t statistic "
    "and retain the largest absolute t per sign vector. Each adjusted p is the "
    "inclusive tail proportion of this maximum distribution beyond the observed "
    "absolute t; no additional Holm correction is applied. Finite-sample strong "
    "familywise error control requires independent participants and joint central "
    "sign symmetry of their true-null contrast vectors. Marginal symmetry alone "
    "is insufficient; these assumptions cannot be established with nine participants. "
    "Effects are alternative minus reference error in millimetres; negative "
    "favors the alternative. All four direct representation contrasts use tokens minus "
    "encoder features, so positive favors features. Gap contrasts subtract proposed's token-minus-feature "
    "gap from the other decoder's gap, so positive favors proposed. Exact 95% pointwise location intervals invert the "
    "univariate sign-flip test, assuming independent differences symmetric about "
    "a common effect, equal to the mean when finite. Intervals are not simultaneous and do not assign stars. "
    "All nine leave-one-participant-out mean differences describe influence. "
    "Histories, pretraining and some training protocols differ, so these compare "
    "fitted systems. The method was fixed before the max-T outcomes were computed. "
    "The previous global Holm analysis is retained as sensitivity output. "
    "Non-rejection does not establish equivalence."
)


def exact_paired_test(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != (9,) or b.shape != (9,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Expected nine paired finite participant means")
    differences = a - b
    observed = float(differences.mean())
    signs = np.asarray(list(product((-1., 1.), repeat=len(differences))))
    null = np.mean(signs * differences, axis=1)
    tolerance = 100 * np.finfo(np.float64).eps * max(1., abs(observed))
    # The exact sign-flip distribution is symmetric. Include tied outcomes;
    # no Monte Carlo +1 adjustment is applied when all outcomes are enumerated.
    pvalue = float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    return observed, pvalue, differences.tolist()


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted.tolist()


def analyze(rows, groups):
    by_key = {(row["campaign"], row["arm"]): row for row in rows}
    output = {"method": LEGACY_METHOD, "alpha": ALPHA, "replication_unit": "participant",
              "seeds_averaged_before_testing": 3, "sources": [METHOD_SOURCE, CORRECTION_SOURCE],
              "groups": [], "excluded": {
                "latency": "Repeated calls in one benchmark session are descriptive measurements, not independent fitted participants.",
                "model_size_and_payload": "Deterministic tensor sizes and format calculations; no sampling uncertainty."}}
    for group in groups:
        keys = group["variants"]
        selected = [by_key[key] for key in keys]
        if group["id"] == "qwen" and all(r["error_evaluation"] == "best_validation_only" for r in selected):
            output["excluded"]["qwen"] = "One-person validation pilot; no participant-level test."
            continue
        assert all(r["error_evaluation"] == "chronological_test" and r["error_n_participants"] == 9
                   and r["error_n_seeds"] == 3 for r in selected)
        participants = sorted(selected[0]["error_participant_means"], key=int)
        assert len(participants) == 9
        values = []
        for row in selected:
            assert set(row["error_participant_means"]) == set(participants)
            vector = [row["error_participant_means"][p] for p in participants]
            np.testing.assert_allclose(np.mean(vector), row["error_mean"], rtol=0, atol=1e-10)
            values.append(vector)
        pairs = []
        for i, j in combinations(range(len(selected)), 2):
            effect, pvalue, differences = exact_paired_test(values[i], values[j])
            pairs.append({"i": i, "j": j, "a": list(keys[i]), "b": list(keys[j]),
                "label_a": selected[i]["label"], "label_b": selected[j]["label"],
                "mean_difference_a_minus_b": effect, "participant_differences": dict(zip(participants, differences)),
                "p_raw": pvalue, "n_participants": 9, "n_seeds_per_participant": 3, "permutations": 512})
        for pair, adjusted in zip(pairs, holm_adjust([p["p_raw"] for p in pairs])):
            pair["p_holm"] = adjusted
            pair["significant"] = adjusted < ALPHA
        output["groups"].append({"id": group["id"], "label": group["label"],
            "family_size": len(pairs), "participants": participants, "comparisons": pairs})
    return output


def analyze_references_holm(rows, protocol):
    """All declared alternative-versus-CNN contrasts, corrected together."""
    by_key = {(row["campaign"], row["arm"]): row for row in rows}
    references = [tuple(key) for key in protocol["references"]]
    alternatives = [tuple(key) for key in protocol["alternatives"]]
    assert len(by_key) == len(rows) == 16
    assert len(references) == 2 and len(alternatives) == 14
    assert not set(references).intersection(alternatives)
    assert set(references + alternatives) == set(by_key)
    participants = sorted(by_key[references[0]]["error_participant_means"], key=int)
    assert len(participants) == 9
    values = {}
    for key, row in by_key.items():
        assert row["error_evaluation"] == "chronological_test"
        assert row["error_n_participants"] == 9 and row["error_n_seeds"] == 3
        assert set(row["error_participant_means"]) == set(participants)
        values[key] = [row["error_participant_means"][p] for p in participants]
        np.testing.assert_allclose(np.mean(values[key]), row["error_mean"], rtol=0, atol=1e-10)
    pairs = []
    for reference in references:
        for alternative in alternatives:
            effect, raw, differences = exact_paired_test(values[alternative], values[reference])
            pairs.append({
                "i": alternatives.index(alternative), "j": references.index(reference),
                "a": list(alternative), "b": list(reference),
                "alternative": list(alternative), "reference": list(reference),
                "label_a": by_key[alternative]["label"], "label_b": by_key[reference]["label"],
                "mean_difference_a_minus_b": effect,
                "effect_alternative_minus_reference_mm": effect,
                "participant_differences": dict(zip(participants, differences)),
                "p_raw": raw, "n_participants": 9, "n_seeds_per_participant": 3,
                "permutations": 512,
            })
    assert len(pairs) == protocol["family_size"] == 28
    assert [[pair["reference"], pair["alternative"]] for pair in pairs] == protocol["planned_contrasts"]
    for pair, adjusted in zip(pairs, holm_adjust([pair["p_raw"] for pair in pairs])):
        pair["p_holm"] = adjusted
        pair["significant"] = adjusted < ALPHA
    lower_bound = len(pairs) * 2 / (2 ** len(participants))
    assert all(pair["p_holm"] >= lower_bound for pair in pairs)
    return {
        "method": HOLM_METHOD, "alpha": ALPHA, "replication_unit": "participant",
        "seeds_averaged_before_testing": 3,
        "sources": [METHOD_SOURCE, CORRECTION_SOURCE],
        "comparison_protocol": protocol,
        "groups": [{"id": "cnn_references", "label": "Each CNN reference versus all alternatives",
                    "family_size": len(pairs), "participants": participants, "comparisons": pairs}],
        "minimum_possible_raw_p": 2 / (2 ** len(participants)),
        "minimum_possible_holm_p": lower_bound,
        "minimum_observed_holm_p": min(pair["p_holm"] for pair in pairs),
        "significant_comparisons": sum(pair["significant"] for pair in pairs),
        "can_reject_at_alpha_given_resolution": lower_bound < ALPHA,
        "excluded": {"reference_vs_reference": "Not part of the requested CNN-versus-alternative question.",
                     "alternative_vs_alternative": "Earlier within-family contrasts are archived separately.",
                     "published_JNE_aggregate": "No matched participant-level observations under this protocol.",
                     "incomplete_TBME_rerun": "No complete final cohort in this figure.",
                     "resources": "Timing, model bytes and payload are not independent participant-level error measurements."},
    }


def studentized_mean(values, axis=0):
    """A t statistic, with defined behavior for zero-variance columns."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[axis]
    if n < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite paired differences")
    mean = values.mean(axis=axis)
    se = values.std(axis=axis, ddof=1) / np.sqrt(n)
    out = np.where(mean == 0, 0., np.copysign(np.inf, mean))
    return np.divide(mean, se, out=out, where=se > 0)


def inclusive_tail(null, observed):
    observed = np.abs(np.asarray(observed, dtype=np.float64))
    scale = np.where(np.isfinite(observed), np.maximum(1., observed), 1.)
    tolerance = 100 * np.finfo(np.float64).eps * scale
    return np.mean(np.asarray(null) >= observed - tolerance, axis=0)


def exact_max_t(differences):
    """Synchronize participant flips; maximize across the complete family."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or not 2 <= differences.shape[0] <= 16:
        raise ValueError("Expected a participant-by-contrast matrix with 2-16 participants")
    signs = np.asarray(list(product((-1., 1.), repeat=differences.shape[0])))
    observed = studentized_mean(differences)
    flipped = signs[:, :, None] * differences[None, :, :]
    null_t = np.abs(studentized_mean(flipped, axis=1))
    null_max = np.max(null_t, axis=1)
    raw = inclusive_tail(null_t, observed)
    adjusted = inclusive_tail(null_max[:, None], observed)
    assert np.all(adjusted >= raw)
    return observed, raw, adjusted, null_max


def exact_location_interval(differences, alpha=ALPHA):
    """Invert the inclusive univariate sign-flip test; pointwise coverage only.

    For each nonconstant sign vector, extremeness for a candidate location delta
    holds between the means of its positive and negative subsets. Each interval
    contains the full-sample mean, so the accepted set is itself an interval.
    The two constant sign vectors are always extreme. Complementary sign vectors
    are intentionally both retained, exactly as in the enumerated p-value.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or not 2 <= len(d) <= 16 or not np.isfinite(d).all() or not 0 < alpha < 1:
        raise ValueError("Expected 2-16 finite differences and 0 < alpha < 1")
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    required = int(np.ceil(alpha * len(signs))) - 2
    if required <= 0:
        return [-np.inf, np.inf]
    masks = signs[1:-1] > 0
    positive = (masks * d).sum(axis=1) / masks.sum(axis=1)
    negative = ((~masks) * d).sum(axis=1) / (~masks).sum(axis=1)
    lower = np.sort(np.minimum(positive, negative))[required - 1]
    upper = np.sort(np.maximum(positive, negative))[-required]
    return [float(lower), float(upper)]


def analyze_references(rows, protocol):
    """CNN contrasts and declared direct contrasts, corrected jointly."""
    by_key = {(row["campaign"], row["arm"]): row for row in rows}
    references = [tuple(key) for key in protocol["references"]]
    alternatives = [tuple(key) for key in protocol["alternatives"]]
    assert len(by_key) == len(rows) and not set(references).intersection(alternatives)
    assert set(references + alternatives) == set(by_key)
    historical = protocol.get("historical_holm_protocol")
    if historical is None:
        contrasts = [[list(ref), list(alt)] for ref in references[:2] for alt in alternatives]
        historical = dict(protocol, references=[list(ref) for ref in references[:2]],
                          family_size=len(contrasts), planned_contrasts=contrasts)
    historical_keys = {tuple(key) for key in historical["references"] + historical["alternatives"]}
    previous = analyze_references_holm([row for key, row in by_key.items() if key in historical_keys], historical)
    participants = previous["groups"][0]["participants"]
    holm_by_pair = {(tuple(p["reference"]), tuple(p["alternative"])): p["p_holm"]
                    for p in previous["groups"][0]["comparisons"]}
    values = {}
    for key, row in by_key.items():
        assert row["error_evaluation"] == "chronological_test"
        assert row["error_n_participants"] == len(participants) == 9 and row["error_n_seeds"] == 3
        assert set(row["error_participant_means"]) == set(participants)
        values[key] = [row["error_participant_means"][p] for p in participants]
        np.testing.assert_allclose(np.mean(values[key]), row["error_mean"], rtol=0, atol=1e-10)
    direct = {(tuple(c["reference"]), tuple(c["alternative"])): c
              for c in protocol.get("direct_contrasts", [])}
    gaps = {(tuple(c["reference"]), tuple(c["alternative"])): c
            for c in protocol.get("gap_contrasts", [])}
    cnn_keys = {(ref, alt) for ref in references for alt in alternatives}
    cnn_keys.update((tuple(c["reference"]), tuple(c["alternative"]))
                    for c in protocol.get("reference_contrasts", []))
    planned = [(tuple(ref), tuple(alt)) for ref, alt in protocol["planned_contrasts"]]
    assert set(planned) == cnn_keys | set(direct) | set(gaps)
    assert not cnn_keys.intersection(direct) and not set(gaps).intersection(cnn_keys | set(direct))
    assert len({frozenset(pair) for pair in planned}) == len(planned)
    pairs = []
    for reference, alternative in planned:
        if (reference, alternative) in gaps:
            contrast = gaps[(reference, alternative)]
            model_keys = {side: {rep: tuple(key) for rep, key in contrast[f"{side}_models"].items()}
                          for side in ("alternative", "reference")}
            gap_values = {side: np.asarray(values[keys["tokens"]]) - np.asarray(values[keys["features"]])
                          for side, keys in model_keys.items()}
            for keys in model_keys.values():
                assert by_key[keys["tokens"]]["representation"] == "tokens"
                assert by_key[keys["features"]]["representation"] == "continuous"
            effect, raw_p, differences = exact_paired_test(gap_values["alternative"], gap_values["reference"])
            pairs.append({
                "i": list(gaps).index((reference, alternative)), "j": len(references) + 1,
                "contrast_kind": "representation_gap", "contrast_id": contrast["id"],
                "contrast_label": contrast["label"], "display_label": contrast["display_label"],
                "table_label": contrast["table_label"], "effect_direction": contrast["effect_direction"],
                "a": list(alternative), "b": list(reference),
                "alternative": list(alternative), "reference": list(reference),
                "label_a": contrast["table_label"] + " token−feature gap",
                "label_b": "Proposed token−feature gap",
                "alternative_models": contrast["alternative_models"], "reference_models": contrast["reference_models"],
                "mean_difference_a_minus_b": effect, "effect_alternative_minus_reference_mm": effect,
                "participant_differences": dict(zip(participants, differences)),
                "p_raw": raw_p, "p_holm_sensitivity": None,
                "n_participants": len(participants), "n_seeds_per_participant": 3, "permutations": 512})
            continue
        a, b = by_key[alternative], by_key[reference]
        effect, raw_p, differences = exact_paired_test(values[alternative], values[reference])
        is_direct = (reference, alternative) in direct
        pair = {
            "i": (list(direct).index((reference, alternative)) if is_direct else
                  alternatives.index(alternative) if alternative in alternatives else
                  len(alternatives) + references.index(alternative)),
            "j": len(references) if is_direct else references.index(reference),
            "contrast_kind": (direct[(reference, alternative)].get("kind", "proposed_representation")
                              if is_direct else "cnn_reference"),
            "a": list(alternative), "b": list(reference),
            "alternative": list(alternative), "reference": list(reference),
            "label_a": a["label"], "label_b": b["label"],
            "mean_difference_a_minus_b": effect, "effect_alternative_minus_reference_mm": effect,
            "participant_differences": dict(zip(participants, differences)),
            "p_raw": raw_p, "p_holm_sensitivity": holm_by_pair.get((reference, alternative)),
            "n_participants": len(participants), "n_seeds_per_participant": 3, "permutations": 512,
        }
        if is_direct:
            contrast = direct[(reference, alternative)]
            pair.update(contrast_id=contrast["id"], contrast_label=contrast["label"],
                        effect_direction=contrast["effect_direction"])
        pairs.append(pair)
    assert len(pairs) == protocol["family_size"]
    assert [[pair["reference"], pair["alternative"]] for pair in pairs] == protocol["planned_contrasts"]
    assert len({(tuple(pair["a"]), tuple(pair["b"])) for pair in pairs}) == len(pairs)
    family = dict(id="cnn_references", family_size=len(pairs), participants=participants,
                  comparisons=pairs, label="CNN references, within-decoder representations and between-decoder gaps")
    d = np.array([[pair["participant_differences"][p] for pair in pairs] for p in participants])
    observed, raw, adjusted, null_max = exact_max_t(d)
    np.testing.assert_allclose(raw, [pair["p_raw"] for pair in pairs], rtol=0, atol=1e-12)
    for j, pair in enumerate(pairs):
        pair["t_observed"] = float(observed[j])
        pair["p_maxT"] = pair["p_adjusted"] = float(adjusted[j])
        pair["significant"] = bool(adjusted[j] < ALPHA)
        pair["effect_ci95_pointwise_mm"] = exact_location_interval(d[:, j])
        loo = (d[:, j].sum() - d[:, j]) / (len(participants) - 1)
        pair["leave_one_participant_out_effects_mm"] = dict(zip(participants, loo.tolist()))
        pair["leave_one_participant_out_effect_range_mm"] = [float(loo.min()), float(loo.max())]
        pair["participants_favor_alternative"] = int(np.sum(d[:, j] < 0))
        pair["participants_favor_reference"] = int(np.sum(d[:, j] > 0))
    return {
        "method": METHOD, "method_id": "exact_paired_single_step_maxT", "alpha": ALPHA,
        "replication_unit": "participant", "seeds_averaged_before_testing": 3,
        "sources": [MAXT_SOURCE, SIGN_FLIP_SOURCE, METHOD_SOURCE],
        "comparison_protocol": protocol, "groups": [family],
        "minimum_possible_raw_p": 2 / len(null_max),
        "minimum_possible_adjusted_p": 2 / len(null_max),
        "minimum_observed_adjusted_p": float(min(adjusted)),
        "significant_comparisons": int(np.sum(adjusted < ALPHA)),
        "can_reject_at_alpha_given_resolution": 2 / len(null_max) < ALPHA,
        "maxT_reference_distribution": null_max.tolist(),
        "pointwise_interval_method": "Exact inversion of the inclusive two-sided univariate sign-flip location test; not simultaneous.",
        "sensitivity_previous_holm": {
            "family_size": historical["family_size"],
            "significant_comparisons": sum(pair["p_holm_sensitivity"] < ALPHA for pair in pairs
                                           if pair["p_holm_sensitivity"] is not None),
            "minimum_observed_adjusted_p": previous["minimum_observed_holm_p"],
            "archive": protocol.get("holm_analysis_archive", protocol["previous_analysis_archive"]),
            "note": "Previous global Holm correction for the original 28 CNN-reference contrasts; direct and TBME contrasts were not included."
        },
        "excluded": {
            "old_reference_vs_old_reference": "Old matched CNN versus old L1/SWA CNN remains outside the declared family.",
            "alternative_vs_alternative": "Four within-decoder representation pairs and three other-minus-proposed gap contrasts are included; other alternative comparisons are excluded.",
            "published_JNE_aggregate": "No matched participant-level observations under this protocol.",
            "resources": "Resource measurements are not participant-level error observations."},
    }
