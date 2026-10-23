"""Validate the completed TBME cohort and the separate matched cost benchmark."""
import hashlib
import json

import numpy as np

KEY = ("hdemg_tbme2024_v1", "tbme2024_retrained")


def load_accuracy(root):
    sources = {}
    def record(path):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return sources[str(path.relative_to(root))]
    def read(path):
        record(path)
        return json.loads(path.read_text())
    run = root / "results" / KEY[0]
    result = read(run / "results.json")
    protocol = read(run / "protocol.json")
    gate = read(run / "test_gate.json")
    assert result["complete"] and result["verified"] and result["paired_targets_exact"]
    protocol_hash = record(run / "protocol.json")
    assert result["protocol_sha256"] == gate["protocol_sha256"] == protocol_hash
    assert gate["passed"] and gate["fits"] == 27 and result["units"] == "mm"
    by_cell = {(r["subject"], r["seed"]): r for r in result["per_subject_seed"]}
    assert len(by_cell) == len(protocol["cells"]) == 27
    people = {}
    for subject, seed in protocol["cells"]:
        folder = run / f"subject_{subject:02d}/fits/tbme2024_{seed}"
        fit, test = read(folder / "fit.json"), read(folder / "test.json")
        assert fit["complete"] and fit["epochs"] == 50 and fit["steps"] == 950
        assert fit["parameters"] == 27403873
        assert fit["protocol_sha256"] == test["protocol_sha256"] == protocol_hash
        assert fit["best_sha256"] == test["selected_checkpoint_sha256"] == gate["selected_checkpoints"][f"{subject}_{seed}"]
        assert fit["cache_receipt_sha256"] == record(run / f"subject_{subject:02d}/cache_receipt.json")
        assert test["complete"] and test["paired_targets_exact"]
        path = folder / "test_predictions.npz"
        assert record(path) == test["prediction_sha256"]
        with np.load(path) as arrays:
            pred, truth = arrays["prediction"], arrays["target"]
        assert pred.shape == truth.shape == (16, 50, 60)
        assert np.isfinite(pred).all() and np.isfinite(truth).all()
        error = test["error_mm"]
        assert error == by_cell[(subject, seed)]["mean_3d_error"]
        recomputed = np.linalg.norm((pred.astype(np.float64) - truth).reshape(-1, 20, 3), axis=-1).mean()
        np.testing.assert_allclose(recomputed, error, rtol=0, atol=1e-5)
        for campaign, prefix in [("hdemg_within_v1", "tokens"), ("hdemg_raulnet_v1", "raulnet")]:
            paired = root / "results" / campaign / f"subject_{subject:02d}/fits/{prefix}_{seed}/test_predictions.npz"
            record(paired)
            with np.load(paired) as arrays:
                np.testing.assert_array_equal(truth, arrays["target"])
        people.setdefault(subject, {})[seed] = error
    means = {s: float(np.mean(list(people[s].values()))) for s in sorted(people)}
    np.testing.assert_array_equal(list(means.values()), result["participant_means"])
    assert float(np.mean(list(means.values()))) == result["summary"]["mean"]
    row = dict(campaign=KEY[0], arm=KEY[1], representation="raw", panel="TBME",
               evaluation="chronological_test", mean=result["summary"]["mean"],
               n_participants=9, n_seeds=3, person_means=means,
               person_seed_errors=people, constant_pose_error=None,
               source_scope=result["scope"])
    return row, sources


def load_resources(root, reference_benchmark, original_rows):
    directory = root / "results/hdemg_workstation_cost_v1"
    path = directory / "results.json"
    result = json.loads(path.read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert result["complete"] and not result["test_evaluated"] and not result["training_performed"]
    assert result["source_a100_benchmark_sha256"] == digest(root / "results/hdemg_resource_benchmark_v1/results.json")
    assert result["hardware"]["gpu"] == "NVIDIA GeForce RTX 4090"
    for field in ("reference", "native_raw", "continuous_v20"):
        current, previous = result["inputs"][field], reference_benchmark["inputs"][field]
        if field == "reference":
            assert current["identities"] == previous["identities"]
            assert current["arrays"] == previous["arrays"]
        else:
            assert current == previous
    prior = {(row["campaign"], row["arm"]): row for row in original_rows}
    rows = []
    assert len(result["results"]) == 17
    for measurement in result["results"]:
        key = (measurement["campaign"], measurement["arm"])
        assert measurement["subject"] == 1 and measurement["seed"] == 260911
        assert measurement["loaded_state_unchanged"] and measurement["warmup_calls"] == 12 and measurement["measured_calls"] == 48
        assert measurement["parameter_dtypes"] == ["torch.float32"]
        if measurement["campaign"] == "hdemg_llm_pilot_v1" and measurement["arm"] != "tokens_bypass":
            assert measurement["frozen_backbone_matches_fitted_pilot"] is True
        assert measurement["model_tensor_bytes"] == sum(measurement["component_tensor_bytes"].values())
        assert measurement["model_tensor_mb"] == measurement["model_tensor_bytes"] / 1e6
        times = np.asarray(measurement["samples_ms"])
        assert times.shape == (48,) and np.isfinite(times).all() and (times > 0).all()
        assert measurement["updates_per_second"] == 1000 / np.median(times)
        if key in prior:
            row = dict(prior[key])
            for field in ("checkpoint_sha256", "model_source_sha256", "parameter_count", "model_tensor_bytes"):
                assert measurement[field] == row[field], (key, field)
        else:
            assert key == KEY and measurement["parameter_count"] == 27403873
            fit = json.loads((root / "results" / KEY[0] / "subject_01/fits/tbme2024_260911/fit.json").read_text())
            assert measurement["checkpoint_sha256"] == fit["best_sha256"]
            row = dict(family="TBME", label="CNN (Sîmpetru 2024, TBME; retrained)",
                       experiment_group="Retrained TBME CNN", experiment_group_id="tbme_reference")
        row.update(measurement, latency_mean_ms=float(times.mean()), latency_sd_ms=float(times.std(ddof=1)),
                   timing_n_calls=48, resource_session=result["hardware"])
        rows.append(row)
    assert {(r["campaign"], r["arm"]) for r in rows} == set(prior) | {KEY}
    assert result["frozen_random_backbone_archive_sha256"] == "d259cc4d0c42511bcf159e8092ec9c533ed38e9f7efe2058c8b177dc1948599c"
    sources = {str(path.relative_to(root)): digest(path),
               str((directory / "protocol.json").relative_to(root)): digest(directory / "protocol.json")}
    return rows, result, sources
