"""Verify all matched fits, paired initializations, schedules and predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def report(root, output):
    p = json.loads((root / "protocol.json").read_text())
    ph = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    rows = []
    baselines = []
    fits = []
    for subject in p["subjects"]:
        folder = root / f"subject_{subject:02d}"
        with np.load(folder / "baseline_training_targets.npz") as data:
            mean = data["target"].reshape(-1, 60).mean(0)
        reference = None
        initial = {}
        statistics = set()
        for arm in p["arms"]:
            for seed in p["seeds"]:
                out = folder / f"fits/{arm}_{seed}"
                fit = json.loads((out / "fit.json").read_text())
                assert fit["complete"] and fit["protocol_sha256"] == ph
                assert (
                    fit["successful_updates"] == 15000
                    and fit["endpoint_presentations"] == 1500000
                )
                assert (
                    len(fit["history"]) == 150
                    and fit["sampling_sha256"] == p["sampling_sha256"][str(seed)]
                )
                assert fit["parameters"] == p["parameters"][arm]
                fits.append(
                    {
                        k: fit[k]
                        for k in (
                            "subject",
                            "arm",
                            "seed",
                            "parameters",
                            "successful_updates",
                            "endpoint_presentations",
                            "peak_gpu_bytes",
                            "seconds_this_invocation",
                            "amp_retries",
                        )
                    }
                )
                initial[(arm, seed)] = fit["initial_state_sha256"]
                statistics.add(fit["target_stats_sha256"])
                with np.load(out / "test_predictions.npz") as data:
                    pred, target = data["prediction"], data["target"]
                    assert np.all(data["subject"] == subject)
                assert (
                    pred.shape == target.shape == (16, 50, 60)
                    and np.isfinite(pred).all()
                )
                if reference is None:
                    reference = target.copy()
                assert np.array_equal(reference, target)
                delta = pred - target
                error = float(np.linalg.norm(delta.reshape(-1, 20, 3), axis=-1).mean())
                assert np.isclose(
                    error,
                    json.loads((out / "test.json").read_text())["mean_3d_error"],
                    rtol=2e-6,
                    atol=2e-6,
                )
                rows.append(
                    {
                        "subject": subject,
                        "arm": arm,
                        "seed": seed,
                        "mean_3d_error": error,
                        "coordinate_mae": float(np.abs(delta).mean()),
                        "coordinate_r2": float(
                            1
                            - (delta**2).sum()
                            / ((target - target.mean((0, 1))) ** 2).sum()
                        ),
                    }
                )
        assert len(statistics) == 1
        for kind in ("tcn", "transformer"):
            for seed in p["seeds"]:
                assert (
                    initial[(kind + "_tokens", seed)]
                    == initial[(kind + "_continuous", seed)]
                )
        baselines.append(
            float(np.linalg.norm((reference - mean).reshape(-1, 20, 3), axis=-1).mean())
        )
    assert len(rows) == 135
    summary = {}
    for arm in p["arms"]:
        values = [
            np.mean(
                [
                    r["mean_3d_error"]
                    for r in rows
                    if r["arm"] == arm and r["seed"] == seed
                ]
            )
            for seed in p["seeds"]
        ]
        summary[arm] = {
            "mean": float(np.mean(values)),
            "seed_sd": float(np.std(values, ddof=1)),
        }
    result = {
        "complete": True,
        "verified": True,
        "paired_initializations_exact": True,
        "targets_exact_across_arms": True,
        "sampling_schedules_exact": True,
        "summary": summary,
        "constant_baseline": float(np.mean(baselines)),
        "per_subject_seed": rows,
        "fit_resources": fits,
        "scope": p["scope"],
        "protocol_sha256": ph,
    }
    output.mkdir(parents=True, exist_ok=True)
    for path in (root / "results.json", output / "hdemg_matched_result_audit.json"):
        path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Capacity- and optimization-matched HD-EMG results",
        "",
        "All 135 fits completed and passed prediction, target, sampling-budget and paired-initialization verification.",
        "",
        "| Arm | Trainable parameters | Mean 3D error | Seed SD |",
        "|---|---:|---:|---:|",
    ]
    for arm, v in summary.items():
        lines.append(
            f"| {arm} | {p['parameters'][arm]:,} | {v['mean']:.3f} | {v['seed_sd']:.3f} |"
        )
    lines += [
        f"| Constant pose | — | {np.mean(baselines):.3f} | — |",
        "",
        "All fits have 15,000 optimizer updates and 1.5 million target presentations under the same loss, normalization, sampling and schedule. Trainable model sizes are within 1% of V19. Token/continuous pairs also match architecture, context and initialization.",
        "",
        "Input context, preprocessing, frozen tokenizer parameters/pretraining and FLOPs are not matched to V19. These are same-person/session/task results. Errors are in millimetres; seed SD is not a population confidence interval.",
        "",
        "[Protocol](HDEMG_MATCHED_PROTOCOL.md) · [Audit and per-subject results](hdemg_matched_result_audit.json)",
        "",
        "The main manuscript/PDF has not been automatically updated.",
    ]
    (output / "HDEMG_MATCHED_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    report(a.root, a.output)
