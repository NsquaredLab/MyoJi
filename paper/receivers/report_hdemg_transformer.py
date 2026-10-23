"""Independently verify token-transformer predictions against the frozen TCN."""

import argparse
import json
from pathlib import Path

import numpy as np


def report(root, reference, output):
    new = json.loads((root / "results.json").read_text())
    old = json.loads((reference / "results.json").read_text())
    p = json.loads((root / "protocol.json").read_text())
    assert new["complete"] and old["complete"]
    rows, baselines = [], []
    for subject in p["subjects"]:
        folder = root / f"subject_{subject:02d}"
        with np.load(folder / "baseline_training_targets.npz") as data:
            mean = data["target"].reshape(-1, 60).mean(0)
        for seed in p["seeds"]:
            with np.load(
                folder / "fits" / f"tokens_{seed}" / "test_predictions.npz"
            ) as data:
                pred, target = data["prediction"], data["target"]
                assert np.all(data["subject"] == subject)
            with np.load(
                reference
                / f"subject_{subject:02d}/fits/tokens_{seed}/test_predictions.npz"
            ) as data:
                tcn = data["prediction"]
                assert np.array_equal(target, data["target"])
            assert pred.shape == target.shape == tcn.shape == (16, 50, 60)
            assert np.isfinite(pred).all()
            error = float(
                np.linalg.norm((pred - target).reshape(-1, 20, 3), axis=-1).mean()
            )
            tcn_error = float(
                np.linalg.norm((tcn - target).reshape(-1, 20, 3), axis=-1).mean()
            )
            baseline = float(
                np.linalg.norm((target - mean).reshape(-1, 20, 3), axis=-1).mean()
            )
            r2 = float(
                1
                - ((pred - target) ** 2).sum()
                / ((target - target.mean((0, 1))) ** 2).sum()
            )
            saved = next(
                r
                for r in new["per_subject"][str(subject)]["results"]
                if r["seed"] == seed
            )
            saved_old = next(
                r
                for r in old["per_subject"][str(subject)]["results"]
                if r["seed"] == seed and r["arm"] == "tokens"
            )
            for actual, expected in [
                (error, saved["macro_subject_error"]),
                (tcn_error, saved_old["macro_subject_error"]),
                (baseline, new["constant_subject_train_mean"][str(subject)]),
                (r2, saved["subjects"][str(subject)]["coordinate_r2"]),
            ]:
                assert np.isclose(actual, expected, rtol=2e-6, atol=2e-6)
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "transformer_error": error,
                    "tcn_error": tcn_error,
                    "coordinate_r2": r2,
                }
            )
        baselines.append(baseline)
    assert len(rows) == 27

    def aggregate(key):
        means = [
            np.mean([r[key] for r in rows if r["seed"] == seed]) for seed in p["seeds"]
        ]
        return {"mean": float(np.mean(means)), "seed_sd": float(np.std(means, ddof=1))}

    transformer, tcn = aggregate("transformer_error"), aggregate("tcn_error")
    for key in ("mean", "seed_sd"):
        assert np.isclose(
            transformer[key], new["summary"]["tokens"][key], rtol=2e-6, atol=2e-6
        )
    audit = {
        "verified": True,
        "predictions_checked": 27,
        "paired_tcn_predictions_checked": 27,
        "transformer": transformer,
        "tcn": tcn,
        "constant_baseline": float(np.mean(baselines)),
        "per_subject_seed": rows,
        "scope": p["scope"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "hdemg_transformer_result_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    lines = [
        "# EMG → frozen tokenizer → causal transformer → kinematics",
        "",
        "All 27 transformer fits completed; test predictions and the paired TCN predictions were independently checked against identical targets. Baselines were recomputed from training targets.",
        "",
        "| Decoder/input | Mean 3D error | Seed SD |",
        "|---|---:|---:|",
        f"| Token transformer | {transformer['mean']:.3f} | {transformer['seed_sd']:.3f} |",
        f"| Token TCN (previous run) | {tcn['mean']:.3f} | {tcn['seed_sd']:.3f} |",
        f"| Personalized constant pose | {np.mean(baselines):.3f} | — |",
        "",
        "Errors use stored coordinate units, not verified millimetres. Seed SD describes the three seed macro-subject averages, not population uncertainty. This is an exploratory architecture comparison on a previously inspected same-person/session/task test split; the tokenizer saw these recordings in pretraining. No unseen-person claim follows. Both architectures use the same training settings and data but have different parameter counts.",
        "",
        "| Subject | Transformer (seed mean) | TCN (seed mean) |",
        "|---|---:|---:|",
    ]
    for subject in p["subjects"]:
        r = [r for r in rows if r["subject"] == subject]
        lines.append(
            f"| {subject} | {np.mean([v['transformer_error'] for v in r]):.3f} | {np.mean([v['tcn_error'] for v in r]):.3f} |"
        )
    lines += [
        "",
        "[Protocol](HDEMG_TRANSFORMER_PROTOCOL.md) · [Numeric audit](hdemg_transformer_result_audit.json)",
        "",
        "The main manuscript/PDF has not been automatically updated.",
    ]
    (output / "HDEMG_TRANSFORMER_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({k: v for k, v in audit.items() if k != "per_subject_seed"}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.reference, args.output)
