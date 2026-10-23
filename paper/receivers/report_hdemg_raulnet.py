"""Independent matched-target RaulNet/token comparison from saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


def report(root, reference, transformer, output):
    rows, baselines = [], []
    for subject in range(1, 10):
        folder = root / f"subject_{subject:02d}"
        with np.load(folder / "baseline_training_targets.npz") as f:
            mean = f["target"].reshape(-1, 60).mean(0)
        for seed in (260911, 260912, 260913):
            with np.load(folder / f"fits/raulnet_{seed}/test_predictions.npz") as f:
                pred, target = f["prediction"], f["target"]
                assert np.all(f["subject"] == subject)
            assert (
                pred.shape == target.shape == (16, 50, 60) and np.isfinite(pred).all()
            )
            predictions = {"raulnet_v19_q32": pred}
            for name, source in [
                ("token_tcn", reference),
                ("token_transformer", transformer),
            ]:
                with np.load(
                    source
                    / f"subject_{subject:02d}/fits/tokens_{seed}/test_predictions.npz"
                ) as f:
                    assert np.array_equal(target, f["target"])
                    predictions[name] = f["prediction"]
            for name, pred in predictions.items():
                delta = pred - target
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "model": name,
                        "mean_3d_error": float(
                            np.linalg.norm(delta.reshape(-1, 20, 3), axis=-1).mean()
                        ),
                        "coordinate_mae": float(np.abs(delta).mean()),
                        "coordinate_r2": float(
                            1
                            - (delta**2).sum()
                            / ((target - target.mean((0, 1))) ** 2).sum()
                        ),
                    }
                )
        baselines.append(
            float(np.linalg.norm((target - mean).reshape(-1, 20, 3), axis=-1).mean())
        )
    summary = {}
    for name in predictions:
        means = [
            np.mean(
                [
                    r["mean_3d_error"]
                    for r in rows
                    if r["model"] == name and r["seed"] == seed
                ]
            )
            for seed in (260911, 260912, 260913)
        ]
        summary[name] = {
            "mean": float(np.mean(means)),
            "seed_sd": float(np.std(means, ddof=1)),
        }
    old = json.loads((reference / "results.json").read_text())
    assert np.isclose(
        np.mean(baselines), old["constant_macro_subject_error"], rtol=2e-6
    )
    assert np.isclose(
        summary["token_tcn"]["mean"], old["summary"]["tokens"]["mean"], rtol=2e-6
    )
    assert np.isclose(
        summary["token_transformer"]["mean"],
        json.loads((transformer / "results.json").read_text())["summary"]["tokens"][
            "mean"
        ],
        rtol=2e-6,
    )
    result = {
        "complete": True,
        "verified": True,
        "paired_targets_exact": True,
        "summary": summary,
        "constant_baseline": float(np.mean(baselines)),
        "per_subject_seed": rows,
        "scope": "Matched targets/splits, different native model/input/optimization recipes; exploratory same-person/session/task comparison; not equal compute.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "hdemg_raulnet_result_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        "# Matched-endpoint RaulNet V19 comparison",
        "",
        "All 27 V19 fits completed. The 27 V19 and 54 paired token prediction arrays have exactly matching targets. Errors and personalized constant baselines were independently recomputed.",
        "",
        "| Model | Mean 3D error | Seed SD |",
        "|---|---:|---:|",
    ]
    for name, values in summary.items():
        lines.append(f"| {name} | {values['mean']:.3f} | {values['seed_sd']:.3f} |")
    lines += [
        f"| Personalized constant pose | {np.mean(baselines):.3f} | — |",
        "",
        "These are the same nine subjects, chronological split and instantaneous target timestamps. V19 q32 uses native 156 ms raw/lowpass inputs and its longer supervised training recipe; token heads use frozen pretrained tokens and longer temporal context. This is a system comparison, not an isolated architecture or equal-compute comparison. Physical units remain unverified; seed SD is not population uncertainty. The split was previously inspected, and the tokenizer saw these recordings during pretraining.",
        "",
        "[Protocol](HDEMG_RAULNET_PROTOCOL.md) · [Numeric audit](hdemg_raulnet_result_audit.json)",
        "",
        "The manuscript/PDF has not been automatically updated.",
    ]
    (output / "HDEMG_RAULNET_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference", "transformer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    report(a.root, a.reference, a.transformer, a.output)
