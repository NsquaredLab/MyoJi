"""Independently audit saved noise predictions, paired statistics and paper output."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def collect(root, paper):
    p, result = read(root / "protocol.json"), read(root / "results.json")
    protocol_hash, selection_hash = digest(root / "protocol.json"), digest(root / "selection.json")
    assert result["complete"] and result["protocol_sha256"] == protocol_hash
    assert read(root / "selection.json")["protocol_sha256"] == protocol_hash
    rows = result["rows"]
    key = lambda r: (r["subject"], r["arm"], r["filtering"], r["name"], r["seed"])
    lookup = {key(r): r for r in rows}
    assert len(lookup) == len(rows) == 2106
    errors, hashes = [], {}
    for subject in p["subjects"]:
        paired = []
        for family, arms in (("v19", ["native_v19"]), ("v20", ["token_v20", "continuous_v20"])):
            folder = root / f"test/{family}/subject_{subject:02d}"
            receipt = read(folder / "receipt.json")
            assert receipt["complete"] and receipt["all_corrupted_native_hashes_exact"]
            assert receipt["protocol_sha256"] == protocol_hash and receipt["selection_sha256"] == selection_hash
            for filename, field in (("predictions.h5", "predictions_sha256"), ("results.json", "results_sha256")):
                hashes[str((folder / filename).relative_to(root))] = digest(folder / filename)
                assert hashes[str((folder / filename).relative_to(root))] == receipt[field]
            with h5py.File(folder / "predictions.h5", "r") as h:
                assert h.attrs["complete"] and h.attrs["selection_sha256"] == selection_hash
                paired.append([h[name][:] for name in ("target", "starts", "tasks")])
                target = h["target"][:, -50:].astype(np.float64)
                assert target.shape == (16, 50, 60)
                target_velocity = np.diff(target, axis=1) * 50
                position_variance = ((target-target.mean((0, 1)))**2).sum()
                velocity_variance = ((target_velocity-target_velocity.mean((0, 1)))**2).sum()
                assert position_variance > 0 and velocity_variance > 0
                for condition in p["conditions"]:
                    for arm in arms:
                        for filtering in ("unfiltered", "one_euro"):
                            values = h[f"{condition['name']}/{arm}/{filtering}"][:, :, -50:].astype(np.float64)
                            assert values.shape == (3, 16, 50, 60) and np.isfinite(values).all()
                            for i, seed in enumerate(p["seeds"]):
                                residual = values[i] - target
                                velocity = np.diff(values[i], axis=1) * 50
                                computed = {
                                    "mean_3d_error": np.sqrt((residual.reshape(16, 50, 20, 3)**2).sum(-1)).mean(),
                                    "coordinate_r2": 1 - (residual**2).sum()/position_variance,
                                    "velocity_r2": 1 - ((velocity-target_velocity)**2).sum()/velocity_variance,
                                    "velocity_mse": ((velocity-target_velocity)**2).mean(),
                                    "velocity_energy_ratio": (velocity**2).sum()/(target_velocity**2).sum(),
                                }
                                row = lookup[(subject, arm, filtering, condition["name"], seed)]
                                for metric, value in computed.items():
                                    errors.append(abs(float(value)-row[metric]))
                                    np.testing.assert_allclose(value, row[metric], atol=1e-9, rtol=1e-10)
        for a, b in zip(*paired, strict=True):
            np.testing.assert_array_equal(a, b)

    draws = np.random.default_rng(260930).integers(0, len(p["subjects"]), (10000, len(p["subjects"])))

    def person_means(arm, filtering, snr):
        return np.array([np.mean([r["mean_3d_error"] for r in rows if r["subject"] == subject
                                 and r["arm"] == arm and r["filtering"] == filtering and r["snr_db"] == snr])
                         for subject in p["subjects"]])

    def interval(values):
        return np.quantile(values[draws].mean(1), [.025, .975]).tolist()

    for summary in result["summary"]:
        values = person_means(summary["arm"], summary["filtering"], summary["snr_db"])
        delta = values - person_means(summary["arm"], summary["filtering"], None)
        for field, actual in (("mean_3d_error", values.mean()), ("clean_delta", delta.mean()),
                              ("participant_ci95", interval(values)), ("paired_clean_delta_ci95", interval(delta))):
            np.testing.assert_allclose(actual, summary[field], atol=1e-12, rtol=1e-12)
    comparisons = []
    for filtering in ("unfiltered", "one_euro"):
        clean_delta = person_means("token_v20", filtering, None) - person_means("continuous_v20", filtering, None)
        for snr in (None, 40, 30, 20, 10):
            delta = person_means("token_v20", filtering, snr) - person_means("continuous_v20", filtering, snr)
            comparisons.append({"filtering": filtering, "snr_db": snr, "token_minus_continuous": float(delta.mean()),
                                "paired_ci95": interval(delta), "difference_of_noise_deltas": float((delta-clean_delta).mean()),
                                "noise_delta_difference_ci95": interval(delta-clean_delta)})
    audit = {"complete": True, "checked_at_utc": datetime.now(UTC).isoformat(), "prediction_files": 18,
             "prediction_metric_rows": len(rows), "summary_rows": len(result["summary"]),
             "max_metric_abs_discrepancy": max(errors), "artifact_sha256": hashes,
             "protocol_sha256": protocol_hash, "results_sha256": digest(root / "results.json"),
             "collector_sha256": digest(__file__), "comparisons": comparisons,
             "scope": "Local independent NumPy recalculation of all five metrics from saved predictions and all primary summary intervals. Native-noise replay is attested by the remote receipts, not independently rerun here. Added token/continuous paired contrasts are pointwise."}
    (root / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [None, 40, 30, 20, 10]
    names = {"token_v20": "Token V20", "continuous_v20": "Continuous V20", "native_v19": "Native V19"}
    colors = {"token_v20": "#2463A5", "continuous_v20": "#CE7428", "native_v19": "#23816F"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for arm, label in names.items():
        means = [person_means(arm, "unfiltered", level) for level in levels]
        for ax, values in zip(axes, (means, [v-means[0] for v in means]), strict=True):
            ci = np.array([interval(v) for v in values])
            ax.plot(range(5), [v.mean() for v in values], "o-", color=colors[arm], label=label, markersize=4)
            ax.fill_between(range(5), ci[:, 0], ci[:, 1], color=colors[arm], alpha=.12, linewidth=0)
    axes[0].set_ylabel("Mean 3D error (stored coordinate units)")
    axes[1].set_ylabel("Change from clean (same units)")
    axes[1].axhline(0, color="#666666", linewidth=.7)
    for ax in axes:
        ax.set_xticks(range(5), ["Clean", "40", "30", "20", "10"])
        ax.set_xlabel("Native EMG SNR (dB)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Native EMG noise: pose error and degradation", fontsize=12, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False)
    fig.text(.5, .02, "Unfiltered outputs · 9 participants · paired participant bootstrap, pointwise 95% intervals", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .82))
    output = paper / "figures/hdemg_v20_noise"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("native_noise." + suffix), dpi=180)
    plt.close(fig)
    lines = ["# V20/V19 native-noise follow-up", "", "8 September 2026. All 18 subject jobs and the report completed. All 2,106 prediction-metric rows and 30 summary rows were independently recalculated locally from the saved arrays; artifact hashes match the remote receipts.", "",
             "Nine participants, three decoder seeds, three paired noise draws at each of 40/30/20/10 dB. No network was retrained. Average seeds and noise draws within each person before aggregation. The primary table below uses unfiltered predictions.", "",
             "| Model | Clean error | 10 dB error | Change | Relative change | Paired 95% interval for change |", "|---|---:|---:|---:|---:|---|" ]
    for arm, label in names.items():
        clean, noisy = person_means(arm, "unfiltered", None), person_means(arm, "unfiltered", 10)
        delta = noisy-clean
        ci = interval(delta)
        lines.append(f"| {label} | {clean.mean():.4f} | {noisy.mean():.4f} | {delta.mean():+.4f} | {100*delta.mean()/clean.mean():+.1f}% | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    contrast = next(r for r in comparisons if r["filtering"] == "unfiltered" and r["snr_db"] == 10)
    lines += ["", f"At 10 dB, the token-minus-continuous error contrast is {contrast['token_minus_continuous']:.4f}, paired 95% interval [{contrast['paired_ci95'][0]:.4f}, {contrast['paired_ci95'][1]:.4f}]. The change-from-clean contrast is {contrast['difference_of_noise_deltas']:.4f}, interval [{contrast['noise_delta_difference_ci95'][0]:.4f}, {contrast['noise_delta_difference_ci95'][1]:.4f}]. These are different questions: relative degradation can improve while absolute-error evidence remains uncertain.", "",
              "Native V19 has the lowest average absolute error and the smallest average noise degradation at every tested noisy level. Its context, architecture and training recipe differ; this does not isolate a causal benefit of raw inputs. The ongoing matched-budget experiment addresses part of that confounding.", "",
              "Validation selected One Euro (mincutoff 1.0, beta 0.1, dcutoff 1.0, nominal 50 Hz) for both V20 variants and identity for V19. Accordingly, the report's `one_euro` label means the validation-selected filter candidate, including identity for V19. Filtering modestly lowers V20 pose error, but velocity R² remains negative for every arm at every tested noise level, even after selection. This does not establish usable movement fidelity for device control.", "",
              "The results support measuring compact-representation tradeoffs; they do not support blanket noise superiority over RaulNet V19. The data are same-person/session/task; tokenizer pretraining excluded the validation and test windows. AWGN after clean calibration does not test electrode replacement. Errors are in millimetres.", "",
              "![Native-noise comparison](figures/hdemg_v20_noise/native_noise.png)", "",
              "[Complete report](../results/hdemg_v20_noise_followup_v2/RESULTS.md) · [Independent audit](../results/hdemg_v20_noise_followup_v2/independent_audit.json)"]
    (paper / "HDEMG_V20_NOISE_FOLLOWUP_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: audit[k] for k in ("complete", "prediction_files", "prediction_metric_rows", "summary_rows", "max_metric_abs_discrepancy")}))
    print(json.dumps(contrast))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.paper)
