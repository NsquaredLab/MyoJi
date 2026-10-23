"""Recompute paired noise metrics from saved predictions and audit artifacts."""

import json
from pathlib import Path

import h5py
import numpy as np


def report(root):
    from paper.hdemg import run_dpose_pilot as pilot
    from paper.hdemg.dpose_noise import array_sha, unit_noise

    root = Path(root)
    p = json.loads((root / "protocol.json").read_text())
    levels = [None, 40, 30, 20, 10]
    all_rows, checks, stability = [], [], []
    for subject in p["subjects"]:
        folder = root / f"subject_{subject:02d}"
        receipt = json.loads((folder / "receipt.json").read_text())
        assert receipt["complete"] and not receipt["smoke"]
        assert receipt["protocol_sha256"] == pilot.sha256(root / "protocol.json")
        for name, digest in receipt["artifacts"].items():
            assert pilot.sha256(folder / name) == digest
        result = json.loads((folder / "results.json").read_text())
        assert len(result["metrics"]) == 78 and len(result["diagnostics"]) == 208
        with (
            h5py.File(folder / "paired_noise.h5", "r") as noise,
            h5py.File(folder / "predictions.h5", "r") as f,
        ):
            assert noise.attrs["complete"] and f.attrs["complete"]
            assert (
                f.attrs["protocol_sha256"]
                == noise.attrs["protocol_sha256"]
                == receipt["protocol_sha256"]
            )
            target = f["target"][:].astype(np.float64)
            starts = f["starts"][:]
            assert target.shape == (16, 50, 60)
            clean_ids = f["clean/ids"][:]
            for index in range(16):
                g = noise[f"window_{index:03d}"]
                assert g.attrs["start"] == starts[index]
                assert (
                    g.attrs["subject"] == subject
                    and g.attrs["task"] == f["tasks"][index]
                )
                for seed in p["noise_seeds"]:
                    data = g[f"seed_{seed}"][:]
                    assert array_sha(data) == g[f"seed_{seed}"].attrs["sha256"]
                    regenerated = unit_noise(
                        data.shape,
                        subject,
                        int(g.attrs["task"]),
                        int(g.attrs["start"]),
                        seed,
                    )
                    assert np.array_equal(data, regenerated)
            for condition in p["conditions"]:
                name = condition["name"]
                ids = f[name]["ids"][:]
                agreement = (ids == clean_ids).mean((1, 2))
                scored_agreement = (ids[:, :, -50:] == clean_ids[:, :, -50:]).mean(
                    (1, 2)
                )
                diagnostics = [r for r in result["diagnostics"] if r["name"] == name]
                assert [r["window"] for r in diagnostics] == list(range(16))
                for index, r in enumerate(diagnostics):
                    assert agreement[index] == r["token_agreement"]
                    assert scored_agreement[index] == r["scored_token_agreement"]
                    if name != "clean":
                        draw = noise[
                            f"window_{index:03d}/seed_{condition['noise_seed']}"
                        ]
                        assert draw.attrs["sha256"] == r["unit_noise_sha256"]
                        assert r["positive_power_channels"] > 0
                        assert (
                            abs(r["realized_snr_db_mean"] - condition["snr_db"]) < 0.3
                        )
                stability.append(
                    {
                        "subject": subject,
                        **condition,
                        "agreement": float(agreement.mean()),
                        "scored_agreement": float(scored_agreement.mean()),
                    }
                )
                for arm in p["arms"]:
                    for ki, seed in enumerate(p["decoder_seeds"]):
                        pred = f[name][arm][ki].astype(np.float64)
                        assert pred.shape == target.shape and np.isfinite(pred).all()
                        error = pred - target
                        denominator = np.sum(
                            (target - np.mean(target, axis=(0, 1), keepdims=True)) ** 2
                        )
                        endpoints = (
                            np.floor(
                                (starts[:, None] + np.arange(4039, 6000, 40)[None])
                                * 128
                                / 125
                            )
                            / 2048
                        )
                        dt = np.diff(endpoints, axis=1)[:, :, None]
                        pv, tv = (
                            np.diff(pred, axis=1) / dt,
                            np.diff(target, axis=1) / dt,
                        )
                        vden = np.sum((tv - tv.mean((0, 1), keepdims=True)) ** 2)
                        values = {
                            "mean_3d_point_error": float(
                                np.sqrt((error.reshape(-1, 20, 3) ** 2).sum(-1)).mean()
                            ),
                            "coordinate_mae": float(np.mean(np.abs(error))),
                            "coordinate_r2": float(1 - np.sum(error**2) / denominator),
                            "velocity_r2": float(1 - np.sum((pv - tv) ** 2) / vden),
                        }
                        saved = next(
                            r
                            for r in result["metrics"]
                            if r["name"] == name
                            and r["arm"] == arm
                            and r["decoder_seed"] == seed
                        )
                        for key, value in values.items():
                            assert np.isclose(
                                value, saved[key], atol=1e-10, rtol=1e-10
                            ), (subject, arm, name, key)
                        all_rows.append(
                            {
                                "subject": subject,
                                "arm": arm,
                                "decoder_seed": seed,
                                **condition,
                                **values,
                            }
                        )
            checks.append(
                {
                    "subject": subject,
                    "verified_metrics": 78 * 4,
                    "verified_noise_draws": 48,
                    "identity_replays": len(result["identity_replays"]),
                    "receipt_sha256": pilot.sha256(folder / "receipt.json"),
                }
            )
    # Nested repeats are averaged before any population summary or bootstrap.
    person, summaries = [], []
    for subject in p["subjects"]:
        for arm in p["arms"]:
            for level in levels:
                rows = [
                    r
                    for r in all_rows
                    if r["subject"] == subject
                    and r["arm"] == arm
                    and r["snr_db"] == level
                ]
                assert len(rows) == (3 if level is None else 9)
                keys = (
                    "mean_3d_point_error",
                    "coordinate_mae",
                    "coordinate_r2",
                    "velocity_r2",
                )
                person.append(
                    {
                        "subject": subject,
                        "arm": arm,
                        "snr_db": level,
                        **{k: float(np.mean([r[k] for r in rows])) for k in keys},
                    }
                )
    rng = np.random.default_rng(260924)
    bootstrap = rng.integers(0, len(p["subjects"]), size=(5000, len(p["subjects"])))
    for arm in p["arms"]:
        clean = np.array(
            [
                r["mean_3d_point_error"]
                for r in person
                if r["arm"] == arm and r["snr_db"] is None
            ]
        )
        for level in levels:
            rows = [r for r in person if r["arm"] == arm and r["snr_db"] == level]
            errors = np.array([r["mean_3d_point_error"] for r in rows])
            delta = errors - clean
            summaries.append(
                {
                    "arm": arm,
                    "snr_db": level,
                    "participants": len(rows),
                    "mean_3d_point_error": float(errors.mean()),
                    "mean_error_ci95": np.quantile(
                        errors[bootstrap].mean(1), [0.025, 0.975]
                    ).tolist(),
                    "delta_from_clean": float(delta.mean()),
                    "delta_from_clean_ci95": np.quantile(
                        delta[bootstrap].mean(1), [0.025, 0.975]
                    ).tolist(),
                    "coordinate_r2": float(np.mean([r["coordinate_r2"] for r in rows])),
                    "velocity_r2": float(np.mean([r["velocity_r2"] for r in rows])),
                }
            )
    paired = []
    for level in levels:
        values = {}
        for arm in p["arms"]:
            values[arm] = np.array(
                [
                    r["mean_3d_point_error"]
                    for r in person
                    if r["arm"] == arm and r["snr_db"] == level
                ]
            )
        delta = values["tokens"] - values["continuous"]
        paired.append(
            {
                "snr_db": level,
                "tokens_minus_continuous": float(delta.mean()),
                "ci95": np.quantile(delta[bootstrap].mean(1), [0.025, 0.975]).tolist(),
            }
        )
    output = root / "report"
    output.mkdir(exist_ok=True)
    pilot.write_json(
        output / "results.json",
        {
            "complete": True,
            "protocol_sha256": pilot.sha256(root / "protocol.json"),
            "checks": checks,
            "summary": summaries,
            "paired_comparisons": paired,
            "per_participant": person,
            "token_stability": stability,
        },
    )
    lines = [
        "# Frozen TCN noise evaluation",
        "",
        p["scope"],
        "",
        "Native 2048-Hz independent-channel additive noise starts after a clean calibration prefix. No retraining. SNR is referenced to demeaned clean signal power in the perturbed interval. Noise arrays are retained for a future identical-input V19 comparison. Current results do not compare against V19.",
        "",
        "Mean 3D error uses unverified stored coordinate units. Noise seeds and decoder seeds are averaged within each of nine participants. Intervals resample participants; no multiplicity correction. Velocity R² uses first differences within each scored window and the exact native endpoint intervals. Negative values are retained.",
        "",
        "| Added noise SNR | Token TCN error | Continuous TCN error | Token agreement |",
        "|---|---:|---:|---:|",
    ]
    for level in levels:
        scores = {
            r["arm"]: r["mean_3d_point_error"]
            for r in summaries
            if r["snr_db"] == level
        }
        agreement = np.mean([r["agreement"] for r in stability if r["snr_db"] == level])
        label = "Clean" if level is None else f"{level} dB"
        lines.append(
            f"| {label} | {scores['tokens']:.4f} | {scores['continuous']:.4f} | {agreement:.2%} |"
        )
    lines += [
        "",
        "Token agreement is a diagnostic. Stability without accurate pose/movement prediction is not useful robustness. This synthetic post-calibration noise experiment does not establish robustness to electrode movement, acquisition changes or noisy calibration.",
    ]
    (output / "DPOSE_NOISE_RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    x = np.arange(5)
    for arm, label, color in (
        ("tokens", "Token TCN", "#245c8b"),
        ("continuous", "Continuous TCN", "#c16824"),
    ):
        rows = [
            next(r for r in summaries if r["arm"] == arm and r["snr_db"] == level)
            for level in levels
        ]
        means = np.array([r["mean_3d_point_error"] for r in rows])
        intervals = np.array([r["mean_error_ci95"] for r in rows])
        axes[0].plot(x, means, "o-", label=label, color=color)
        axes[0].fill_between(
            x, intervals[:, 0], intervals[:, 1], color=color, alpha=0.12
        )
    axes[1].plot(
        x,
        [
            100 * np.mean([r["agreement"] for r in stability if r["snr_db"] == level])
            for level in levels
        ],
        "o-",
        color="#245c8b",
    )
    axes[0].set_ylabel("Mean 3D pose error (stored units)")
    axes[1].set_ylabel("Unchanged token IDs (%)")
    axes[1].set_ylim(0, 102)
    for ax in axes:
        ax.set_xticks(x, ["Clean", "40", "30", "20", "10"])
        ax.set_xlabel("Added-noise SNR (dB)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(output / "noise_pose_and_tokens.png", dpi=180)
    fig.savefig(output / "noise_pose_and_tokens.pdf")
    plt.close(fig)
    pilot.write_json(
        output / "completion_receipt.json",
        {
            "complete": True,
            "protocol_sha256": pilot.sha256(root / "protocol.json"),
            "subjects": 9,
            "existing_models": 54,
            "new_fits": 0,
            "recomputed_metrics": 9 * 78 * 4,
            "artifacts": {
                name: pilot.sha256(output / name)
                for name in (
                    "results.json",
                    "DPOSE_NOISE_RESULTS.md",
                    "noise_pose_and_tokens.png",
                    "noise_pose_and_tokens.pdf",
                )
            },
        },
    )
    print(
        json.dumps({"complete": True, "report": str(output), "summary": summaries}),
        flush=True,
    )
