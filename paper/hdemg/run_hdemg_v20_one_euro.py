"""Validation-selected 1€ filtering of frozen V20; no new network fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from paper.common import paths
from paper.hdemg.hdemg_one_euro import CANDIDATES, apply, choose, metrics, step_responses

BASE = Path(__file__).resolve().parents[1]
SEEDS = (260911, 260912, 260913)


def source_path(name):
    """Paper scripts are named relative to paper/, library files (src/...) relative to the repository."""
    return (paths.REPO if name.startswith("src/") else paths.ROOT) / name


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError(args.root / "protocol.json")
    parent = json.loads((args.reference / "protocol.json").read_text())
    results = json.loads((args.reference / "results.json").read_text())
    assert results["complete"] and results["protocol_sha256"] == sha(
        args.reference / "protocol.json"
    )
    assert parent["readout"]["kind"] == "raulnet_v20"
    source_names = (
        "hdemg/run_hdemg_v20_one_euro.py",
        "hdemg/hdemg_one_euro.py",
        "vendor/one_euro_filter.py",
        "vendor/OneEuroFilter-LICENSE",
    )
    p = {
        "reference": str(args.reference),
        "base_protocol_sha256": sha(args.reference / "protocol.json"),
        "base_results_sha256": sha(args.reference / "results.json"),
        "source_sha256": {name: sha(BASE / name) for name in source_names},
        "reference_implementation": "https://github.com/casiez/OneEuroFilter/tree/d78925584245597f2aa9c4c01a802eb0f0b77fb9/python",
        "reference_source_sha256": "cdd79ec70e001a671bd8cc72351c074815e529d33a378d97725c7986404d7d14",
        "subjects": list(range(1, 10)),
        "seeds": list(SEEDS),
        "new_network_fits": 0,
        "candidates": CANDIDATES,
        "selection": "One global configuration minimizing validation macro-person/seed mean 3D error. Identity eligible, then fixed grid; earlier index wins ties. Settings are saved before test predictions are regenerated or filtered.",
        "filter": "Independent scalar 1 Euro filter per stored coordinate, vectorized identically to pinned authors' reference. Raw stored coordinate units, no target-based per-window normalization. Derivative uses previous filtered position. Fixed nominal frequency 50 Hz, derivative cutoff 1 Hz.",
        "state": "Reset at first of 100 V20 frames of every independent input window. Initialize with first prediction and zero derivative. First 50 frames are warmup; score final 50 at original endpoints. No target initialization, future frames, cross-window state, time shifts, or bidirectional filtering.",
        "evaluation": "Before/after mean 3D point error, coordinate R2, within-window 49-interval velocity R2/MSE/energy; average subjects and seeds equally. Reproduce saved unfiltered test predictions before scoring. Synthetic step responses at amplitudes 1, 10 and 100 stored units describe filter delay; they are not measured end-to-end device latency.",
        "scope": "Exploratory post-hoc follow-up prompted by inspected unfiltered V20 test results. Filter parameters use validation only; same-person/session/task and tokenizer-pretraining exposure remain. No claim of an untouched confirmatory test or of fair comparison against untested filtered baselines.",
        "checkpoint_sha256": {},
        "fit_metadata_sha256": {},
    }
    assert (
        sha(BASE / "vendor/one_euro_filter.py") == p["reference_source_sha256"]
    )
    for subject in p["subjects"]:
        for seed in SEEDS:
            folder = args.reference / f"subject_{subject:02d}/fits/tokens_{seed}"
            fit = json.loads((folder / "fit.json").read_text())
            integrity = json.loads((folder / "integrity.json").read_text())
            assert fit["complete"] and integrity["verified"]
            assert (
                fit["best_sha256"]
                == integrity["best_sha256"]
                == sha(folder / "best.pt")
            )
            p["checkpoint_sha256"][
                str((folder / "best.pt").relative_to(args.reference))
            ] = fit["best_sha256"]
            for name in ("fit.json", "integrity.json"):
                p["fit_metadata_sha256"][
                    str((folder / name).relative_to(args.reference))
                ] = sha(folder / name)
    write(args.root / "protocol.json", p)
    print(
        json.dumps(
            {
                "frozen": True,
                "candidate_count": len(CANDIDATES),
                "new_network_fits": 0,
                "protocol_sha256": sha(args.root / "protocol.json"),
            }
        ),
        flush=True,
    )


def verify(args):
    p = json.loads((args.root / "protocol.json").read_text())
    assert str(args.reference) == p["reference"]
    assert p["candidates"] == CANDIDATES
    for name, digest in p["source_sha256"].items():
        assert sha(BASE / name) == digest, name
    assert sha(args.reference / "protocol.json") == p["base_protocol_sha256"]
    assert sha(args.reference / "results.json") == p["base_results_sha256"]
    parent = json.loads((args.reference / "protocol.json").read_text())
    # The parent campaign's sources are this repository's files (no per-run code copy).
    for name, digest in parent["source_sha256"].items():
        assert sha(source_path(name)) == digest, name
    for name, digest in p["fit_metadata_sha256"].items():
        assert sha(args.reference / name) == digest, name
    return p, parent


def export(args, p, parent, split):
    import h5py
    import torch

    if split == "test":
        selected = json.loads((args.root / "selection.json").read_text())
        assert selected["protocol_sha256"] == sha(args.root / "protocol.json")
        assert all(
            sha(args.root / "val" / name) == digest
            for name, digest in selected["validation_sha256"].items()
        )
    from paper.hdemg import run_hdemg_v20 as v20

    torch.set_num_threads(4)
    for subject in p["subjects"]:
        folder = v20.verify_subject(args.reference, parent, subject)
        data = v20.pilot.load_split(folder, split, "tokens")
        book = data["book"]
        assert v20.book_digest(book) == parent["codebook_float32_sha256"]
        output = args.root / split / f"subject_{subject:02d}.npz"
        if output.exists():
            with np.load(output) as f:
                assert str(f["protocol_sha256"]) == sha(args.root / "protocol.json")
            continue
        predictions = []
        with v20.backend(book), torch.inference_mode():
            for seed in SEEDS:
                fit_folder = folder / f"fits/tokens_{seed}"
                path = fit_folder / "best.pt"
                assert (
                    sha(path)
                    == p["checkpoint_sha256"][str(path.relative_to(args.reference))]
                )
                state = torch.load(path, map_location="cpu", weights_only=False)
                assert torch.equal(state["model"]["codebook"], book)
                model = (
                    v20.pilot.build("tokens", parent, state["config"])
                    .to(args.device)
                    .eval()
                )
                model.load_state_dict(state["model"])
                batches = []
                for first in range(0, len(data["subjects"]), 2):
                    ix = np.arange(first, min(first + 2, len(data["subjects"])))
                    out = v20.predict_tokens(
                        model, data, ix, "tokens", args.device
                    ).cpu()
                    batches.append((out * state["scale"] + state["mean"]).numpy())
                pred = np.concatenate(batches)
                assert pred.shape == (16, 100, 60) and np.isfinite(pred).all()
                if split == "test":
                    with np.load(fit_folder / "test_predictions.npz") as previous:
                        np.testing.assert_array_equal(
                            previous["target"], data["targets"].numpy()[:, -50:]
                        )
                        np.testing.assert_allclose(
                            previous["prediction"], pred[:, -50:], atol=1e-4, rtol=2e-6
                        )
                else:
                    fit = json.loads((fit_folder / "fit.json").read_text())
                    actual = np.linalg.norm(
                        (pred[:, -50:] - data["targets"].numpy()[:, -50:]).reshape(
                            -1, 20, 3
                        ),
                        axis=-1,
                    ).mean()
                    assert np.isclose(
                        actual, fit["validation_best"], atol=1e-4, rtol=2e-6
                    )
                predictions.append(pred)
                del model
        with h5py.File(folder / "features.h5", "r") as f:
            starts, tasks = f[f"{split}/starts"][:], f[f"{split}/tasks"][:]
            display_coordinate = int(
                f["train/targets"][:, -50:].reshape(-1, 60).var(0).argmax()
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            prediction=np.stack(predictions),
            target=data["targets"].numpy(),
            subjects=data["subjects"],
            starts=starts,
            tasks=tasks,
            display_coordinate=display_coordinate,
            protocol_sha256=sha(args.root / "protocol.json"),
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "exported": split,
                    "subject": subject,
                    "models": 3,
                    "replay_verified": True,
                }
            ),
            flush=True,
        )


def load_split(root, split):
    values = {}
    for subject in range(1, 10):
        with np.load(root / split / f"subject_{subject:02d}.npz") as f:
            values[subject] = {name: f[name] for name in f.files}
    return values


def select(root):
    path = root / "selection.json"
    fingerprints = {
        f.name: sha(f) for f in sorted((root / "val").glob("subject_*.npz"))
    }
    assert len(fingerprints) == 9
    if path.exists():
        selected = json.loads(path.read_text())
        assert selected["validation_sha256"] == fingerprints
        assert selected["protocol_sha256"] == sha(root / "protocol.json")
        return selected
    selected = choose(load_split(root, "val"))
    selected.update(
        validation_sha256=fingerprints, protocol_sha256=sha(root / "protocol.json")
    )
    write(path, selected)
    path.chmod(0o444)
    print(
        json.dumps(
            {
                "validation_selected": selected["selected"]["config"],
                "validation_error": selected["selected"]["mean_3d_error"],
                "identity_validation_error": selected["candidates"][0]["mean_3d_error"],
            }
        ),
        flush=True,
    )
    return selected


def report(root, selection):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = selection["selected"]["config"]
    test = load_split(root, "test")
    rows = []
    for subject, data in test.items():
        filtered = apply(data["prediction"], config)
        out = root / "filtered" / f"subject_{subject:02d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, prediction=filtered, selection_sha256=sha(root / "selection.json")
        )
        for condition, pred in (
            ("unfiltered", data["prediction"]),
            ("one_euro", filtered),
        ):
            rows.extend(
                dict(score, subject=subject, seed=SEEDS[index], condition=condition)
                for index, score in enumerate(metrics(pred, data["target"]))
            )
    summary = {}
    for condition in ("unfiltered", "one_euro"):
        selected = [r for r in rows if r["condition"] == condition]
        seed_means = [
            np.mean([r["mean_3d_error"] for r in selected if r["seed"] == seed])
            for seed in SEEDS
        ]
        summary[condition] = {
            key: float(np.mean([r[key] for r in selected]))
            for key in (
                "mean_3d_error",
                "coordinate_r2",
                "velocity_r2",
                "velocity_mse",
                "velocity_energy_ratio",
            )
        }
        summary[condition]["seed_sd"] = float(np.std(seed_means, ddof=1))
    protocol = json.loads((root / "protocol.json").read_text())
    result = {
        "complete": True,
        "protocol_sha256": sha(root / "protocol.json"),
        "selection_sha256": sha(root / "selection.json"),
        "selected_config": config,
        "summary": summary,
        "per_subject_seed": rows,
        "step_responses": step_responses(config),
        "prediction_sha256": {
            str(f.relative_to(root)): sha(f)
            for split in ("val", "test", "filtered")
            for f in sorted((root / split).glob("*.npz"))
        },
        "scope": protocol["scope"],
    }
    write(root / "results.json", result)
    out = root / "report"
    out.mkdir(exist_ok=True)
    lines = [
        "# Validation-selected One Euro filtering of V20",
        "",
        f"Selected one global configuration using validation only: `{json.dumps(config)}`. The 43 candidates included unfiltered output. All 27 validation and 27 unfiltered test replays passed. No network was retrained.",
        "",
        "| V20 output | Mean 3D error | Seed SD | Coordinate R² | Velocity R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, v in summary.items():
        lines.append(
            f"| {condition} | {v['mean_3d_error']:.3f} | {v['seed_sd']:.3f} | {v['coordinate_r2']:.3f} | {v['velocity_r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Errors use unverified stored coordinate units. Mean across nine people and three seeds; seed SD describes the three seed macro-person means. Velocity R² uses only 49 within-window differences. Scoring retains original time alignment, so additional filter lag counts as error. Filters reset per independent window, warm up on the first 50 predictions, and score the final 50; no ground-truth state initialization.",
            "",
            "Synthetic filter response (50 Hz, exact clean steps; not end-to-end device latency):",
            "",
            "| Step amplitude, stored units | 50% response delay, ms | 90% response delay, ms |",
            "|---|---:|---:|",
        ]
    )
    for row in result["step_responses"]:
        lines.append(
            f"| {row['step_amplitude_stored_units']:g} | {row['t50_ms']} | {row['t90_ms']} |"
        )
    lines.extend(
        [
            "",
            "No filtered V19 or filtered token TCN comparator was fitted/tuned in this experiment. A reduction in jitter does not prove recovery of neural structure. "
            + protocol["scope"],
            "",
            "[Reference implementation](https://github.com/casiez/OneEuroFilter/tree/d78925584245597f2aa9c4c01a802eb0f0b77fb9/python) · [Protocol](../protocol.json) · [Selection](../selection.json) · [Results](../results.json)",
        ]
    )
    (out / "HDEMG_V20_ONE_EURO_RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    colors = ("#8C8F99", "#176B87")
    for ax, key, title in zip(
        axes[:2],
        ("mean_3d_error", "velocity_r2"),
        ("Pose error (lower is better)", "Velocity R² (higher is better)"),
    ):
        for subject in range(1, 10):
            vals = [
                np.mean(
                    [
                        r[key]
                        for r in rows
                        if r["subject"] == subject and r["condition"] == c
                    ]
                )
                for c in ("unfiltered", "one_euro")
            ]
            ax.plot([0, 1], vals, color="#CED1D6", linewidth=1, zorder=1)
            ax.scatter([0, 1], vals, c=colors, s=22, zorder=2)
        ax.set_xticks([0, 1], ["V20", "V20 + 1€"])
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    example = test[1]
    coordinate = int(example["display_coordinate"])
    time = 1.0195 + np.arange(100) / 50
    axes[2].plot(
        time,
        example["target"][0, :, coordinate],
        color="#222222",
        label="Target",
        linewidth=1.6,
    )
    axes[2].plot(
        time,
        example["prediction"][0, 0, :, coordinate],
        color=colors[0],
        label="V20",
        linewidth=0.9,
    )
    axes[2].plot(
        time,
        apply(example["prediction"], config)[0, 0, :, coordinate],
        color=colors[1],
        label="V20 + 1€",
        linewidth=1.3,
    )
    axes[2].axvspan(time[0], time[50], color="#EFEFEF", zorder=-1)
    axes[2].set(
        xlabel="Time in input window (s)",
        ylabel="Stored coordinate units",
        title=f"Person 1, window 1, coordinate {coordinate}",
    )
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("V20 with validation-selected causal One Euro filtering", fontsize=12)
    fig.savefig(out / "one_euro_comparison.png", dpi=180)
    fig.savefig(out / "one_euro_comparison.pdf")
    plt.close(fig)
    write(
        out / "completion_receipt.json",
        {
            "complete": True,
            "protocol_sha256": result["protocol_sha256"],
            "selection_sha256": result["selection_sha256"],
            "artifacts": {
                name: sha(out / name)
                for name in (
                    "HDEMG_V20_ONE_EURO_RESULTS.md",
                    "one_euro_comparison.png",
                    "one_euro_comparison.pdf",
                )
            },
            "results_sha256": sha(root / "results.json"),
        },
    )
    print(
        json.dumps(
            {
                "complete": True,
                "selected_config": config,
                "summary": summary,
                "step_responses": result["step_responses"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=paths.RESULTS / "hdemg_v20_v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.root, args.reference = args.root.resolve(), args.reference.resolve()
    if args.mode == "freeze":
        freeze(args)
        return
    p, parent = verify(args)
    export(args, p, parent, "val")
    selection = select(args.root)
    export(args, p, parent, "test")
    report(args.root, selection)


if __name__ == "__main__":
    main()
