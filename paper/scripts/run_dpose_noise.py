"""Frozen DPose token/continuous TCNs under paired native-rate additive noise."""

import argparse
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import paths
import run_dpose_pilot as pilot
import torch
from dpose_noise import (
    NOISE_SEEDS,
    SNR_DB,
    array_sha,
    native_bounds,
    perturb_native,
    pose_metrics,
    resampled_window,
    unit_noise,
)

ARMS = ("tokens", "continuous")
SEEDS = (260911, 260912, 260913)
SUBJECTS = tuple(range(1, 10))
CONDITIONS = [{"name": "clean", "snr_db": None, "noise_seed": None}] + [
    {"name": f"snr{snr}_seed{seed}", "snr_db": snr, "noise_seed": seed}
    for snr in SNR_DB
    for seed in NOISE_SEEDS
]


def freeze(args):
    root, ref = args.root, args.reference
    if (root / "protocol.json").exists():
        raise FileExistsError(root / "protocol.json")
    old = json.loads((ref / "protocol.json").read_text())
    assert old["subjects"] == list(SUBJECTS) and old["readout"]["kind"] == "tcn"
    assert json.loads((ref / "results.json").read_text())["complete"]
    # Library files are named src/... (relative to the repository), scripts relative to paper/.
    sources = [
        str(x.relative_to(paths.REPO)) for x in sorted((paths.REPO / "src").rglob("*.py"))
    ] + [
        "scripts/" + name
        for name in (
            "dpose_noise.py",
            "run_dpose_noise.py",
            "report_dpose_noise.py",
            "run_dpose_pilot.py",
        )
    ]
    p = {
        "version": "dpose-native-awgn-v1",
        "scope": "Exploratory frozen-model noise evaluation on the previously inspected within-person/session/task DPose test split. Tokenizer pretraining saw these recordings. No model fitting, tuning, or new-person claim.",
        "reference": str(ref.resolve()),
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": pilot.sha256(args.checkpoint),
        "reference_protocol_sha256": pilot.sha256(ref / "protocol.json"),
        "reference_results_sha256": pilot.sha256(ref / "results.json"),
        "selection_sha256": pilot.sha256(ref / "selection.json"),
        "subjects": list(SUBJECTS),
        "arms": list(ARMS),
        "decoder_seeds": list(SEEDS),
        "conditions": CONDITIONS,
        "noise_seeds": list(NOISE_SEEDS),
        "split": "test",
        "windows_per_subject": 16,
        "noise": "Independent-channel float32 Gaussian draws at native 2048 Hz, added after the 500 ms clean calibration prefix. Channel SD uses demeaned clean raw EMG over the perturbed part of that window; nominal SNR is benchmark input scaling, not an encoder statistic. Zero-power channels receive zero noise. Identical draws are scaled across SNRs, shared across models/seeds, and saved for V19 replay. No noisy-startup, physical electrode-shift or general acquisition-robustness claim.",
        "timing": "Native origins are multiples of 128 with >=128 samples of past FIR support. Resample using the exact prior causal 125/128 filter, retain its 20 ms delay and absolute phase, then crop the same 6000 samples. Noise onset follows the 1000-sample calibration prefix; no resampling/window reset at the scored targets. Score identical final 50 endpoint targets.",
        "identity_tolerances": {
            "tokens": "bit exact",
            "continuous_float16": "bit exact",
            "prediction_atol": 1e-4,
            "prediction_rtol": 2e-6,
        },
        "statistics": "Average noise repetitions and decoder seeds within participant; participant is the resampling unit. Report absolute and clean-relative pose error, coordinate R2 and within-window velocity R2. Keep negative R2. Token agreement is secondary, never evidence of useful robustness alone.",
        "source_sha256": {x: pilot.sha256(pilot.source_path(x)) for x in sources},
        "source_archives": old["files"],
        "references": {},
    }
    assert p["checkpoint_sha256"] == old["checkpoint_sha256"]
    for subject in SUBJECTS:
        folder = ref / f"subject_{subject:02d}"
        record = {"protocol_sha256": pilot.sha256(folder / "protocol.json"), "fits": {}}
        for arm in ARMS:
            for seed in SEEDS:
                fitdir = folder / "fits" / f"{arm}_{seed}"
                fit = json.loads((fitdir / "fit.json").read_text())
                assert (
                    fit["complete"]
                    and fit["protocol_sha256"] == record["protocol_sha256"]
                )
                assert pilot.sha256(fitdir / "best.pt") == fit["best_sha256"]
                record["fits"][f"{arm}_{seed}"] = {
                    "best_sha256": fit["best_sha256"],
                    "predictions_sha256": pilot.sha256(fitdir / "test_predictions.npz"),
                }
        record["baseline_sha256"] = pilot.sha256(
            folder / "baseline_training_targets.npz"
        )
        with h5py.File(folder / "features.h5", "r") as f:
            assert (
                f.attrs["complete"]
                and f.attrs["protocol_sha256"] == record["protocol_sha256"]
            )
            assert f["test/tokens"].shape == (16, 320, 100)
            record["arrays"] = {
                k: array_sha(f[k][:])
                for k in (
                    "codebook",
                    "descriptors",
                    "test/tokens",
                    "test/targets",
                    "test/starts",
                    "test/tasks",
                    "test/subjects",
                )
            }
        p["references"][str(subject)] = record
    pilot.write_json(root / "protocol.json", p)
    print(
        json.dumps(
            {
                "frozen": True,
                "subjects": 9,
                "existing_models": 54,
                "new_fits": 0,
                "noisy_windows": 144 * 12,
                "protocol_sha256": pilot.sha256(root / "protocol.json"),
            }
        ),
        flush=True,
    )


def verify_protocol(root):
    p = json.loads((root / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        assert pilot.sha256(pilot.source_path(name)) == digest, name
    ref = Path(p["reference"])
    for name, key in (
        ("protocol.json", "reference_protocol_sha256"),
        ("results.json", "reference_results_sha256"),
        ("selection.json", "selection_sha256"),
    ):
        assert pilot.sha256(ref / name) == p[key], name
    assert pilot.sha256(Path(p["checkpoint"])) == p["checkpoint_sha256"]
    return p


def load_models(p, subject, device):
    folder = Path(p["reference"]) / f"subject_{subject:02d}"
    record = p["references"][str(subject)]
    assert pilot.sha256(folder / "protocol.json") == record["protocol_sha256"]
    result, originals = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}_{seed}"
            fitdir = folder / "fits" / key
            r = record["fits"][key]
            assert pilot.sha256(fitdir / "best.pt") == r["best_sha256"]
            assert (
                pilot.sha256(fitdir / "test_predictions.npz") == r["predictions_sha256"]
            )
            saved = torch.load(
                fitdir / "best.pt", map_location="cpu", weights_only=False
            )
            assert saved["protocol_sha256"] == record["protocol_sha256"]
            assert saved["arm"] == arm and saved["seed"] == seed
            model = (
                pilot.build(arm, saved["protocol"], saved["config"])
                .to(device)
                .eval()
                .requires_grad_(False)
            )
            model.load_state_dict(saved["model"])
            result[key] = (model, saved["mean"], saved["scale"])
            with np.load(fitdir / "test_predictions.npz") as data:
                originals[key] = {k: data[k].copy() for k in ("prediction", "target")}
    return result, originals


@torch.inference_mode()
def encode(tokenizer, signal, device):
    raw = torch.from_numpy(signal)[None].to(device)
    normalized, centre, scale = pilot.calibration_normalize(raw, 1000)
    ids, vectors = [], []
    for channel in range(0, 320, 32):
        output = tokenizer.model(
            normalized[:, channel : channel + 32],
            sensor="hd",
            channel_descriptors=torch.zeros(
                32, tokenizer.config.geometry_dim, device=device
            ),
            include_spatial=False,
        )
        ids.append(output.tokens[0, :, -100:].cpu().numpy().astype(np.uint16))
        vectors.append(output.continuous[0, :, -100:].half().cpu().numpy())
    return (
        np.concatenate(ids),
        np.concatenate(vectors),
        normalized[0].cpu().numpy(),
        centre.cpu().numpy(),
        scale.cpu().numpy(),
    )


@torch.inference_mode()
def predict(models, ids, continuous, book, descriptors, device):
    features = {
        "tokens": book[torch.from_numpy(ids.astype(np.int64)).to(device)][None],
        "continuous": torch.from_numpy(continuous).to(device).float()[None],
    }
    zero = torch.zeros(1, 320, 1, device=device)
    results = {}
    for arm in ARMS:
        for seed in SEEDS:
            model, mean, scale = models[f"{arm}_{seed}"]
            results[f"{arm}_{seed}"] = (
                model(features[arm], descriptors, zero, torch.ones_like(zero)).cpu()
                * scale
                + mean
            )[0, -50:].numpy()
    return results


def relative_distance(value, reference):
    a, b = value.astype(np.float32), reference.astype(np.float32)
    return float(
        np.sqrt(
            np.sum((a - b) ** 2, dtype=np.float64)
            / max(np.sum(b**2, dtype=np.float64), 1e-30)
        )
    )


def evaluate_subject(root, p, subject, device, smoke=False):
    started = time.monotonic()
    refdir = Path(p["reference"]) / f"subject_{subject:02d}"
    outdir = root / ("smoke" if smoke else f"subject_{subject:02d}")
    outdir.mkdir(parents=True, exist_ok=True)
    receipt_path = outdir / "receipt.json"
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        assert r["complete"] and r["protocol_sha256"] == pilot.sha256(
            root / "protocol.json"
        )
        for name, digest in r["artifacts"].items():
            assert pilot.sha256(outdir / name) == digest
        print(json.dumps({"already_complete": subject, "smoke": smoke}), flush=True)
        return
    for filename in ("paired_noise.h5", "predictions.h5"):
        if (outdir / filename).exists():
            raise FileExistsError(
                f"Incomplete output requires explicit recovery: {outdir / filename}"
            )
    models, originals = load_models(p, subject, device)
    tokenizer = pilot.Tokenizer(p["checkpoint"], device=device)
    before = {k: pilot.state_hash(v[0]) for k, v in models.items()}
    before["tokenizer"] = pilot.state_hash(tokenizer.model)
    paths = [
        Path(p["data"]) / f"Sub{subject}_{kind}.pkl" for kind in ("emg", "kinematics")
    ]
    for path in paths:
        assert pilot.sha256(path) == p["source_archives"][path.name]
    emg, kin = [joblib.load(path) for path in paths]
    conditions = CONDITIONS if not smoke else [CONDITIONS[0], CONDITIONS[-3]]
    noise_seeds = NOISE_SEEDS if not smoke else (NOISE_SEEDS[0],)
    diagnostics, replays = [], []
    with (
        h5py.File(refdir / "features.h5", "r") as cached,
        h5py.File(outdir / "paired_noise.h5", "w") as noises,
        h5py.File(outdir / "predictions.h5", "w") as output,
    ):
        for name, digest in p["references"][str(subject)]["arrays"].items():
            assert array_sha(cached[name][:]) == digest, name
        starts, tasks = cached["test/starts"][:], cached["test/tasks"][:]
        indices = np.arange(1 if smoke else len(starts))
        targets = cached["test/targets"][:, -50:][indices]
        book_array = cached["codebook"][:]
        book = torch.from_numpy(book_array).to(device)
        descriptors = torch.from_numpy(cached["descriptors"][:]).to(device)
        for file in (noises, output):
            file.attrs["complete"] = False
            file.attrs["protocol_sha256"] = pilot.sha256(root / "protocol.json")
        output["target"], output["starts"], output["tasks"] = (
            targets,
            starts[indices],
            tasks[indices],
        )
        for condition in conditions:
            group = output.create_group(condition["name"])
            for arm in ARMS:
                group.create_dataset(arm, shape=(3, len(indices), 50, 60), dtype="f4")
            group.create_dataset("ids", shape=(len(indices), 320, 100), dtype="u2")
        for index in indices:
            start, task = int(starts[index]), int(tasks[index])
            task_key = next(k for k in emg if int(k) == task)
            origin, onset, stop = native_bounds(start)
            raw = np.asarray(emg[task_key][:, origin:stop], dtype=np.float32)
            assert raw.shape == (320, stop - origin)
            target = pilot.target_at(kin[task_key], start)[-50:]
            assert np.array_equal(target, targets[index])
            for old in originals.values():
                assert np.array_equal(old["target"][index], target)
            signal = resampled_window(raw, start, origin, pilot.resample)
            clean_ids, clean_cont, clean_norm, centre, scale = encode(
                tokenizer, signal, device
            )
            assert np.array_equal(clean_norm, cached["test/raw"][index]), (
                "raw identity replay"
            )
            assert np.array_equal(clean_ids, cached["test/tokens"][index]), (
                "token identity replay"
            )
            assert np.array_equal(clean_cont, cached["test/continuous"][index]), (
                "continuous identity replay"
            )
            group = noises.create_group(f"window_{index:03d}")
            for k, v in {
                "start": start,
                "task": task,
                "subject": subject,
                "origin": origin,
                "onset": onset,
                "stop": stop,
                "source_segment_sha256": array_sha(raw),
                "clean_normalized_sha256": array_sha(clean_norm),
            }.items():
                group.attrs[k] = v
            draws = {
                seed: unit_noise((320, stop - onset), subject, task, start, seed)
                for seed in noise_seeds
            }
            for seed, draw in draws.items():
                dataset = group.create_dataset(f"seed_{seed}", data=draw)
                dataset.attrs["sha256"] = array_sha(draw)
            clean_pred = None
            for condition in conditions:
                name, snr, seed = (
                    condition["name"],
                    condition["snr_db"],
                    condition["noise_seed"],
                )
                diag = {
                    "subject": subject,
                    "window": int(index),
                    "task": task,
                    **condition,
                }
                if name == "clean":
                    ids, cont = clean_ids, clean_cont
                    changed = raw
                else:
                    changed, measured = perturb_native(
                        raw, onset - origin, draws[seed], snr
                    )
                    assert np.array_equal(
                        changed[:, : onset - origin], raw[:, : onset - origin]
                    )
                    noisy_signal = resampled_window(
                        changed, start, origin, pilot.resample
                    )
                    assert np.array_equal(noisy_signal[:, :1000], signal[:, :1000]), (
                        "clean calibration prefix"
                    )
                    ids, cont, _, noisy_centre, noisy_scale = encode(
                        tokenizer, noisy_signal, device
                    )
                    assert np.array_equal(centre, noisy_centre) and np.array_equal(
                        scale, noisy_scale
                    )
                    diag.update(measured, unit_noise_sha256=array_sha(draws[seed]))
                diag.update(
                    corrupted_native_sha256=array_sha(changed),
                    token_agreement=float(np.mean(ids == clean_ids)),
                    scored_token_agreement=float(
                        np.mean(ids[:, -50:] == clean_ids[:, -50:])
                    ),
                    continuous_relative_l2=relative_distance(cont, clean_cont),
                    quantized_relative_l2=relative_distance(
                        book_array[ids], book_array[clean_ids]
                    ),
                )
                predictions = predict(models, ids, cont, book, descriptors, device)
                if name == "clean":
                    clean_pred = predictions
                    for key, pred in predictions.items():
                        previous = originals[key]["prediction"][index]
                        np.testing.assert_allclose(
                            pred,
                            previous,
                            atol=p["identity_tolerances"]["prediction_atol"],
                            rtol=p["identity_tolerances"]["prediction_rtol"],
                        )
                        replays.append(
                            {
                                "window": int(index),
                                "model": key,
                                "max_absolute_prediction_difference": float(
                                    np.max(np.abs(pred - previous))
                                ),
                            }
                        )
                group_out = output[name]
                group_out["ids"][index] = ids
                for arm in ARMS:
                    for ki, decoder_seed in enumerate(SEEDS):
                        key = f"{arm}_{decoder_seed}"
                        pred = predictions[key]
                        assert np.isfinite(pred).all()
                        group_out[arm][ki, index] = pred
                        diag[f"{key}_pose_change"] = float(
                            np.linalg.norm(
                                (pred - clean_pred[key]).reshape(-1, 20, 3), axis=-1
                            ).mean()
                        )
                diagnostics.append(diag)
            noises.flush()
            output.flush()
            print(
                json.dumps(
                    {
                        "subject": subject,
                        "completed_windows": int(index + 1),
                        "total_windows": len(indices),
                        "elapsed_seconds": time.monotonic() - started,
                        "smoke": smoke,
                    }
                ),
                flush=True,
            )
        metrics = []
        for condition in conditions:
            for arm in ARMS:
                for ki, seed in enumerate(SEEDS):
                    metrics.append(
                        {
                            "subject": subject,
                            "arm": arm,
                            "decoder_seed": seed,
                            **condition,
                            **pose_metrics(
                                output[condition["name"]][arm][ki],
                                targets,
                                starts[indices],
                            ),
                        }
                    )
        for file in (noises, output):
            file.attrs["complete"] = True
    assert before["tokenizer"] == pilot.state_hash(tokenizer.model)
    assert all(before[k] == pilot.state_hash(v[0]) for k, v in models.items())
    pilot.write_json(
        outdir / "results.json",
        {
            "complete": True,
            "smoke": smoke,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "identity_replays": replays,
        },
    )
    pilot.write_json(
        receipt_path,
        {
            "complete": True,
            "subject": subject,
            "smoke": smoke,
            "protocol_sha256": pilot.sha256(root / "protocol.json"),
            "runtime": tokenizer.metadata(),
            "numpy_version": np.__version__,
            "elapsed_seconds": time.monotonic() - started,
            "models_unchanged": True,
            "new_fits": 0,
            "artifacts": {
                name: pilot.sha256(outdir / name)
                for name in ("paired_noise.h5", "predictions.h5", "results.json")
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "smoke", "subject", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=paths.RESULTS / "dpose_within_v1")
    parser.add_argument("--data", type=Path, default=paths.HDEMG)
    parser.add_argument("--checkpoint", type=Path, default=paths.LEARNED_TOKENIZER)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "freeze":
        freeze(args)
        return
    p = verify_protocol(args.root)
    if args.mode == "report":
        from report_dpose_noise import report

        report(args.root)
    else:
        subject = 1 if args.mode == "smoke" else args.index + 1
        assert subject in SUBJECTS
        if args.mode == "subject":
            smoke = json.loads((args.root / "smoke/receipt.json").read_text())
            assert smoke["complete"] and smoke["protocol_sha256"] == pilot.sha256(
                args.root / "protocol.json"
            )
        evaluate_subject(args.root, p, subject, args.device, smoke=args.mode == "smoke")
        if args.mode == "subject" and all(
            (args.root / f"subject_{s:02d}" / "receipt.json").exists() for s in SUBJECTS
        ):
            from report_dpose_noise import report

            report(args.root)


if __name__ == "__main__":
    main()
