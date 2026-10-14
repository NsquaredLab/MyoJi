"""Frozen, paired native-noise comparison of token V20, continuous V20 and V19.

Separate validation exports precede global filter selection. Test jobs require
that selection and its checkpoint hashes. No network is fitted by this script.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import paths
import torch
from dpose_noise import (
    ENDS,
    NOISE_SEEDS,
    SNR_DB,
    array_sha,
    native_bounds,
    perturb_native,
)
from dpose_one_euro import CANDIDATES, apply, choose, metrics

BASE = Path(__file__).resolve().parents[1]
SUBJECTS = tuple(range(1, 10))
SEEDS = (260911, 260912, 260913)
ARMS = ("token_v20", "continuous_v20", "native_v19")
CONDITIONS = [{"name": "clean", "snr_db": None, "noise_seed": None}] + [
    {"name": f"snr{snr}_seed{seed}", "snr_db": snr, "noise_seed": seed}
    for snr in SNR_DB
    for seed in NOISE_SEEDS
]


def source_path(name):
    """Paper scripts are named relative to paper/, library files (src/...) relative to the repository."""
    return (paths.REPO if name.startswith("src/") else paths.ROOT) / name


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def freeze(args):
    path = args.root / "protocol.json"
    if path.exists():
        raise FileExistsError(path)
    parents = {
        "token_v20": "dpose_v20_v1",
        "continuous_v20": "dpose_v20_continuous_v1",
        "native_v19": "dpose_raulnet_v1",
        "noise": "dpose_noise_v1",
        "token_filter": "dpose_v20_one_euro_v1",
        "within": "dpose_within_v1",
    }
    references = {k: str((args.runs / v).resolve()) for k, v in parents.items()}
    noise = json.loads((Path(references["noise"]) / "protocol.json").read_text())
    assert noise["conditions"] == CONDITIONS
    native_root = Path(references["native_v19"])
    validation_gpus = {}
    for subject in SUBJECTS:
        gpus = {
            json.loads(
                (
                    native_root / f"subject_{subject:02d}/fits/raulnet_{seed}/fit.json"
                ).read_text()
            )["gpu"]
            for seed in SEEDS
        }
        assert len(gpus) == 1, (subject, gpus)
        validation_gpus[str(subject)] = gpus.pop()
    evaluation_logs = list(native_root.glob("evaluate-*.out"))
    assert len(evaluation_logs) == 1
    evaluation_log = evaluation_logs[0]
    test_gpus = {
        line.split(",", 1)[0]
        for line in evaluation_log.read_text().splitlines()
        if line.startswith(("NVIDIA ", "Tesla "))
    }
    assert len(test_gpus) == 1, test_gpus
    # The library (including the vendored V20 module) is hashed through src/.
    source_names = [
        "scripts/run_dpose_v20_noise_followup.py",
        "scripts/dpose_noise.py",
        "scripts/dpose_one_euro.py",
        "scripts/dpose_raulnet_model.py",
        "scripts/run_dpose_pilot.py",
        "scripts/vendor/raulnet_v19.py",
        "src/myoji/_vendor/LICENSE-GPL-3.0",
    ]
    sources = source_names + [
        str(x.relative_to(paths.REPO)) for x in (paths.REPO / "src").rglob("*.py")
    ]
    p = {
        "version": "v20-v19-paired-noise-v2",
        "references": references,
        "parent_protocol_sha256": {
            k: sha(Path(v) / "protocol.json") for k, v in references.items()
        },
        "source_sha256": {x: sha(source_path(x)) for x in sorted(sources)},
        "data": noise["data"],
        "source_archives": noise["source_archives"],
        "tokenizer_checkpoint": noise["checkpoint"],
        "tokenizer_sha256": noise["checkpoint_sha256"],
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "conditions": CONDITIONS,
        "new_network_fits": 0,
        "candidates": CANDIDATES,
        "selection": "One global One Euro setting per model family, all 27 clean-validation predictions, macro-person/seed mean 3D error; identity eligible. Same 43 candidates for each. All settings and model hashes frozen before test/noise export; reused across every SNR/noise repetition.",
        "filter_state": "Reset independently at first of 100 predictions per input window, initialize first prediction/zero derivative, warm first 50, score last 50. Frequency 50 Hz; no targets, future data, shifts or cross-window state.",
        "noise": "Replay the exact previously saved native 2048-Hz Gaussian draws after a clean 500-ms prefix, same channel scaling at 40/30/20/10 dB and three noise seeds. Verify corrupted-native hashes and token IDs against the old campaign. Native V19 receives these same native samples; V20 gets the original causal resampling/calibration/tokenizer. No new random corruption.",
        "precision": "Original inference precision preserved: V20 float32; V19 float16 autocast in original Torch 2.10 environment. Continuous V20 consumes pre-rounding float32 vectors. No compile optimization in this scientific evaluation.",
        "native_validation_gpus": validation_gpus,
        "native_test_gpu": test_gpus.pop(),
        "native_evaluation_log": str(evaluation_log),
        "native_evaluation_log_sha256": sha(evaluation_log),
        "hardware_replay_amendment": "V1 stopped on validation for subject 7 seed 260911: 7.2842674363 on V100 versus recorded 7.2846288681 from RTX 3080. Replay tolerances remain atol=2e-4, rtol=4e-6. Validate each V19 participant on its original training GPU model; require all nine hash-bound validation receipts before filter selection. Run V19 clean/noisy test inference on the original test-evaluation GPU model. No filter was selected and no follow-up test/noise prediction was exported before this amendment.",
        "native_warmup": "Regenerate all 100 native 320-sample endpoint windows. Process first/last 50 separately in the original flattened 64-endpoint batches, preserving scored clean V19 batching and numerical replay.",
        "statistics": "Absolute pose and movement errors plus paired clean deltas. Average decoder seeds/noise repetitions within person; bootstrap people, 10000 draws, seed 260930. Velocity diagnostics use nominal 50 Hz as in the filter study. Token agreement remains secondary.",
        "limitations": "Previously inspected same-person/session/task test split; tokenizer saw recordings. V19 has a different input context and larger training recipe. AWGN after clean startup is not electrode displacement or general deployment robustness. Stored target units are unverified.",
    }
    write(path, p)
    print(
        json.dumps({"frozen": True, "new_fits": 0, "models": 81, "conditions": 13}),
        flush=True,
    )


def verify(root):
    p = json.loads((root / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        assert sha(source_path(name)) == digest, name
    for key, ref in p["references"].items():
        assert sha(Path(ref) / "protocol.json") == p["parent_protocol_sha256"][key]
    assert sha(Path(p["native_evaluation_log"])) == p["native_evaluation_log_sha256"]
    return p


def fit_folder(p, arm, subject, seed):
    prefix = {
        "token_v20": "tokens",
        "continuous_v20": "continuous",
        "native_v19": "raulnet",
    }[arm]
    return Path(p["references"][arm]) / f"subject_{subject:02d}/fits/{prefix}_{seed}"


def checkpoints(p, arm):
    records = {}
    # Global 27-fit gate before exporting any family validation or test data.
    for subject in SUBJECTS:
        for seed in SEEDS:
            folder = fit_folder(p, arm, subject, seed)
            fit = json.loads((folder / "fit.json").read_text())
            protocol = (
                Path(p["references"][arm]) / "protocol.json"
                if arm == "native_v19"
                else folder.parents[1] / "protocol.json"
            )
            assert fit["complete"] and fit["protocol_sha256"] == sha(protocol)
            assert fit["best_sha256"] == sha(folder / "best.pt")
            records[f"{subject}_{seed}"] = fit["best_sha256"]
    return records


def arrays(p, subject, split):
    folder = Path(p["references"]["within"]) / f"subject_{subject:02d}"
    with h5py.File(folder / "features.h5", "r") as f:
        assert f.attrs["complete"]
        return {
            name: f[f"{split}/{name}"][:]
            for name in ("targets", "tasks", "starts", "subjects")
        }


def load_models(p, arm, subject, device):
    if arm == "native_v19":
        from dpose_raulnet_model import CONFIG, RaulNetV19, configure_eager
    else:
        from myoji._vendor.raulnet_v20 import RaulNetV20
    models = []
    for seed in SEEDS:
        saved = torch.load(
            fit_folder(p, arm, subject, seed) / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        if arm == "native_v19":
            model = configure_eager(RaulNetV19(**CONFIG)).to(device)
            model.load_state_dict(saved["state_dict"])
        else:
            model = RaulNetV20(saved["model"]["codebook"]).to(device)
            model.load_state_dict(saved["model"])
        models.append((model.eval().requires_grad_(False), saved))
    return models


@torch.inference_mode()
def predict_v20(models, values, arm, device):
    x = torch.from_numpy(values).to(device)
    if arm == "token_v20":
        x = x.to(torch.int64)
    else:
        assert x.dtype == torch.float32
    rows = []
    for model, saved in models:
        pred = model(x) if arm == "token_v20" else model.forward_embeddings(x)
        rows.append((pred.cpu() * saved["scale"] + saved["mean"]).numpy())
    return np.stack(rows)


def native_frames(segment, start, origin=0):
    indices = (
        ((int(start) + ENDS) * 128 // 125)[:, None]
        - origin
        - np.arange(319, -1, -1)[None]
    )
    assert indices.min() >= 0 and indices.max() < segment.shape[-1]
    return segment[:, indices].transpose(1, 0, 2).copy()


@torch.inference_mode()
def predict_native(models, raw, device):
    from dpose_raulnet_model import dual_input

    assert raw.shape == (16, 100, 320, 320)
    result = np.empty((3, 16, 100, 60), np.float32)
    for begin in (0, 50):
        half = raw[:, begin : begin + 50].reshape(-1, 320, 320)
        predictions = [[] for _ in models]
        for index in range(0, len(half), 64):
            x = dual_input(torch.from_numpy(half[index : index + 64]).to(device))
            for seed_index, (model, _) in enumerate(models):
                with torch.autocast("cuda", dtype=torch.float16):
                    predictions[seed_index].append(model(x).float().cpu().numpy())
        for seed_index, parts in enumerate(predictions):
            result[seed_index, :, begin : begin + 50] = np.concatenate(parts).reshape(
                16, 50, 60
            )
    assert np.isfinite(result).all()
    return result


def native_validation_subjects(p, gpu):
    subjects = [s for s in SUBJECTS if p["native_validation_gpus"][str(s)] == gpu]
    if not subjects:
        raise ValueError(f"No original V19 validation fits on GPU {gpu!r}")
    return subjects


def complete_validation(root, p, arm, model_hashes):
    folder = root / f"val/{arm}"
    missing = [s for s in SUBJECTS if not (folder / f"subject_{s:02d}.json").exists()]
    if missing:
        print(
            json.dumps({"arm": arm, "validation_pending_subjects": missing}), flush=True
        )
        return False
    records = {}
    for subject in SUBJECTS:
        record = json.loads((folder / f"subject_{subject:02d}.json").read_text())
        assert (
            record["complete"] and record["subject"] == subject and record["arm"] == arm
        )
        assert record["protocol_sha256"] == sha(root / "protocol.json")
        assert record["checkpoint_sha256"] == {
            str(seed): model_hashes[f"{subject}_{seed}"] for seed in SEEDS
        }
        assert record["validation_sha256"] == sha(folder / f"subject_{subject:02d}.npz")
        if arm == "native_v19":
            assert record["gpu"] == p["native_validation_gpus"][str(subject)]
        records[str(subject)] = record
    write(
        folder / "receipt.json",
        {
            "complete": True,
            "checkpoints": model_hashes,
            "validation_sha256": {
                s: r["validation_sha256"] for s, r in records.items()
            },
            "per_subject_receipts": records,
            "protocol_sha256": sha(root / "protocol.json"),
            "torch": torch.__version__,
        },
    )
    return True


def validate(args, p):
    arms = ("native_v19",) if args.family == "v19" else ("token_v20", "continuous_v20")
    for arm in arms:
        model_hashes = checkpoints(p, arm)
        gpu = torch.cuda.get_device_name(args.device)
        subjects = (
            native_validation_subjects(p, gpu) if arm == "native_v19" else SUBJECTS
        )
        for subject in subjects:
            meta = arrays(p, subject, "val")
            out = args.root / f"val/{arm}/subject_{subject:02d}.npz"
            models = load_models(p, arm, subject, args.device)
            if arm == "native_v19":
                path = Path(p["data"]) / f"Sub{subject}_emg.pkl"
                assert sha(path) == p["source_archives"][path.name]
                emg = joblib.load(path)
                raw = np.stack(
                    [
                        native_frames(emg[str(task)], int(start))
                        for task, start in zip(
                            meta["tasks"], meta["starts"], strict=True
                        )
                    ]
                )
                with h5py.File(
                    Path(p["references"][arm]) / f"subject_{subject:02d}/native.h5", "r"
                ) as f:
                    np.testing.assert_array_equal(raw[:, -50:], f["val/raw"][:])
                pred = predict_native(models, raw, args.device)
                del raw, emg
            else:
                cache = (
                    Path(p["references"][arm]) / f"subject_{subject:02d}/features.h5"
                )
                name = "tokens" if arm == "token_v20" else "continuous"
                with h5py.File(cache, "r") as f:
                    pred = np.concatenate(
                        [
                            predict_v20(
                                models, f[f"val/{name}"][i : i + 2], arm, args.device
                            )
                            for i in range(0, 16, 2)
                        ],
                        axis=1,
                    )
            scores = metrics(pred, meta["targets"])
            for k, seed in enumerate(SEEDS):
                fit = json.loads(
                    (fit_folder(p, arm, subject, seed) / "fit.json").read_text()
                )
                expected = (
                    fit["best_validation_3d"]
                    if arm == "native_v19"
                    else fit["validation_best"]
                )
                assert np.isclose(
                    scores[k]["mean_3d_error"], expected, atol=2e-4, rtol=4e-6
                ), (arm, subject, seed, scores[k], expected)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                prediction=pred,
                target=meta["targets"],
                tasks=meta["tasks"],
                starts=meta["starts"],
            )
            write(
                out.with_suffix(".json"),
                {
                    "complete": True,
                    "subject": subject,
                    "arm": arm,
                    "gpu": gpu,
                    "torch": torch.__version__,
                    "protocol_sha256": sha(args.root / "protocol.json"),
                    "checkpoint_sha256": {
                        str(seed): model_hashes[f"{subject}_{seed}"] for seed in SEEDS
                    },
                    "validation_sha256": sha(out),
                    "scores": scores,
                },
            )
            print(json.dumps({"validated": arm, "subject": subject}), flush=True)
            del models, pred
            gc.collect()
            torch.cuda.empty_cache()
        complete_validation(args.root, p, arm, model_hashes)


def select(root, p):
    path = root / "selection.json"
    if path.exists():
        raise FileExistsError(path)
    selected = {}
    for arm in ARMS:
        receipt = json.loads((root / f"val/{arm}/receipt.json").read_text())
        assert receipt["complete"] and receipt["protocol_sha256"] == sha(
            root / "protocol.json"
        )
        assert checkpoints(p, arm) == receipt["checkpoints"]
        validation = {}
        for subject in SUBJECTS:
            source = root / f"val/{arm}/subject_{subject:02d}.npz"
            assert sha(source) == receipt["validation_sha256"][str(subject)]
            with np.load(source) as f:
                validation[subject] = {k: f[k] for k in ("prediction", "target")}
        selected[arm] = dict(choose(validation), validation_receipt=receipt)
    previous = json.loads(
        (Path(p["references"]["token_filter"]) / "selection.json").read_text()
    )
    assert selected["token_v20"]["selected"]["config"] == previous["selected"]["config"]
    write(
        path,
        {
            "complete": True,
            "protocol_sha256": sha(root / "protocol.json"),
            "arms": selected,
        },
    )
    path.chmod(0o444)


def require_selection(root, p):
    selection = json.loads((root / "selection.json").read_text())
    assert selection["complete"] and selection["protocol_sha256"] == sha(
        root / "protocol.json"
    )
    for arm in ARMS:
        assert (
            checkpoints(p, arm)
            == selection["arms"][arm]["validation_receipt"]["checkpoints"]
        )
    return selection


def replay_segments(p, subject):
    path = Path(p["data"]) / f"Sub{subject}_emg.pkl"
    assert sha(path) == p["source_archives"][path.name]
    emg = joblib.load(path)
    noise_root = Path(p["references"]["noise"]) / f"subject_{subject:02d}"
    receipt = json.loads((noise_root / "receipt.json").read_text())
    assert receipt["complete"]
    for name in ("paired_noise.h5", "predictions.h5", "results.json"):
        assert sha(noise_root / name) == receipt["artifacts"][name]
    diag = json.loads((noise_root / "results.json").read_text())["diagnostics"]
    hashes = {(d["window"], d["name"]): d["corrupted_native_sha256"] for d in diag}
    return emg, noise_root, hashes


def corrupted_window(emg, saved, index, task, start, condition, hashes):
    g = saved[f"window_{index:03d}"]
    origin, onset, stop = native_bounds(int(start))
    assert [int(g.attrs[k]) for k in ("task", "start", "origin", "onset", "stop")] == [
        int(task),
        int(start),
        origin,
        onset,
        stop,
    ]
    raw = emg[str(task)][:, origin:stop].astype(np.float32)
    assert array_sha(raw) == g.attrs["source_segment_sha256"]
    if condition["name"] != "clean":
        ds = g[f"seed_{condition['noise_seed']}"]
        draw = ds[:]
        assert array_sha(draw) == ds.attrs["sha256"]
        raw, _ = perturb_native(raw, onset - origin, draw, condition["snr_db"])
    assert array_sha(raw) == hashes[(index, condition["name"])]
    return raw, origin


@torch.inference_mode()
def encode_float32(tokenizer, segment, start, origin, device):
    import run_dpose_pilot as pilot
    from dpose_noise import resampled_window

    signal = resampled_window(segment, start, origin, pilot.resample)
    raw, _, _ = pilot.calibration_normalize(
        torch.from_numpy(signal)[None].to(device), 1000
    )
    ids, vectors = [], []
    for c in range(0, 320, 32):
        out = tokenizer.model(
            raw[:, c : c + 32],
            sensor="hd",
            include_spatial=False,
            channel_descriptors=torch.zeros(
                32, tokenizer.config.geometry_dim, device=device
            ),
        )
        ids.append(out.tokens[0, :, -100:].cpu().numpy().astype(np.uint16))
        vectors.append(out.continuous[0, :, -100:].float().cpu().numpy())
    return np.concatenate(ids), np.concatenate(vectors), raw[0].cpu().numpy()


def test_subject(args, p):
    selection = require_selection(args.root, p)
    if args.family == "v19":
        assert torch.cuda.get_device_name(args.device) == p["native_test_gpu"]
    subject = args.index + 1
    assert subject in SUBJECTS
    arms = ("native_v19",) if args.family == "v19" else ("token_v20", "continuous_v20")
    models = {arm: load_models(p, arm, subject, args.device) for arm in arms}
    emg, noise_root, hashes = replay_segments(p, subject)
    meta = arrays(p, subject, "test")
    outdir = args.root / f"test/{args.family}/subject_{subject:02d}"
    if (outdir / "predictions.h5").exists():
        raise FileExistsError(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.family == "v20":
        import run_dpose_pilot as pilot

        assert sha(Path(p["tokenizer_checkpoint"])) == p["tokenizer_sha256"]
        tokenizer = pilot.Tokenizer(Path(p["tokenizer_checkpoint"]), device=args.device)
        before = pilot.state_hash(tokenizer.model)
    replays, rows = [], []
    with (
        h5py.File(noise_root / "paired_noise.h5", "r") as noise,
        h5py.File(noise_root / "predictions.h5", "r") as old,
        h5py.File(outdir / "predictions.h5", "w") as out,
    ):
        out.attrs["complete"] = False
        out.attrs["selection_sha256"] = sha(args.root / "selection.json")
        out["target"], out["tasks"], out["starts"] = (
            meta["targets"],
            meta["tasks"],
            meta["starts"],
        )
        np.testing.assert_array_equal(old["target"][:], meta["targets"][:, -50:])
        for condition in CONDITIONS:
            name = condition["name"]
            predictions = {arm: np.empty((3, 16, 100, 60), np.float32) for arm in arms}
            native = []
            for index, (task, start) in enumerate(
                zip(meta["tasks"], meta["starts"], strict=True)
            ):
                segment, origin = corrupted_window(
                    emg, noise, index, task, start, condition, hashes
                )
                if args.family == "v19":
                    native.append(native_frames(segment, int(start), origin))
                else:
                    ids, vectors, normalized = encode_float32(
                        tokenizer, segment, int(start), origin, args.device
                    )
                    np.testing.assert_array_equal(ids, old[f"{name}/ids"][index])
                    if name == "clean":
                        cache = (
                            Path(p["references"]["continuous_v20"])
                            / f"subject_{subject:02d}/features.h5"
                        )
                        with h5py.File(cache, "r") as f:
                            np.testing.assert_array_equal(
                                vectors, f["test/continuous"][index]
                            )
                        assert (
                            array_sha(normalized)
                            == noise[f"window_{index:03d}"].attrs[
                                "clean_normalized_sha256"
                            ]
                        )
                    for arm, values in (
                        ("token_v20", ids),
                        ("continuous_v20", vectors),
                    ):
                        predictions[arm][:, index : index + 1] = predict_v20(
                            models[arm], values[None], arm, args.device
                        )
            if args.family == "v19":
                predictions["native_v19"] = predict_native(
                    models["native_v19"], np.stack(native), args.device
                )
                del native
            for arm, pred in predictions.items():
                assert np.isfinite(pred).all()
                if name == "clean":
                    for k, seed in enumerate(SEEDS):
                        with np.load(
                            fit_folder(p, arm, subject, seed) / "test_predictions.npz"
                        ) as f:
                            np.testing.assert_array_equal(
                                f["target"], meta["targets"][:, -50:]
                            )
                            np.testing.assert_allclose(
                                pred[k, :, -50:], f["prediction"], atol=2e-4, rtol=4e-6
                            )
                            replays.append(
                                {
                                    "arm": arm,
                                    "seed": seed,
                                    "max_abs_difference": float(
                                        np.max(
                                            np.abs(pred[k, :, -50:] - f["prediction"])
                                        )
                                    ),
                                }
                            )
                config = selection["arms"][arm]["selected"]["config"]
                for label, values in (
                    ("unfiltered", pred),
                    ("one_euro", apply(pred, config)),
                ):
                    out[f"{name}/{arm}/{label}"] = values
                    rows.extend(
                        dict(
                            row,
                            subject=subject,
                            seed=SEEDS[k],
                            arm=arm,
                            filtering=label,
                            **condition,
                        )
                        for k, row in enumerate(metrics(values, meta["targets"]))
                    )
            out.flush()
            print(
                json.dumps(
                    {
                        "subject": subject,
                        "family": args.family,
                        "condition": name,
                        "complete": True,
                    }
                ),
                flush=True,
            )
        out.attrs["complete"] = True
    if args.family == "v20":
        assert before == pilot.state_hash(tokenizer.model)
    write(
        outdir / "results.json",
        {"complete": True, "rows": rows, "clean_replay": replays},
    )
    write(
        outdir / "receipt.json",
        {
            "complete": True,
            "protocol_sha256": sha(args.root / "protocol.json"),
            "selection_sha256": sha(args.root / "selection.json"),
            "new_fits": 0,
            "all_corrupted_native_hashes_exact": True,
            "torch": torch.__version__,
            "predictions_sha256": sha(outdir / "predictions.h5"),
            "results_sha256": sha(outdir / "results.json"),
        },
    )


def report(root, p):
    require_selection(root, p)
    rows = []
    for subject in SUBJECTS:
        targets = []
        for family in ("v20", "v19"):
            folder = root / f"test/{family}/subject_{subject:02d}"
            r = json.loads((folder / "receipt.json").read_text())
            assert r["complete"] and r["selection_sha256"] == sha(
                root / "selection.json"
            )
            assert r["predictions_sha256"] == sha(folder / "predictions.h5")
            assert r["results_sha256"] == sha(folder / "results.json")
            rows.extend(json.loads((folder / "results.json").read_text())["rows"])
            with h5py.File(folder / "predictions.h5", "r") as f:
                targets.append(f["target"][:])
        np.testing.assert_array_equal(*targets)
    summary = []
    draws = np.random.default_rng(260930).integers(0, 9, (10000, 9))
    for arm in ARMS:
        for filtering in ("unfiltered", "one_euro"):
            clean = np.array(
                [
                    np.mean(
                        [
                            r["mean_3d_error"]
                            for r in rows
                            if r["arm"] == arm
                            and r["filtering"] == filtering
                            and r["subject"] == s
                            and r["snr_db"] is None
                        ]
                    )
                    for s in SUBJECTS
                ]
            )
            for snr in (None, *SNR_DB):
                selected = [
                    r
                    for r in rows
                    if r["arm"] == arm
                    and r["filtering"] == filtering
                    and r["snr_db"] == snr
                ]
                values = np.array(
                    [
                        np.mean(
                            [r["mean_3d_error"] for r in selected if r["subject"] == s]
                        )
                        for s in SUBJECTS
                    ]
                )
                delta = values - clean
                summary.append(
                    {
                        "arm": arm,
                        "filtering": filtering,
                        "snr_db": snr,
                        "mean_3d_error": float(values.mean()),
                        "clean_delta": float(delta.mean()),
                        "participant_ci95": np.percentile(
                            values[draws].mean(1), [2.5, 97.5]
                        ).tolist(),
                        "paired_clean_delta_ci95": np.percentile(
                            delta[draws].mean(1), [2.5, 97.5]
                        ).tolist(),
                        "coordinate_r2": float(
                            np.mean([r["coordinate_r2"] for r in selected])
                        ),
                        "velocity_r2": float(
                            np.mean([r["velocity_r2"] for r in selected])
                        ),
                    }
                )
    write(
        root / "results.json",
        {
            "complete": True,
            "summary": summary,
            "rows": rows,
            "protocol_sha256": sha(root / "protocol.json"),
            "selection_sha256": sha(root / "selection.json"),
            "limitations": p["limitations"],
        },
    )
    lines = [
        "# Matched V20/V19 native-noise evaluation",
        "",
        p["limitations"],
        "",
        "| Model | Filter | SNR dB | Mean 3D error | Change from clean | Velocity R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['arm']} | {r['filtering']} | {r['snr_db'] if r['snr_db'] is not None else 'clean'} | {r['mean_3d_error']:.4f} | {r['clean_delta']:.4f} | {r['velocity_r2']:.3f} |"
        for r in summary
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("freeze", "validate", "select", "test", "report")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=paths.RESULTS)
    parser.add_argument("--family", choices=("v20", "v19"), default="v20")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "freeze":
        freeze(args)
        return
    p = verify(args.root)
    if args.mode == "validate":
        validate(args, p)
    elif args.mode == "select":
        select(args.root, p)
    elif args.mode == "test":
        test_subject(args, p)
    else:
        report(args.root, p)


if __name__ == "__main__":
    main()
