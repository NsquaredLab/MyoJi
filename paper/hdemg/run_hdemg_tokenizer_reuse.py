"""Fresh, donor-only tokenizers followed by nine personalized HD-EMG readouts.

Existing inspected results are kept intact. Subjects 10--13 supply unlabeled
tokenizer training data; subjects 1--9 supply only new downstream decoder data.
Both routes reuse the same audited spatial CNN/TCN fitting implementation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
import torch

from paper.common import paths
from paper.hdemg import run_hdemg_spar_simple as decoder
from paper.hdemg.run_hdemg_spar_full import Monitor
from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize
from myoji.spar import grid_layout, spar_simple_dataset_batched

BASE = Path(__file__).resolve().parents[1]
DONORS = (10, 11, 12, 13)
SUBJECTS = tuple(range(1, 10))
ROUTES = ("spar", "learned")
ARMS, SEEDS = decoder.ARMS, decoder.SEEDS
SPLITS = {"train": 64, "val": 16, "test": 16}
sha, write = decoder.sha, decoder.write


def read(path):
    return json.loads(Path(path).read_text())


def check_partition(donors, held_out):
    if (not donors or not held_out or len(set(donors)) != len(donors)
            or len(set(held_out)) != len(held_out) or set(donors) & set(held_out)):
        raise ValueError("Tokenizer donors and evaluation people must be unique and disjoint")


def donor_files(directory, donors=DONORS, held_out=SUBJECTS):
    check_partition(donors, held_out)
    paths = sorted(Path(directory).glob("Sub*_emg.pkl"))
    expected = {f"Sub{s}_emg.pkl" for s in donors}
    if {p.name for p in paths} != expected:
        raise ValueError("Donor directory must contain exactly the permitted EMG recordings")
    # A renamed link to an excluded recording must not bypass the allowlist.
    if any(p.resolve().name != p.name or not p.is_file() for p in paths):
        raise ValueError("Donor filenames must match their actual resolved recordings")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Duplicate underlying donor recordings")
    return paths


def child_args(args, route, subject):
    return SimpleNamespace(root=args.root / route / f"subject_{subject:02d}", device="cuda")


def implementation():
    # SPAR, the codebook and the vendored V20 module are library files under src/.
    names = ["hdemg/run_hdemg_tokenizer_reuse.py", "pretraining/train_mixed_sources.py",
             "hdemg/run_hdemg_spar_simple.py", "hdemg/run_hdemg_spar_full.py", "common/hdemg_figure_statistics.py"]
    names += sorted(str(x.relative_to(paths.REPO)) for x in (paths.REPO / "src").rglob("*.py"))
    return {name: sha(decoder.source_path(name)) for name in names}


def freeze(args):
    path = args.root / "protocol.json"
    if path.exists():
        return verify(args)
    args.root.mkdir(parents=True, exist_ok=True)
    check_partition(DONORS, SUBJECTS)
    reference = paths.RESULTS / "hdemg_within_v1"
    spar_reference = paths.RESULTS / "hdemg_spar_simple_full_v1"
    old = read(reference / "protocol.json")
    old_spar = read(spar_reference / "protocol.json")
    windows = read(reference / "selection.json")["windows"]
    if old_spar["windows"] != windows or old_spar["decoder"]["training"] != old["training"]:
        raise ValueError("Existing neural and SPAR comparisons must share windows and fit settings")
    for subject in SUBJECTS:
        if {s: sum(w["subject"] == subject and w["split"] == s for w in windows) for s in SPLITS} != SPLITS:
            raise ValueError("Unexpected decoder window selection")
    donor_dir = args.root / "donor_emg"
    donor_dir.mkdir(exist_ok=True)
    for subject in DONORS:
        source = args.data / f"Sub{subject}_emg.pkl"
        link = donor_dir / source.name
        if not link.exists():
            link.symlink_to(source.resolve())
    donors = donor_files(donor_dir)
    # Read only hyperparameters/provenance from the earlier run, never its weights.
    recipe = read(args.recipe)["training_arguments"].copy()
    for key in ("ramp_dir", "mirpni_dir", "ramp_batches_per_recording", "mirpni_batches_per_recording"):
        recipe.pop(key, None)
    recipe.update(hdemg_dir=str(donor_dir.resolve()), output=str(args.root / "learned/tokenizer.pt"),
                  emg2pose_root=str(paths.EMG2POSE),
                  emg2pose_metadata=str(paths.EMG2POSE / "emg2pose_metadata.csv"),
                  emg2pose_window_cache=str(paths.RESULTS / "clean_base_v1/train.h5"),
                  seed=260916, resume=None)
    if tuple(recipe[k] for k in ("continuous_steps", "calibration_steps", "quantized_steps")) != (50000, 128, 250000):
        raise ValueError("Expected the full original 300000-update learned-tokenizer budget")
    upstream_contract = read(args.recipe)["data_contract"]
    for field, key in (("emg2pose_metadata", "metadata_sha256"), ("emg2pose_window_cache", "window_cache_sha256")):
        if sha(Path(recipe[field])) != upstream_contract[key]:
            raise ValueError("Changed official Meta training data")
    p = dict(name="hdemg_tokenizer_reuse_v1", tokenizer_donors=list(DONORS), decoder_subjects=list(SUBJECTS),
             routes=list(ROUTES), arms=list(ARMS), decoder_seeds=list(SEEDS), windows=windows, split_counts=SPLITS,
             scope="Transfer of newly fitted, frozen tokenizers to nine tokenizer-unseen HD-EMG participants, with supervised personalized decoders. Not zero-shot pose decoding or cross-session transfer.",
             initialization="New learned encoder and codebook from scratch; new donor-only SPAR scaling/codebook. No old tokenizer or decoder weights reused. One tokenizer fit per route, shared by all nine people and all decoder seeds.",
             data_boundary="No EMG, features, feature statistics, or labels from subjects 1--9 enter either tokenizer fit. Fixed per-window causal prefix median/MAD remains permitted inference preprocessing. Only the downstream decoder and its target scaling use each new participant's training block; validation selects the decoder checkpoint.",
             learned_training_arguments=recipe, meta_training_contract=upstream_contract,
             donor_files={x.name: dict(path=str(x.resolve()), sha256=sha(x)) for x in donors},
             spar=copy.deepcopy(old_spar["tokenizer"]), feature=old_spar["feature"], geometry=old_spar["geometry"],
             geometry_confirmation=old_spar["geometry_confirmation"], training=old["training"],
             donor_spar_selection="First chronological 60% of each donor task, nonoverlapping 6000-sample windows after causal resampling; up to eight evenly spaced windows per task, selected using EMG length only. No kinematics file is opened for donors.",
             fit_selection="Final fixed-step learned tokenizer and fixed-budget SPAR quantizer; no downstream-based tokenizer selection. Decoder checkpoint with minimum validation 3D error, established patience. All 54 fits per route audited before that route's test inference.",
             metric="Unfiltered mean Euclidean error over 20 wrist-relative landmarks, final 50 of 100 frames, millimetres; average seeds within participant before statistics.",
             statistics="Two predeclared token-minus-continuous contrasts (one per route), exact two-sided participant paired sign-flip tests and Holm over both. Participant bootstrap 95% intervals, 10000 draws, seed 260916. Nine people are the units; three head seeds are not independent people. Non-significance does not establish equivalence.",
             reference=str(reference), spar_reference=str(spar_reference),
             reference_protocol_sha256=sha(reference / "protocol.json"), reference_selection_sha256=sha(reference / "selection.json"),
             spar_reference_protocol_sha256=sha(spar_reference / "protocol.json"),
             source_sha256=implementation(), resources=dict(gpus=1, cpu_threads=4, partition="rtx3080,v100", concurrent_fits=1, walltime="08:00:00"))
    p["spar"].update(training_scope="One shared scaler and codebook, fit only on subjects 10--13; frozen across all nine unseen people and three decoder seeds.")
    write(path, p)
    for route in ROUTES:
        for subject in SUBJECTS:
            c = child_args(args, route, subject)
            write(c.root / "protocol.json", dict(name=p["name"], route=route, subjects=[subject], seeds=list(SEEDS), arms=list(ARMS),
                  windows=[w for w in windows if w["subject"] == subject], split_counts=SPLITS, scope=p["scope"],
                  campaign_protocol_sha256=sha(path),
                  decoder=dict(training=p["training"], trainable_parameters=689468 if route == "spar" else 704700)))
    write(args.root / "status.json", dict(stage="frozen", expected_decoder_fits=108, tokenizer_donors=list(DONORS), decoder_subjects=list(SUBJECTS)))
    return p


def verify(args):
    p = read(args.root / "protocol.json")
    check_partition(p["tokenizer_donors"], p["decoder_subjects"])
    if p["source_sha256"] != implementation():
        raise ValueError("Frozen implementation changed")
    paths = donor_files(args.root / "donor_emg", p["tokenizer_donors"], p["decoder_subjects"])
    for path in paths:
        record = p["donor_files"][path.name]
        if str(path.resolve()) != record["path"] or sha(path) != record["sha256"]:
            raise ValueError("Donor recording changed")
    for folder, filename, key in ((p["reference"], "protocol.json", "reference_protocol_sha256"),
                                  (p["reference"], "selection.json", "reference_selection_sha256"),
                                  (p["spar_reference"], "protocol.json", "spar_reference_protocol_sha256")):
        if sha(Path(folder) / filename) != p[key]:
            raise ValueError("Reference split/protocol changed")
    return p


def select_donor_windows(emg):
    rows = []
    for task in sorted(emg, key=int):
        last = (emg[task].shape[-1] * 125 // 128) * 3 // 5
        starts = list(range(0, last - 6000 + 1, 6000))
        if not starts:
            raise ValueError("Donor task has no complete training window")
        selected = np.linspace(0, len(starts) - 1, min(8, len(starts)), dtype=int)
        rows.extend(dict(task=str(task), start=starts[i]) for i in selected)
    return rows


def fit_spar(args, p):
    output = args.root / "spar/tokenizer.npz"
    receipt_path = output.with_suffix(".json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["sha256"] != sha(output) or receipt["protocol_sha256"] != sha(args.root / "protocol.json"):
            raise ValueError("SPAR tokenizer receipt mismatch")
        return
    from scipy.signal import firwin, upfirdn
    layout = grid_layout(p["geometry"]["grids"])
    mask = np.tile(layout.mask, (1, 6))
    taps = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
    indices = decoder.ENDS[:, None] - np.arange(319, -1, -1)[None]
    features, manifest = [], []
    for path in donor_files(args.root / "donor_emg"):
        subject = int(path.stem.split("_")[0][3:])
        emg = joblib.load(path)  # Deliberately no kinematics or held-out participant data.
        rows = select_donor_windows(emg)
        for task in sorted(emg, key=int):
            signal = upfirdn(taps, emg[task].astype(np.float32), up=125, down=128, axis=-1)
            signal = signal[..., :emg[task].shape[-1] * 125 // 128].astype(np.float32)
            for row in rows:
                if row["task"] != str(task):
                    continue
                raw = decoder.normalize_raw(signal[:, row["start"]:row["start"] + 6000])
                analysis = raw[:, indices].transpose(1, 2, 0).copy()
                values = spar_simple_dataset_batched(analysis, layout=layout, channelwise=True, batch_size=16, device="cuda")
                features.append(values.transpose(1, 0, 2))
                manifest.append(dict(subject=subject, **row))
            print(json.dumps(dict(stage="donor_spar", subject=subject, task=str(task), windows=len(features))), flush=True)
        del emg, signal
    x = np.stack(features)
    del features
    mean, scale = fit_standardization(x, mask)
    z = standardize(x, mean, scale, mask)
    cfg = p["spar"]
    book, fitted = fit_codebook(z, vocabulary=cfg["vocabulary"], seed=cfg["seed"], max_vectors=cfg["max_vectors"], iterations=cfg["iterations"], device="cuda")
    np.savez(output, codebook=book, mask=mask, feature_mean=mean, feature_scale=scale)
    output.chmod(0o444)
    write(receipt_path, dict(complete=True, sha256=sha(output), protocol_sha256=sha(args.root / "protocol.json"),
                            training_subjects=list(DONORS), windows=manifest, fit=fitted, no_pose_labels=True))
    del x, z
    gc.collect()
    torch.cuda.empty_cache()


def training_command(p, output, smoke=False, resume=None):
    values = p["learned_training_arguments"].copy()
    values.update(output=str(output), resume=str(resume) if resume else None)
    if smoke:
        values.update(continuous_steps=40, calibration_steps=12, quantized_steps=40, checkpoint_every=40, report_every=40, seed=260917)
    command = [sys.executable, "-u", "-m", "paper.pretraining.train_mixed_sources"]
    for key, value in values.items():
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    return command


def audit_learned(state, p, smoke=False):
    expected_names = {f"Sub{s}_emg.pkl" for s in p["tokenizer_donors"]}
    seen = state["source_seen_recordings"].get("hdemg", [])
    if not seen or not {Path(x).name for x in seen}.issubset(expected_names):
        raise ValueError("Learned tokenizer saw a forbidden HD-EMG recording")
    contract = state["data_contract"]
    if ({k: contract.get(k) for k in p["meta_training_contract"]} != p["meta_training_contract"]
            or ("sources" in contract and [s["available_recordings"] for s in contract["sources"]
                                           if s["name"] == "hdemg"] != [len(DONORS)])):
        raise ValueError("Wrong neural pretraining data contract")
    for key, value in p["learned_training_arguments"].items():
        if key in ("output", "resume") or (smoke and key in ("continuous_steps", "calibration_steps", "quantized_steps", "checkpoint_every", "report_every", "seed")):
            continue
        if state["training_args"][key] != value:
            raise ValueError("Changed tokenizer training argument: " + key)
    expected = (40, 40) if smoke else (50000, 250000)
    if state["progress"] != dict(phase="complete", continuous_step=expected[0], quantized_step=expected[1]):
        raise ValueError("Tokenizer training is incomplete")
    if not smoke and {Path(x).name for x in seen} != expected_names:
        raise ValueError("Production tokenizer did not sample all four donors")


def fit_learned(args, p, smoke=False):
    from myoji.learned.stream import export_checkpoint
    folder = args.root / ("smoke" if smoke else "learned")
    output = folder / "tokenizer.pt"
    folder.mkdir(exist_ok=True)
    if not output.exists():
        last = folder / "tokenizer.last.pt"
        # Recovery may resume only the checkpoint located within this new run.
        subprocess.run(training_command(p, output, smoke, last if last.exists() else None), check=True, cwd=paths.REPO)
    state = torch.load(output, map_location="cpu", weights_only=False)
    audit_learned(state, p, smoke)
    release = folder / "release.pt"
    if not release.exists():
        export_checkpoint(output, release, calibration_samples=1000)
    else:
        saved = torch.load(release, map_location="cpu", weights_only=True)
        if saved["source_sha256"] != sha(output):
            raise ValueError("Release does not match newly trained tokenizer")
    write(folder / "tokenizer-receipt.json", dict(complete=True, smoke_only=smoke, checkpoint_sha256=sha(output),
          release_sha256=sha(release), source_seen_recordings=state["source_seen_recordings"],
          protocol_sha256=sha(args.root / "protocol.json"), progress=state["progress"], fresh_initialization=True))
    del state
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def encode_window(tokenizer, raw):
    vectors, ids = [], []
    for c in range(0, 320, 32):
        x = torch.as_tensor(raw[c:c + 32], device="cuda")[None]
        encoded = tokenizer.model(x, sensor="hd", include_spatial=False,
                                  channel_descriptors=torch.zeros(32, tokenizer.config.geometry_dim, device="cuda"))
        vectors.append(encoded.continuous[0, :, -100:].float().cpu().numpy())
        ids.append(encoded.tokens[0, :, -100:].cpu().numpy().astype(np.uint16))
    return np.concatenate(vectors), np.concatenate(ids)


def prepare(args, p, route, subject):
    c = child_args(args, route, subject)
    protocol_sha = sha(c.root / "protocol.json")
    out = c.root / "features.h5"
    receipt_path = c.root / "cache-receipt.json"
    if receipt_path.exists():
        r = read(receipt_path)
        if r["sha256"] != sha(out) or r["protocol_sha256"] != protocol_sha:
            raise ValueError("Cache integrity mismatch")
        return
    if out.exists() or out.with_suffix(".partial.h5").exists():
        raise FileExistsError("Incomplete feature cache needs explicit recovery")
    old_path = Path(p["spar_reference"] if route == "spar" else p["reference"]) / c.root.name / "features.h5"
    tokenizer = None
    if route == "spar":
        with np.load(args.root / "spar/tokenizer.npz") as state:
            book, mask, mean, scale = (state[k] for k in ("codebook", "mask", "feature_mean", "feature_scale"))
        tokenizer_sha = sha(args.root / "spar/tokenizer.npz")
    else:
        from myoji.learned.stream import Tokenizer
        tokenizer = Tokenizer(args.root / "learned/release.pt", device="cuda")
        before = decoder.tensor_hash(tokenizer.model.state_dict())
        book = tokenizer.model.quantizer.codebook.cpu().numpy()
        mask = np.ones((320, 256), dtype=np.float32)
        tokenizer_sha = sha(args.root / "learned/release.pt")
    selected = [w for w in p["windows"] if w["subject"] == subject]
    temporary = out.with_suffix(".partial.h5")
    with h5py.File(old_path, "r") as old, h5py.File(temporary, "w") as f:
        if not old.attrs["complete"] or old.attrs["protocol_sha256"] != sha(old_path.parent / "protocol.json"):
            raise ValueError("Original cache has invalid provenance")
        f.attrs["complete"] = False
        f.attrs["protocol_sha256"] = protocol_sha
        f.attrs["tokenizer_sha256"] = tokenizer_sha
        f["codebook"], f["mask"] = book, mask
        if route == "spar":
            f["feature_mean"], f["feature_scale"] = mean, scale
        for split, count in SPLITS.items():
            rows = [w for w in selected if w["split"] == split]
            group = f.create_group(split)
            for field in ("subject", "task", "start"):
                expected = [int(w[field]) for w in rows]
                np.testing.assert_array_equal(old[f"{split}/{field}s"][:], expected)
                group[field + "s"] = expected
            group["targets"] = old[f"{split}/targets"][:]
            v = group.create_dataset("continuous", (count, 320, 100, book.shape[1]), dtype="f4")
            t = group.create_dataset("tokens", (count, 320, 100), dtype="u2")
            for index in range(count):
                if route == "spar":
                    values = standardize(old[f"{split}/spar"][index:index + 1], mean, scale, mask)
                    tokens = tokenize(values, book, device="cuda")
                    v[index], t[index] = values[0], tokens[0]
                else:
                    # Old raw cache is fixed causal median/MAD EMG, not learned features.
                    values, tokens = encode_window(tokenizer, old[f"{split}/raw"][index])
                    v[index], t[index] = values, tokens
            if not np.isfinite(v[:]).all():
                raise ValueError("Nonfinite newly encoded features")
            print(json.dumps(dict(stage="new_cache", route=route, subject=subject, split=split)), flush=True)
        if tokenizer is not None and before != decoder.tensor_hash(tokenizer.model.state_dict()):
            raise ValueError("Frozen encoder/codebook changed during inference")
        f.attrs["complete"] = True
    temporary.replace(out)
    out.chmod(0o444)
    write(receipt_path, dict(complete=True, protocol_sha256=protocol_sha, sha256=sha(out), tokenizer_sha256=tokenizer_sha,
                            upstream_cache=str(old_path), upstream_protocol_sha256=sha(old_path.parent / "protocol.json"),
                            reuses_only_fixed_raw_preprocessing_or_unstandardized_spar=True))
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def audit_subject(args, route, subject):
    c = child_args(args, route, subject)
    digest = sha(c.root / "protocol.json")
    receipt = read(c.root / "cache-receipt.json")
    if receipt["protocol_sha256"] != digest or receipt["sha256"] != sha(c.root / "features.h5"):
        raise ValueError("Decoder cache changed")
    with h5py.File(c.root / "features.h5", "r") as f:
        book, mask = (torch.from_numpy(f[k][:]) for k in ("codebook", "mask"))
        target = torch.from_numpy(f["train/targets"][:, -50:]).reshape(-1, 60)
        mean, scale = target.mean(0), target.std(0, correction=0).clamp_min(1e-4)
        if route == "spar":
            with np.load(args.root / "spar/tokenizer.npz") as donor:
                for key in ("codebook", "mask", "feature_mean", "feature_scale"):
                    np.testing.assert_array_equal(f[key][:], donor[key])
        else:
            bundle = torch.load(args.root / "learned/release.pt", map_location="cpu", weights_only=True)
            torch.testing.assert_close(book, bundle["state_dict"]["quantizer.codebook"], rtol=0, atol=0)
    fits = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = c.root / "fits" / f"{arm}_{seed}"
            fit = read(folder / "fit.json")
            if not fit["complete"] or fit["protocol_sha256"] != digest or fit["best_sha256"] != sha(folder / "best.pt"):
                raise ValueError("Missing or changed decoder fit")
            state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            if state["protocol_sha256"] != digest or state["seed"] != seed or state["arm"] != arm:
                raise ValueError("Wrong decoder checkpoint")
            for actual, expected in ((state["model"]["decoder.codebook"], book), (state["model"]["mask"], mask),
                                     (state["mean"], mean), (state["scale"], scale)):
                if not torch.equal(actual, expected):
                    raise ValueError("Tokenizer changed or target normalization was not train-only")
            if state["epoch"] != min(fit["history"], key=lambda x: x["val_error_mm"])["epoch"]:
                raise ValueError("Checkpoint was not selected by validation")
            fits[(arm, seed)] = fit
    for seed in SEEDS:
        if fits[("tokens", seed)]["initial_state_sha256"] != fits[("continuous", seed)]["initial_state_sha256"]:
            raise ValueError("Paired arms did not start with identical decoder weights")
    result = dict(passed=True, subject=subject, fits=6, protocol_sha256=digest,
                  tokenizer_sha256=receipt["tokenizer_sha256"], cache_sha256=receipt["sha256"],
                  checkpoints={f"{a}_{s}": f["best_sha256"] for (a, s), f in fits.items()})
    write(c.root / "fit-audit.json", result)
    return result


def global_gate(args, route):
    receipts = [audit_subject(args, route, s) for s in SUBJECTS]
    if sum(r["fits"] for r in receipts) != 54 or len({r["tokenizer_sha256"] for r in receipts}) != 1:
        raise ValueError("Require all 54 fits using exactly one shared frozen tokenizer")
    write(args.root / route / "global-fit-gate.json", dict(passed=True, subjects=receipts, fits=54,
                                                        protocol_sha256=sha(args.root / "protocol.json")))


def evaluate(args, p, route):
    global_gate(args, route)
    scores = []
    for subject in SUBJECTS:
        c = child_args(args, route, subject)
        for arm in ARMS:
            data = decoder.load(c, "test", arm)
            for seed in SEEDS:
                folder = c.root / "fits" / f"{arm}_{seed}"
                state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                model = decoder.SparPose(data["book"], data["mask"]).to("cuda")
                model.load_state_dict(state["model"])
                pred = decoder.prediction(model, data, arm, state["mean"], state["scale"])
                target = data["targets"][:, -50:].numpy()
                test_rows = [w for w in p["windows"] if w["subject"] == subject and w["split"] == "test"]
                np.savez_compressed(folder / "test_predictions.npz", prediction=pred, target=target, tasks=data["tasks"],
                                    subject=np.full(len(pred), subject), starts=[w["start"] for w in test_rows])
                scores.append(dict(subject=subject, arm=arm, seed=seed, error_mm=decoder.error_mm(pred, target),
                                   per_task={str(t): decoder.error_mm(pred[data["tasks"] == t], target[data["tasks"] == t]) for t in np.unique(data["tasks"])},
                                   training_mean_baseline_mm=decoder.error_mm(np.broadcast_to(state["mean"].numpy(), target.shape), target),
                                   predictions_sha256=sha(folder / "test_predictions.npz")))
                del model
            del data
    write(args.root / route / "results.json", dict(complete=True, route=route, scope=p["scope"], scores=scores,
                                                   protocol_sha256=sha(args.root / "protocol.json")))


def summarize(args, p):
    summary, tests = {}, []
    for route in ROUTES:
        scores = read(args.root / route / "results.json")["scores"]
        means = {arm: np.array([np.mean([x["error_mm"] for x in scores if x["arm"] == arm and x["subject"] == s]) for s in SUBJECTS]) for arm in ARMS}
        summary[route] = {arm: dict(mean_mm=float(x.mean()), participant_sd_mm=float(x.std(ddof=1)), participant_means_mm=x.tolist()) for arm, x in means.items()}
        delta = means["tokens"] - means["continuous"]
        null = (np.array(list(itertools.product((-1, 1), repeat=len(delta)))) * delta).mean(-1)
        pval = float(np.mean(np.abs(null) >= abs(delta.mean()) - 1e-12))
        rng = np.random.default_rng(260916)
        boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(-1)
        tests.append(dict(route=route, contrast="tokens minus continuous", mean_difference_mm=float(delta.mean()),
                          participant_differences_mm=delta.tolist(), bootstrap95_mm=np.quantile(boot, [.025, .975]).tolist(), p_raw=pval))
    running = 0.
    for rank, i in enumerate(sorted(range(len(tests)), key=lambda i: tests[i]["p_raw"])):
        running = max(running, min(1., tests[i]["p_raw"] * (len(tests) - rank)))
        tests[i]["p_holm"] = running
    result = dict(complete=True, scope=p["scope"], summary=summary, comparisons=tests, protocol_sha256=sha(args.root / "protocol.json"))
    write(args.root / "results.json", result)
    lines = ["# HD-EMG tokenizer reuse", "", p["scope"], "", "Tokenizer fitting: subjects 10–13 only. New decoder fitting/evaluation: subjects 1–9; three seeds each.", "", "| Route | Input | 3D error (mm), mean ± participant SD |", "|---|---|---:|"]
    for route, arms in summary.items():
        for arm, row in arms.items():
            lines.append(f"| {route} | {arm} | {row['mean_mm']:.3f} ± {row['participant_sd_mm']:.3f} |")
    lines += ["", "One new tokenizer fit per route. This tests personalized adaptation using frozen representations, with no claim of zero-shot pose decoding or equivalence from nonsignificance.", ""]
    (args.root / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "all", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=paths.HDEMG)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    args.root, args.data = args.root.resolve(), args.data.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = freeze(args)
    if args.stage == "freeze":
        print(json.dumps(dict(stage="frozen", root=str(args.root), decoder_fits=108)), flush=True)
        return
    if args.stage == "summarize":
        summarize(args, p)
        return
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible allocated GPU is required")
    started = time.monotonic()
    with Monitor(args.root) as monitor:
        write(args.root / "runtime.json", dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                                               gpu=torch.cuda.get_device_name(), host=os.uname().nodename, job_id=os.environ.get("SLURM_JOB_ID")))
        monitor.phase = "learned tokenizer smoke: donor-only data, discarded weights"
        fit_learned(args, p, smoke=True)
        monitor.phase = "shared donor-only SPAR tokenizer"
        fit_spar(args, p)
        for route in ROUTES:
            if route == "learned":
                monitor.phase = "fresh learned tokenizer: 50000 continuous + 250000 quantized updates"
                fit_learned(args, p)
            for subject in SUBJECTS:
                monitor.phase = f"{route}: encode subject {subject} with frozen shared tokenizer"
                prepare(args, p, route, subject)
                c = child_args(args, route, subject)
                child = read(c.root / "protocol.json")
                for arm in ARMS:
                    for seed in SEEDS:
                        monitor.phase = f"{route}: decoder subject {subject} {arm} seed {seed}"
                        decoder.fit(c, child, arm, seed)
                write(args.root / "status.json", dict(stage="decoders", route=route, completed_subject=subject,
                      route_completed_fits=subject * 6, route_expected_fits=54, elapsed_seconds=time.monotonic() - started))
            monitor.phase = f"{route}: 54-fit integrity gate and test evaluation"
            evaluate(args, p, route)
        monitor.phase = "participant-paired analysis"
        summarize(args, p)
        write(args.root / "status.json", dict(stage="complete", decoder_fits=108, elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
