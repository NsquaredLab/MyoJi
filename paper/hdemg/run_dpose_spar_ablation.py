"""SPAR feature ablation on the nine-person campaign, reusing the frozen pilot's model and fit loop.

Same windows, codebook size, seeds, decoder and gates as dpose_spar_simple_full_v1; only the
per-channel descriptor changes, chosen by SPAR_ABLATION_VARIANT:
  diagonal  the six matrix-log diagonal entries D0 (one per band) of the stored v64 SPAR features;
            D0 is an entry of log(S), so neighbours still enter through the matrix logarithm.
  logpower  log of the centre electrode's own band power from the same weighted spectra
            (SPAR's bin weighting and 1e-6 relative ridge), with no matrix logarithm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
from scipy.signal import firwin, upfirdn
import torch

from paper.common import paths
from paper.hdemg import run_dpose_spar_simple as pilot
from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize
from myoji.spar import grid_layout, spds_tensor
from paper.common.dpose_figure_statistics import exact_paired_test, holm_adjust

SUBJECTS = tuple(range(1, 10))
SEEDS, ARMS, ENDS = pilot.SEEDS, pilot.ARMS, pilot.ENDS
sha, write = pilot.sha, pilot.write
VARIANT = os.environ.get("SPAR_ABLATION_VARIANT", "")
DIMS = 6                    # one value per frequency band
TRAINABLE_6D = 688700       # proposed decoder with 6-dimensional channel vectors (checked in tests)
V1 = paths.RESULTS / "dpose_spar_simple_full_v1"
V64 = paths.RESULTS / "dpose_spar_simple_full_v64"


@torch.inference_mode()
def centre_log_power(analysis, layout, device, batch_size=16):
    """(frames,samples,channels) -> (frames,channels,bands): log weighted band power of each centre."""
    out = []
    for start in range(0, len(analysis), batch_size):
        spds = spds_tensor(analysis[start:start + batch_size], layout=layout, device=device)
        out.append(spds[..., 0, 0].real.clamp_min(1e-12).log().permute(0, 2, 1).cpu().numpy())
    return np.concatenate(out)
SOURCES = (*pilot.SOURCES, "hdemg/run_dpose_spar_ablation.py", "common/dpose_figure_statistics.py")


def read(path):
    return json.loads(path.read_text())


def subject_args(args, subject):
    return SimpleNamespace(root=args.root / f"subject_{subject:02d}", data=args.data, device=args.device)


def reference_path(subject, arm, seed):
    """Test predictions of the learned-token (V20) campaigns, run into paths.RESULTS."""
    name = "dpose_v20_v1" if arm == "tokens" else "dpose_v20_continuous_v1"
    return paths.RESULTS / name / f"subject_{subject:02d}/fits/{arm}_{seed}/test_predictions.npz"


def freeze(args):
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root / "protocol.json"
    if path.exists():
        p = read(path)
        assert p["feature"]["variant"] == VARIANT, "protocol and SPAR_ABLATION_VARIANT disagree"
        for source, digest in p["source_sha256"].items():
            assert sha(pilot.source_path(source)) == digest, "Frozen implementation changed: " + source
        assert sha(args.reference / "protocol.json") == p["reference_protocol_sha256"]
        assert sha(args.reference / "selection.json") == p["reference_selection_sha256"]
        for subject in SUBJECTS:
            child = read(subject_args(args, subject).root / "protocol.json")
            assert child["campaign_protocol_sha256"] == sha(path)
        return p
    old = read(args.pilot / "protocol.json")
    confirmed = read(args.pilot / "geometry-confirmation.json")
    assert confirmed["exact_match_to_used_mapping"] and confirmed["protocol_sha256"] == sha(args.pilot / "protocol.json")
    assert old["geometry"] == json.loads(json.dumps(grid_layout().record()))
    geometry_hash = hashlib.sha256(json.dumps(old["geometry"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert geometry_hash == confirmed["geometry_canonical_json_sha256"]
    reference = read(args.reference / "protocol.json")
    selected = read(args.reference / "selection.json")["windows"]
    windows = [w for w in selected if w["subject"] in SUBJECTS]
    for subject in SUBJECTS:
        counts = {split: sum(w["subject"] == subject and w["split"] == split for w in windows) for split in ("train", "val", "test")}
        assert counts == dict(train=64, val=16, test=16)
    assert reference["training"] == old["decoder"]["training"]
    p = copy.deepcopy(old)
    assert VARIANT in ("diagonal", "logpower"), "set SPAR_ABLATION_VARIANT to diagonal or logpower"
    p["feature"] = dict(p["feature"], variant=VARIANT, channel_vector_dim=DIMS, compact_values_per_frame=320 * DIMS,
                        layout=("Each channel carries the matrix-log diagonal D0 of each band (no coupling entries)." if VARIANT == "diagonal" else
                                "Each channel carries log of its own weighted band power per band (no matrix logarithm, no couplings)."),
                        caveat=("D0 is an entry of log(S); neighbours still enter through the matrix logarithm." if VARIANT == "diagonal" else
                                "Bin weights pool spectral power over all channels, as in SPAR; the 1e-6 relative ridge is retained."))
    p["decoder"] = dict(p["decoder"], family="Proposed spatial CNN + TCN, adapted to 6-dimensional inputs", trainable_parameters=TRAINABLE_6D)
    p.update(name=f"dpose_spar_ablation_{VARIANT}_v1", subjects=list(SUBJECTS), windows=windows,
             derived_from="dpose_spar_simple_full_v1 with each channel's 18 SPAR values replaced by the six-value " + VARIANT +
                          " descriptor; codebook size (1,024), windows, seeds, decoder training and gates are unchanged",
             scope="Nine-participant personalized within-session feature ablation of SPAR; same protocol as dpose_spar_simple_full_v1.",
             geometry_status="Existing row-major DPose montage confirmed by dataset owner on 2026-09-12",
             geometry_confirmation=confirmed, pilot_protocol_sha256=sha(args.pilot / "protocol.json"),
             source_sha256={name: sha(pilot.source_path(name)) for name in SOURCES},
             data_sha256={f"Sub{s}_{kind}.pkl": reference["files"][f"Sub{s}_{kind}.pkl"] for s in SUBJECTS for kind in ("emg", "kinematics")},
             reference_protocol_sha256=sha(args.reference / "protocol.json"), reference_selection_sha256=sha(args.reference / "selection.json"),
             evaluation="All 54 fits must pass a global checkpoint, cache, frozen-codebook and training-statistics gate before any test inference. Unfiltered mean Euclidean error over 20 wrist-relative landmarks, final 50 frames, in mm. Average seeds within participant before participant-level analysis.",
             resources=dict(cluster="tinygpu", partition="v100", gpus=1, cpu_threads=4, concurrent_fits=1, walltime="00:45:00"),
             statistics=dict(method="Two-sided exact paired sign-flip of the mean difference, all 512 participant sign patterns; inclusive ties. Holm correction over the two declared contrasts. Independent participants and sign-symmetric paired differences are assumed. Paper estimates come from hdemg/spar_ablation_statistics.py.",
                             contrasts=["SPAR tokens minus existing learned tokens", "SPAR tokens minus continuous SPAR"],
                             unit="Nine participants, each averaged over three decoder seeds", alpha=.05))
    p["tokenizer"]["training_scope"] = "Separate codebook per participant, fitted only to that participant's training SPAR descriptors; no pose labels. Frozen and shared by their three decoder seeds."
    p["total_split_counts"] = {split: count * len(SUBJECTS) for split, count in p["split_counts"].items()}
    p["split_count_convention"] = "split_counts are per participant; total_split_counts cover all nine."
    p["reference_predictions_sha256"] = {
        str(reference_path(s, a, k).relative_to(paths.RESULTS)): sha(reference_path(s, a, k))
        for s in SUBJECTS for a in ARMS for k in SEEDS}
    write(path, p)
    for subject in SUBJECTS:
        child = copy.deepcopy(p)
        child.update(name=f"{p['name']}_subject_{subject:02d}", subjects=[subject],
                     windows=[w for w in windows if w["subject"] == subject],
                     campaign_protocol_sha256=sha(path),
                     data_sha256={k: v for k, v in p["data_sha256"].items() if k.startswith(f"Sub{subject}_")})
        write(subject_args(args, subject).root / "protocol.json", child)
    return p


def logpower_features(args, p, f, selected):
    """Re-extract from raw EMG with the v1 resampling and normalization; return the normalized-window digest."""
    subject = p["subjects"][0]
    emg = joblib.load(args.data / f"Sub{subject}_emg.pkl")
    kin = joblib.load(args.data / f"Sub{subject}_kinematics.pkl")
    layout = grid_layout(p["geometry"]["grids"])
    taps = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
    indices = ENDS[:, None] - np.arange(319, -1, -1)[None]
    raw_hash, t0 = hashlib.sha256(), time.monotonic()
    needed = {str(int(r["task"])) for rows in selected.values() for r in rows}
    for task in sorted(emg, key=int):
        if str(int(task)) not in needed:
            continue
        assert emg[task].shape[0] == 320 and kin[task].shape == (21, 3, emg[task].shape[-1])
        signal = upfirdn(taps, emg[task].astype(np.float32), up=125, down=128, axis=-1)[..., :emg[task].shape[-1] * 125 // 128].astype(np.float32)
        for split, rows in selected.items():
            for wi, row in enumerate(rows):
                if int(row["task"]) != int(task):
                    continue
                start = row["start"]
                raw = pilot.normalize_raw(signal[:, start:start + 6000])
                raw_hash.update(raw.tobytes())
                analysis = raw[:, indices].transpose(1, 2, 0).copy()
                features = centre_log_power(analysis, layout, args.device)
                assert features.shape == (100, 320, DIMS) and np.isfinite(features).all()
                f[f"{split}/spar"][wi] = features.transpose(1, 0, 2)
                target_indices = ((start + ENDS) * 128) // 125
                points = kin[task][..., target_indices].transpose(2, 0, 1).astype(np.float32)
                f[f"{split}/targets"][wi] = (points[:, 1:] - points[:, :1]).reshape(100, 60)
        f.flush()
        print(json.dumps(dict(stage="logpower", task=int(task), elapsed_seconds=time.monotonic() - t0)), flush=True)
    return raw_hash.hexdigest(), "raw EMG, re-extracted with the v1 resampling and normalization"


def diagonal_features(args, p, f, selected):
    """D0 entries of the stored v64 SPAR descriptors, whose windows are identical to v1's."""
    subject = p["subjects"][0]
    source = V64 / f"subject_{subject:02d}"
    receipt = read(source / "cache-receipt.json")
    assert sha(source / "features.h5") == receipt["sha256"] and receipt["matches_v1_normalized_raw_sha256"]
    with h5py.File(source / "features.h5", "r") as s:
        assert s.attrs["complete"]
        for split, rows in selected.items():
            for field in ("subject", "task", "start"):
                np.testing.assert_array_equal(s[f"{split}/{field}s"][:], [int(r[field]) for r in rows])
            f[f"{split}/spar"][:] = s[f"{split}/spar"][:][..., 0::3]
            f[f"{split}/targets"][:] = s[f"{split}/targets"][:]
    return receipt["normalized_raw_sha256"], f"{source.parent.name}/{source.name}/features.h5 sha256 {receipt['sha256']}"


# Same codebook pipeline as v1; only the per-channel descriptor differs (VARIANT).
def prepare(args, p):
    subject = p["subjects"][0]
    assert p["feature"]["variant"] == VARIANT, "protocol and SPAR_ABLATION_VARIANT disagree"
    path = args.root / "features.h5"
    if (args.root / "cache-receipt.json").exists():
        receipt = json.loads((args.root / "cache-receipt.json").read_text())
        assert sha(path) == receipt["sha256"] and receipt["protocol_sha256"] == sha(args.root / "protocol.json")
        return
    if path.exists():
        raise FileExistsError("Incomplete cache: preserve it and use a fresh run directory")
    v1_receipt = read(V1 / f"subject_{subject:02d}" / "cache-receipt.json")
    mask = np.ones((320, DIMS), dtype=bool)
    selected = {split: [r for r in p["windows"] if r["split"] == split] for split in p["split_counts"]}
    t0 = time.monotonic()
    with h5py.File(path, "w") as f:
        f.attrs["complete"] = False
        f.attrs["protocol_sha256"] = sha(args.root / "protocol.json")
        f["mask"] = mask
        for split, rows in selected.items():
            g = f.create_group(split)
            g.create_dataset("spar", (len(rows), 320, 100, DIMS), dtype="f4")
            g.create_dataset("targets", (len(rows), 100, 60), dtype="f4")
            for field in ("task", "start", "subject"):
                g[field + "s"] = [int(r[field]) for r in rows]
        digest, source = (diagonal_features if VARIANT == "diagonal" else logpower_features)(args, p, f, selected)
        assert digest == v1_receipt["normalized_raw_sha256"], "normalized windows differ from the v1 campaign"
        training = f["train/spar"][:]
        assert np.isfinite(training).all()
        mean, scale = fit_standardization(training, mask)
        z = standardize(training, mean, scale, mask)
        config = p["tokenizer"]
        assert config["vocabulary"] == 1024
        book, quantizer_receipt = fit_codebook(z, vocabulary=config["vocabulary"], seed=config["seed"],
                                              max_vectors=config["max_vectors"], iterations=config["iterations"], device=args.device)
        f["codebook"], f["feature_mean"], f["feature_scale"] = book, mean, scale
        del training, z
        usage = {}
        for split in selected:
            values = standardize(f[f"{split}/spar"][:], mean, scale, mask)
            tokens = tokenize(values, book, device=args.device)
            f[f"{split}/continuous"], f[f"{split}/tokens"] = values, tokens
            counts = np.bincount(tokens.ravel(), minlength=len(book))
            error = (values - book[tokens] * mask[None, :, None, :]) ** 2
            usage[split] = dict(used_codes=int((counts > 0).sum()), masked_reconstruction_mse=float(error.sum() / (mask.sum() * len(values) * 100)),
                                token_count=int(tokens.size), histogram=counts.tolist())
        f.attrs["complete"] = True
    path.chmod(0o444)
    write(args.root / "quantizer-fit.json", quantizer_receipt)
    write(args.root / "cache-receipt.json", dict(complete=True, sha256=sha(path), protocol_sha256=sha(args.root / "protocol.json"),
                                                variant=VARIANT, feature_source=source,
                                                normalized_raw_sha256=digest, matches_v1_normalized_raw_sha256=True,
                                                data_gate="normalized windows identical to dpose_spar_simple_full_v1",
                                                seconds=time.monotonic() - t0, usage=usage))
    print(json.dumps(dict(stage="prepare_complete", variant=VARIANT, seconds=time.monotonic() - t0)), flush=True)



def validate_subject(args, p):
    """Completion/integrity gate uses training targets only, never test inference."""
    subject = p["subjects"][0]
    assert len(p["subjects"]) == 1
    digest = sha(args.root / "protocol.json")
    cache = read(args.root / "cache-receipt.json")
    assert cache["complete"] and cache["protocol_sha256"] == digest
    assert sha(args.root / "features.h5") == cache["sha256"]
    with h5py.File(args.root / "features.h5", "r") as f:
        assert f.attrs["complete"] and f.attrs["protocol_sha256"] == digest
        book, mask = torch.from_numpy(f["codebook"][:]), torch.from_numpy(f["mask"][:])
        training = torch.from_numpy(f["train/targets"][:, -50:]).reshape(-1, 60)
        mean, scale = training.mean(0), training.std(0, correction=0).clamp_min(1e-4)
        feature_mean, feature_scale = fit_standardization(f["train/spar"][:], f["mask"][:])
        np.testing.assert_array_equal(feature_mean, f["feature_mean"][:])
        np.testing.assert_array_equal(feature_scale, f["feature_scale"][:])
        for split, count in p["split_counts"].items():
            assert f[f"{split}/tokens"].shape == (count, 320, 100)
            rows = [w for w in p["windows"] if w["split"] == split]
            for field in ("subject", "task", "start"):
                np.testing.assert_array_equal(f[f"{split}/{field}s"][:], [int(w[field]) for w in rows])
    fits = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = args.root / "fits" / f"{arm}_{seed}"
            fit = read(folder / "fit.json")
            assert fit["complete"] and fit["protocol_sha256"] == digest
            assert fit["arm"] == arm and fit["seed"] == seed
            assert fit["best_sha256"] == sha(folder / "best.pt")
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            assert checkpoint["protocol_sha256"] == digest
            assert checkpoint["arm"] == arm and checkpoint["seed"] == seed
            assert torch.equal(checkpoint["model"]["decoder.codebook"], book)
            assert torch.equal(checkpoint["model"]["mask"], mask)
            assert torch.equal(checkpoint["mean"], mean) and torch.equal(checkpoint["scale"], scale)
            assert checkpoint["epoch"] == min(fit["history"], key=lambda x: x["val_error_mm"])["epoch"]
            assert fit["frozen_book_sha256"] == pilot.tensor_hash({"codebook": book})
            fits[(arm, seed)] = fit
    for seed in SEEDS:
        assert fits[("continuous", seed)]["initial_state_sha256"] == fits[("tokens", seed)]["initial_state_sha256"]
    receipt = dict(passed=True, subject=subject, fits=6, protocol_sha256=digest,
                   cache_sha256=cache["sha256"], codebook_float32_bytes_sha256=hashlib.sha256(book.numpy().tobytes()).hexdigest(),
                   exact_train_feature_and_target_statistics=True, initial_weights_matched_per_seed=True,
                   checkpoint_sha256={f"{a}_{s}": f["best_sha256"] for (a, s), f in fits.items()})
    write(args.root / "fit-audit.json", receipt)
    return receipt


def global_gate(args, p):
    receipts = []
    for subject in p["subjects"]:
        child_args = subject_args(args, subject)
        child = read(child_args.root / "protocol.json")
        assert child["campaign_protocol_sha256"] == sha(args.root / "protocol.json")
        receipts.append(validate_subject(child_args, child))
    assert len(receipts) == 9 and sum(r["fits"] for r in receipts) == 54
    receipt = dict(passed=True, fits=54, protocol_sha256=sha(args.root / "protocol.json"), subjects=receipts)
    write(args.root / "global-fit-gate.json", receipt)
    return receipt


def evaluate_subject(args, p, subject):
    child_args = subject_args(args, subject)
    child = read(child_args.root / "protocol.json")
    scores = []
    for arm in ARMS:
        data = pilot.load(child_args, "test", arm)
        for seed in SEEDS:
            folder = child_args.root / "fits" / f"{arm}_{seed}"
            saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            model = pilot.SparPose(data["book"], data["mask"])
            model.load_state_dict(saved["model"])
            model.to(args.device)
            pred = pilot.prediction(model, data, arm, saved["mean"], saved["scale"])
            target = data["targets"][:, -50:].numpy()
            old_path = reference_path(subject, arm, seed)
            assert sha(old_path) == p["reference_predictions_sha256"][str(old_path.relative_to(paths.RESULTS))]
            with np.load(old_path) as old:
                np.testing.assert_array_equal(target, old["target"])
                np.testing.assert_array_equal(old["subject"], subject)
                old_error = pilot.error_mm(old["prediction"], target)
            test_rows = [w for w in child["windows"] if w["split"] == "test"]
            np.savez_compressed(folder / "test_predictions.npz", prediction=pred, target=target,
                                subject=np.full(len(pred), subject), tasks=data["tasks"], starts=[w["start"] for w in test_rows])
            # Recompute from serialized predictions, independently of the live model.
            with np.load(folder / "test_predictions.npz") as checked:
                error = float(np.sqrt(((checked["prediction"] - checked["target"]).reshape(-1, 20, 3) ** 2).sum(-1)).mean())
            assert error == pilot.error_mm(pred, target)
            scores.append(dict(subject=subject, arm=arm, seed=seed, error_mm=error,
                               existing_learned_error_mm=old_error,
                               per_task={str(t): pilot.error_mm(pred[data["tasks"] == t], target[data["tasks"] == t]) for t in np.unique(data["tasks"])},
                               training_mean_baseline_mm=pilot.error_mm(np.broadcast_to(saved["mean"].numpy(), target.shape), target),
                               reference_targets_exact=True, predictions_sha256=sha(folder / "test_predictions.npz")))
    write(child_args.root / "results.json", dict(complete=True, subject=subject, scores=scores,
                                                protocol_sha256=sha(child_args.root / "protocol.json"),
                                                cache_sha256=read(child_args.root / "cache-receipt.json")["sha256"]))
    return scores


def summarize(scores):
    summary = {}
    for arm in ARMS:
        for feature, field in (("spar", "error_mm"), ("existing_learned", "existing_learned_error_mm")):
            matrix = np.array([[next(r[field] for r in scores if r["subject"] == subject and r["arm"] == arm and r["seed"] == seed)
                                for seed in SEEDS] for subject in SUBJECTS])
            means = matrix.mean(1)
            summary[f"{feature}_{arm}"] = dict(mean_mm=float(means.mean()), participant_sd_mm=float(means.std(ddof=1)),
                                               macro_seed_sd_mm=float(matrix.mean(0).std(ddof=1)), participant_means_mm=means.tolist(),
                                               subject_by_seed_mm=matrix.tolist())
    tests = []
    a = summary["spar_tokens"]["participant_means_mm"]
    for ref in ("existing_learned_tokens", "spar_continuous"):
        effect, pvalue, differences = exact_paired_test(a, summary[ref]["participant_means_mm"])
        tests.append(dict(alternative="spar_tokens", reference=ref, mean_difference_mm=effect,
                          participant_differences_mm=differences, p_raw=pvalue))
    for test, adjusted in zip(tests, holm_adjust([t["p_raw"] for t in tests])):
        test.update(p_holm=adjusted, annotation="***" if adjusted < .001 else "**" if adjusted < .01 else "*" if adjusted < .05 else "n.s.")
    return summary, tests


def report(args, p, scores):
    summary, comparisons = summarize(scores)
    result = dict(complete=True, subjects=list(SUBJECTS), seeds=list(SEEDS), fits=54, scores=scores, summary=summary,
                  comparisons=comparisons, statistics=p["statistics"], scope=p["scope"], geometry_status=p["geometry_status"],
                  protocol_sha256=sha(args.root / "protocol.json"), global_gate_sha256=sha(args.root / "global-fit-gate.json"))
    write(args.root / "results.json", result)
    names = {"spar_tokens": f"{VARIANT} tokens", "spar_continuous": f"Continuous {VARIANT}", "existing_learned_tokens": "Existing learned tokens", "existing_learned_continuous": "Existing learned features"}
    lines = [f"# Nine-participant SPAR ablation: {VARIANT}", "", p["scope"], "", "Mean unfiltered 3D error in mm. Seeds are averaged within each participant; SD is across nine participants.", "",
             "| Representation | Mean (mm) | Participant SD (mm) |", "|---|---:|---:|"]
    for key, name in names.items():
        r = summary[key]
        lines.append(f"| {name} | {r['mean_mm']:.3f} | {r['participant_sd_mm']:.3f} |")
    lines += ["", "| Participant | SPAR tokens | Continuous SPAR | Existing tokens | Existing features |", "|---|---:|---:|---:|---:|"]
    for i, subject in enumerate(SUBJECTS):
        lines.append("| " + str(subject) + " | " + " | ".join(f"{summary[k]['participant_means_mm'][i]:.3f}" for k in names) + " |")
    lines += ["", p["statistics"]["method"], "", "| Contrast | Difference (mm) | Raw p | Holm p | Annotation |", "|---|---:|---:|---:|---|"]
    for t in comparisons:
        lines.append(f"| SPAR tokens minus {names[t['reference']]} | {t['mean_difference_mm']:.3f} | {t['p_raw']:.6f} | {t['p_holm']:.6f} | {t['annotation']} |")
    lines += ["", "Each participant has a separate training-only SPAR codebook. Only the two SPAR arms isolate quantization. Existing learned representations differ in front end, pretraining data and input projection. Non-rejection is not equivalence.", ""]
    (args.root / "REPORT.md").write_text("\n".join(lines))
    return result


class Monitor:
    """Sample only this process's allocated GPU, never other GPUs on the node."""
    def __init__(self, root):
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        matches = {line.split(",")[1].strip() for line in out.splitlines() if line.split(",")[0].strip() == str(os.getpid())}
        if len(matches) != 1:
            raise RuntimeError(f"Could not identify exactly one allocated GPU for this process: {matches}")
        self.uuid = matches.pop()
        self.path, self.phase, self.samples = root / "resource-monitor.json", "starting", []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.start = time.monotonic()

    def snapshot(self, finished=False, successful=False):
        write(self.path, dict(gpu_uuid=self.uuid, gpu_name=torch.cuda.get_device_name(0), job_id=os.environ.get("SLURM_JOB_ID"),
                             allocated_gpus=1, cpu_threads=4, samples=self.samples, finished=finished, successful=successful,
                             elapsed_seconds=time.monotonic() - self.start, process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

    def collect(self):
        while not self.stop.wait(5):
            try:
                out = subprocess.check_output(["nvidia-smi", "-i", self.uuid, "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=4)
                util, memory, power = [float(x.strip()) for x in out.strip().split(",")]
                self.samples.append(dict(time=time.time(), phase=self.phase, gpu_utilization_pct=util, memory_mib=memory, power_w=power))
                self.snapshot()
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                self.samples.append(dict(time=time.time(), phase=self.phase, monitor_error=str(error)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=6)
        self.snapshot(finished=True, successful=kind is None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "all", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=paths.REFERENCE / "dpose_within_v1")
    parser.add_argument("--pilot", type=Path, default=paths.REFERENCE / "dpose_spar_simple_pilot_v1")
    parser.add_argument("--data", type=Path, default=paths.HDEMG)
    args = parser.parse_args()
    args.device = "cuda"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = freeze(args)
    if args.stage == "freeze":
        return
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This campaign requires exactly one visible, allocated GPU")
    with Monitor(args.root) as monitor:
        write(args.root / "runtime.json", dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                                               gpu=torch.cuda.get_device_name(0), host=os.uname().nodename,
                                               job_id=os.environ.get("SLURM_JOB_ID"), source_sha256=p["source_sha256"]))
        if args.stage == "all":
            for subject in SUBJECTS:
                child_args = subject_args(args, subject)
                child = read(child_args.root / "protocol.json")
                monitor.phase = f"subject {subject}: SPAR extraction and training-only codebook"
                prepare(child_args, child)
                for arm in ARMS:
                    for seed in SEEDS:
                        monitor.phase = f"fit subject {subject} {arm} {seed}"
                        pilot.fit(child_args, child, arm, seed)
                write(args.root / "progress.json", dict(last_completed_subject=subject, completed_fits=subject * 6, expected_fits=54, test_evaluation_started=False))
        monitor.phase = "global 54-fit integrity gate"
        global_gate(args, p)
        scores = []
        for subject in SUBJECTS:
            monitor.phase = f"gated test evaluation subject {subject}"
            scores.extend(evaluate_subject(args, p, subject))
        monitor.phase = "paired participant report"
        report(args, p, scores)
        write(args.root / "progress.json", dict(complete=True, completed_fits=54, expected_fits=54, test_evaluation_complete=True))


if __name__ == "__main__":
    main()
