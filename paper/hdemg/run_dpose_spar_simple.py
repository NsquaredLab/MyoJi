"""Frozen Subject-1 pilot: EMG -> SPAR_SIMPLE -> VQ -> spatial CNN + TCN."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy.signal import firwin, upfirdn
import torch
from torch import nn
from torch.nn import functional as F

from paper.common import paths
from myoji.codebook import fit_codebook, fit_standardization, standardize, tokenize
from myoji.decoders import SparPose
from myoji.preprocessing import normalize_raw
from myoji.spar import DEFAULT_FREQUENCY_BANDS, grid_layout, spar_simple_dataset_batched

ENDS = np.arange(2039, 6000, 40)
SEEDS = (260911, 260912, 260913)
ARMS = ("continuous", "tokens")
SOURCES = ("src/myoji/spar.py", "src/myoji/codebook.py", "hdemg/run_dpose_spar_simple.py",
           "src/myoji/_vendor/raulnet_v20.py", "src/myoji/decoders.py", "src/myoji/preprocessing.py")


def source_path(name):
    """Paper scripts are named relative to paper/, library files (src/...) relative to the repository."""
    return (paths.REPO if name.startswith("src/") else paths.ROOT) / name


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_hash(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze(args):
    args.root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.root / "protocol.json"
    if protocol_path.exists():
        p = json.loads(protocol_path.read_text())
        for source, digest in p["source_sha256"].items():
            assert sha(source_path(source)) == digest, "Frozen implementation changed: " + source
        return p
    reference = json.loads((args.reference / "protocol.json").read_text())
    selected = json.loads((args.reference / "selection.json").read_text())
    layout = grid_layout(json.loads(args.grid_map.read_text()) if args.grid_map else None)
    assert layout.channels == 320
    windows = [row for row in selected["windows"] if row["subject"] == 1]
    counts = {split: sum(row["split"] == split for row in windows) for split in ("train", "val", "test")}
    assert counts == dict(train=64, val=16, test=16)
    p = dict(name="dpose_spar_simple_pilot_v1", subjects=[1], seeds=list(SEEDS), arms=list(ARMS), windows=windows,
             split_counts=counts, geometry=layout.record(), geometry_status=args.geometry_status,
             scope="Exploratory Subject-1 same-session chronological pilot; not participant-level statistical evidence.",
             feature=dict(bands=DEFAULT_FREQUENCY_BANDS, input_hz=2000, analysis_samples=320, segment_ms=64., hop_ms=16.,
                          update_hz=50, frames_per_window=100, frame_end_samples=ENDS.tolist(),
                          compact_values_per_frame=layout.entries_per_band * 6, channel_vector_dim=18,
                          layout="Each channel carries [D0,O02,O03] per band. Missing outgoing edges are structural zeros.",
                          alignment="None; no cross-day alignment is evaluated in this same-session pilot.",
                          caveat="Real matrix-log entries are not pure log power and not phase-invariant."),
             tokenizer=dict(kind="Train-only standardized Lloyd vector quantizer", vocabulary=1024, seed=260911,
                            max_vectors=131072, iterations=40, shared_across_decoder_seeds=True,
                            training_scope="Subject 1 training descriptors only; no pose labels; frozen before decoder fitting.",
                            initialization="Seeded uniform training examples; no neural encoder or old EMG codebook weights."),
             decoder=dict(family="Proposed spatial CNN + TCN, adapted to 18-dimensional SPAR inputs", trainable_parameters=689468,
                          training=reference["training"], float32=True, tf32=False, filtering="None for primary 3D error"),
             evaluation="All six decoder fits and checkpoint hashes must pass before test scoring. Score the final 50 frames of each fixed test window; wrist-relative points 1..20, mean Euclidean error in mm (units confirmed by dataset owner). Report each seed and their mean/SD, not population significance.",
             existing_comparison="Original token and continuous V20 use a different 256-D learned representation, more decoder parameters and a different unlabeled pretraining data budget. Only the two new SPAR arms isolate quantization.",
             reference_protocol_sha256=sha(args.reference / "protocol.json"), reference_selection_sha256=sha(args.reference / "selection.json"),
             source_sha256={name: sha(source_path(name)) for name in SOURCES},
             data_sha256={name: digest for name, digest in reference["files"].items() if name.startswith("Sub1_")},
             upstream=dict(project="MyoKit", author="Amin Olamazadeh", commit="e19a2fcf8dbbcf836e586eadc7dd9dec2942d252", license="MIT"))
    write(protocol_path, p)
    return p


def prepare(args, p):
    path = args.root / "features.h5"
    if (args.root / "cache-receipt.json").exists():
        receipt = json.loads((args.root / "cache-receipt.json").read_text())
        assert sha(path) == receipt["sha256"] and receipt["protocol_sha256"] == sha(args.root / "protocol.json")
        return
    if path.exists():
        raise FileExistsError("Incomplete cache: preserve it and use a fresh run directory")
    for name, digest in p["data_sha256"].items():
        assert sha(args.data / name) == digest, name
    emg = joblib.load(args.data / "Sub1_emg.pkl")
    kin = joblib.load(args.data / "Sub1_kinematics.pkl")
    layout = grid_layout(p["geometry"]["grids"])
    mask = np.tile(layout.mask, (1, 6))
    selected = {split: [r for r in p["windows"] if r["split"] == split] for split in p["split_counts"]}
    taps = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
    indices = ENDS[:, None] - np.arange(319, -1, -1)[None]
    t0 = time.monotonic()
    raw_hash = hashlib.sha256()
    with h5py.File(path, "w") as f:
        f.attrs["complete"] = False
        f.attrs["protocol_sha256"] = sha(args.root / "protocol.json")
        f["mask"] = mask
        for split, rows in selected.items():
            g = f.create_group(split)
            g.create_dataset("spar", (len(rows), 320, 100, 18), dtype="f4")
            g.create_dataset("targets", (len(rows), 100, 60), dtype="f4")
            for field in ("task", "start", "subject"):
                g[field + "s"] = [int(r[field]) for r in rows]
        for task in sorted(emg, key=int):
            assert emg[task].shape[0] == 320 and kin[task].shape == (21, 3, emg[task].shape[-1])
            signal = upfirdn(taps, emg[task].astype(np.float32), up=125, down=128, axis=-1)[..., :emg[task].shape[-1] * 125 // 128].astype(np.float32)
            for split, rows in selected.items():
                for wi, row in enumerate(rows):
                    if int(row["task"]) != int(task):
                        continue
                    start = row["start"]
                    raw = normalize_raw(signal[:, start:start + 6000])
                    raw_hash.update(raw.tobytes())
                    analysis = raw[:, indices].transpose(1, 2, 0).copy()
                    features = spar_simple_dataset_batched(analysis, layout=layout, channelwise=True, batch_size=16, device=args.device)
                    assert features.shape == (100, 320, 18) and np.isfinite(features).all()
                    f[f"{split}/spar"][wi] = features.transpose(1, 0, 2)
                    target_indices = ((start + ENDS) * 128) // 125
                    points = kin[task][..., target_indices].transpose(2, 0, 1).astype(np.float32)
                    f[f"{split}/targets"][wi] = (points[:, 1:] - points[:, :1]).reshape(100, 60)
            f.flush()
            print(json.dumps(dict(stage="spar", task=int(task), elapsed_seconds=time.monotonic() - t0)), flush=True)
        del emg, kin, signal
        training = f["train/spar"][:]
        mean, scale = fit_standardization(training, mask)
        z = standardize(training, mean, scale, mask)
        config = p["tokenizer"]
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
                                                normalized_raw_sha256=raw_hash.hexdigest(), seconds=time.monotonic() - t0, usage=usage))
    print(json.dumps(dict(stage="prepare_complete", seconds=time.monotonic() - t0)), flush=True)


def load(args, split, arm):
    with h5py.File(args.root / "features.h5", "r") as f:
        assert f.attrs["complete"]
        features = torch.from_numpy(f[f"{split}/{arm}"][:])
        if arm == "tokens":
            features = features.long()
        return dict(features=features.to(args.device), targets=torch.from_numpy(f[f"{split}/targets"][:]),
                    book=torch.from_numpy(f["codebook"][:]), mask=torch.from_numpy(f["mask"][:]), tasks=f[f"{split}/tasks"][:])


@torch.inference_mode()
def prediction(model, data, arm, mean, scale):
    model.eval()
    values = []
    for start in range(0, len(data["features"]), 2):
        values.append((model(data["features"][start:start + 2], arm).cpu() * scale + mean)[:, -50:])
    return torch.cat(values).numpy()


def error_mm(pred, target):
    return float(np.linalg.norm((pred - target).reshape(-1, 20, 3), axis=-1).mean())


def fit(args, p, arm, seed):
    folder = args.root / "fits" / f"{arm}_{seed}"
    if (folder / "fit.json").exists():
        saved = json.loads((folder / "fit.json").read_text())
        assert saved["protocol_sha256"] == sha(args.root / "protocol.json") and saved["best_sha256"] == sha(folder / "best.pt")
        return
    folder.mkdir(parents=True, exist_ok=True)
    train, val = load(args, "train", arm), load(args, "val", arm)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SparPose(train["book"], train["mask"]).to(args.device)
    initial_hash = tensor_hash(model.state_dict())
    book_hash = tensor_hash({"codebook": model.decoder.codebook})
    assert sum(v.numel() for v in model.parameters() if v.requires_grad) == p["decoder"]["trainable_parameters"]
    targets = train["targets"][:, -50:].reshape(-1, 60)
    mean, scale = targets.mean(0), targets.std(0, correction=0).clamp_min(1e-4)
    config = p["decoder"]["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaled_targets = ((train["targets"] - mean) / scale).to(args.device)
    best, stale, history, t0 = math.inf, 0, [], time.monotonic()
    for epoch in range(config["epochs"]):
        model.train()
        torch.manual_seed(seed + epoch)
        order = np.random.default_rng(seed + epoch).permutation(len(train["features"]))
        losses = []
        for begin in range(0, len(order), config["batch_size"]):
            ix = order[begin:begin + config["batch_size"]]
            predicted = model(train["features"][ix], arm)
            loss = F.smooth_l1_loss(predicted[:, -50:], scaled_targets[ix, -50:])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = error_mm(prediction(model, val, arm, mean, scale), val["targets"][:, -50:].numpy())
        row = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_error_mm=validation, seconds=time.monotonic() - t0)
        history.append(row)
        if validation < best:
            best, stale = validation, 0
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, mean=mean, scale=scale,
                            arm=arm, seed=seed, epoch=epoch + 1, protocol_sha256=sha(args.root / "protocol.json")), folder / "best.pt")
        else:
            stale += 1
        write(folder / "progress.json", dict(arm=arm, seed=seed, history=history))
        print(json.dumps(dict(stage="fit", arm=arm, seed=seed, **row)), flush=True)
        if stale >= config["patience"]:
            break
    assert book_hash == tensor_hash({"codebook": model.decoder.codebook})
    write(folder / "fit.json", dict(complete=True, arm=arm, seed=seed, best_validation_mm=best, history=history,
                                   initial_state_sha256=initial_hash, frozen_book_sha256=book_hash,
                                   best_sha256=sha(folder / "best.pt"), protocol_sha256=sha(args.root / "protocol.json")))


def evaluate(args, p):
    # Global six-fit gate; test data loading occurs only after every fit is verified.
    fits = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = args.root / "fits" / f"{arm}_{seed}"
            fit_record = json.loads((folder / "fit.json").read_text())
            assert fit_record["complete"] and fit_record["best_sha256"] == sha(folder / "best.pt")
            assert fit_record["protocol_sha256"] == sha(args.root / "protocol.json")
            fits[(arm, seed)] = fit_record
    for seed in SEEDS:
        assert fits[("tokens", seed)]["initial_state_sha256"] == fits[("continuous", seed)]["initial_state_sha256"]
    scores = []
    for arm in ARMS:
        data = load(args, "test", arm)
        for seed in SEEDS:
            folder = args.root / "fits" / f"{arm}_{seed}"
            saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            model = SparPose(data["book"], data["mask"])
            model.load_state_dict(saved["model"])
            assert torch.equal(model.decoder.codebook, data["book"])
            model.to(args.device)
            pred = prediction(model, data, arm, saved["mean"], saved["scale"])
            target = data["targets"][:, -50:].numpy()
            np.savez_compressed(folder / "test_predictions.npz", pred=pred, target=target, tasks=data["tasks"])
            per_task = {str(task): error_mm(pred[data["tasks"] == task], target[data["tasks"] == task]) for task in np.unique(data["tasks"])}
            scores.append(dict(arm=arm, seed=seed, error_mm=error_mm(pred, target), per_task=per_task,
                               train_mean_baseline_mm=error_mm(np.broadcast_to(saved["mean"].numpy(), target.shape), target)))
    summary = {arm: dict(mean_error_mm=float(np.mean([s["error_mm"] for s in scores if s["arm"] == arm])),
                         seed_sd_mm=float(np.std([s["error_mm"] for s in scores if s["arm"] == arm], ddof=1))) for arm in ARMS}
    result = dict(complete=True, scope=p["scope"], geometry_status=p["geometry_status"], subjects=[1], decoder_seeds=list(SEEDS),
                  scores=scores, summary=summary, protocol_sha256=sha(args.root / "protocol.json"),
                  tokenizer_sha256=json.loads((args.root / "cache-receipt.json").read_text())["sha256"],
                  payload=dict(compact_continuous_float32_MB_per_min=p["feature"]["compact_values_per_frame"] * 4 * 50 * 60 / 1e6,
                               tokens_10bit_packed_MB_per_min=320 * 10 / 8 * 50 * 60 / 1e6,
                               tokens_uint16_MB_per_min=320 * 2 * 50 * 60 / 1e6),
                  statistic="Seed SD describes optimization variability for this one person. No population significance test.")
    write(args.root / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


class Monitor:
    def __init__(self, output):
        self.output, self.stop, self.samples, self.phase = output, threading.Event(), [], "starting"
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.wait(2):
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=5)
                self.samples.append(dict(time=time.time(), phase=self.phase, values=out.strip()))
                write(self.output, dict(samples=self.samples, finished=False, allocated_gpus=1, cpu_threads=4))
            except (OSError, subprocess.SubprocessError):
                pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=6)
        write(self.output, dict(samples=self.samples, finished=True, successful=kind is None, allocated_gpus=1, cpu_threads=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare", "fit", "evaluate", "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=paths.HDEMG)
    parser.add_argument("--reference", type=Path, default=paths.REFERENCE / "dpose_within_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-map", type=Path)
    parser.add_argument("--geometry-status", default="Provisional existing montage; physical channel ordering not yet confirmed by dataset owner")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = freeze(args)
    if args.stage == "freeze":
        return
    with Monitor(args.root / "resource-monitor.json") as monitor:
        if args.stage in ("prepare", "all"):
            monitor.phase = "SPAR extraction and training-only codebook"
            prepare(args, p)
        if args.stage in ("fit", "all"):
            for arm in ARMS:
                for seed in SEEDS:
                    monitor.phase = f"fit {arm} {seed}"
                    fit(args, p, arm, seed)
        if args.stage in ("evaluate", "all"):
            monitor.phase = "gated test evaluation"
            evaluate(args, p)


if __name__ == "__main__":
    main()
