"""Cache frozen features, fit causal readouts, then evaluate the locked comparison."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
from paper.common import paths
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from myoji.data import (
    EMG2POSE_ARCHIVE_URL,
    EMG2PoseWindowDataset,
    RemoteTarHDF5Cache,
    normalized_dynamic_landmarks,
    recording_stratified_indices,
)
from myoji.decoders import CachedPoseDecoder, RawPoseDecoder
from myoji.learned import (
    atomic_torch_save,
    calibration_normalize,
    load_universal_checkpoint,
)
from myoji.metrics import (
    normalized_landmark_metrics,
    normalized_landmark_pose_velocity_metrics,
)

ARMS = ("token_frame", "token_tcn", "token_transformer", "continuous_transformer")
SCHEMA = 1


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    # Undefined R2 is represented by null, never nonstandard JSON NaN.
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    temporary.write_text(json.dumps(clean(value), indent=2) + "\n")
    temporary.replace(path)


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--metadata", type=Path, required=True)
    cache.add_argument("--data-root", type=Path, default=paths.EMG2POSE)
    cache.add_argument("--tar-index", type=Path)
    cache.add_argument("--archive-url", default=EMG2POSE_ARCHIVE_URL)
    cache.add_argument("--recording-cache", type=Path)
    cache.add_argument("--cache-gib", type=float, default=20)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    cache.add_argument("--window-seconds", type=float, default=3)
    cache.add_argument("--context-frames", type=int, default=100)
    cache.add_argument("--score-frames", type=int, default=50)
    cache.add_argument("--batch-size", type=int, default=8)
    cache.add_argument(
        "--omit-raw",
        dest="include_raw",
        action="store_false",
        help="omit normalized raw inputs when no raw baseline is needed",
    )
    cache.add_argument("--train-windows", type=int, default=8192)
    cache.add_argument("--val-windows", type=int, default=2048)
    cache.add_argument("--test-windows", type=int, default=4096)
    cache.add_argument("--train-recordings", type=int, default=512)
    cache.add_argument("--val-recordings", type=int, default=256)
    cache.add_argument("--test-recordings", type=int, default=512)
    fit = commands.add_parser("fit")
    fit.add_argument("--arm", choices=(*ARMS, "raw_transformer"), required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--patience", type=int, default=8)
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=0.01)
    fit.add_argument("--velocity-weight", type=float, default=0.1)
    fit.add_argument("--width", type=int, default=256)
    fit.add_argument("--slots", type=int, default=8)
    fit.add_argument("--heads", type=int, default=4)
    fit.add_argument("--layers", type=int, default=2)
    fit.add_argument("--dropout", type=float, default=0.1)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument(
        "--input-mode",
        choices=("full", "features_only", "calibration_only"),
        default="full",
    )
    fit.add_argument("--experiment-label", default="legacy")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--runs", nargs="+", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--bootstrap-resamples", type=int, default=2000)
    for sub in (fit, evaluate):
        sub.add_argument("--cache", type=Path, required=True)
        sub.add_argument("--batch-size", type=int, default=16)
        sub.add_argument("--workers", type=int, default=2)
        sub.add_argument("--seed", type=int, default=260826)
    for sub in (cache, fit, evaluate):
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        sub.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


@torch.inference_mode()
def run_cache(args):
    if args.output.exists():
        raise FileExistsError(f"immutable cache already exists: {args.output}")
    if not 2 <= args.score_frames <= args.context_frames:
        raise ValueError("require 2 <= score frames <= context frames")
    if min(args.batch_size, args.window_seconds, args.cache_gib) <= 0:
        raise ValueError("cache controls must be positive")
    checkpoint_digest, metadata_digest = digest(args.checkpoint), digest(args.metadata)
    device = device_for(args.device)
    module, checkpoint = load_universal_checkpoint(args.checkpoint, device)
    if checkpoint.get("progress", {}).get("phase") != "complete":
        raise ValueError("cache requires a completed tokenizer")
    module.eval().requires_grad_(False)
    config = module.tokenizer.config
    if config.codebook_size > 65536:
        raise ValueError("uint16 cache cannot represent this codebook")
    samples = round(args.window_seconds * config.sample_rate_hz)
    remote = None
    root = args.data_root
    if args.tar_index:
        root = (
            args.recording_cache
            or Path(os.environ.get("TMPDIR", "/tmp")) / "emg2pose-cache"
        )
        remote = RemoteTarHDF5Cache(
            args.archive_url,
            args.tar_index,
            root,
            max_cache_bytes=round(args.cache_gib * 1024**3),
        )
    manifest = {
        "schema": SCHEMA,
        "checkpoint_sha256": checkpoint_digest,
        "metadata_sha256": metadata_digest,
        "checkpoint": str(args.checkpoint),
        "token_rate_hz": config.token_rate_hz,
        "window_samples": samples,
        "context_frames": args.context_frames,
        "score_frames": args.score_frames,
        "calibration_samples": module.calibration_samples,
        "config": checkpoint["config"],
        "raw_cached": args.include_raw,
        "splits": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(
            f"incomplete cache exists; inspect it before retrying: {temporary}"
        )
    with h5py.File(temporary, "w") as out:
        out.create_dataset(
            "codebook", data=module.tokenizer.quantizer.codebook.cpu().numpy()
        )
        out.create_dataset(
            "descriptors", data=module.tokenizer.sensor_positions("wrist").cpu().numpy()
        )
        for split in args.splits:
            max_windows = getattr(args, split + "_windows")
            max_recordings = getattr(args, split + "_recordings")
            if min(max_windows, max_recordings) < 1:
                raise ValueError("window and recording caps must be positive")
            dataset = EMG2PoseWindowDataset(
                root,
                args.metadata,
                split=split,
                window_samples=samples,
                stride_samples=samples,
                jitter=False,
                remote_cache=remote,
            )
            indices = recording_stratified_indices(
                dataset, max_windows, max_recordings, seed=0, random_recordings=False
            )
            group = out.create_group(split)
            count = len(indices)
            seen, identities = 0, []
            for batch in DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            ):
                normalized, centre, scale = calibration_normalize(
                    batch["emg"].to(device), module.calibration_samples
                )
                encoded = module.tokenizer(
                    normalized, sensor="wrist", quantize=True, include_spatial=False
                )
                n = encoded.tokens.shape[-1]
                if n < args.context_frames:
                    raise ValueError("window too short for requested context")
                selection = torch.arange(n - args.context_frames, n)
                if not module.scored_token_mask(n, device)[selection.to(device)].all():
                    raise ValueError("cached context overlaps calibration preamble")
                sample_indices = (
                    config.receptive_samples - 1 + selection * config.hop_samples
                )
                angles = (
                    batch["joint_angles"]
                    .index_select(-1, sample_indices)
                    .transpose(1, 2)
                )
                targets = normalized_dynamic_landmarks(angles).flatten(-2)
                valid = batch["valid_pose"].index_select(
                    -1, sample_indices
                ) & torch.isfinite(targets).all(-1)
                values = {
                    "tokens": encoded.tokens[..., -args.context_frames :]
                    .cpu()
                    .numpy()
                    .astype(np.uint16),
                    "continuous": encoded.continuous[..., -args.context_frames :, :]
                    .cpu()
                    .half()
                    .numpy(),
                    "targets": torch.nan_to_num(targets).numpy(),
                    "valid": valid.numpy(),
                    "centre": centre.cpu().numpy(),
                    "scale": scale.cpu().numpy(),
                    "start_sample": batch["start_sample"].numpy(),
                    "held_out_user": batch["held_out_user"].numpy(),
                    "held_out_stage": batch["held_out_stage"].numpy(),
                }
                if args.include_raw:
                    values["raw"] = normalized.cpu().numpy().astype(np.float32)
                if not all(
                    np.isfinite(values[k]).all()
                    for k in ("continuous", "centre", "scale")
                ):
                    raise ValueError("nonfinite cached features or calibration")
                batch_count = len(targets)
                for key, value in values.items():
                    if key not in group:
                        group.create_dataset(
                            key,
                            (count, *value.shape[1:]),
                            dtype=value.dtype,
                            chunks=(1, *value.shape[1:]),
                        )
                    group[key][seen : seen + batch_count] = value
                for key in ("user", "filename", "session", "stage", "side"):
                    if key not in group:
                        group.create_dataset(key, (count,), dtype=h5py.string_dtype())
                    group[key][seen : seen + batch_count] = batch[key]
                identities.extend(
                    zip(batch["filename"], batch["start_sample"].tolist())
                )
                seen += batch_count
                if seen % (args.batch_size * 20) == 0 or seen == count:
                    print(
                        json.dumps(
                            {
                                "phase": "cache",
                                "split": split,
                                "windows": seen,
                                "total": count,
                            }
                        ),
                        flush=True,
                    )
            # Approximate remote lengths can clamp multiple indices to the same tail.
            # Preserve the first actual window only; duplicate samples bias fitting.
            unique = list(dict.fromkeys(identities))
            first = {}
            for index, identity in enumerate(identities):
                first.setdefault(identity, index)
            group.create_dataset(
                "unique_indices",
                data=np.array([first[key] for key in unique], dtype=np.int64),
            )
            manifest["splits"][split] = {
                "windows": len(unique),
                "windows_extracted": count,
                "recordings": len({key[0] for key in unique}),
                "users": len(set(group["user"].asstr()[:])),
                "selected_dataset_indices": indices,
                "window_identity_sha256": hashlib.sha256(
                    json.dumps(unique).encode()
                ).hexdigest(),
            }
        if (
            digest(args.checkpoint) != checkpoint_digest
            or digest(args.metadata) != metadata_digest
        ):
            raise RuntimeError("checkpoint or metadata changed during extraction")
        manifest["cache_id"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        out.attrs["manifest"] = json.dumps(manifest)
        out.attrs["complete"] = True
    temporary.replace(args.output)
    write_json(args.output.with_suffix(".json"), manifest)
    return manifest


class FeatureCache(Dataset):
    def __init__(self, path, split, continuous=False, raw=False):
        self.path, self.split, self.continuous = str(path), split, continuous or raw
        self.raw = raw
        self.handle = None
        with h5py.File(path, "r") as cache:
            if not cache.attrs.get("complete", False):
                raise ValueError("incomplete feature cache")
            self.manifest = json.loads(cache.attrs["manifest"])
            if self.manifest["schema"] != SCHEMA:
                raise ValueError("unsupported cache schema")
            if raw and "raw" not in cache[split]:
                raise ValueError("raw baseline requires cached normalized raw EMG")
            self.indices = cache[split]["unique_indices"][:]
            self.codebook = torch.from_numpy(cache["codebook"][:])
            self.descriptors = torch.from_numpy(cache["descriptors"][:])
            self.embedding_dim = (
                cache[split]["continuous"].shape[-1]
                if continuous and not raw
                else self.codebook.shape[-1]
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        group, index = self.handle[self.split], int(self.indices[index])
        key = "raw" if self.raw else "continuous" if self.continuous else "tokens"
        return {
            "features": torch.from_numpy(
                group[key][index].astype(np.float32 if self.continuous else np.int64)
            ),
            **{
                k: torch.from_numpy(np.asarray(group[k][index]))
                for k in ("targets", "valid", "centre", "scale")
            },
        }

    def metadata(self):
        with h5py.File(self.path, "r") as cache:
            g = cache[self.split]
            return {
                k: (
                    g[k].asstr()[:][self.indices]
                    if k == "user"
                    else g[k][:][self.indices]
                )
                for k in ("user", "held_out_user", "held_out_stage")
            }

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def train_statistics(dataset):
    with h5py.File(dataset.path, "r") as cache:
        group = cache[dataset.split]
        targets = torch.from_numpy(group["targets"][:][dataset.indices])
        valid = torch.from_numpy(group["valid"][:][dataset.indices])
    frames = dataset.manifest["score_frames"]
    targets, valid = targets[:, -frames:], valid[:, -frames:]
    values = targets[valid]
    if len(values) < 2:
        raise ValueError("training split needs valid scored targets")
    velocity = (targets[:, 1:] - targets[:, :-1]) * dataset.manifest["token_rate_hz"]
    velocity_values = velocity[valid[:, 1:] & valid[:, :-1]]
    if len(velocity_values) < 2:
        raise ValueError("training split needs valid consecutive targets")
    return {
        "mean": values.mean(0),
        "scale": values.std(0, unbiased=False).clamp_min(1e-4),
        "velocity_scale": velocity_values.std(0, unbiased=False).clamp_min(1e-4),
    }


def pose_loss(prediction, target, valid, stats, rate, score_frames, velocity_weight):
    prediction, target, valid = (
        prediction[:, -score_frames:],
        target[:, -score_frames:],
        valid[:, -score_frames:],
    )
    if not valid.any():
        return prediction.sum() * 0
    pose = ((prediction - target) / stats["scale"])[valid].square().mean()
    adjacent = valid[:, 1:] & valid[:, :-1]
    if not adjacent.any() or velocity_weight == 0:
        return pose
    residual = (prediction[:, 1:] - prediction[:, :-1]) - (
        target[:, 1:] - target[:, :-1]
    )
    velocity = (residual * rate / stats["velocity_scale"])[adjacent].square().mean()
    return pose + velocity_weight * velocity


def loader_for(dataset, args, shuffle=False, generator=None):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
    )


def forward_batch(model, batch, dataset, codebook, descriptors, device, stats):
    features = batch["features"].to(device, non_blocking=True)
    if not dataset.continuous:
        features = F.embedding(features.long(), codebook)
    else:
        features = features.float()
    output = model(
        features, descriptors, batch["centre"].to(device), batch["scale"].to(device)
    )
    return output * stats["scale"] + stats["mean"]


@torch.inference_mode()
def predict(model, loader, dataset, device, stats):
    model.eval()
    codebook, descriptors = dataset.codebook.to(device), dataset.descriptors.to(device)
    predictions, targets, masks = [], [], []
    frames = dataset.manifest["score_frames"]
    for batch in loader:
        p = forward_batch(model, batch, dataset, codebook, descriptors, device, stats)
        if not torch.isfinite(p).all():
            raise ValueError("nonfinite pose prediction")
        predictions.append(p[:, -frames:].cpu())
        targets.append(batch["targets"][:, -frames:])
        masks.append(batch["valid"][:, -frames:])
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def metrics_for(prediction, target, valid):
    if not valid.any():
        return {}
    return {**normalized_landmark_metrics(prediction, target, valid)}


def summarize_predictions(prediction, target, valid, metadata, rate):
    def metrics(p, t, v):
        result = metrics_for(p, t, v)
        if result and (v[:, 1:] & v[:, :-1]).any():
            result.update(normalized_landmark_pose_velocity_metrics(p, t, v, rate))
        return result

    users = np.asarray(metadata["user"])
    per_user = {}
    for user in sorted(set(users)):
        index = np.flatnonzero(users == user)
        row = metrics(prediction[index], target[index], valid[index])
        if row:
            per_user[str(user)] = row
    if not per_user:
        raise ValueError("evaluation split has no valid users")
    mean_mae = float(
        np.mean([row["landmark_mae_palm_lengths"] for row in per_user.values()])
    )
    subgroups = {}
    held_user, held_stage = metadata["held_out_user"], metadata["held_out_stage"]
    for name, mask in (
        ("unseen_user_only", held_user & ~held_stage),
        ("unseen_stage_only", held_stage & ~held_user),
        ("both", held_user & held_stage),
    ):
        index = np.flatnonzero(mask)
        if len(index):
            subgroups[name] = metrics(prediction[index], target[index], valid[index])
    return {
        "pooled": metrics(prediction, target, valid),
        "macro_user_mae": mean_mae,
        "per_user": per_user,
        "subgroups": subgroups,
        "windows": len(target),
    }


def model_kwargs(dataset, args):
    kwargs = {
        "embedding_dim": dataset.embedding_dim,
        "geometry_dim": dataset.descriptors.shape[-1],
        "channels": dataset.descriptors.shape[0],
        "kind": args.arm.split("_", 1)[1],
        "width": args.width,
        "slots": args.slots,
        "heads": args.heads,
        "layers": args.layers,
        "context_frames": dataset.manifest["context_frames"],
        "dropout": args.dropout,
        "input_mode": getattr(args, "input_mode", "full"),
    }
    if args.arm == "raw_transformer":
        kwargs["raw_config"] = dataset.manifest["config"]
    return kwargs


def build_pose_model(kwargs):
    return (
        RawPoseDecoder(**kwargs)
        if "raw_config" in kwargs
        else CachedPoseDecoder(**kwargs)
    )


def run_fit(args):
    if (
        min(args.epochs, args.patience, args.batch_size, args.learning_rate) <= 0
        or min(args.weight_decay, args.velocity_weight, args.workers) < 0
    ):
        raise ValueError("invalid fitting controls")
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    continuous = args.arm.startswith("continuous")
    train, val = [
        FeatureCache(args.cache, split, continuous, raw=args.arm == "raw_transformer")
        for split in ("train", "val")
    ]
    stats_cpu = train_statistics(train)
    stats = {k: v.to(device) for k, v in stats_cpu.items()}
    kwargs = model_kwargs(train, args)
    model = build_pose_model(kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader_for(train, args, True, generator)
    val_loader = loader_for(
        val, args, generator=torch.Generator().manual_seed(args.seed)
    )
    descriptors, codebook = train.descriptors.to(device), train.codebook.to(device)
    signature = {
        k: v
        for k, v in vars(args).items()
        if k not in ("cache", "output_dir", "resume", "device", "threads", "workers")
    }
    signature["cache_id"] = train.manifest["cache_id"]
    signature["implementation_sha256"] = {
        "runner": digest(__file__),
        "decoder": digest(inspect.getfile(CachedPoseDecoder)),
    }
    runtime = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output / "last.pt", output / "best.pt"
    begin, best_mae, stale, history = 0, math.inf, 0, []
    if last_path.exists():
        if not args.resume:
            raise FileExistsError("existing run requires --resume")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("resume configuration/cache differs")
        if (output / "fit_summary.json").exists():
            return json.loads((output / "fit_summary.json").read_text())
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        begin, best_mae, stale, history = (
            state["epoch"],
            state["best_mae"],
            state["stale"],
            state["history"],
        )
        torch.set_rng_state(state["rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif any(output.iterdir()):
        raise FileExistsError(
            "output contains artifacts without a resumable checkpoint"
        )
    start = time.monotonic()
    rate, score_frames = train.manifest["token_rate_hz"], train.manifest["score_frames"]
    for epoch in range(begin, args.epochs):
        if stale >= args.patience:
            break
        # Epoch-local RNG makes restart behavior independent of worker startup.
        torch.manual_seed(args.seed + epoch)
        generator.manual_seed(args.seed + epoch)
        model.train()
        total, updates = 0.0, 0
        epoch_start = time.monotonic()
        for batch in train_loader:
            target, valid = batch["targets"].to(device), batch["valid"].to(device)
            if not valid[:, -score_frames:].any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_batch(
                model, batch, train, codebook, descriptors, device, stats
            )
            loss = pose_loss(
                prediction,
                target,
                valid,
                stats,
                rate,
                score_frames,
                args.velocity_weight,
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            updates += 1
        if updates == 0:
            raise ValueError("epoch had no valid training batches")
        predictions = predict(model, val_loader, val, device, stats)
        validation = summarize_predictions(*predictions, val.metadata(), rate)
        val_mae = validation["macro_user_mae"]
        row = {
            "epoch": epoch + 1,
            "loss": total / updates,
            "validation_macro_mae": val_mae,
            "validation_velocity_r2": validation["pooled"].get("landmark_velocity_r2"),
            "seconds": time.monotonic() - epoch_start,
            "updates": updates,
        }
        history.append(row)
        print(json.dumps({"arm": args.arm, "seed": args.seed, **row}), flush=True)
        payload = {
            "signature": signature,
            "runtime": runtime,
            "model_kwargs": kwargs,
            "stats": stats_cpu,
            "model": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "epoch": epoch + 1,
        }
        if val_mae < best_mae:
            best_mae, stale = val_mae, 0
            atomic_torch_save({**payload, "validation": validation}, best_path)
        else:
            stale += 1
        atomic_torch_save(
            {
                **payload,
                "optimizer": optimizer.state_dict(),
                "best_mae": best_mae,
                "stale": stale,
                "history": history,
                "rng": torch.get_rng_state(),
                "loader_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
            },
            last_path,
        )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {
        "arm": args.arm,
        "runtime": runtime,
        "seed": args.seed,
        "cache_id": train.manifest["cache_id"],
        "signature": signature,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": best["epoch"],
        "validation": best["validation"],
        "history": history,
        "invocation_seconds": time.monotonic() - start,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else 0,
    }
    write_json(output / "fit_summary.json", result)
    train.close()
    val.close()
    return result


def bootstrap(values, seed, resamples):
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(1)
    return {
        "point": float(values.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "users": len(values),
    }


def paired_comparisons(results, seed, resamples):
    grouped = {}
    for row in results:
        if row["seed"] in grouped.setdefault(row["arm"], {}):
            raise ValueError("duplicate arm/seed")
        grouped[row["arm"]][row["seed"]] = row["metrics"]["per_user"]
    comparisons = {}
    candidate = grouped.get("token_transformer", {})
    for control in (
        "token_frame",
        "token_tcn",
        "continuous_transformer",
        "raw_transformer",
    ):
        if control not in grouped or not candidate:
            continue
        if set(candidate) != set(grouped[control]):
            raise ValueError("paired comparisons require identical seeds")
        for metric in ("landmark_mae_palm_lengths", "landmark_velocity_r2"):
            users = set.intersection(
                *[
                    set(rows)
                    for arm in (candidate, grouped[control])
                    for rows in arm.values()
                ]
            )
            deltas = []
            for user in sorted(users):
                pairs = [
                    (
                        candidate[s][user].get(metric),
                        grouped[control][s][user].get(metric),
                    )
                    for s in candidate
                ]
                if all(
                    a is not None
                    and b is not None
                    and math.isfinite(a)
                    and math.isfinite(b)
                    for a, b in pairs
                ):
                    deltas.append(float(np.mean([a - b for a, b in pairs])))
            if deltas:
                comparisons[f"token_transformer_minus_{control}/{metric}"] = bootstrap(
                    deltas, seed, resamples
                )
    return comparisons


def run_evaluate(args):
    device = device_for(args.device)
    results = []
    baseline = None
    for directory in args.runs:
        state = torch.load(
            directory / "best.pt", map_location="cpu", weights_only=False
        )
        arm = state["signature"]["arm"]
        dataset = FeatureCache(
            args.cache,
            args.split,
            arm.startswith("continuous"),
            raw=arm == "raw_transformer",
        )
        if state["signature"]["cache_id"] != dataset.manifest["cache_id"]:
            raise ValueError("evaluation cache differs from the training cache")
        model = build_pose_model(state["model_kwargs"]).to(device)
        model.load_state_dict(state["model"])
        stats = {k: v.to(device) for k, v in state["stats"].items()}
        predictions, targets, valid = predict(
            model, loader_for(dataset, args), dataset, device, stats
        )
        metadata, rate = dataset.metadata(), dataset.manifest["token_rate_hz"]
        summary = summarize_predictions(predictions, targets, valid, metadata, rate)
        if baseline is None:
            constant = state["stats"]["mean"].expand_as(targets)
            baseline = summarize_predictions(constant, targets, valid, metadata, rate)
        results.append(
            {
                "arm": arm,
                "seed": state["signature"]["seed"],
                "best_epoch": state["epoch"],
                "checkpoint_sha256": digest(directory / "best.pt"),
                "metrics": summary,
            }
        )
        dataset.close()
    summary = {
        "split": args.split,
        "cache_id": dataset.manifest["cache_id"],
        "results": results,
        "train_mean_baseline": baseline,
        "paired": paired_comparisons(results, args.seed, args.bootstrap_resamples),
    }
    write_json(args.output, summary)
    print(
        json.dumps({"output": str(args.output), "paired": summary["paired"]}),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    {"cache": run_cache, "fit": run_fit, "evaluate": run_evaluate}[args.command](args)
