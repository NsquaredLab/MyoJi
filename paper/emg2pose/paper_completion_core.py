"""Shared data, compression and analysis for the fixed Meta completion study."""

import hashlib
import json
import struct
from pathlib import Path

from paper.emg2pose import compare_temporal_pose as pose
import h5py
import numpy as np
import torch

HEAD_SEEDS = (260826, 260827, 260828)
ENCODER_SEEDS = (260905, 260906, 260907)
SCOPES = ("official", "session")
PCA_LEVELS = {
    "pca4q8": (4, 4, 4, 4),
    "pca4q10": (8, 8, 4, 4),
    "pca4q16": (16, 16, 16, 16),
}
CONTROL_LABELS = (*PCA_LEVELS, "fsq1000", "features67", "kmeans256", "kmeans1024")


def configure_encoder_replay(device):
    """Match the arithmetic that produced all three canonical token caches."""
    if str(device) != "cuda" or not torch.cuda.is_available():
        raise ValueError("canonical feature extraction requires CUDA on RTX 3080")
    name = torch.cuda.get_device_name()
    if name != "NVIDIA GeForce RTX 3080" or str(torch.__version__) != "2.6.0+cu124":
        raise ValueError("canonical cache replay requires RTX 3080 and torch2.6.0+cu124")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return {"gpu": name, "torch": str(torch.__version__), "cudnn_allow_tf32": True,
            "matmul_allow_tf32": False, "stored_feature_dtype": "float32"}


def configure_runtime_precision():
    """Use float32 convolution/matmul for same-device causal replay and timing."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {"cudnn_allow_tf32": False, "matmul_allow_tf32": False,
            "dtype": "float32", "parity_reference": "same-device full window"}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes(code):
    return {
        str(p.relative_to(code)): pose.digest(p)
        for p in sorted(Path(code).rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and (p.suffix in (".py", ".sbatch", ".yaml", ".md") or p.name == "LICENSE")
    }


def verify(root, require_smoke=True):
    root = Path(root)
    p = json.loads((root / "protocol.json").read_text())
    if source_hashes(root / "code") != p["source_sha256"]:
        raise ValueError("completion-study source changed")
    for name, expected in p["parent_protocols"].items():
        if pose.digest(name) != expected:
            raise ValueError("parent protocol changed: " + name)
    if require_smoke:
        receipt = json.loads((root / "smoke.json").read_text())
        if not receipt["complete"] or receipt["protocol_sha256"] != pose.digest(root / "protocol.json"):
            raise ValueError("matching preflight required")
    return p


def read_indices(dataset, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("selection contains duplicate rows")
    order = np.argsort(indices)
    if np.array_equal(order, np.arange(len(indices))):
        if len(indices) and np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))):
            return dataset[int(indices[0]):int(indices[-1])+1]
        return dataset[indices]
    values = dataset[indices[order]]
    return values[np.argsort(order)]


class ResidentCache(pose.FeatureCache):
    """Exact selected-row caching, including noncontiguous session views."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        key = "raw" if self.raw else "continuous" if self.continuous else "tokens"
        self.feature_key = key
        with h5py.File(self.path, "r") as h:
            self.arrays = {
                k: read_indices(h[self.split][k], self.indices)
                for k in (key, "targets", "valid", "centre", "scale")
            }
        for i in np.unique(np.linspace(0, len(self) - 1, min(8, len(self)), dtype=int)):
            disk, memory = super().__getitem__(i), self[i]
            if any(not torch.equal(disk[k], memory[k]) for k in disk):
                raise ValueError("resident inputs differ from original disk reads")
        self.close()
        print(json.dumps({
            "resident_split": self.split,
            "windows": len(self),
            "bytes": sum(v.nbytes for v in self.arrays.values()),
            "disk_replay_passed": True,
        }), flush=True)

    def __getitem__(self, index):
        return {
            "features": torch.from_numpy(self.arrays[self.feature_key][index].astype(
                np.float32 if self.continuous else np.int64, copy=False
            )),
            **{k: torch.from_numpy(np.asarray(self.arrays[k][index]))
               for k in ("targets", "valid", "centre", "scale")},
        }


def cache_identities(path, split):
    with h5py.File(path, "r") as h:
        g = h[split]
        ix = g["unique_indices"][:]
        keys = ("filename", "start_sample", "user", "session", "stage", "side")
        values = {k: read_indices(g[k], ix) for k in keys if k in g}
        rows = [
            [v[i].decode() if isinstance(v[i], bytes) else
             v[i].item() if isinstance(v[i], np.generic) else v[i]
             for v in values.values()]
            for i in range(len(ix))
        ]
    return json_hash({"keys": list(values), "rows": rows})


def compact_metadata(source, target):
    """Copy only selected metadata; never attach a misindexed raw-data link."""
    for split in ("train", "val", "test"):
        origin, group = source[split], target.create_group(split)
        ix = origin["unique_indices"][:]
        group["unique_indices"] = np.arange(len(ix), dtype=np.int64)
        group["parent_indices"] = ix
        for key, dataset in origin.items():
            if key not in ("raw", "continuous", "tokens", "unique_indices", "parent_indices"):
                group.create_dataset(key, data=read_indices(dataset, ix), dtype=dataset.dtype)
    target["descriptors"] = source["descriptors"][:]


def finish_cache(out, parent_manifest, **metadata):
    manifest = {**parent_manifest, **metadata}
    manifest.pop("cache_id", None)
    manifest["cache_id"] = json_hash(manifest)
    out.attrs["manifest"] = json.dumps(manifest)
    out.attrs["complete"] = True
    return manifest


def state_hash(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pack_ids(ids, bits):
    """Actual fixed-width packet: magic, width, count, then little-endian bits."""
    values = np.asarray(ids)
    if values.dtype.kind not in "iu" or not 1 <= bits <= 16:
        raise ValueError("integer IDs and a 1..16 bit width are required")
    if np.any(values < 0) or np.any(values >= 2**bits):
        raise ValueError("ID exceeds declared bit width")
    flat = values.reshape(-1).astype(np.uint32)
    binary = ((flat[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    return struct.pack("<4sBI", b"EMGC", bits, len(flat)) + np.packbits(
        binary.reshape(-1), bitorder="little"
    ).tobytes()


def unpack_ids(packet):
    if len(packet) < 9:
        raise ValueError("incomplete packet")
    magic, bits, count = struct.unpack("<4sBI", packet[:9])
    if magic != b"EMGC" or not 1 <= bits <= 16 or len(packet) != 9 + (count * bits + 7) // 8:
        raise ValueError("invalid packet header or length")
    digits = np.unpackbits(np.frombuffer(packet[9:], dtype=np.uint8), bitorder="little")
    digits = digits[:count * bits].reshape(count, bits).astype(np.uint32)
    return (digits << np.arange(bits, dtype=np.uint32)).sum(1).astype(np.uint16)


def scalar_encode(values, lo, hi, levels):
    levels = torch.as_tensor(levels, device=values.device)
    digits = ((values - lo) / (hi - lo).clamp_min(1e-6) * (levels - 1)).round()
    digits = torch.minimum(digits.clamp_min(0), levels - 1).long()
    multipliers = torch.cat((levels.new_ones(1), levels[:-1].cumprod(0)))
    return (digits * multipliers).sum(-1)


def scalar_codebook(lo, hi, levels):
    levels = torch.as_tensor(levels, device=lo.device)
    multipliers = torch.cat((levels.new_ones(1), levels[:-1].cumprod(0)))
    ids = torch.arange(int(levels.prod()), device=lo.device)
    digits = (ids[:, None] // multipliers) % levels
    return lo + digits * (hi - lo).clamp_min(1e-6) / (levels - 1)


def nearest_centres(values, centres, chunk=8192):
    values = values.reshape(-1, values.shape[-1])
    ids = []
    for x in values.split(chunk):
        distances = x.square().sum(1, keepdim=True) + centres.square().sum(1)[None] - 2 * x @ centres.T
        ids.append(distances.argmin(1))
    return torch.cat(ids)


def lloyd_kmeans(values, count, seed, iterations=40):
    if len(values) < count:
        raise ValueError("fewer fitting observations than clusters")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    centres = values[torch.randperm(len(values), generator=generator, device=values.device)[:count]].clone()
    history = []
    for step in range(iterations):
        ids = nearest_centres(values, centres)
        sums = torch.zeros_like(centres).index_add_(0, ids, values)
        counts = torch.bincount(ids, minlength=count)
        present = counts > 0
        centres[present] = sums[present] / counts[present, None]
        history.append(float((values - centres[ids]).square().mean()))
    return centres, history


def features67(raw, frames, config):
    from paper.emg2pose.prepare_reviewer_features import causal_features
    return torch.cat([causal_features(raw, frames, config, name)
                      for name in ("rms3", "spectral64")], dim=-1)


def movement_reference(target, valid, score_frames=50, rate=50):
    target, valid = np.asarray(target)[:, -score_frames:], np.asarray(valid)[:, -score_frames:]
    speed = np.linalg.norm(np.diff(target, axis=1).reshape(len(target), -1, target.shape[-1] // 3, 3), axis=-1).mean(-1) * rate
    adjacent = valid[:, 1:] & valid[:, :-1]
    per_window = np.array([x[m].mean() if m.any() else np.nan for x, m in zip(speed, adjacent)])
    useful = per_window[np.isfinite(per_window)]
    if len(useful) < 2:
        raise ValueError("insufficient training movement support")
    return {"speed_tertiles": np.quantile(useful, [1/3, 2/3]).tolist(),
            "lag_speed_floor": float(np.quantile(useful, .25)), "rate_hz": rate}


def lag_diagnostic(prediction, target, max_lag=10):
    """Fixed interior support; positive lag means prediction follows target."""
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 2 or len(p) <= 2 * max_lag + 2:
        raise ValueError("lag diagnostic needs one sufficiently long paired window")
    interior = np.arange(max_lag, len(p) - max_lag)
    y = t[interior] - t[interior].mean(0)
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        x = p[interior + lag] - p[interior + lag].mean(0)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        correlations.append(float((x * y).sum() / denominator) if denominator > 1e-12 else None)
    candidates = [i for i, x in enumerate(correlations) if x is not None]
    if not candidates:
        return {"lag_frames": None, "peak_correlation": None, "at_boundary": None}
    best = max(candidates, key=lambda i: (correlations[i], -abs(i - max_lag), -i))
    return {"lag_frames": best - max_lag, "peak_correlation": correlations[best],
            "at_boundary": best in (0, 2 * max_lag)}


def motion_rows(prediction, target, valid, users, stages, reference, score_frames=50):
    """Window-level diagnostics, retaining negative R² and uninformative lag flags."""
    p, t, v = np.asarray(prediction)[:, -score_frames:], np.asarray(target)[:, -score_frames:], np.asarray(valid)[:, -score_frames:]
    rate, rows = reference["rate_hz"], []
    for index, (x, y, mask) in enumerate(zip(p, t, v)):
        if not mask.any():
            continue
        adjacent = mask[1:] & mask[:-1]
        delta = x - y
        velocity_p, velocity_t = rate * np.diff(x, axis=0), rate * np.diff(y, axis=0)
        speed = float(np.linalg.norm(velocity_t[adjacent].reshape(-1, y.shape[-1] // 3, 3), axis=-1).mean()) if adjacent.any() else None
        error = float(np.linalg.norm(delta[mask].reshape(-1, y.shape[-1] // 3, 3), axis=-1).mean())
        variance = float(np.square(y[mask] - y[mask].mean(0)).sum())
        row = {"window": index, "user": str(users[index]), "stage": str(stages[index]),
               "valid_frames": int(mask.sum()), "pose_error": error,
               "target_speed": speed,
               "motion_stratum": int(np.searchsorted(reference["speed_tertiles"], speed)) if speed is not None else None,
               "pose_r2": 1 - float(np.square(delta[mask]).sum()) / variance if variance > 1e-12 else None,
               "target_coordinate_variance": variance / int(mask.sum()) / y.shape[-1],
               "velocity_mae": float(np.abs(velocity_p[adjacent] - velocity_t[adjacent]).mean()) if adjacent.any() else None,
               "amplitude_ratio": float(np.linalg.norm(x[mask] - x[mask].mean(0)) / np.sqrt(variance)) if variance > 1e-12 else None}
        row["lag_eligible"] = bool(mask.all() and speed is not None and speed > max(1e-8, reference["lag_speed_floor"]))
        row.update(lag_diagnostic(x, y) if row["lag_eligible"] else
                   {"lag_frames": None, "peak_correlation": None, "at_boundary": None})
        rows.append(row)
    return rows


def paired_summary(rows, reference, comparator, metric="landmark_mae_palm_lengths"):
    a = [r["per_user"] for r in rows if r["label"] == reference]
    b = [r["per_user"] for r in rows if r["label"] == comparator]
    if len(a) != 3 or len(b) != 3:
        raise ValueError("three head seeds required in each paired condition")
    users = sorted(a[0])
    if any(set(v) != set(users) for v in a + b):
        raise ValueError("participants differ across paired conditions")
    x = np.array([np.mean([v[u][metric] for v in a]) for u in users])
    y = np.array([np.mean([v[u][metric] for v in b]) for u in users])
    draws = np.random.default_rng(260908).integers(0, len(users), (10000, len(users)))
    return {"users": users, "mean_difference": float((x - y).mean()),
            "difference_ci95": np.quantile((x-y)[draws].mean(1), [.025, .975]).tolist(),
            "ratio": float(x.mean()/y.mean()),
            "ratio_ci95": np.quantile(x[draws].mean(1)/y[draws].mean(1), [.025, .975]).tolist(),
            "per_user_difference": dict(zip(users, (x-y).tolist()))}
