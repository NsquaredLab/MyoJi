"""Prepare train-fitted compact controls on an immutable pose cache.

FSQ here is a post-hoc latent-compression baseline, not an end-to-end replacement
of spectral VQ pretraining. No pose labels or test features fit its bottleneck.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from paper.emg2pose.compare_temporal_pose import digest, write_json
from torch import nn
from torch.nn import functional as F


class LatentFSQ(nn.Module):
    def __init__(self, dimension=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, 128), nn.GELU(), nn.Linear(128, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 128), nn.GELU(), nn.Linear(128, dimension)
        )
        self.register_buffer("levels", torch.tensor([8, 5, 5, 5]))

    def quantize(self, x):
        bounded = (self.encoder(x).tanh() + 1) * (self.levels - 1) / 2
        rounded = bounded + (bounded.round() - bounded).detach()
        q = 2 * rounded / (self.levels - 1) - 1
        indices = (
            rounded.detach().long() * torch.tensor([1, 8, 40, 200], device=x.device)
        ).sum(-1)
        return q, indices

    def forward(self, x):
        return self.decoder(self.quantize(x)[0])


def sample_latents(cache, split, count, seed):
    """Uniform window sampling followed by uniform channel/time sampling."""
    rng = np.random.default_rng(seed)
    g = cache[split]
    valid_windows = g["unique_indices"][:]
    windows = rng.choice(
        valid_windows, size=min(1024, len(valid_windows)), replace=False
    )
    per_window = max(1, int(np.ceil(count / len(windows))))
    rows = []
    for w in sorted(windows):
        x = g["continuous"][int(w)].reshape(-1, g["continuous"].shape[-1])
        rows.append(x[rng.choice(len(x), size=min(per_window, len(x)), replace=False)])
    return torch.from_numpy(np.concatenate(rows)[:count].astype(np.float32))


def causal_features(raw, frames, config, kind):
    """Trailing 480 ms windows at precisely the existing cached token ends."""
    receptive = round(
        config.get("sample_rate_hz", 2000)
        * config.get("receptive_field_ms", 160)
        / 1000
    )
    hop = config.get("sample_rate_hz", 2000) // config.get("token_rate_hz", 50)
    # Derived config properties are not serialized; these are fixed by this study.
    if receptive != 320 or hop != 40:
        raise ValueError("this campaign requires the audited 320/40 grid")
    all_steps = (raw.shape[-1] - 320) // 40 + 1
    ends = 319 + torch.arange(all_steps - frames, all_steps, device=raw.device) * 40
    indices = ends[:, None] - torch.arange(959, -1, -1, device=raw.device)
    if indices.min() < 1000:
        raise ValueError("feature context must follow calibration")
    patches = raw[..., indices]
    if kind == "rms3":
        return torch.stack(
            (
                patches.square().mean(-1).sqrt(),
                patches.abs().mean(-1),
                patches.diff(dim=-1).abs().mean(-1),
            ),
            -1,
        ).log1p()
    if kind == "spectral64":
        patches = patches - patches.mean(-1, keepdim=True)
        power = (
            torch.fft.rfft(
                patches * torch.hann_window(960, periodic=False, device=raw.device)
            )
            .abs()
            .square()[..., 1:]
        )
        return (
            F.adaptive_avg_pool1d(power.reshape(-1, 1, 480), 64)
            .reshape(*power.shape[:-1], 64)
            .log1p()
        )
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fsq-steps", type=int, default=5000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(260906)
    device = torch.device(args.device)
    base = args.cache.resolve()
    with h5py.File(base, "r") as source:
        assert source.attrs["complete"]
        manifest = json.loads(source.attrs["manifest"])
        train = sample_latents(source, "train", 131072, 260906).to(device)
        val = sample_latents(source, "val", 16384, 260907).to(device)
        mean = train.mean(0)
        centred = train - mean
        covariance = centred.T @ centred / len(train)
        _, vectors = torch.linalg.eigh(covariance)
        basis = vectors.flip(1)
        scale = centred.square().mean(0).sqrt().clamp_min(1e-4)
        normalized, validation = centred / scale, (val - mean) / scale
        model = LatentFSQ(train.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        best, best_state, history = float("inf"), None, []
        for step in range(1, args.fsq_steps + 1):
            model.train()
            x = normalized[torch.randint(len(normalized), (1024,), device=device)]
            loss = F.mse_loss(model(x), x)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite FSQ loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 250 == 0 or step == args.fsq_steps:
                model.eval()
                with torch.no_grad():
                    error = float(F.mse_loss(model(validation), validation))
                history.append({"step": step, "validation_latent_mse": error})
                print(json.dumps(history[-1]), flush=True)
                if error < best:
                    best = error
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
        model.load_state_dict(best_state)
        model.eval().requires_grad_(False)
        artifact = args.output / "transforms.pt"
        torch.save(
            {
                "mean": mean.cpu(),
                "scale": scale.cpu(),
                "pca_basis": basis.cpu(),
                "fsq": best_state,
                "fsq_history": history,
                "base_cache_id": manifest["cache_id"],
                "seed": 260906,
                "fit_split": "train",
                "selection_split": "val",
            },
            artifact,
        )
        transform_hash = digest(artifact)
        sizes = {}
        for kind in ("rms3", "spectral64", "pca4", "pca16", "pca64", "fsq1000"):
            path = args.output / (kind + ".h5")
            with h5py.File(path, "w") as out:
                modified = {
                    **manifest,
                    "representation": kind,
                    "base_cache_id": manifest["cache_id"],
                    "transform_sha256": transform_hash,
                    "preparation_sha256": digest(__file__),
                }
                modified["cache_id"] = hashlib.sha256(
                    json.dumps(modified, sort_keys=True).encode()
                ).hexdigest()
                for key in source:
                    if key not in ("train", "val", "test"):
                        out[key] = h5py.ExternalLink(str(base), "/" + key)
                if kind == "fsq1000":
                    del out["codebook"]
                    ids = torch.arange(1000)
                    digits = (
                        ids[:, None] // torch.tensor([1, 8, 40, 200])
                    ) % torch.tensor([8, 5, 5, 5])
                    out["codebook"] = (
                        (2 * digits / torch.tensor([7, 4, 4, 4]) - 1)
                        .numpy()
                        .astype(np.float32)
                    )
                for split in ("train", "val", "test"):
                    g, original = out.create_group(split), source[split]
                    feature_key = "tokens" if kind == "fsq1000" else "continuous"
                    for key in original:
                        if key != feature_key:
                            g[key] = h5py.ExternalLink(
                                str(base), "/" + split + "/" + key
                            )
                    for start in range(0, len(original["tokens"]), 8):
                        stop = min(start + 8, len(original["tokens"]))
                        with torch.no_grad():
                            if kind.startswith("pca") or kind == "fsq1000":
                                x = torch.from_numpy(
                                    original["continuous"][start:stop].astype(
                                        np.float32
                                    )
                                ).to(device)
                                if kind.startswith("pca"):
                                    values = (
                                        ((x - mean) @ basis[:, : int(kind[3:])])
                                        .cpu()
                                        .numpy()
                                        .astype(np.float16)
                                    )
                                else:
                                    values = (
                                        model.quantize((x - mean) / scale)[1]
                                        .cpu()
                                        .numpy()
                                        .astype(np.uint16)
                                    )
                            else:
                                raw = torch.from_numpy(original["raw"][start:stop]).to(
                                    device
                                )
                                values = (
                                    causal_features(
                                        raw,
                                        manifest["context_frames"],
                                        manifest["config"],
                                        kind,
                                    )
                                    .cpu()
                                    .numpy()
                                    .astype(np.float16)
                                )
                        if not np.isfinite(values).all():
                            raise ValueError("nonfinite derived features")
                        if feature_key not in g:
                            g.create_dataset(
                                feature_key,
                                (len(original["tokens"]), *values.shape[1:]),
                                dtype=values.dtype,
                                chunks=(1, *values.shape[1:]),
                            )
                        g[feature_key][start:stop] = values
                    print(
                        json.dumps(
                            {"representation": kind, "split": split, "complete": True}
                        ),
                        flush=True,
                    )
                out.attrs["manifest"] = json.dumps(modified)
                out.attrs["complete"] = True
                sizes[kind] = {
                    "train_feature_bytes": out["train"][
                        feature_key
                    ].id.get_storage_size(),
                    # Read the already-open source directly: resolving an external
                    # link from a writable output can request a conflicting mode.
                    "train_unique_windows": len(source["train"]["unique_indices"]),
                    "feature_shape": list(out["train"][feature_key].shape),
                    "shared_codebook_bytes": 16000 if kind == "fsq1000" else 0,
                    "cache_id": modified["cache_id"],
                }
        write_json(
            args.output / "receipt.json",
            {
                "representations": sizes,
                "transform_sha256": transform_hash,
                "fsq_scope": "post-hoc latent compression; 1000 codes; train-only fitting, validation latent-MSE selection; not end-to-end spectral FSQ",
                "causal_feature_samples": 960,
                "sampled_train_latents": len(train),
                "sampled_val_latents": len(val),
            },
        )


if __name__ == "__main__":
    main()
