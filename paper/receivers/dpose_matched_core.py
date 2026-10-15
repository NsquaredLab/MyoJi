"""Shared capacity, sampling and optimization mechanics for DPose controls."""

import hashlib
import math

import numpy as np
import torch
from paper.receivers.dpose_raulnet_model import CONFIG, RaulNetV19, configure_eager
from torch.nn import functional as F

from myoji.decoders import CachedPoseDecoder

ARMS = (
    "tcn_tokens",
    "tcn_continuous",
    "transformer_tokens",
    "transformer_continuous",
    "v19",
)
COUNTS = {
    "tcn_tokens": 7549186,
    "tcn_continuous": 7549186,
    "transformer_tokens": 7445082,
    "transformer_continuous": 7445082,
    "v19": 7489436,
}
SEEDS = (260911, 260912, 260913)
EPOCHS, UPDATES_PER_EPOCH, WINDOWS_PER_UPDATE, SCORE_FRAMES = 150, 100, 2, 50


def sample_windows(seed, epoch, windows=64):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, 8192])).integers(
        0, windows, size=(UPDATES_PER_EPOCH, WINDOWS_PER_UPDATE), dtype=np.int64
    )


def schedule_digest(seed):
    h = hashlib.sha256()
    for epoch in range(EPOCHS):
        h.update(sample_windows(seed, epoch).tobytes())
    return h.hexdigest()


def learning_rate(step):
    warmup, total = 500, EPOCHS * UPDATES_PER_EPOCH
    if step < warmup:
        return 3e-4 * (step + 1) / warmup
    progress = (step - warmup) / (total - warmup - 1)
    return 3e-6 + (3e-4 - 3e-6) * 0.5 * (1 + math.cos(math.pi * progress))


def build(arm):
    if arm == "v19":
        model = configure_eager(RaulNetV19(**CONFIG))
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout3d)):
                module.p = 0.1
    else:
        kind = arm.split("_")[0]
        model = CachedPoseDecoder(
            embedding_dim=256,
            geometry_dim=14,
            channels=320,
            kind=kind,
            width=404 if kind == "tcn" else 468,
            slots=4,
            heads=4,
            layers=2,
            context_frames=100,
            output_dim=60,
            dropout=0.1,
            input_mode="features_only",
        )
    count = sum(v.numel() for v in model.parameters() if v.requires_grad)
    assert count == COUNTS[arm] and abs(count / COUNTS["v19"] - 1) < 0.01
    return model


def cached_forward(model, data, windows, arm):
    x = data["features"][windows]
    if arm.endswith("tokens"):
        x = data["book"][x.long()]
    else:
        x = x.float()
    zero = torch.zeros(len(windows), 320, 1, device=x.device)
    return model(x, data["descriptors"], zero, torch.ones_like(zero))[:, -50:]


def loss_parts(model, data, windows, arm, mean, scale):
    """Each update weights exactly the same 100 endpoint-coordinate targets."""
    if arm == "v19":
        # Native V19 has no batch-dependent normalization; two 50-example
        # microbatches bound memory, accumulating one 100-endpoint update.
        for window in windows:
            pred = model(data["features"][window])
            target = (data["targets"][window] - mean) / scale
            yield F.smooth_l1_loss(pred.float(), target) / len(windows)
    else:
        pred = cached_forward(model, data, windows, arm)
        target = (data["targets"][windows] - mean) / scale
        yield F.smooth_l1_loss(pred.float(), target)


def update(model, data, windows, arm, mean, scale, opt, scaler, step, seed):
    """AMP overflow retries do not consume an optimizer update or new targets."""
    for retry in range(8):
        torch.manual_seed(seed + step)
        model.train()
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = learning_rate(step)
        value = 0.0
        with torch.autocast("cuda", dtype=torch.float16):
            for loss in loss_parts(model, data, windows, arm, mean, scale):
                assert torch.isfinite(loss)
                value += float(loss.detach())
                scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            assert torch.isfinite(norm)
            return value, retry
    raise FloatingPointError("AMP overflow persisted for eight retries")
