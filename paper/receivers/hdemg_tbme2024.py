"""Original TBME 2024 CNN, with endpoint-matched data and explicit provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
import torch

from paper.vendor.tbme2024.model import CNNApproximator

SUBJECTS = tuple(range(1, 10))
SEEDS = (260911, 260912, 260913)
EPOCHS, EFFECTIVE_BATCH, MICRO_BATCH = 50, 512, 64
CONFIG = dict(
    learning_rate=1e-4, nr_of_input_channels=2, input_length__samples=192,
    nr_of_outputs=60, cnn_encoder_channels=(256, 32, 64),
    mlp_encoder_channels=(512, 512), event_search_kernel_length=31,
    event_search_kernel_stride=8,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def build():
    # The archived forward, normalization, layers and initialization are intact.
    return CNNApproximator(**CONFIG)


def preprocess(raw):
    """Archived window-local notch and R+20 filtering; no post-endpoint samples."""
    raw = np.asarray(raw)
    if raw.shape[-2:] != (320, 192) or not np.isfinite(raw).all():
        raise ValueError("Expected finite 320 x 192 native endpoint windows")
    notch = filtfilt(*iirnotch(50, 75, fs=2048), raw, padtype=None, axis=-1)
    low = sosfiltfilt(butter(4, 20, fs=2048, output="sos"), notch, axis=-1)
    result = np.stack((notch, low), axis=-3).astype(np.float32)
    grids = result.reshape(-1, 2, 5, 64, 192)
    if not np.isfinite(result).all() or np.any(np.ptp(grids, axis=(-2, -1)) == 0):
        raise ValueError("Nonfinite or constant grid would invalidate original normalization")
    return result


def augment(x, rng):
    """Published two augmentation distributions, with an explicit seeded RNG.

    Preserve the archived Gaussian implementation's int16-rounded additive noise;
    magnitude warping uses six cubic spline points per grid, shared across the two
    representations. Unlike its unseeded default_rng calls, this is reproducible.
    """
    x = np.asarray(x, dtype=np.float32)
    rms = np.sqrt((np.square(x.astype(np.float64)).mean(-1, keepdims=True) + 1e-15)
                  / 10 ** (5.0 / 10))
    noise = rng.normal(size=x.shape) * rms
    noisy = x + noise.astype(np.int16)
    shape = x.shape[:-3] + (5, 6)
    controls = rng.normal(1.0, 0.35, size=shape)
    curves = interp1d(np.linspace(0, 192, 6), controls, kind="cubic", axis=-1)(np.arange(192))
    curves = np.repeat(curves, 64, axis=-2)
    warped = x.astype(np.float64) * np.expand_dims(curves, -3)
    return noisy.astype(np.float32), warped.astype(np.float32)


def target_stats(train):
    flat = np.asarray(train).reshape(-1, 60)
    minimum = flat.min(0)
    span = flat.max(0) - minimum
    # Constant coordinates remain finite and reversible; never inspect val/test.
    span = np.where(span > 0, span, 1).astype(np.float32)
    return minimum.astype(np.float32), span


def error_mm(prediction, target):
    pred, truth = np.asarray(prediction), np.asarray(target)
    if pred.shape != truth.shape or pred.shape[-1] != 60:
        raise ValueError("Mismatched 60-coordinate predictions and targets")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite predictions or targets")
    return float(np.linalg.norm((pred - truth).reshape(-1, 20, 3), axis=-1).mean())


@torch.inference_mode()
def predict(model, x, minimum, span):
    model.eval()
    rows = []
    for batch in x.split(MICRO_BATCH):
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(batch)
        rows.append(((result.float() + 1) * span / 2 + minimum).cpu().numpy())
    result = np.concatenate(rows)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite inference")
    return result


def optimizer(model, total_steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, amsgrad=True, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-4 * 10 ** 1.5, total_steps=total_steps,
        anneal_strategy="cos", three_phase=False, div_factor=10 ** 1.5,
        final_div_factor=1e3,
    )
    return opt, schedule


def train_step(model, x, y, indices, opt, scaler, seed):
    # Retry an AMP overflow with the same dropout RNG and BatchNorm state, so a
    # skipped optimizer update cannot silently consume targets or alter buffers.
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    for retry in range(8):
        if retry:
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        torch.manual_seed(seed)
        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for part in indices.split(MICRO_BATCH):
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(x[part])
                loss = torch.nn.functional.l1_loss(output.float(), y[part]) * len(part) / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            total_loss += float(loss.detach())
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        previous = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= previous:
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            return total_loss, retry
    raise FloatingPointError("AMP overflow persisted for eight attempts")
