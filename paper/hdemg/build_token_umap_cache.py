"""Encode one HD-EMG task into both token representations, frame-aligned to kinematics.

Both routes see identical frames. The first 23 frames are dropped from both because
the learned encoder needs 960 samples of history; dropping them from SPAR too keeps
the two arms comparable, which is the only thing the headline figure measures.

Calibration follows the codebook's own regime: 6000-sample windows tiled at a
4000-sample stride, each normalized by its own first 1000 samples, each emitting the
protocol's 100 frames. Calibrating once per recording instead was measured at 0.2221
masked reconstruction MSE against 0.1863 val / 0.1743 test in this regime, because the
log-determinant entry does not shift linearly under per-channel rescaling. The stride
is a multiple of the 40-sample hop, so the frame grid stays aligned to the kinematics.

    uv run python -m paper.hdemg.build_token_umap_cache --task 15
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import firwin, upfirdn

ROOT = Path(__file__).resolve().parents[1]
from paper.common import paths
from myoji.spar import spar_simple_dataset_batched
from paper.firmware.raw_tokenizer_benchmark.prepare import torch_learned

RAW = paths.HDEMG / "Sub1_emg.pkl"   # participant 1: task -> (320, T) int16 at 2048 Hz
CYCLES = paths.RESULTS / "hdemg_movement_video/kinematic-cycle-selection.npz"
# Participant 1's SPAR codebook and standardization, byte-identical to the full campaign's subject_01 cache.
SPAR_SOURCE = paths.ASSETS / "spar/codebooks-1024.npz"
CHECKPOINT = paths.LEARNED_TOKENIZER
OUT = paths.RESULTS / "token_umap"
CHANNELS, PATCH, HOP = 320, 320, 40
WINDOW, STRIDE, LEAD = 6000, 4000, 2039   # the protocol's window, and its first frame
TIPS = [0, 4, 8, 12, 16]         # tip-first chains: fingertips in the 20-point truth
BLOCK = 128


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resample(emg: np.ndarray) -> np.ndarray:
    """The dataset export's own 2048 -> 2000 Hz path, reproduced exactly."""
    taps = firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
    return upfirdn(taps, emg.astype(np.float32), up=125, down=128,
                   axis=-1)[..., : emg.shape[-1] * 125 // 128].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)

    raw = joblib.load(RAW)[str(args.task)]
    assert raw.dtype == np.int16 and raw.shape[0] == CHANNELS, (raw.dtype, raw.shape)
    signal = resample(raw)

    cycles = np.load(CYCLES, allow_pickle=False)
    all_ends = cycles[f"task_{args.task}_ends"]
    all_truth = cycles[f"task_{args.task}_truth"]

    # One window per calibration, exactly as the codebook was fitted.
    starts = np.arange(0, signal.shape[1] - WINDOW + 1, STRIDE)
    assert STRIDE % HOP == 0 and LEAD % HOP == 39 % HOP, "stride would break the frame grid"
    ends = np.concatenate([s + LEAD + HOP * np.arange(100) for s in starts])
    keep = (ends - 39) // HOP
    assert np.array_equal(all_ends[keep], ends), "frame grid does not line up with the kinematics"
    assert keep.size == np.unique(keep).size, "windows overlap in frames"
    truth = all_truth[keep]
    frames = len(ends)

    x = torch.from_numpy(signal)
    centres, scales, pieces = [], [], []
    for start in starts:
        block = x[:, start:start + WINDOW]
        centre = block[:, :1000].sort(-1).values[:, 499:500]
        scale = (1.4826 * (block[:, :1000] - centre).abs().sort(-1).values[:, 499:500]).clamp_min(1e-4)
        centres.append(centre.ravel().numpy()); scales.append(scale.ravel().numpy())
        pieces.append(((block - centre) / scale).clamp(-12, 12))

    spar_source = np.load(SPAR_SOURCE)
    mean, spread_, mask = (spar_source[f"subject_01/{k}"] for k in ("feature_mean", "feature_scale", "mask"))
    spar_book = torch.from_numpy(spar_source["subject_01/codebook"])
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"]
    learned_book = state["quantizer.codebook"]

    path = OUT / f"sub01_task{args.task}.h5"
    with h5py.File(path, "w") as store, torch.inference_mode():
        store.create_dataset("ends", data=ends)
        store.create_dataset("truth", data=truth.astype(np.float32))
        spar_out = store.create_dataset("spar_continuous", (frames, CHANNELS, 18), "f4")
        spar_ids = store.create_dataset("spar_ids", (frames, CHANNELS), "u2")
        learned_out = store.create_dataset("learned_continuous", (frames, CHANNELS, 256), "f2",
                                           chunks=(32, CHANNELS, 256))
        learned_ids = store.create_dataset("learned_ids", (frames, CHANNELS), "u2")
        residual = 0.0

        for lo in range(0, frames, BLOCK):
            hi = min(lo + BLOCK, frames)
            windows = np.stack([
                pieces[f // 100].numpy()[:, local - PATCH + 1: local + 1]
                for f, local in ((f, ends[f] - starts[f // 100]) for f in range(lo, hi))])
            windows = windows.transpose(0, 2, 1).copy()               # (n, 320 samples, 320 ch)
            values = spar_simple_dataset_batched(windows, channelwise=True)
            values = ((values - mean) / spread_) * mask[None]
            flat = torch.from_numpy(values.reshape(-1, 18))
            d = flat.square().sum(-1, keepdim=True) + spar_book.square().sum(-1)[None] - 2 * flat @ spar_book.T
            ids = d.argmin(-1)
            residual += float(((flat - spar_book[ids]) ** 2).sum())
            spar_out[lo:hi] = values
            spar_ids[lo:hi] = ids.reshape(hi - lo, CHANNELS).numpy().astype("uint16")

        # Learned: per window, so both routes see the same calibration.
        for w, piece in enumerate(pieces):
            rows = slice(w * 100, (w + 1) * 100)
            for c in range(0, CHANNELS, 16):
                out = torch_learned(piece[c:c + 16], state)           # (16, 143, 256)
                picked = out[:, 43:143]
                flat = picked.reshape(-1, 256)
                d = (flat.square().sum(-1, keepdim=True) - 2 * flat @ learned_book.T
                     + learned_book.square().sum(-1)[None])
                ids = d.argmin(-1).reshape(16, 100)
                learned_out[rows, c:c + 16] = picked.permute(1, 0, 2).numpy().astype(np.float16)
                learned_ids[rows, c:c + 16] = ids.T.numpy().astype("uint16")

        tips = np.linalg.norm(truth[:, TIPS, :], axis=2).mean(axis=1)
        store.create_dataset("fingertip_spread_mm", data=tips.astype(np.float32))
        store.create_dataset("window_starts", data=starts)
        store.create_dataset("calibration_centre", data=np.stack(centres))
        store.create_dataset("calibration_scale", data=np.stack(scales))
        live = int(mask.sum())
        store.attrs.update(
            task=args.task, subject=1, frames=frames, channels=CHANNELS,
            first_frame_index=int(keep[0]), sample_rate_hz=2000, token_rate_hz=50,
            windows=len(starts), window_samples=WINDOW, window_stride=STRIDE,
            patch_samples=PATCH, hop_samples=HOP,
            resampler="firwin(10241, 800, fs=256000, kaiser 8.6)*125; upfirdn(125,128); "
                      "truncate to n*125//128",
            calibration="per 6000-sample window, its own first 1000 samples, lower median and "
                        "lower MAD*1.4826, floor 1e-4, clip +/-12 - the regime the codebook was fitted in",
            calibration_alternative_measured="calibrating once per recording gave masked MSE 0.2221 "
                                             "against 0.1863 val / 0.1743 test here, so it was rejected",
            masked_reconstruction_mse=residual / (frames * live),
            reference_reconstruction_mse="0.1863 val / 0.1743 test, recomputed here from the released "
                                         "per-window features with this codebook",
            learned_continuous_dtype="float16 (ids computed in float32 before the downcast)",
            checkpoint_sha256=sha(CHECKPOINT), spar_source_sha256=sha(SPAR_SOURCE),
            # C-order little-endian int16 samples: the same digest as the former raw int16 export.
            raw_int16_sha256=hashlib.sha256(raw.astype("<i2").tobytes()).hexdigest(),
            builder_sha256=sha(Path(__file__)))
        summary = dict(path=str(path.relative_to(ROOT)), frames=frames,
                       masked_reconstruction_mse=round(residual / (frames * live), 4),
                       spar_codes_used=int(np.unique(spar_ids[:]).size),
                       learned_codes_used=int(np.unique(learned_ids[:]).size),
                       spread_mm=[round(float(tips.min()), 1), round(float(tips.max()), 1)])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
