"""Measured packet and full inference costs on a fixed native-EMG replay."""

import argparse
import json
import platform
import struct
import time
from pathlib import Path

from paper.emg2pose import compare_temporal_pose as pose
import h5py
import numpy as np
from paper.emg2pose import paper_completion_biocodec as codec
from paper.emg2pose import paper_completion_core as core
from paper.common import paths
import torch
from torch.nn import functional as F


class CachedReadout:
    """Cache only time-local operations, identically for every representation."""

    def __init__(self, model, descriptors, centre, scale):
        if model.training or model.kind != "transformer":
            raise ValueError("frozen causal transformer required")
        self.model, self.descriptors = model, descriptors
        self.centre, self.scale, self.frames = centre, scale, []

    def push(self, features):
        m = self.model
        if features.shape[2] != 1:
            raise ValueError("one frame required")
        centre, scale = self.centre, self.scale
        if m.input_mode == "calibration_only":
            features = torch.zeros_like(features)
        if m.input_mode == "features_only":
            centre, scale = torch.zeros_like(centre), torch.ones_like(scale)
        spatial = m.pool.spatialize(m.project(features), self.descriptors)
        pooled = m.pool(spatial, self.descriptors, spatialized=True)
        batch, slots, steps, width = pooled.shape
        pooled = pooled.permute(0, 2, 1, 3).reshape(batch, steps, slots*width)
        ids = torch.zeros(batch, 3, dtype=torch.long, device=features.device)
        style = m.style(centre, scale, ids).unsqueeze(1)
        self.frames.append(m.frame(torch.cat((pooled, style), dim=-1)))
        self.frames = self.frames[-m.context_frames:]
        frames = torch.cat(self.frames, dim=1)
        steps = frames.shape[1]
        mask = torch.ones(steps, steps, dtype=torch.bool, device=frames.device).triu(1)
        return m.output(m.temporal(frames + m.position[:steps], mask=mask))[:, -1:]


def engineered_frame(raw):
    """Same trailing960 samples as the offline RMS3+spectral64 transform."""
    patches = raw[..., -960:]
    rms = torch.stack((patches.square().mean(-1).sqrt(), patches.abs().mean(-1),
                       patches.diff(dim=-1).abs().mean(-1)), -1).log1p()
    centred = patches - patches.mean(-1, keepdim=True)
    power = torch.fft.rfft(centred * torch.hann_window(960, periodic=False, device=raw.device)).abs().square()[..., 1:]
    spectral = F.adaptive_avg_pool1d(power.reshape(-1, 1, 480), 64).reshape(*raw.shape[:-1], 64).log1p()
    return torch.cat((rms, spectral), -1).unsqueeze(2)


def floats_packet(values):
    array = np.asarray(values, dtype="<f4")
    return struct.pack("<4sBI", b"EMGF", 32, array.size) + array.tobytes()


def tensor_bytes(values):
    return sum(t.numel()*t.element_size() for t in values)


def auxiliary_bytes(label, transforms, fsq):
    """Sender transforms omitted from the frozen tokenizer's state_dict."""
    if label in core.PCA_LEVELS:
        return tensor_bytes(transforms[k] for k in ("mean", "pca_basis", "pca_lo", "pca_hi"))
    if label == "fsq1000":
        return tensor_bytes(fsq.encoder.state_dict().values()) + tensor_bytes(transforms[k] for k in ("mean", "scale")) + tensor_bytes([fsq.levels])
    if label in ("features67", "kmeans256", "kmeans1024"):
        value = tensor_bytes(transforms[k] for k in ("feature_mean", "feature_scale"))
        return value + (tensor_bytes([transforms["clusters"][label.removeprefix("kmeans")]]) if label.startswith("kmeans") else 0)
    return 0


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def distribution(values):
    values = np.asarray(values)
    return {"count": len(values), "p50_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, .95)), "max_ms": float(values.max()),
            "fraction_over_20ms": float((values > 20).mean())}


@torch.inference_mode()
def run(args):
    from paper.emg2pose.prepare_reviewer_features import LatentFSQ

    from myoji.learned import load_universal_checkpoint
    from myoji.preprocessing import calibration_normalize
    p = core.verify(args.root)
    runtime_precision = core.configure_runtime_precision()
    replay = args.root / "assets/pose.npz"
    if pose.digest(replay) != p["runtime"]["replay_sha256"]:
        raise ValueError("runtime input changed")
    with np.load(replay) as archive:
        raw_cpu = archive["raw"].copy()
        descriptors = torch.from_numpy(archive["descriptors"]).to(args.device)
        ends = archive["end_samples"].copy()
        expected_ids = archive["tokens"].copy()
    sync(args.device)
    cold = time.monotonic()
    module, _ = load_universal_checkpoint(paths.LEARNED_TOKENIZER_FULL, torch.device(args.device))
    encoder = module.tokenizer.eval().requires_grad_(False)
    normalized, centre, scale = calibration_normalize(torch.from_numpy(raw_cpu)[None].to(args.device), 1000)
    transforms = torch.load(args.root / "controls/official/transforms.pt", map_location=args.device, weights_only=False)
    fsq = LatentFSQ(256).to(args.device).eval()
    fsq.load_state_dict(transforms["fsq"], strict=True)
    codec_model = codec.build_model().to(args.device).eval().requires_grad_(False)
    codec_path = args.root / "biocodec/seed_260940/model.pt"
    codec_model.load_state_dict(torch.load(codec_path, map_location="cpu", weights_only=False)["model"], strict=True)
    sync(args.device)
    initialization_ms = 1000*(time.monotonic()-cold)
    # Timing uses this device's arithmetic. Canonical training caches came from
    # RTX3080/TF32; record portability separately from same-device stream parity.
    device_reference = encoder(normalized, sensor="wrist", quantize=True, include_spatial=False)
    reference_ids = device_reference.tokens[:, :, -100:].cpu().numpy()[0]
    cache_mismatches = int((reference_ids != expected_ids).sum())
    rows = []
    cases = [r for r in p["cases"] + p["references"] if r["scope"] == "official" and r["head_seed"] == 260826 and
             (r["encoder_seed"] in (260905, 260940) or r["label"] == "raw_supervised")]
    for case in cases:
        label = case["label"]
        checkpoint = Path(case["fit"]) / "best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = pose.build_pose_model(saved["model_kwargs"]).to(args.device).eval()
        model.load_state_dict(saved["model"], strict=True)
        stats = {k: v.to(args.device) for k, v in saved["stats"].items()}
        with h5py.File(case["cache"], "r") as h:
            book = torch.from_numpy(h["codebook"][:]).to(args.device)
        is_codec = label.startswith("biocodec")
        is_raw = label == "raw_supervised"
        enc = model.encoder if is_raw else encoder
        head = model.readout if is_raw else model
        decoder_bytes = tensor_bytes(head.state_dict().values())
        representation_bytes = (sum(t.numel()*t.element_size() for t in codec_model.encoder.state_dict().values()) +
                                sum(t.numel()*t.element_size() for t in codec_model.quantizer.state_dict().values())) if is_codec else (
                                    0 if label in ("features67", "kmeans256", "kmeans1024") else
                                    tensor_bytes(enc.state_dict().values()))
        auxiliary = auxiliary_bytes(label, transforms, fsq)
        lookup_bytes = book.numel()*book.element_size() if not is_codec and case["arm"].startswith("token") else 0
        timings, components, packets, maximum_difference = [], [], [], 0.
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(3):
            stream = CachedReadout(head, descriptors, centre, scale)
            reference_features, predictions = [], []
            for j, end in enumerate(ends):
                sync(args.device)
                started = time.monotonic()
                # Includes native-EMG CPU->device transfer and fixed calibration application.
                segment = torch.from_numpy(raw_cpu[:, :int(end)+1] if is_codec else
                                           raw_cpu[:, int(end)-959:int(end)+1])[None].to(args.device)
                current = ((segment-centre)/scale).clamp(-12, 12)
                sync(args.device)
                preprocessing_end = time.monotonic()
                bits, ids = None, None
                if is_codec:
                    codes = codec.encode_at(codec_model, current, [end])
                    ids, bits = codes[:, :, :case["depth"], :], 8
                elif label in ("features67", "kmeans256", "kmeans1024"):
                    feature = (engineered_frame(current)-transforms["feature_mean"])/transforms["feature_scale"]
                    if label.startswith("kmeans"):
                        ids = core.nearest_centres(feature, transforms["clusters"][label.removeprefix("kmeans")]).reshape(1, 16, 1)
                        bits = 8 if label == "kmeans256" else 10
                else:
                    encoded = enc(current, sensor="wrist", quantize=label.startswith("vq_"), include_spatial=False)
                    feature = encoded.continuous[:, :, -1:]
                    if label.startswith("vq_"):
                        ids, bits = encoded.tokens[:, :, -1:], 10
                        np.testing.assert_array_equal(ids[0, :, 0].cpu().numpy(), reference_ids[:, j])
                    elif label in core.PCA_LEVELS:
                        values = (feature-transforms["mean"]) @ transforms["pca_basis"]
                        ids = core.scalar_encode(values, transforms["pca_lo"], transforms["pca_hi"], core.PCA_LEVELS[label])
                        bits = {"pca4q8": 8, "pca4q10": 10, "pca4q16": 16}[label]
                    elif label == "fsq1000":
                        ids, bits = fsq.quantize((feature-transforms["mean"])/transforms["scale"])[1], 10
                sync(args.device)
                encoding_end = time.monotonic()
                if bits is None:
                    packet = floats_packet(feature.cpu().numpy())
                    feature = torch.from_numpy(np.frombuffer(packet, dtype="<f4", offset=9).copy()).reshape(feature.shape).to(args.device)
                else:
                    packet = core.pack_ids(ids.cpu().numpy(), bits)
                    received = torch.from_numpy(core.unpack_ids(packet).astype(np.int64)).reshape(ids.shape).to(args.device)
                    if is_codec:
                        padded = torch.zeros(1, 16, 6, 1, dtype=torch.long, device=args.device)
                        padded[:, :, :case["depth"]] = received
                        feature = codec.decode_codes(codec_model, padded, case["depth"])
                    else:
                        feature = F.embedding(received, book)
                sync(args.device)
                transport_end = time.monotonic()
                prediction = stream.push(feature)*stats["scale"] + stats["mean"]
                prediction.cpu().numpy()  # Actual result available to the consumer.
                sync(args.device)
                finished = time.monotonic()
                if repeat > 0 and j >= 50:
                    timings.append(1000*(finished-started))
                    components.append([1000*(preprocessing_end-started), 1000*(encoding_end-preprocessing_end),
                                       1000*(transport_end-encoding_end), 1000*(finished-transport_end)])
                if repeat == 0:
                    packets.append(len(packet))
                    reference_features.append(feature)
                    predictions.append(prediction)
            if repeat == 0:
                full_features = torch.cat(reference_features, 2)
                full = head(full_features, descriptors, centre, scale)*stats["scale"] + stats["mean"]
                online = torch.cat(predictions, 1)
                torch.testing.assert_close(online, full, atol=3e-4, rtol=3e-4)
                maximum_difference = float((online-full).abs().max())
                if label.startswith("continuous_"):
                    batch = encoder(normalized, sensor="wrist", quantize=False, include_spatial=False).continuous[:, :, -100:]
                    torch.testing.assert_close(full_features, batch, atol=3e-4, rtol=3e-4)
        calibration_setup = json.dumps({"sample_rate": 2000, "token_rate": 50, "channels": 16,
                                       "first_end_sample": int(ends[0]), "case_id": case["id"],
                                       "model_sha256": pose.digest(checkpoint),
                                       "centre": centre.cpu().tolist(), "scale": scale.cpu().tolist(),
                                       "descriptors": descriptors.cpu().tolist()}, separators=(",", ":")).encode()
        rows.append({"case": case, "latency": distribution(timings),
                     "component_latency": {name: distribution(np.asarray(components)[:, i]) for i, name in
                                           enumerate(("preprocessing", "encoding", "pack_copy_unpack_lookup", "readout_output"))},
                     "mean_packet_bytes_per_50hz_frame": float(np.mean(packets)),
                     "packet_bytes_per_second": float(np.mean(packets)*50),
                     "setup_json_bytes": len(calibration_setup), "decoder_tensor_bytes": decoder_bytes,
                     "representation_tensor_bytes": representation_bytes,
                     "sender_auxiliary_tensor_bytes": auxiliary,
                     "lookup_codebook_bytes": lookup_bytes,
                     "sum_component_tensor_bytes": representation_bytes + decoder_bytes + auxiliary + lookup_bytes,
                     "parity_max_abs": maximum_difference, "checkpoint_sha256": pose.digest(checkpoint),
                     "peak_gpu_bytes_process": torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") else None,
                     "codec_native_full_rate_bits_per_channel_second": case.get("depth", 0)*8*1000/18 if is_codec else None})
        print(json.dumps({"runtime": label, **rows[-1]["latency"]}), flush=True)
    pose.write_json(args.root / ("runtime_" + args.device.replace(":", "_") + ".json"), {
        "complete": True, "protocol_sha256": pose.digest(args.root / "protocol.json"), "rows": rows,
        "input_sha256": pose.digest(replay), "initialization_ms_joint_models": initialization_ms,
        "runtime_precision": runtime_precision,
        "canonical_cache_token_portability": {"mismatched_ids": cache_mismatches,
                                               "total_ids": int(expected_ids.size),
                                               "agreement": float((reference_ids == expected_ids).mean()),
                                               "scope": "Fixed replay only; exact online parity uses the same device's full-window reference. Canonical cache remains unchanged."},
        "calibration_acquisition_ms": 500,
        "native_raw_float32_reference_packet_bytes_per_second": len(floats_packet(raw_cpu[:, :40]))*50,
        "storage_accounting": "Named inference tensor components, including post-hoc transforms and receiver lookup tables. Raw-supervised encoder and readout counted separately. Instantiated tokenizers retain unused buffers/spatial parameters; no pruning claim. Fixed signal-processing constants, Python/library memory and checkpoint serialization overhead excluded from tensor counts; process GPU allocation reported separately. Native-raw reference uses actual float32 values and the same9-byte framing, for bytes only; raw-supervised runtime sends its intermediate features.",
        "device": args.device, "hardware": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else platform.platform(),
        "torch": str(torch.__version__), "threads": torch.get_num_threads(),
        "scope": "One prespecified native2kHz 16-channel three-second replay, first repetition discarded, last50 frames timed in two repeats. Includes CPU/device copies and actual packets in one process; no physical network/ADC/device actuation. Cached time-local readout shared by all methods; BioCodec recomputes increasing causal prefixes (up to3s), no optimized LSTM streaming or equal-context claim. Meta output has no filter. Model initialization is joint, not per-arm cold-start ranking. Process GPU peak includes other loaded study models, not isolated per-arm footprint.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)
