"""Gate expensive jobs on real-source replay, gradient, timing and storage checks."""

import argparse
import json
import time
from pathlib import Path

import compare_temporal_pose as pose
import h5py
import numpy as np
import paper_completion_biocodec as codec
import paper_completion_core as core
import paths
import torch
from paper_completion_runtime import CachedReadout, sync


def run(args):
    import train_mixed_sources as mixed

    from myoji.decoders import CachedPoseDecoder
    from myoji.learned import TokenizerConfig, calibration_normalize, load_universal_checkpoint

    p = core.verify(args.root, require_smoke=False)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("real CUDA preflight required")
    precision = core.configure_encoder_replay(args.device)
    storage = Path(p["storage"])
    probe = storage / ".gpu_access_probe.h5"
    with h5py.File(probe, "x") as h:
        h["values"] = np.arange(100)
    with h5py.File(probe, "r") as h:
        np.testing.assert_array_equal(h["values"][:], np.arange(100))
    probe.unlink()
    module, _ = load_universal_checkpoint(paths.LEARNED_TOKENIZER_FULL, torch.device(args.device))
    encoder = module.tokenizer.eval().requires_grad_(False)
    before = core.state_hash(encoder)
    with h5py.File(p["base_cache"], "r") as h:
        manifest = json.loads(h.attrs["manifest"])
        if manifest["cache_id"] != p["base_cache_id"]:
            raise ValueError("base cache changed")
        if TokenizerConfig(**manifest["config"]).effective_receptive_samples != 960:
            raise ValueError("runtime history does not match the actual encoder")
        raw = torch.from_numpy(h["train/raw"][:8]).to(args.device)
        descriptors = torch.from_numpy(h["descriptors"][:]).to(args.device)
        centre, scale = [torch.from_numpy(h["train/" + k][:2]).to(args.device) for k in ("centre", "scale")]
        with torch.no_grad():
            encoded = encoder(raw, sensor="wrist", quantize=True, include_spatial=False)
            np.testing.assert_array_equal(encoded.tokens[:, :, -100:].cpu().numpy(), h["train/tokens"][:8])
            np.testing.assert_allclose(encoded.continuous[:, :, -100:].cpu().numpy(),
                                       h["train/continuous"][:8].astype(np.float32), atol=6e-5, rtol=6e-4)
            canonical = encoded
            runtime_precision = core.configure_runtime_precision()
            encoded = encoder(raw, sensor="wrist", quantize=True, include_spatial=False)
            portability = {
                "tokens": int(encoded.tokens[:, :, -100:].numel()),
                "canonical_cache_id_mismatches": int((encoded.tokens[:, :, -100:] != canonical.tokens[:, :, -100:]).sum()),
                "continuous_max_abs_difference": float((encoded.continuous-canonical.continuous).abs().max()),
                "scope": "first eight training windows; float32 runtime versus canonical TF32 extraction",
            }
            for j in (0, 50, 99):
                end = 2039 + 40*j
                online = encoder(raw[..., end-959:end+1], sensor="wrist", quantize=True, include_spatial=False)
                torch.testing.assert_close(online.continuous[:, :, -1], encoded.continuous[:, :, -100+j], atol=3e-4, rtol=3e-4)
                torch.testing.assert_close(online.tokens[:, :, -1], encoded.tokens[:, :, -100+j], rtol=0, atol=0)
    if before != core.state_hash(encoder):
        raise ValueError("frozen encoder mutated")
    head_timings = {}
    for dimension in (4, 67, 128, 256):
        head = CachedPoseDecoder(dimension, 14, 16, kind="transformer").to(args.device)
        values = torch.randn(2, 16, 100, dimension, device=args.device)
        start = time.monotonic()
        output = head(values, descriptors, centre, scale)
        output.square().mean().backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in head.parameters()):
            raise ValueError("nonfinite head gradients")
        sync(args.device)
        head_timings[str(dimension)] = time.monotonic()-start
        head.eval()
        with torch.no_grad():
            stream = CachedReadout(head, descriptors, centre, scale)
            actual = torch.cat([stream.push(values[:, :, i:i+1]) for i in range(100)], 1)
            expected = head(values, descriptors, centre, scale)
            torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    del head, values, encoder, module
    # The codec will train on V100 using full-precision convolutions.
    torch.backends.cudnn.allow_tf32 = False
    model = codec.build_model().to(args.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005, betas=(.5, .9))
    sources, _ = codec.source_setup(p)
    rng = np.random.default_rng(260939)
    times, loss_rows, loads = [], [], []
    for source in sources:
        for repeat in range(2):
            started = time.monotonic()
            batch = mixed.sample_source_batch(source, 2, TokenizerConfig(), 4000, rng, torch.device(args.device))
            normalized, _, _ = calibration_normalize(batch["emg"], 1000)
            flat = normalized.reshape(-1, 4000)
            choice = rng.choice(len(flat), 32, replace=len(flat) < 32)
            signal = codec.downsample(flat[choice][:, None])
            sync(args.device)
            loaded = time.monotonic()
            optimizer.zero_grad()
            output, commitment, _ = model(signal)
            loss, _ = codec.original_nonamp_loss(signal, output, commitment, args.device)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite real-source codec loss")
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise ValueError("nonfinite real-source codec gradient")
            optimizer.step()
            sync(args.device)
            elapsed = time.monotonic()-loaded
            times.append(elapsed)
            loads.append({"source": source.name, "repeat": repeat, "seconds": loaded-started})
            loss_rows.append({"source": source.name, "repeat": repeat, "loss": float(loss)})
            print(json.dumps({"smoke_source": source.name, "step_seconds": elapsed}), flush=True)
    model.eval().requires_grad_(False)
    before = core.state_hash(model)
    with torch.no_grad():
        full = model.encoder(codec.downsample(raw[:1]))
        for end in (2039, 3039, 5999):
            index = codec.frame_indices([end]).item()
            prefix = model.encoder(codec.downsample(raw[:1, :, :end+1]))
            torch.testing.assert_close(full[..., :index+1], prefix[..., :index+1], atol=3e-4, rtol=3e-4)
        codes = codec.encode_at(model, raw[:1], [2039, 3039, 5999])
        for depth in (1, 3, 6):
            vectors = codec.decode_codes(model, codes, depth)
            assert vectors.shape == (1, 16, 3, 128) and torch.isfinite(vectors).all()
    if before != core.state_hash(model):
        raise ValueError("codec evaluation mutates codebook")
    # GPU smoke weights are discarded; fresh scientific seeds start from step0.
    core.verify(args.root, require_smoke=False)
    pose.write_json(args.root / "smoke.json", {
        "complete": True, "protocol_sha256": pose.digest(args.root / "protocol.json"),
        "vault_gpu_read_write": True, "original_ids_replayed": True,
        "continuous_fp32_replayed": True, "causal_encoder_and_head_parity": True,
        "codec_causal_frame_availability": True, "codec_frozen_eval": True,
        "real_source_losses": loss_rows, "source_load_timings": loads,
        "codec_step_seconds": times, "head_forward_backward_seconds": head_timings,
        "canonical_encoder_precision": precision,
        "runtime_precision": runtime_precision,
        "canonical_to_runtime_portability": portability,
        "codec_timing_scope": "RTX3080 with TF32 disabled; timing is not a measured V100 training ETA",
        "codec_compute_hours_40k_excluding_loading_and_extraction": float(np.median(times[1:])*40000/3600),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__), "discarded_smoke_weights": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)
