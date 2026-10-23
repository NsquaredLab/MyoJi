"""Pinned BioCodec EMG adaptation; original architecture, loss and quantizer."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from paper.emg2pose import compare_temporal_pose as pose
import h5py
import numpy as np
from paper.emg2pose import paper_completion_core as core
from paper.common import paths
import torch
from torch.nn import functional as F


def upstream():
    # BioCodec (https://github.com/klean2050/BioCodec-release, commit 575bf6e, MIT) is
    # not redistributed: put a clone on PYTHONPATH or point MYOJI_BIOCODEC at it.
    if os.environ.get("MYOJI_BIOCODEC"):
        sys.path.insert(0, os.environ["MYOJI_BIOCODEC"])
    from biocodec.model import BioCodecModel
    return BioCodecModel


def build_model():
    return upstream()._get_emg_model(
        sample_rate=1000, causal=True, signal_normalize=False, segment=None,
        n_q=6, q_bins=256,
    )


def downsample(raw):
    from scipy.signal import firwin
    kernel = torch.as_tensor(firwin(41, .45), device=raw.device, dtype=raw.dtype).flip(0)
    flat = raw.reshape(-1, 1, raw.shape[-1])
    return F.conv1d(F.pad(flat, (40, 0)), kernel[None, None], stride=2)


def frame_indices(ends):
    # Downsample sample i arrives at original sample 2*i. Encoder stride is18.
    indices = ((torch.as_tensor(ends, dtype=torch.long) // 2 + 1) // 18) - 1
    if torch.any(indices < 0):
        raise ValueError("no fully observed codec frame")
    return indices


@torch.inference_mode()
def encode_at(model, normalized_raw, ends):
    if model.training:
        raise ValueError("code extraction requires a frozen evaluation model")
    sampled = downsample(normalized_raw)
    codes = model.encode(sampled)[0][0]
    ix = frame_indices(ends).to(codes.device)
    selected = codes.index_select(-1, ix)
    return selected.reshape(*normalized_raw.shape[:2], 6, len(ix))


@torch.inference_mode()
def decode_codes(model, codes, depth):
    if not 1 <= depth <= 6:
        raise ValueError("invalid RVQ depth")
    b, c, _, t = codes.shape
    flat = codes[:, :, :depth].reshape(b*c, depth, t).permute(1, 0, 2)
    vectors = model.quantizer.decode(flat)
    return vectors.transpose(1, 2).reshape(b, c, t, 128)


def original_nonamp_loss(signal_in, signal_out, commitment, device):
    from biocodec.objective import total_loss
    losses = total_loss(signal_in, signal_out, sr=1000, is_emg=True, device=device)
    # Equivalent to upstream loss_g.backward(retain_graph=True); loss_w.backward().
    return .2 * losses["l_t"] + losses["l_f"] + 2 * commitment.mean(), losses


def source_setup(p):
    from paper.emg2pose import run_tokenizer_seeds as seeds
    from paper.pretraining import train_mixed_sources as mixed
    parent = json.loads((paths.RESULTS / "tokenizer_seeds_v1/protocol.json").read_text())
    if seeds.catalogue(parent["training_arguments"]) != parent["mixed_source_catalogue"]:
        raise ValueError("mixed-source files changed")
    values = parent["training_arguments"].copy()
    for key in ("emg2pose_root", "emg2pose_metadata", "emg2pose_window_cache", "hdemg_dir"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    if pose.digest(values["emg2pose_window_cache"]) != parent["data_contract"]["window_cache_sha256"]:
        raise ValueError("pretraining cache changed")
    args = SimpleNamespace(**values)
    # The public reference checkpoint records no per-source list; the file pool is pinned by the
    # seed campaign's catalogue and window-cache hash checked above.
    sources = mixed.build_sources(args, 4000)
    track_sources(sources)
    return sources, parent


def track_sources(sources):
    """Record cache identities without serializing multi-gigabyte recordings."""
    for source in sources:
        source._completion_current = None
        for name in ("_next_path", "_next_session"):
            if not hasattr(source, name):
                continue
            original = getattr(source, name)

            def choose(rng, original=original, source=source):
                value = original(rng)
                source._completion_current = value
                return value

            setattr(source, name, choose)


def source_states(sources):
    return [{"name": s.name, "current": s._completion_current,
             "seen": sorted(s.seen_recordings),
             "uses": getattr(s, "_cache_uses", None),
             "paths": getattr(s, "_path_cycle", None),
             "sessions": getattr(s, "_session_cycle", None)} for s in sources]


def restore_sources(sources, states):
    for source, state in zip(sources, states, strict=True):
        if source.name != state["name"]:
            raise ValueError("resume source order changed")
        current = state["current"]
        if current is not None:
            name = "_next_session" if hasattr(source, "_next_session") else "_next_path"
            original = getattr(source, name)
            setattr(source, name, lambda rng, value=current: value)
            try:
                # Reload the exact cached file without consuming the training RNG.
                if name == "_next_session":
                    source._load_session(np.random.default_rng(0))
                else:
                    source._recording(np.random.default_rng(0))
            finally:
                setattr(source, name, original)
            source._completion_current = current
            source._cache_uses = state["uses"]
        if state["paths"] is not None:
            source._path_cycle = state["paths"]
        if state["sessions"] is not None:
            source._session_cycle = state["sessions"]
        source.seen_recordings = set(state["seen"])


@torch.inference_mode()
def reconstruction_check(model, p, device):
    from biocodec.objective import total_loss
    # Fixed first16 validation windows; no labels, test input or downstream selection.
    with h5py.File(p["base_cache"], "r") as h:
        raw = torch.from_numpy(h["val/raw"][:16, :, :4000]).to(device)
    values = downsample(raw)
    sums, n = {}, 0
    model.eval()
    for x in values.split(32):
        output = model(x)
        losses = total_loss(x, output, sr=1000, is_emg=True, device=device)
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.) + float(value) * len(x)
        n += len(x)
    model.train()
    return {k: v/n for k, v in sums.items()}


def train(args):
    from paper.pretraining import train_mixed_sources as mixed

    from myoji.learned import TokenizerConfig, calibration_normalize
    p = core.verify(args.root)
    seed = p["biocodec"]["seeds"][args.index]
    folder = args.root / f"biocodec/seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "last.pt"
    if path.exists() and not args.resume:
        raise FileExistsError("codec training already started; explicit recovery required")
    if (folder / "training_receipt.json").exists():
        raise FileExistsError("codec training already completed")
    torch.manual_seed(seed)
    model = build_model().to(args.device).train()
    from biocodec.scheduler import WarmupCosineLrScheduler
    opt = torch.optim.Adam(model.parameters(), lr=.0005, betas=(.5, .9))
    scheduler = WarmupCosineLrScheduler(opt, max_iter=40000, warmup_iter=10000,
                                      warmup_ratio=5e-4, warmup="linear")
    sources, parent = source_setup(p)
    rng = np.random.default_rng(seed)
    config = TokenizerConfig()
    step, history, diagnostics, duration, exposures = 0, [], [], 0., {}
    if args.resume:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["protocol_sha256"] != pose.digest(args.root / "protocol.json") or state["seed"] != seed:
            raise ValueError("codec resume identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_sources(sources, state["source_state"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, history, diagnostics = state["step"], state["history"], state["diagnostics"]
        duration, exposures = state["seconds"], state["exposures"]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    start_step = step + 1
    for step in range(start_step, 40001):
        source = sources[(step - 1) % len(sources)]
        batch = mixed.sample_source_batch(source, 2, config, 4000, rng, torch.device(args.device))
        normalized, _, _ = calibration_normalize(batch["emg"], 1000)
        flat = normalized.reshape(-1, 4000)
        choices = rng.choice(len(flat), 32, replace=len(flat) < 32)
        x = downsample(flat[choices][:, None])
        opt.zero_grad()
        output, penalty, _ = model(x)
        loss, metrics = original_nonamp_loss(x, output, penalty, args.device)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite codec objective")
        loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
            raise ValueError("nonfinite codec gradients")
        opt.step()
        scheduler.step()
        exposures[source.name] = exposures.get(source.name, 0) + 32
        if step % 250 == 0:
            row = {"step": step, "loss": float(loss), "time_loss": float(metrics["l_t"]),
                   "frequency_loss": float(metrics["l_f"]), "commitment": float(penalty.mean()),
                   "elapsed_seconds": duration + time.monotonic() - started}
            history.append(row)
            print(json.dumps({"codec_seed": seed, **row}), flush=True)
        if step in (20000, 40000):
            diagnostics.append({"step": step, "validation": reconstruction_check(model, p, args.device)})
        if step % 2000 == 0 or step == 40000:
            saved = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "seed": seed, "history": history, "diagnostics": diagnostics,
                "protocol_sha256": pose.digest(args.root / "protocol.json"),
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": rng.bit_generator.state,
                "source_state": source_states(sources),
                "exposures": exposures, "seconds": duration + time.monotonic() - started,
            }
            pose.atomic_torch_save(saved, path)
    core.verify(args.root)
    source_setup(p)  # Recheck source catalogue/content after fitting.
    final = folder / "model.pt"
    pose.atomic_torch_save({
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "seed": seed, "steps": 40000,
        "protocol_sha256": pose.digest(args.root / "protocol.json"),
        "source_seen_recordings": {s.name: sorted(s.seen_recordings) for s in sources},
    }, final)
    pose.write_json(folder / "training_receipt.json", {
        "complete": True, "steps": 40000, "seed": seed,
        "protocol_sha256": pose.digest(args.root / "protocol.json"),
        "model_sha256": pose.digest(final), "exposures": exposures,
        "parameters": sum(v.numel() for v in model.parameters()),
        "seconds": duration + time.monotonic() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "history": history, "unlabeled_validation_diagnostics": diagnostics,
        "same_file_pool": parent["data_contract"],
    })
    prepare(args, p, seed)


@torch.inference_mode()
def prepare(args, p, seed):
    folder = args.root / f"biocodec/seed_{seed}"
    receipt = json.loads((folder / "training_receipt.json").read_text())
    if not receipt["complete"] or receipt["model_sha256"] != pose.digest(folder / "model.pt"):
        raise ValueError("unverified codec weights")
    model = build_model().to(args.device).eval().requires_grad_(False)
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    before = core.state_hash(model)
    output = Path(p["storage"]) / f"biocodec_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    depths = (1, 3, 6) if seed == 260940 else (6,)
    with h5py.File(p["base_cache"], "r") as source:
        handles = {}
        try:
            for depth in depths:
                h = h5py.File(output / f"official_q{depth}.h5", "x")
                h.attrs["complete"] = False
                handles[depth] = h
                for key in ("codebook", "descriptors"):
                    h[key] = h5py.ExternalLink(p["base_cache"], "/" + key)
                for split in ("train", "val", "test"):
                    original, g = source[split], h.create_group(split)
                    for key in original:
                        if key not in ("continuous", "tokens"):
                            g[key] = h5py.ExternalLink(p["base_cache"], f"/{split}/{key}")
                    n = len(original["tokens"])
                    g.create_dataset("continuous", shape=(n, 16, 100, 128), dtype="<f4")
                    g.create_dataset("rvq_ids", shape=(n, 16, depth, 100), dtype="u1")
            ends = 2039 + 40 * torch.arange(100)
            for split in ("train", "val", "test"):
                for start in range(0, len(source[split]["raw"]), 8):
                    raw = torch.from_numpy(source[split]["raw"][start:start+8]).to(args.device)
                    codes = encode_at(model, raw, ends)
                    for depth, out in handles.items():
                        vectors = decode_codes(model, codes, depth)
                        out[split]["continuous"][start:start+len(raw)] = vectors.cpu().numpy()
                        out[split]["rvq_ids"][start:start+len(raw)] = codes[:, :, :depth].cpu().numpy()
                print(json.dumps({"codec_seed": seed, "prepared": split}), flush=True)
            for depth, h in handles.items():
                core.finish_cache(h, json.loads(source.attrs["manifest"]),
                                  representation=f"biocodec_rvq{depth}_quantized_vectors",
                                  codec_seed=seed, rvq_depth=depth, embedding_dim=128,
                                  checkpoint_sha256=receipt["model_sha256"],
                                  completion_protocol_sha256=pose.digest(args.root / "protocol.json"),
                                  tokens_are_rvq_ids=True)
        finally:
            for h in handles.values():
                h.close()
    selection = paths.RESULTS / "meta_sessions_v1/features.h5"
    for depth in depths:
        parent = output / f"official_q{depth}.h5"
        with h5py.File(selection, "r") as sel, h5py.File(parent, "r") as source, h5py.File(output / f"session_q{depth}.h5", "x") as h:
            for key in ("codebook", "descriptors"):
                h[key] = h5py.ExternalLink(str(parent), "/" + key)
            for split, origin in (("train", "train"), ("val", "val"), ("test", "train")):
                g = h.create_group(split)
                for key in source[origin]:
                    if key != "unique_indices":
                        g[key] = h5py.ExternalLink(str(parent), f"/{origin}/{key}")
                g["unique_indices"] = sel[split]["unique_indices"][:]
            core.finish_cache(h, json.loads(sel.attrs["manifest"]),
                              representation=f"biocodec_rvq{depth}_quantized_vectors",
                              codec_seed=seed, rvq_depth=depth, embedding_dim=128,
                              checkpoint_sha256=receipt["model_sha256"],
                              completion_protocol_sha256=pose.digest(args.root / "protocol.json"),
                              tokens_are_rvq_ids=True)
    if core.state_hash(model) != before:
        raise ValueError("codec state changed during extraction")
    pose.write_json(folder / "cache_receipt.json", {
        "complete": True, "model_unchanged": True,
        "protocol_sha256": pose.digest(args.root / "protocol.json"),
        "files": {str(path): pose.digest(path) for path in output.glob("*.h5")},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.prepare_only:
        p = core.verify(args.root)
        prepare(args, p, p["biocodec"]["seeds"][args.index])
    else:
        train(args)


if __name__ == "__main__":
    main()
