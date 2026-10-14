"""Staged round-robin tokenizer pretraining on HD-EMG and emg2pose training windows."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import paths
import torch

from myoji.data import EMG2POSE_MINI_SPLITS, EMG2PoseWindowDataset
from myoji.learned import (
    TokenizerConfig,
    UniversalEMGTokenizerModule,
    atomic_torch_save,
    build_default_tokenizer,
    calibration_normalize,
    last_checkpoint_path,
    restore_training_state,
    sample_calibration_latents,
    training_state,
)
from myoji.learned.sources import (
    DPosePickleBatchSource,
    EMG2PoseBatchSource,
    FrozenEMG2PoseSource,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emg2pose-root", type=Path, default=paths.EMG2POSE)
    parser.add_argument("--emg2pose-metadata", type=Path)
    parser.add_argument("--emg2pose-window-cache", type=Path)
    parser.add_argument(
        "--emg2pose-split-protocol",
        choices=("official", "legacy-mini"),
        default="official",
    )
    parser.add_argument("--dpose-dir", type=Path, default=paths.HDEMG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--continuous-steps", type=int, default=40)
    parser.add_argument("--calibration-steps", type=int, default=12)
    parser.add_argument("--quantized-steps", type=int, default=40)
    parser.add_argument("--calibration-tokens", type=int, default=65_536)
    parser.add_argument("--codebook-size", type=int, default=1_024)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--codebook-update-tokens", type=int, default=4_096)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--spectral-bins", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--usage-weight", type=float, default=0.1)
    parser.add_argument("--usage-target-fraction", type=float, default=0.25)
    parser.add_argument("--usage-temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume a run from a periodic .last.pt training checkpoint",
    )
    parser.add_argument("--dpose-batches-per-recording", type=int, default=1)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_sources(args: argparse.Namespace, window_samples: int):
    sources = []
    emg2pose_root = args.emg2pose_root
    nested_mini = emg2pose_root / "emg2pose_dataset_mini"
    if nested_mini.is_dir():
        emg2pose_root = nested_mini
    metadata_candidates = (
        [args.emg2pose_metadata]
        if args.emg2pose_metadata is not None
        else [
            emg2pose_root / "metadata.csv",
            emg2pose_root.parent / "emg2pose_metadata.csv",
        ]
    )
    metadata = next((path for path in metadata_candidates if path.is_file()), None)
    if args.emg2pose_window_cache is not None:
        if args.emg2pose_metadata is None:
            raise ValueError("frozen windows require official metadata")
        sources.append(
            FrozenEMG2PoseSource(args.emg2pose_window_cache, args.emg2pose_metadata)
        )
    elif emg2pose_root.is_dir() and metadata is not None:
        dataset = EMG2PoseWindowDataset(
            emg2pose_root,
            metadata,
            split="train",
            filenames=EMG2POSE_MINI_SPLITS["train"]
            if args.emg2pose_split_protocol == "legacy-mini"
            else None,
            window_samples=window_samples,
            stride_samples=window_samples,
            jitter=True,
            seed=args.seed,
        )
        sources.append(EMG2PoseBatchSource(dataset))

    dpose_paths = sorted(args.dpose_dir.glob("Sub*_emg.pkl"))
    if dpose_paths:
        sources.append(
            DPosePickleBatchSource(
                dpose_paths,
                batches_per_participant=args.dpose_batches_per_recording,
            )
        )

    if not sources:
        raise FileNotFoundError(
            "none of the configured mixed EMG sources are available"
        )
    return sources


def source_batch_size(source, requested: int) -> int:
    return 1 if source.montage.channels >= 64 else requested


def sample_source_batch(
    source,
    requested_batch_size: int,
    config: TokenizerConfig,
    window_samples: int,
    rng: np.random.Generator,
    device: torch.device,
):
    batch = source.sample_batch(
        source_batch_size(source, requested_batch_size),
        config.sample_rate_hz,
        window_samples,
        rng,
    )
    for name, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[name] = value.to(device, non_blocking=True)
    return batch


@torch.no_grad()
def continuous_latents(
    module: UniversalEMGTokenizerModule,
    batch,
) -> torch.Tensor:
    normalized, _, _ = calibration_normalize(batch["emg"], module.calibration_samples)
    return module.tokenizer(
        normalized,
        sensor=batch["sensor"],
        channel_descriptors=batch["channel_descriptors"],
        quantize=False,
        include_spatial=False,
    ).continuous


def train_steps(
    module: UniversalEMGTokenizerModule,
    optimizer: torch.optim.Optimizer,
    sources,
    steps: int,
    phase: str,
    args: argparse.Namespace,
    config: TokenizerConfig,
    window_samples: int,
    rng: np.random.Generator,
    device: torch.device,
    history: list[dict[str, object]],
    start_step: int = 0,
    checkpoint_callback: Callable[[str, int], None] | None = None,
) -> None:
    module.use_quantization = phase == "quantized"
    module.train()
    for local_step in range(start_step, steps):
        source = sources[local_step % len(sources)]
        batch = sample_source_batch(
            source,
            args.batch_size,
            config,
            window_samples,
            rng,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics = module.compute_loss(batch)
        metrics["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 10.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {
            "phase": phase,
            "step": local_step,
            "source": source.name,
            "channels": source.montage.channels,
            "batch_size": int(batch["emg"].shape[0]),
            "loss": float(metrics["loss"].detach()),
            "spectral_loss": float(metrics["spectral_loss"].detach()),
            "commitment_loss": float(metrics["commitment_loss"].detach()),
            "usage_loss": float(metrics["usage_loss"].detach()),
            "soft_perplexity": float(metrics["soft_perplexity"].detach()),
            "perplexity": float(metrics["perplexity"].detach()),
            "active_fraction": float(metrics["active_fraction"].detach()),
            "gradient_norm": float(gradient_norm),
            "step_ms": 1_000 * (time.perf_counter() - started),
        }
        should_report = (
            local_step < len(sources)
            or local_step == steps - 1
            or local_step % args.report_every < len(sources)
        )
        if should_report:
            history.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        completed_steps = local_step + 1
        if checkpoint_callback is not None:
            checkpoint_callback(phase, completed_steps)


@torch.no_grad()
def evaluate_codebook(
    module,
    sources,
    args,
    config,
    window_samples,
    rng,
    device,
) -> dict[str, object]:
    module.eval()
    counts = torch.zeros(config.codebook_size, device=device)
    by_source = {}
    for source in sources:
        batch = sample_source_batch(
            source,
            args.batch_size,
            config,
            window_samples,
            rng,
            device,
        )
        normalized, _, _ = calibration_normalize(
            batch["emg"], module.calibration_samples
        )
        output = module.tokenizer(
            normalized,
            sensor=batch["sensor"],
            channel_descriptors=batch["channel_descriptors"],
            include_spatial=False,
        )
        source_counts = torch.bincount(
            output.tokens.reshape(-1), minlength=config.codebook_size
        )
        counts += source_counts
        by_source[source.name] = {
            "perplexity": float(output.perplexity),
            "active_codes": int(torch.count_nonzero(source_counts)),
            "tokens": int(output.tokens.numel()),
        }
    probabilities = counts / counts.sum().clamp_min(1)
    occupied = probabilities > 0
    return {
        "active_codes": int(torch.count_nonzero(counts)),
        "active_fraction": float(occupied.float().mean()),
        "perplexity": float(
            (-(probabilities[occupied] * probabilities[occupied].log()).sum()).exp()
        ),
        "tokens": int(counts.sum()),
        "by_source": by_source,
    }


def main() -> None:
    args = parse_args()
    if min(args.continuous_steps, args.calibration_steps, args.quantized_steps) <= 0:
        raise ValueError("every training stage needs at least one step")
    positive_controls = (
        args.report_every,
        args.checkpoint_every,
        args.dpose_batches_per_recording,
    )
    if min(positive_controls) <= 0:
        raise ValueError(
            "report, checkpoint, and source reuse controls must be positive"
        )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    window_samples = round(args.window_seconds * 2_000)
    config = TokenizerConfig(
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
        ema_decay=args.ema_decay,
        codebook_update_tokens=args.codebook_update_tokens,
    )
    sources = build_sources(args, window_samples)
    from myoji.learned.stream import sha256

    data_contract = {
        "split_protocol": args.emg2pose_split_protocol,
        "window_cache_sha256": sha256(args.emg2pose_window_cache)
        if args.emg2pose_window_cache
        else None,
        "metadata_sha256": sha256(args.emg2pose_metadata)
        if args.emg2pose_metadata
        else None,
        "sources": [
            {"name": source.name, "available_recordings": source.available_recordings}
            for source in sources
        ],
    }
    module = UniversalEMGTokenizerModule(
        build_default_tokenizer(config),
        spectral_bins=args.spectral_bins,
        learning_rate=args.learning_rate,
        usage_weight=args.usage_weight,
        usage_target_fraction=args.usage_target_fraction,
        usage_temperature=args.usage_temperature,
    ).to(device)
    optimizer = module.configure_optimizers()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress: dict[str, object] = {
        "phase": "continuous",
        "continuous_step": 0,
        "quantized_step": 0,
    }
    history: list[dict[str, object]] = []
    if args.resume is not None:
        saved_contract = torch.load(
            args.resume, map_location="cpu", weights_only=False
        ).get("data_contract")
        if saved_contract != data_contract:
            raise ValueError(
                "resume requires an identical recorded training-data contract"
            )
        progress, history = restore_training_state(
            args.resume,
            module,
            optimizer,
            config,
            sources,
            rng,
            device,
        )
        if (
            progress.get("phase") == "complete"
            and int(progress.get("quantized_step", 0)) < args.quantized_steps
        ):
            progress["phase"] = "quantized"

    state_path = last_checkpoint_path(args.output)

    def save_last() -> None:
        state = training_state(
            module,
            optimizer,
            config,
            sources,
            progress,
            history,
            rng,
            device,
        )
        state["data_contract"] = data_contract
        state["training_args"] = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        atomic_torch_save(state, state_path)

    def checkpoint_callback(phase: str, completed_steps: int) -> None:
        progress[f"{phase}_step"] = completed_steps
        if completed_steps % args.checkpoint_every == 0:
            save_last()
            print(
                json.dumps(
                    {
                        "phase": "checkpoint",
                        "training_phase": phase,
                        "completed_steps": completed_steps,
                        "path": str(state_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if progress.get("phase") == "continuous":
        train_steps(
            module,
            optimizer,
            sources,
            args.continuous_steps,
            "continuous",
            args,
            config,
            window_samples,
            rng,
            device,
            history,
            start_step=int(progress.get("continuous_step", 0)),
            checkpoint_callback=checkpoint_callback,
        )
        progress["continuous_step"] = args.continuous_steps
        progress["phase"] = "calibration"
        save_last()

    if progress.get("phase") == "calibration":
        reservoir = []
        for step in range(args.calibration_steps):
            source = sources[step % len(sources)]
            batch = sample_source_batch(
                source,
                args.batch_size,
                config,
                window_samples,
                rng,
                device,
            )
            reservoir.append(
                sample_calibration_latents(
                    continuous_latents(module, batch),
                    total_tokens=args.calibration_tokens,
                    calibration_steps=args.calibration_steps,
                )
            )
        calibration = torch.cat(reservoir)
        if len(calibration) > args.calibration_tokens:
            calibration = calibration[
                torch.randperm(len(calibration), device=device)[
                    : args.calibration_tokens
                ]
            ]
        module.tokenizer.quantizer.initialize_codebook(calibration)
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "sources": [source.name for source in sources],
                    "tokens": len(calibration),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        progress["phase"] = "quantized"
        save_last()

    if progress.get("phase") == "quantized":
        train_steps(
            module,
            optimizer,
            sources,
            args.quantized_steps,
            "quantized",
            args,
            config,
            window_samples,
            rng,
            device,
            history,
            start_step=int(progress.get("quantized_step", 0)),
            checkpoint_callback=checkpoint_callback,
        )
        progress["quantized_step"] = args.quantized_steps
        progress["phase"] = "complete"
        save_last()

    if progress.get("phase") != "complete":
        raise ValueError(f"unsupported resume phase {progress.get('phase')!r}")
    evaluation = evaluate_codebook(
        module,
        sources,
        args,
        config,
        window_samples,
        rng,
        device,
    )
    summary = {
        "config": asdict(config),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "sources": [
            {
                "name": source.name,
                "channels": source.montage.channels,
                "sample_rate_hz": source.source_rate_hz,
                "seen_recordings": len(source.seen_recordings),
                "available_recordings": source.available_recordings,
            }
            for source in sources
        ],
        "continuous_steps": args.continuous_steps,
        "calibration_steps": args.calibration_steps,
        "quantized_steps": args.quantized_steps,
        "evaluation": evaluation,
        "history": history,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
    }
    final_state = training_state(
        module,
        optimizer,
        config,
        sources,
        progress,
        history,
        rng,
        device,
    )
    final_state["summary"] = summary
    final_state["data_contract"] = data_contract
    final_state["training_args"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    atomic_torch_save(final_state, args.output)
    results_path = args.output.with_suffix(".json")
    results_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "results": str(results_path),
                "evaluation": evaluation,
                "peak_memory_gib": summary["peak_memory_gib"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
