"""Independent clean tokenizer seeds, with frozen official and session readouts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import compare_temporal_pose as pose
import h5py
import numpy as np
import paths
import run_meta_sessions as sessions
import torch
import train_mixed_sources as mixed

NEW_SEEDS = (260906, 260907)
HEAD_SEEDS = sessions.SEEDS
SCOPES = ("official", "session")


def catalogue(arguments):
    result = {}
    for key, pattern in (("dpose_dir", "Sub*_emg.pkl"),):
        result[key] = [
            {
                "path": str(p.resolve()),
                "bytes": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in sorted(Path(arguments[key]).glob(pattern))
        ]
        if not result[key]:
            raise ValueError("missing mixed source: " + key)
    return result


def training_command(arguments, output, seed, smoke=False):
    values = {
        **arguments,
        "output": str(output),
        "seed": seed,
        "device": "cuda",
        "resume": None,
    }
    if smoke:
        values.update(
            continuous_steps=40,
            calibration_steps=12,
            quantized_steps=40,
            checkpoint_every=40,
            report_every=40,
        )
    accepted = vars(mixed.parse_args(["--output", str(output)]))
    command = [sys.executable, str(Path(__file__).with_name("train_mixed_sources.py"))]
    for key, value in values.items():
        if value is not None and key in accepted:
            command += ["--" + key.replace("_", "-"), str(value)]
    return command


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError("tokenizer seed campaign already frozen")
    reference = paths.LEARNED_TOKENIZER_FULL
    state = torch.load(reference, map_location="cpu", weights_only=False)
    if state["progress"]["phase"] != "complete" or state["training_args"].get("resume"):
        raise ValueError(
            "expected the original, independently initialized clean tokenizer"
        )
    arguments = state["training_args"].copy()
    arguments["emg2pose_root"] = str(paths.EMG2POSE)
    arguments["emg2pose_metadata"] = str(paths.EMG2POSE / "emg2pose_metadata.csv")
    arguments["emg2pose_window_cache"] = str(paths.RESULTS / "clean_base_v1/train.h5")
    arguments["dpose_dir"] = str(paths.HDEMG)
    if (
        pose.digest(arguments["emg2pose_window_cache"])
        != state["data_contract"]["window_cache_sha256"]
    ):
        raise ValueError("pretraining cache changed")
    if (
        pose.digest(arguments["emg2pose_metadata"])
        != state["data_contract"]["metadata_sha256"]
    ):
        raise ValueError("official metadata changed")
    session_root = paths.RESULTS / "meta_sessions_v1"
    session_protocol = json.loads((session_root / "protocol.json").read_text())
    if session_protocol["pretraining_data_contract"] != state["data_contract"]:
        raise ValueError("independent seeds would expose different session data")
    protocol = {
        "name": "Independent tokenizer seeds v1",
        "reference_seed": arguments["seed"],
        "new_seeds": list(NEW_SEEDS),
        "head_seeds": list(HEAD_SEEDS),
        "reference_checkpoint": str(reference),
        "reference_sha256": pose.digest(reference),
        "reference_gpu": state["summary"]["gpu"],
        "training_arguments": arguments,
        "data_contract": state["data_contract"],
        "mixed_source_catalogue": catalogue(arguments),
        "catalogue_scope": "Paths, sizes and modification times pinned now; historical content hashes were not recorded for these source families. Meta training cache is content-hashed.",
        "base_cache": str(paths.RESULTS / "clean_temporal_v1/full.h5"),
        "session_root": str(session_root),
        "session_protocol_sha256": pose.digest(session_root / "protocol.json"),
        "reference_official_root": str(paths.RESULTS / "reviewer_controls_v1"),
        "scope": "Two fresh 300000-update tokenizers, same recipe, reference pretraining sources and RTX3080 hardware as original. Each gets three fresh readout seeds for official unseen-user/stage and whole-session tests. No pose labels enter tokenizer training.",
        "selection": "Final fixed-step tokenizer, never selected on downstream scores; each decoder selected by its fixed validation macro-user MAE. Test scoring gated on every required fit.",
        "statistics": "Report every tokenizer replicate and participant-paired results. Three tokenizers are a small seed sample; head seeds are averaged within tokenizer and participant, never treated as independent people.",
        "source_sha256": sessions.source_hashes(args.root / "code"),
    }
    pose.write_json(args.root / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "frozen": True,
                "new_seeds": NEW_SEEDS,
                "new_tokenizers": 2,
                "new_decoders": 12,
            }
        )
    )


def verify(root):
    p = json.loads((root / "protocol.json").read_text())
    if sessions.source_hashes(root / "code") != p["source_sha256"]:
        raise ValueError("frozen implementation changed")
    if catalogue(p["training_arguments"]) != p["mixed_source_catalogue"]:
        raise ValueError("mixed-source files changed")
    if (
        pose.digest(Path(p["session_root"]) / "protocol.json")
        != p["session_protocol_sha256"]
    ):
        raise ValueError("session selection changed")
    return p


def check_training_state(path, p, seed, smoke=False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    a = state["training_args"]
    expected = (40, 12, 40) if smoke else (50000, 128, 250000)
    if (
        # The reference records only these contract fields, not a per-source list.
        {k: state["data_contract"].get(k) for k in p["data_contract"]}
        != p["data_contract"]
        or a["seed"] != seed
        or state["progress"]["phase"] != "complete"
        or tuple(
            a[k] for k in ("continuous_steps", "calibration_steps", "quantized_steps")
        )
        != expected
        or state["summary"]["gpu"] != p["reference_gpu"]
    ):
        raise ValueError("tokenizer training recipe, completion or hardware mismatch")
    return state


def smoke(args):
    p = verify(args.root)
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != p["reference_gpu"]
    ):
        raise ValueError("replication requires the original RTX3080 GPU model")
    output = args.root / "smoke/tokenizer.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(
        training_command(p["training_arguments"], output, 260908, smoke=True),
        check=True,
    )
    check_training_state(output, p, 260908, smoke=True)
    pose.write_json(
        args.root / "smoke.json",
        {
            "passed": True,
            "weights_discarded_from_scientific_fits": True,
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
        },
    )


def train(args):
    p = verify(args.root)
    receipt = json.loads((args.root / "smoke.json").read_text())
    if not receipt["passed"] or receipt["protocol_sha256"] != pose.digest(
        args.root / "protocol.json"
    ):
        raise ValueError("successful preflight required")
    if torch.cuda.get_device_name() != p["reference_gpu"]:
        raise ValueError("different training GPU model")
    seed = NEW_SEEDS[args.index]
    directory = args.root / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "tokenizer.pt"
    if output.exists() or output.with_name("tokenizer.last.pt").exists():
        raise FileExistsError("existing tokenizer requires explicit recovery")
    subprocess.run(training_command(p["training_arguments"], output, seed), check=True)
    state = check_training_state(output, p, seed)
    verify(args.root)
    pose.write_json(
        directory / "training_receipt.json",
        {
            "complete": True,
            "tokenizer_seed": seed,
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "checkpoint_sha256": pose.digest(output),
            "torch": str(torch.__version__),
            "gpu": state["summary"]["gpu"],
            "progress": state["progress"],
            "data_contract": state["data_contract"],
        },
    )
    from myoji.learned import export_checkpoint

    export_checkpoint(output, directory / "release.pt", calibration_samples=1000)
    prepare(args, p, seed)


@torch.inference_mode()
def prepare(args, p, seed):
    from myoji.learned import load_universal_checkpoint

    directory = args.root / f"seed_{seed}"
    model, _ = load_universal_checkpoint(
        directory / "tokenizer.pt", torch.device(args.device)
    )
    model.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.tokenizer.state_dict().items()}
    with h5py.File(p["base_cache"], "r") as source:
        parent = json.loads(source.attrs["manifest"])
        # Check the normalized-input path against original cached IDs, before
        # generating any new features. Never normalize cached raw EMG twice.
        original, _ = load_universal_checkpoint(
            Path(p["reference_checkpoint"]), torch.device(args.device)
        )
        original.eval().requires_grad_(False)
        x = torch.from_numpy(source["train/raw"][:2]).to(args.device)
        replay = original.tokenizer(
            x, sensor="wrist", quantize=True, include_spatial=False
        )
        np.testing.assert_array_equal(
            replay.tokens[:, :, -100:].cpu().numpy(), source["train/tokens"][:2]
        )
        del original
        with h5py.File(directory / "official.h5", "x") as out:
            out.attrs["complete"] = False
            # Store only IDs and the new lookup table; all raw data, targets and
            # calibration arrays remain external links to the existing cache.
            out["codebook"] = model.tokenizer.quantizer.codebook.detach().cpu().numpy()
            out["descriptors"] = h5py.ExternalLink(p["base_cache"], "/descriptors")
            for split in ("train", "val", "test"):
                group = out.create_group(split)
                for key in source[split]:
                    if key not in ("tokens", "continuous"):
                        group[key] = h5py.ExternalLink(
                            p["base_cache"], f"/{split}/{key}"
                        )
                shape = source[f"{split}/tokens"].shape
                target = group.create_dataset("tokens", shape=shape, dtype="<u2")
                for start in range(0, shape[0], 8):
                    raw = torch.from_numpy(
                        source[f"{split}/raw"][start : start + 8]
                    ).to(args.device)
                    encoded = model.tokenizer(
                        raw, sensor="wrist", quantize=True, include_spatial=False
                    )
                    target[start : start + len(raw)] = (
                        encoded.tokens[:, :, -100:].cpu().numpy().astype(np.uint16)
                    )
                print(
                    json.dumps({"prepared": split, "tokenizer_seed": seed}), flush=True
                )
            manifest = dict(parent)
            manifest.update(
                parent_cache_id=parent["cache_id"],
                tokenizer_seed=seed,
                checkpoint_sha256=pose.digest(directory / "tokenizer.pt"),
                checkpoint=str(directory / "tokenizer.pt"),
                representation="tokens_only_no_continuous_array",
                seed_protocol_sha256=pose.digest(args.root / "protocol.json"),
            )
            manifest.pop("cache_id")
            manifest["cache_id"] = sessions.json_hash(manifest)
            out.attrs["manifest"] = json.dumps(manifest)
            out.attrs["complete"] = True
    if any(
        not torch.equal(v, model.tokenizer.state_dict()[k]) for k, v in before.items()
    ):
        raise ValueError("frozen tokenizer changed during extraction")
    with (
        h5py.File(Path(p["session_root"]) / "features.h5", "r") as selection,
        h5py.File(directory / "session.h5", "x") as out,
    ):
        for key in ("codebook", "descriptors"):
            out[key] = h5py.ExternalLink(str(directory / "official.h5"), "/" + key)
        for split, origin in (
            ("train", "train"),
            ("val", "val"),
            ("test", "train"),
        ):
            group = out.create_group(split)
            with h5py.File(directory / "official.h5", "r") as source:
                for key in source[origin]:
                    if key != "unique_indices":
                        group[key] = h5py.ExternalLink(
                            str(directory / "official.h5"), f"/{origin}/{key}"
                        )
            group["unique_indices"] = selection[f"{split}/unique_indices"][:]
        meta = json.loads(selection.attrs["manifest"])
        meta.update(
            tokenizer_seed=seed,
            checkpoint_sha256=manifest["checkpoint_sha256"],
            checkpoint=manifest["checkpoint"],
            representation="tokens_only_no_continuous_array",
            seed_protocol_sha256=pose.digest(args.root / "protocol.json"),
        )
        meta.pop("cache_id")
        meta["cache_id"] = sessions.json_hash(meta)
        out.attrs["manifest"] = json.dumps(meta)
        out.attrs["complete"] = True
    pose.write_json(
        directory / "cache_receipt.json",
        {
            "complete": True,
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "tokenizer_unchanged": True,
            "original_training_ids_replayed_exactly": True,
            "official_sha256": pose.digest(directory / "official.h5"),
            "session_sha256": pose.digest(directory / "session.h5"),
        },
    )


def fit(args):
    verify(args.root)
    tokenizer_seed = NEW_SEEDS[args.index // 6]
    scope = SCOPES[(args.index // 3) % 2]
    seed = HEAD_SEEDS[args.index % 3]
    directory = args.root / f"seed_{tokenizer_seed}"
    receipt = json.loads((directory / "cache_receipt.json").read_text())
    if (
        not receipt["complete"]
        or receipt["protocol_sha256"] != pose.digest(args.root / "protocol.json")
        or receipt[scope + "_sha256"] != pose.digest(directory / f"{scope}.h5")
    ):
        raise ValueError("feature provenance mismatch")
    config = pose.parse_args(
        [
            "fit",
            "--arm",
            "token_transformer",
            "--experiment-label",
            "tokens_cal",
            "--cache",
            str(directory / f"{scope}.h5"),
            "--seed",
            str(seed),
            "--output-dir",
            str(directory / scope / "fits" / f"tokens_cal_{seed}"),
            "--device",
            args.device,
        ]
    )
    config.tokenizer_seed = tokenizer_seed
    config.seed_protocol_sha256 = pose.digest(args.root / "protocol.json")
    pose.run_fit(config)


@torch.inference_mode()
def evaluate(args):
    p = verify(args.root)
    if (args.root / "results.json").exists():
        raise FileExistsError("replicate evaluation already exists")
    plans = []
    for tokenizer_seed in (p["reference_seed"], *NEW_SEEDS):
        for scope in SCOPES:
            if tokenizer_seed == p["reference_seed"]:
                root = Path(
                    p["reference_official_root"]
                    if scope == "official"
                    else p["session_root"]
                )
                cache = (
                    Path(p["base_cache"])
                    if scope == "official"
                    else root / "features.h5"
                )
            else:
                root = args.root / f"seed_{tokenizer_seed}" / scope
                cache = root.parent / f"{scope}.h5"
            for seed in HEAD_SEEDS:
                directory = root / "fits" / f"tokens_cal_{seed}"
                summary = json.loads((directory / "fit_summary.json").read_text())
                if (
                    summary["seed"] != seed
                    or summary["signature"]["experiment_label"] != "tokens_cal"
                ):
                    raise ValueError("incorrect readout identity")
                plans.append(
                    (
                        tokenizer_seed,
                        scope,
                        seed,
                        directory,
                        cache,
                        pose.digest(directory / "best.pt"),
                    )
                )
    pose.write_json(
        args.root / "test_gate.json",
        {
            "complete_fits": len(plans),
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "checkpoint_sha256": {
                f"{s}/{scope}/{h}": sha for s, scope, h, _, _, sha in plans
            },
        },
    )
    rows = []
    for tokenizer_seed, scope, seed, directory, cache, sha in plans:
        state = torch.load(
            directory / "best.pt", map_location="cpu", weights_only=False
        )
        data = pose.FeatureCache(cache, "test")
        if state["signature"]["cache_id"] != data.manifest["cache_id"]:
            raise ValueError("readout/cache mismatch")
        model = pose.build_pose_model(state["model_kwargs"]).to(args.device)
        model.load_state_dict(state["model"], strict=True)
        stats = {k: v.to(args.device) for k, v in state["stats"].items()}
        prediction, target, valid = pose.predict(
            model, pose.loader_for(data, args), data, args.device, stats
        )
        metadata = data.metadata()
        scopes = {"all": np.arange(len(data))}
        if scope == "official":
            scopes["unseen_users"] = np.flatnonzero(metadata["held_out_user"])
        result = {}
        for name, ix in scopes.items():
            result[name] = pose.summarize_predictions(
                prediction[ix],
                target[ix],
                valid[ix],
                {k: v[ix] for k, v in metadata.items()},
                data.manifest["token_rate_hz"],
            )
        if pose.digest(directory / "best.pt") != sha:
            raise ValueError("checkpoint changed")
        rows.append(
            {
                "tokenizer_seed": tokenizer_seed,
                "scope": scope,
                "head_seed": seed,
                "scores": result,
                "checkpoint_sha256": sha,
            }
        )
        data.close()
        print(json.dumps({"evaluated": [tokenizer_seed, scope, seed]}), flush=True)
    grouped = {}
    for scope in SCOPES:
        grouped[scope] = {}
        for tokenizer_seed in (p["reference_seed"], *NEW_SEEDS):
            subset = [
                r
                for r in rows
                if r["scope"] == scope and r["tokenizer_seed"] == tokenizer_seed
            ]
            grouped[scope][str(tokenizer_seed)] = {
                name: float(
                    np.mean([r["scores"][name]["macro_user_mae"] for r in subset])
                )
                for name in subset[0]["scores"]
            }
    pose.write_json(
        args.root / "results.json",
        {
            "complete": True,
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "rows": rows,
            "macro_user_mae_by_tokenizer": grouped,
            "inference": p["statistics"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "smoke", "train", "fit", "evaluate")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
