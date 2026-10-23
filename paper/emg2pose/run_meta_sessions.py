"""Whole-session transfer using only the already prepared Meta pose cache.

Selection uses metadata only. Every held-out session is absent from the frozen
tokenizer's full Meta pretraining manifest and from every new decoder fit.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from paper.emg2pose import compare_temporal_pose as pose
import h5py
import numpy as np
import torch

SEEDS = (260826, 260827, 260828)
ARMS = {
    "tokens_cal": "token_transformer",
    "continuous256": "continuous_transformer",
    "raw_supervised": "raw_transformer",
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def choose_sessions(records, pretrained_sessions):
    """One session per eligible user; keep a same-hand, same-stage partner.

    A recording category is a task-level match, not an identical trajectory.
    All hands and stages from the selected session are excluded from fitting.
    """
    users = defaultdict(list)
    session_users = defaultdict(set)
    for row in records:
        users[row["user"]].append(row)
        session_users[row["session"]].add(row["user"])
    if any(len(v) != 1 for v in session_users.values()):
        raise ValueError("a session maps to multiple participants")
    selected, pairs = set(), []
    for user, rows in sorted(users.items()):
        candidates = defaultdict(list)
        for row in rows:
            if row["session"] in pretrained_sessions:
                continue
            partners = [
                r
                for r in rows
                if r["session"] != row["session"]
                and (r["side"], r["stage"]) == (row["side"], row["stage"])
            ]
            if partners:
                candidates[row["session"]].append((row, partners))
        if not candidates:
            continue
        # Frozen metadata-only tie-break; do not select on labels or errors.
        session = min(candidates, key=lambda s: json_hash([260908, user, s]))
        selected.add(session)
        for row, partners in candidates[session]:
            pairs.append(
                {
                    "user": user,
                    "side": row["side"],
                    "stage": row["stage"],
                    "test_filename": row["filename"],
                    "test_session": session,
                    "train_filenames": sorted(r["filename"] for r in partners),
                    "train_sessions": sorted({r["session"] for r in partners}),
                    "has_different_date_partner": any(
                        r["session"][:10] != session[:10] for r in partners
                    ),
                }
            )
    return selected, pairs


def read_metadata(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result = {Path(r["filename"]).stem: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("nonunique official recording IDs")
    return result


def source_hashes(code):
    return {
        str(f.relative_to(code)): pose.digest(f)
        for f in sorted(code.rglob("*"))
        if f.is_file()
        and f.suffix in (".py", ".sbatch")
        and "__pycache__" not in f.parts
    }


def freeze(args):
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "protocol.json").exists() or (args.root / "features.h5").exists():
        raise FileExistsError("session campaign already frozen")
    metadata = read_metadata(args.metadata)
    metadata_hash = pose.digest(args.metadata)
    with h5py.File(args.pretrain_cache, "r") as f:
        pre = json.loads(f.attrs["manifest"])
    if pre["split"] != "train" or pre["metadata_sha256"] != metadata_hash:
        raise ValueError("incompatible pretraining metadata")
    pre_names = {Path(r["filename"]).stem for r in pre["items"]}
    if not pre_names or any(metadata[n]["split"] != "train" for n in pre_names):
        raise ValueError("pretraining exposure cannot be established")
    pre_sessions = {metadata[n]["session"] for n in pre_names}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint["data_contract"]
    if (
        contract["metadata_sha256"] != metadata_hash
        or contract["window_cache_sha256"] != pose.digest(args.pretrain_cache)
        or checkpoint["training_args"].get("resume") is not None
        or checkpoint.get("progress", {}).get("phase") != "complete"
        or "universal_state_dict" in checkpoint
    ):
        raise ValueError("unexpected tokenizer lineage or data contract")
    seen = {Path(n).stem for n in checkpoint["source_seen_recordings"]["emg2pose"]}
    if not seen or not seen <= pre_names:
        raise ValueError("checkpoint exposure exceeds declared cache")

    with h5py.File(args.cache, "r") as f:
        parent = json.loads(f.attrs["manifest"])
        if (
            not f.attrs.get("complete")
            or parent["metadata_sha256"] != metadata_hash
            or parent["checkpoint_sha256"] != pose.digest(args.checkpoint)
        ):
            raise ValueError("base cache/checkpoint mismatch")
        arrays = {}
        for split in ("train", "val", "test"):
            group = f[split]
            arrays[split] = {
                k: group[k].asstr()[:]
                for k in ("filename", "user", "session", "stage", "side")
            }
            arrays[split]["indices"] = group["unique_indices"][:]
            for i in arrays[split]["indices"]:
                official = metadata[Path(arrays[split]["filename"][i]).stem]
                if official["split"] != split or any(
                    arrays[split][k][i] != official[k]
                    for k in ("user", "session", "stage", "side")
                ):
                    raise ValueError(
                        "cached recording metadata disagrees with official CSV"
                    )
        train = arrays["train"]
        records = [
            metadata[Path(n).stem]
            for n in sorted(set(train["filename"][train["indices"]]))
        ]
        selected, pairs = choose_sessions(records, pre_sessions)
        if len({p["user"] for p in pairs}) < 5:
            raise ValueError("fewer than five usable paired-session participants")
        test_names = {p["test_filename"] for p in pairs}
        indices = {
            "train": np.array(
                [i for i in train["indices"] if train["session"][i] not in selected]
            ),
            "val": arrays["val"]["indices"],
            "test": np.array(
                [i for i in train["indices"] if train["filename"][i] in test_names]
            ),
        }
        origins = {"train": "train", "val": "val", "test": "train"}
        # Check whole-session exclusion, including nonselected hands/stages.
        fit_sessions = set(train["session"][indices["train"]])
        if selected & (fit_sessions | pre_sessions | set(arrays["val"]["session"])):
            raise ValueError("a test session is exposed in training or selection")
        retained_names = set(train["filename"][indices["train"]])
        if any(not set(pair["train_filenames"]) <= retained_names for pair in pairs):
            raise ValueError("a matched training partner was also held out")
        counts = {}
        for split, ix in indices.items():
            a = arrays[origins[split]]
            counts[split] = {
                "windows": len(ix),
                "recordings": len(set(a["filename"][ix])),
                "users": len(set(a["user"][ix])),
                "sessions": len(set(a["session"][ix])),
                "parent_split": origins[split],
                "parent_indices_sha256": json_hash(ix.tolist()),
            }
        protocol = {
            "name": "Meta whole-session transfer v1",
            "schema": 1,
            "scope": "Known-person transfer to an unseen recording session, same hand and movement category; not proof of electrode replacement or identical trajectories.",
            "base_cache": str(args.cache.resolve()),
            "base_cache_id": parent["cache_id"],
            "metadata_sha256": metadata_hash,
            "tokenizer_checkpoint_sha256": pose.digest(args.checkpoint),
            "pretraining_manifest_sha256": json_hash(pre),
            "pretraining_data_contract": contract,
            "pretraining_sessions": sorted(pre_sessions),
            "held_out_sessions": sorted(selected),
            "matched_recordings": pairs,
            "counts": counts,
            "arms": ARMS,
            "seeds": list(SEEDS),
            "fits": 9,
            "selection": "Metadata-only fixed hash tie-break, one session per eligible user, official-train cache only. Fresh decoder weights; validation remains official unseen-user validation. Best validation macro-user pose MAE, at most 50 epochs, patience 8.",
            "normalization": "Original causal per-window calibration: 1000 unlabelled EMG samples, then fixed centering/scaling/clipping. Target statistics fitted on retained training windows only.",
            "training": "Same two-layer width-256 causal transformer, eight slots, AdamW 3e-4, weight decay .01, velocity loss weight .1, batch 16. Raw baseline jointly learns the same encoder architecture from scratch.",
            "inference": "Average decoder seeds within participant; 10000 paired participant bootstrap draws, seed 260908. Report pose error, velocity metrics and token/continuous error ratio.",
            "original_official_test_reused_for_training": False,
            "source_sha256": source_hashes(args.root / "code"),
        }
        pose.write_json(args.root / "protocol.json", protocol)
        manifest = dict(parent)
        manifest.update(
            parent_cache_id=parent["cache_id"],
            splits=counts,
            session_protocol_sha256=pose.digest(args.root / "protocol.json"),
        )
        manifest.pop("cache_id")
        manifest["cache_id"] = json_hash(manifest)
        with h5py.File(args.root / "features.h5", "x") as out:
            for key in ("codebook", "descriptors"):
                out[key] = h5py.ExternalLink(str(args.cache.resolve()), "/" + key)
            for split, origin in origins.items():
                group = out.create_group(split)
                for key in f[origin]:
                    if key != "unique_indices":
                        group[key] = h5py.ExternalLink(
                            str(args.cache.resolve()), f"/{origin}/{key}"
                        )
                group["unique_indices"] = indices[split]
            out.attrs["manifest"] = json.dumps(manifest)
            out.attrs["complete"] = True
    print(json.dumps({"frozen": True, "counts": counts, "pairs": len(pairs)}, indent=2))


def verify(root):
    p = json.loads((root / "protocol.json").read_text())
    if p["source_sha256"] != source_hashes(root / "code"):
        raise ValueError("frozen source changed")
    with h5py.File(root / "features.h5", "r") as f:
        m = json.loads(f.attrs["manifest"])
        if not f.attrs.get("complete") or m["session_protocol_sha256"] != pose.digest(
            root / "protocol.json"
        ):
            raise ValueError("session feature protocol mismatch")
        for split in ("train", "val", "test"):
            ix = f[split]["unique_indices"][:]
            if json_hash(ix.tolist()) != p["counts"][split]["parent_indices_sha256"]:
                raise ValueError("session membership changed")
    return p


def fit_arguments(root, label, seed, device):
    args = pose.parse_args(
        [
            "fit",
            "--cache",
            str(root / "features.h5"),
            "--arm",
            ARMS[label],
            "--experiment-label",
            label,
            "--seed",
            str(seed),
            "--device",
            device,
            "--output-dir",
            str(root / "fits" / f"{label}_{seed}"),
        ]
    )
    args.session_protocol_sha256 = pose.digest(root / "protocol.json")
    return args


def smoke(args):
    verify(args.root)
    torch.manual_seed(260908)
    for label, arm in ARMS.items():
        config = fit_arguments(args.root, label, SEEDS[0], args.device)
        data = pose.FeatureCache(
            args.root / "features.h5",
            "train",
            arm.startswith("continuous"),
            raw=arm == "raw_transformer",
        )
        original_indices = data.indices.copy()
        data.indices = data.indices[:2]
        stats = {k: v.to(args.device) for k, v in pose.train_statistics(data).items()}
        model = pose.build_pose_model(pose.model_kwargs(data, config)).to(args.device)
        batch = next(
            iter(
                pose.loader_for(
                    data,
                    argparse.Namespace(batch_size=2, workers=0, device=args.device),
                    False,
                )
            )
        )
        prediction = pose.forward_batch(
            model,
            batch,
            data,
            data.codebook.to(args.device),
            data.descriptors.to(args.device),
            args.device,
            stats,
        )
        loss = prediction.square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(
            not torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        ):
            raise ValueError("nonfinite forward/backward preflight")
        data.indices = original_indices
        data.close()
    pose.write_json(
        args.root / "smoke.json",
        {
            "passed": True,
            "arms": list(ARMS),
            "train_inputs_only": True,
            "weights_discarded": True,
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "torch": str(torch.__version__),
            "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        },
    )


def fit(args):
    verify(args.root)
    receipt = json.loads((args.root / "smoke.json").read_text())
    if not receipt["passed"] or receipt["protocol_sha256"] != pose.digest(
        args.root / "protocol.json"
    ):
        raise ValueError("missing successful current preflight")
    label = list(ARMS)[args.index // len(SEEDS)]
    config = fit_arguments(
        args.root, label, SEEDS[args.index % len(SEEDS)], args.device
    )
    config.resume = args.resume
    pose.run_fit(config)


@torch.inference_mode()
def evaluate(args):
    from paper.common.reporting_support import (
        recovery_provenance,
        require_scored_participants,
        scoring_support,
    )

    p = verify(args.root)
    recovery = recovery_provenance(args.root, getattr(args, "recovery_manifest", None))
    if (args.root / "results.json").exists():
        raise FileExistsError("session evaluation already exists")
    runs = []
    # Gate and freeze all nine checkpoints before the first test prediction.
    for label, arm in ARMS.items():
        for seed in SEEDS:
            directory = args.root / "fits" / f"{label}_{seed}"
            fit_summary = json.loads((directory / "fit_summary.json").read_text())
            signature = fit_summary["signature"]
            if (
                signature["session_protocol_sha256"]
                != pose.digest(args.root / "protocol.json")
                or signature["seed"] != seed
                or signature["arm"] != arm
                or signature["experiment_label"] != label
            ):
                raise ValueError("fit belongs to a different session protocol")
            runs.append(
                (label, arm, seed, directory, pose.digest(directory / "best.pt"))
            )
    pose.write_json(
        args.root / "test_gate.json",
        {
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "checkpoint_sha256": {
                f"{label}_{seed}": sha for label, _, seed, _, sha in runs
            },
        },
    )
    rows, support = [], None
    for label, arm, seed, directory, sha in runs:
        state = torch.load(
            directory / "best.pt", map_location="cpu", weights_only=False
        )
        data = pose.FeatureCache(
            args.root / "features.h5",
            "test",
            arm.startswith("continuous"),
            raw=arm == "raw_transformer",
        )
        if (
            state["signature"]["cache_id"] != data.manifest["cache_id"]
            or state["signature"]["seed"] != seed
            or state["signature"]["arm"] != arm
            or state["signature"]["experiment_label"] != label
            or state["signature"]["session_protocol_sha256"]
            != pose.digest(args.root / "protocol.json")
        ):
            raise ValueError("checkpoint uses a different data split")
        model = pose.build_pose_model(state["model_kwargs"]).to(args.device)
        model.load_state_dict(state["model"], strict=True)
        stats = {k: v.to(args.device) for k, v in state["stats"].items()}
        prediction, target, valid = pose.predict(
            model, pose.loader_for(data, args), data, args.device, stats
        )
        if (
            not torch.isfinite(prediction).all()
            or pose.digest(directory / "best.pt") != sha
        ):
            raise ValueError(
                "nonfinite prediction or checkpoint changed during evaluation"
            )
        scores = pose.summarize_predictions(
            prediction, target, valid, data.metadata(), 50
        )
        current_support = scoring_support(data.metadata()["user"], valid.numpy())
        require_scored_participants(scores, current_support)
        if support is not None and current_support != support:
            raise ValueError("session heads have different scored-label support")
        support = current_support
        prediction_path = args.root / "evaluation" / f"{label}_{seed}.npz"
        prediction_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(prediction_path, prediction=prediction.numpy(), target=target.numpy(),
                            valid=valid.numpy(), users=np.asarray(data.metadata()["user"], dtype=str))
        rows.append(
            {"arm": label, "seed": seed, "checkpoint_sha256": sha, "scores": scores,
             "predictions": str(prediction_path), "predictions_sha256": pose.digest(prediction_path)}
        )
        data.close()
        print(json.dumps({"evaluated": label, "seed": seed}), flush=True)
    planned_users = sorted({r["user"] for r in p["matched_recordings"]})
    if set(planned_users) != set(support["per_participant"]):
        raise ValueError("evaluated participant identities differ from the frozen plan")
    users = support["scored_participants"]
    values = {}
    for label in ARMS:
        values[label] = np.array(
            [
                np.mean(
                    [
                        r["scores"]["per_user"][u]["landmark_mae_palm_lengths"]
                        for r in rows
                        if r["arm"] == label
                    ]
                )
                for u in users
            ]
        )
    rng = np.random.default_rng(260908)
    draw = rng.integers(0, len(users), size=(10000, len(users)))
    comparisons = {}
    for other in ("continuous256", "raw_supervised"):
        ratio = values["tokens_cal"][draw].mean(1) / values[other][draw].mean(1)
        delta = (values["tokens_cal"] - values[other])[draw].mean(1)
        comparisons["tokens_cal_vs_" + other] = {
            "ratio": float(values["tokens_cal"].mean() / values[other].mean()),
            "ratio_ci95": np.quantile(ratio, [0.025, 0.975]).tolist(),
            "mean_difference": float((values["tokens_cal"] - values[other]).mean()),
            "difference_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        }
    pose.write_json(
        args.root / "results.json",
        {
            "complete": True,
            "protocol_sha256": pose.digest(args.root / "protocol.json"),
            "scope": p["scope"],
            "participants": users,
            "planned_participants": planned_users,
            "scoring_support": support,
            "reporting_recovery": recovery,
            "rows": rows,
            "macro_user_mae": {k: float(v.mean()) for k, v in values.items()},
            "paired_comparisons": comparisons,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "smoke", "fit", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--pretrain-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--index", type=int, choices=range(9), default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recovery-manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
