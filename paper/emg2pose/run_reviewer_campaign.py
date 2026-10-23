"""Run the locked control comparisons without modifying old runs."""

import argparse
import json
from pathlib import Path

from paper.emg2pose import compare_temporal_pose as pose
import numpy as np
import torch

SEEDS = (260826, 260827, 260828)
CORE = ("tokens_cal", "tokens_only", "calibration_only", "continuous256")
COMPACT = ("rms3", "spectral64", "pca4", "pca16", "pca64", "fsq1000")


def specification(root, base, label):
    return {
        "cache": base if label in CORE else root / "features" / (label + ".h5"),
        "arm": "token_transformer"
        if label in ("tokens_cal", "tokens_only", "calibration_only", "fsq1000")
        else "continuous_transformer",
        "input_mode": {
            "tokens_only": "features_only",
            "calibration_only": "calibration_only",
        }.get(label, "full"),
    }


def fit(args):
    labels = CORE if args.stage == "core" else COMPACT
    label, seed = labels[args.index // 3], SEEDS[args.index % 3]
    spec = specification(args.root, args.cache, label)
    fit_args = pose.parse_args(
        [
            "fit",
            "--cache",
            str(spec["cache"]),
            "--arm",
            spec["arm"],
            "--input-mode",
            spec["input_mode"],
            "--experiment-label",
            label,
            "--output-dir",
            str(args.root / "fits" / f"{label}_{seed}"),
            "--seed",
            str(seed),
            "--device",
            args.device,
        ]
    )
    fit_args.campaign_implementation_sha256 = pose.digest(__file__)
    pose.run_fit(fit_args)


@torch.inference_mode()
def evaluate(args):
    output = args.root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    # Require all planned fits before any test predictions are computed.
    for label in (*CORE, *COMPACT):
        for seed in SEEDS:
            if not (
                args.root / "fits" / f"{label}_{seed}" / "fit_summary.json"
            ).exists():
                raise ValueError(f"incomplete campaign: {label}/{seed}")
    results, expected_manifest = [], None
    for label in (*CORE, *COMPACT):
        for seed in SEEDS:
            spec = specification(args.root, args.cache, label)
            path = args.root / "fits" / f"{label}_{seed}" / "best.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            data = pose.FeatureCache(
                spec["cache"], "test", spec["arm"].startswith("continuous")
            )
            if data.manifest["cache_id"] != state["signature"]["cache_id"]:
                raise ValueError("evaluation cache differs from fitted cache")
            identities = data.manifest["splits"]
            if expected_manifest is None:
                expected_manifest = identities
            if identities != expected_manifest:
                raise ValueError("comparators have different input selection manifests")
            model = pose.build_pose_model(state["model_kwargs"]).to(args.device).eval()
            model.load_state_dict(state["model"])
            stats = {k: v.to(args.device) for k, v in state["stats"].items()}
            book, geometry = (
                data.codebook.to(args.device),
                data.descriptors.to(args.device),
            )
            predictions, targets, masks = [], [], []
            for batch in pose.loader_for(data, args):
                p = pose.forward_batch(
                    model, batch, data, book, geometry, args.device, stats
                )
                if not torch.isfinite(p).all():
                    raise ValueError("nonfinite test prediction")
                predictions.append(p.cpu())
                targets.append(batch["targets"])
                masks.append(batch["valid"])
            p, t, v = map(torch.cat, (predictions, targets, masks))
            frames = data.manifest["score_frames"]
            if p.shape[1] <= frames:
                raise ValueError("corrected hold requires a preceding context frame")
            hold = p[:, -frames - 1 : -frames].expand(-1, frames, -1)
            p, t, v = p[:, -frames:], t[:, -frames:], v[:, -frames:]
            metadata, rate = data.metadata(), data.manifest["token_rate_hz"]
            scopes = {}
            for scope, mask in (
                ("all", np.ones(len(p), dtype=bool)),
                ("unseen_users", metadata["held_out_user"]),
                ("unseen_stages", metadata["held_out_stage"]),
                ("both", metadata["held_out_user"] & metadata["held_out_stage"]),
            ):
                index = np.flatnonzero(mask)
                if not len(index):
                    continue
                meta = {k: a[index] for k, a in metadata.items()}
                scopes[scope] = {
                    "model": pose.summarize_predictions(
                        p[index], t[index], v[index], meta, rate
                    ),
                    "context_hold": pose.summarize_predictions(
                        hold[index], t[index], v[index], meta, rate
                    ),
                }
            results.append(
                {
                    "label": label,
                    "seed": seed,
                    "checkpoint_sha256": pose.digest(path),
                    "scopes": scopes,
                }
            )
            data.close()
            pose.write_json(
                args.root / "evaluation_progress.json",
                {"completed_fits": len(results), "required": 30},
            )
            print(json.dumps({"evaluated": label, "seed": seed}), flush=True)
    grouped = {
        label: {r["seed"]: r for r in results if r["label"] == label}
        for label in (*CORE, *COMPACT)
    }
    paired = {}
    for label in (*CORE, *COMPACT):
        if label == "tokens_cal":
            continue
        for scope in ("all", "unseen_users", "unseen_stages", "both"):
            candidate = [
                grouped["tokens_cal"][seed]["scopes"][scope]["model"]["per_user"]
                for seed in SEEDS
            ]
            control = [
                grouped[label][seed]["scopes"][scope]["model"]["per_user"]
                for seed in SEEDS
            ]
            users = set(candidate[0])
            if any(set(rows) != users for rows in candidate + control):
                raise ValueError("paired users differ")
            deltas = [
                np.mean(
                    [
                        a[u]["landmark_mae_palm_lengths"]
                        - b[u]["landmark_mae_palm_lengths"]
                        for a, b in zip(candidate, control)
                    ]
                )
                for u in sorted(users)
            ]
            paired[f"tokens_cal_minus_{label}/{scope}"] = pose.bootstrap(
                deltas, 260906, 2000
            )
    pose.write_json(
        output,
        {
            "complete": True,
            "results": results,
            "paired_macro_user_mae": paired,
            "hold_definition": "last model prediction before scored interval; no ground truth",
            "interpretation": "FSQ is post-hoc frozen-latent compression",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("fit", "evaluate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--stage", choices=("core", "compact"), default="core")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(4)
    {"fit": fit, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
