"""Frozen paired Meta replication and compression comparisons."""

import argparse
import json
from pathlib import Path

from paper.emg2pose import compare_temporal_pose as pose
import h5py
import numpy as np
from paper.emg2pose import paper_completion_core as core
from paper.common import paths
import torch


def study_cases(root, storage):
    cases = []
    for encoder_seed in core.ENCODER_SEEDS:
        for scope in core.SCOPES:
            for seed in core.HEAD_SEEDS:
                cases.append({
                    "id": f"continuous_{encoder_seed}_{scope}_{seed}",
                    "group": "continuous", "label": f"continuous_{encoder_seed}",
                    "scope": scope, "head_seed": seed, "encoder_seed": encoder_seed,
                    "arm": "continuous_transformer",
                    "cache": str(storage / f"encoder_{encoder_seed}/{scope}.h5"),
                })
    for scope in core.SCOPES:
        for label in core.CONTROL_LABELS:
            for seed in core.HEAD_SEEDS:
                cases.append({
                    "id": f"{label}_{scope}_{seed}", "group": "controls", "label": label,
                    "scope": scope, "head_seed": seed, "encoder_seed": 260905,
                    "arm": "continuous_transformer" if label == "features67" else "token_transformer",
                    "cache": str(root / f"controls/{scope}/{label}.h5"),
                })
    for encoder_seed in (260940, 260941, 260942):
        depths = (1, 3, 6) if encoder_seed == 260940 else (6,)
        for scope in core.SCOPES:
            for depth in depths:
                for seed in core.HEAD_SEEDS:
                    cases.append({
                        "id": f"biocodec{depth}_{encoder_seed}_{scope}_{seed}",
                        "group": "codec", "label": f"biocodec{depth}_{encoder_seed}",
                        "scope": scope, "head_seed": seed, "encoder_seed": encoder_seed,
                        "depth": depth, "arm": "continuous_transformer",
                        "cache": str(storage / f"biocodec_{encoder_seed}/{scope}_q{depth}.h5"),
                    })
    for row in cases:
        row["fit"] = str(root / "fits" / row["id"])
        row["new_fit"] = True
    references = []
    for encoder_seed in core.ENCODER_SEEDS:
        for scope in core.SCOPES:
            original = encoder_seed == 260905
            base = paths.RESULTS
            folder = (
                base / ("reviewer_controls_v1" if scope == "official" else "meta_sessions_v1")
                if original else base / f"tokenizer_seeds_v1/seed_{encoder_seed}/{scope}"
            )
            cache = (
                base / ("clean_temporal_v1/full.h5" if scope == "official" else "meta_sessions_v1/features.h5")
                if original else base / f"tokenizer_seeds_v1/seed_{encoder_seed}/{scope}.h5"
            )
            for seed in core.HEAD_SEEDS:
                references.append({
                    "id": f"vq_{encoder_seed}_{scope}_{seed}", "group": "reference",
                    "label": f"vq_{encoder_seed}", "scope": scope,
                    "head_seed": seed, "encoder_seed": encoder_seed,
                    "arm": "token_transformer", "cache": str(cache),
                    "fit": str(folder / f"fits/tokens_cal_{seed}"), "new_fit": False,
                })
    for scope in core.SCOPES:
        for seed in core.HEAD_SEEDS:
            folder = paths.RESULTS / (
                f"clean_temporal_v1/full/raw_transformer_{seed}" if scope == "official"
                else f"meta_sessions_v1/fits/raw_supervised_{seed}"
            )
            references.append({
                "id": f"raw_{scope}_{seed}", "group": "reference", "label": "raw_supervised",
                "scope": scope, "head_seed": seed, "encoder_seed": None,
                "arm": "raw_transformer", "cache": str(paths.RESULTS / (
                    "clean_temporal_v1/full.h5" if scope == "official" else "meta_sessions_v1/features.h5"
                )), "fit": str(folder), "new_fit": False,
            })
    return cases, references


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError("completion protocol already frozen")
    args.storage.mkdir(parents=True, exist_ok=True)
    parents = [paths.RESULTS / f"{name}/protocol.json"
               for name in ("meta_sessions_v1", "tokenizer_seeds_v1", "hdemg_v20_noise_followup_v2")]
    with h5py.File(paths.RESULTS / "clean_temporal_v1/full.h5") as h:
        manifest = json.loads(h.attrs["manifest"])
        if not h.attrs["complete"] or manifest["calibration_samples"] != 1000:
            raise ValueError("unexpected base cache")
    cases, references = study_cases(args.root, args.storage)
    protocol = {
        "name": "Meta paper completion v1", "schema": 1,
        "project": str(paths.ROOT), "storage": str(args.storage),
        "base_cache": str(paths.RESULTS / "clean_temporal_v1/full.h5"),
        "base_cache_id": manifest["cache_id"],
        "parent_protocols": {str(p): pose.digest(p) for p in parents},
        "source_sha256": core.source_hashes(args.root / "code"),
        "cases": cases, "references": references,
        "new_head_fits": len(cases),
        "scope": "Exploratory fixed comparisons on prepared Meta official and eleven same-day held-session partitions. No clinical, replacement, universal manifold or optimal-vocabulary claim.",
        "primary": "Participant-macro normalized-landmark error; head seeds averaged within participant. Paired VQ versus corresponding float32 continuous encoder for each tokenizer; original VQ versus each control; VQ versus adapted BioCodec with every encoder seed shown.",
        "statistics": "10000 paired participant bootstrap draws, fixed seed 260908. Intervals conditional on encoders; display encoder variation separately and average errors over all three encoder repetitions for VQ/continuous and VQ/six-book BioCodec. This averages errors, not ensemble predictions. No new equivalence margin; all planned comparisons retained, pointwise intervals without simultaneous coverage claim.",
        "fit": "Existing two-layer width256 causal transformer, 8 slots, 50 epochs maximum, patience8, batch16, AdamW 3e-4 wd.01, velocity weight.1, identical three head seeds. Natural feature dimensions affect input projection parameter counts; report them.",
        "continuous": "Three corresponding frozen encoders, float32 features, all IDs replay against their original token caches; identical global feature precision. No extra tokenizer training.",
        "controls": "Per scope training-only PCA4 with 8/10/16 packed bits, FSQ1000 with 5000 updates and validation latent MSE selection, standardized causal RMS3+spectral64, Lloyd K-means256/1024. Original encoder only; refit every transform after session exclusion.",
        "compression": "Fixed 50Hz comparisons. Pack bit widths explicitly with a 9-byte header; no entropy-coded rate inference. PCA/FSQ are post-hoc compression of learned features; K-means uses engineered EMG features. Record component and packet bytes separately.",
        "runtime": {
            "replay_sha256": pose.digest(args.root / "assets/pose.npz"),
            "replay_metadata_sha256": pose.digest(args.root / "assets/pose.json"),
            "selection": "Existing prespecified demonstration recording, selected without predictions. Timing only, not an independent accuracy test.",
            "recipe": "Every original-encoder condition and BioCodec depths1/3/6, official head seed260826. 16 channels native2kHz, 50Hz output, three repeats; discard first repeat and first50 frames. Same time-local readout cache for all representations, strict full-prefix parity. CPU and V100 recorded separately; no torch.compile speedup claim. Actual9-byte-header packets, CPU/device transfer, extraction, lookup and unnormalized pose output included. Encoder calibration/model startup separate; no physical network or actuation. BioCodec uses causal-prefix recomputation, not optimized recurrent streaming; native code rate and downselected50Hz operating rate reported separately.",
        },
        "motion": "Training-only speed tertiles, all participant/task windows; absolute velocity error, variance and amplitude ratio. Lag searches -10..10 frames on fixed interior support; positive means prediction delay. Lag descriptive only, no shifted primary scoring; no joining independent windows.",
        "biocodec": {
            "commit": "575bf6ea7ae14a34754b217d9b940597a16951bd",
            "seeds": [260940, 260941, 260942], "steps": 40000,
            "channels_per_step": 32, "learning_rate": .0005,
            "scope": "Fresh adaptation of official EMG architecture/loss on the reference pretraining sources, fixed compute-limited recipe; not released EMG weights or equal pretraining compute.",
            "training": "Official Adam (0.5,0.9), fp32, official WarmupCosineLrScheduler with 10000 warmup steps and warmup_ratio5e-4; one backward of .2 time+1 frequency+2 commitment, reproducing the upstream non-AMP double-commit gradient. Final fixed-step model; report 20k/40k unlabeled validation reconstruction diagnostics without downstream selection.",
            "input": "Original 500ms calibration, causal 41-tap FIR downsample2 retaining10ms delay, 1000Hz; no full-window normalization. 6 residual codebooks of256, 128D; native stride18. At each50Hz target use latest fully observed codec frame, no future interpolation.",
            "depths": "1/3/6 residual layers for seed260940, depth6 for all three seeds. Same trained six-layer codec; these are residual truncation operating points, not separate training algorithms.",
        },
        "execution": "Immutable code, all new weights start fresh; groups gated on every required fit before test. Selected-row resident input has disk parity checks. Large arrays on dedicated storage, original caches read-only.",
    }
    pose.write_json(args.root / "protocol.json", protocol)
    print(json.dumps({"new_head_fits": len(cases), "new_biocodec_fits": 3,
                      "groups": {g: sum(r["group"] == g for r in cases)
                                 for g in ("continuous", "controls", "codec")}}))


def parent_cache(p, encoder_seed):
    if encoder_seed == 260905:
        return Path(p["base_cache"]), paths.LEARNED_TOKENIZER_FULL
    root = paths.RESULTS / f"tokenizer_seeds_v1/seed_{encoder_seed}"
    receipt = json.loads((root / "cache_receipt.json").read_text())
    if not receipt["complete"] or pose.digest(root / "official.h5") != receipt["official_sha256"]:
        raise ValueError("unverified independent tokenizer cache")
    return root / "official.h5", root / "tokenizer.pt"


@torch.inference_mode()
def prepare_continuous(args):
    from myoji.learned import load_universal_checkpoint
    p = core.verify(args.root)
    precision = core.configure_encoder_replay(args.device)
    seed = core.ENCODER_SEEDS[args.index]
    base, checkpoint = parent_cache(p, seed)
    module, _ = load_universal_checkpoint(checkpoint, torch.device(args.device))
    module.eval().requires_grad_(False)
    before = core.state_hash(module.tokenizer)
    directory = Path(p["storage"]) / f"encoder_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "official.h5"
    if path.exists():
        raise FileExistsError(path)
    with h5py.File(base, "r") as source, h5py.File(path, "x") as out:
        manifest = json.loads(source.attrs["manifest"])
        out.attrs["complete"] = False
        for k in ("codebook", "descriptors"):
            out[k] = h5py.ExternalLink(str(base), "/" + k)
        for split in ("train", "val", "test"):
            origin, group = source[split], out.create_group(split)
            for key in origin:
                if key != "continuous":
                    group[key] = h5py.ExternalLink(str(base), f"/{split}/{key}")
            shape = (*origin["tokens"].shape, 256)
            target = group.create_dataset("continuous", shape=shape, dtype="<f4")
            for start in range(0, shape[0], 8):
                raw = torch.from_numpy(origin["raw"][start:start+8]).to(args.device)
                encoded = module.tokenizer(raw, sensor="wrist", quantize=True, include_spatial=False)
                ids = encoded.tokens[..., -100:].cpu().numpy()
                np.testing.assert_array_equal(ids, origin["tokens"][start:start+len(raw)])
                vectors = encoded.continuous[..., -100:, :].float().cpu().numpy()
                if not np.isfinite(vectors).all():
                    raise ValueError("nonfinite continuous features")
                if seed == 260905:
                    # Existing half storage defines a rounding-error envelope, not fp32 truth.
                    old = origin["continuous"][start:start+len(raw)].astype(np.float32)
                    np.testing.assert_allclose(vectors, old, atol=6e-5, rtol=6e-4)
                target[start:start+len(raw)] = vectors
            print(json.dumps({"continuous_seed": seed, "prepared_split": split}), flush=True)
        core.finish_cache(out, manifest, parent_cache_id=manifest["cache_id"],
                          representation="corresponding_encoder_float32",
                          completion_protocol_sha256=pose.digest(args.root / "protocol.json"),
                          tokenizer_seed=seed, checkpoint_sha256=pose.digest(checkpoint))
    if core.state_hash(module.tokenizer) != before:
        raise ValueError("frozen encoder state changed")
    session = paths.RESULTS / "meta_sessions_v1/features.h5"
    with h5py.File(session, "r") as selected, h5py.File(directory / "session.h5", "x") as out:
        for key in ("codebook", "descriptors"):
            out[key] = h5py.ExternalLink(str(path), "/" + key)
        for split, original in (("train", "train"), ("val", "val"), ("test", "train")):
            group = out.create_group(split)
            with h5py.File(path, "r") as source:
                for key in source[original]:
                    if key != "unique_indices":
                        group[key] = h5py.ExternalLink(str(path), f"/{original}/{key}")
            group["unique_indices"] = selected[split]["unique_indices"][:]
        core.finish_cache(out, json.loads(selected.attrs["manifest"]),
                          representation="corresponding_encoder_float32",
                          completion_protocol_sha256=pose.digest(args.root / "protocol.json"),
                          tokenizer_seed=seed, checkpoint_sha256=pose.digest(checkpoint))
    pose.write_json(directory / "receipt.json", {
        "complete": True, "all_token_ids_exact": True, "tokenizer_unchanged": True,
        "protocol_sha256": pose.digest(args.root / "protocol.json"),
        "source_checkpoint_sha256": pose.digest(checkpoint),
        "canonical_encoder_precision": precision,
        "files": {s: pose.digest(directory / (s + ".h5")) for s in core.SCOPES},
    })


def sample_engineered(source, count=65536, seed=260950, device="cuda"):
    rng = np.random.default_rng(seed)
    g = source["train"]
    indices = rng.choice(g["unique_indices"][:], min(1024, len(g["unique_indices"])), replace=False)
    per_window = int(np.ceil(count / len(indices)))
    config = json.loads(source.attrs["manifest"])["config"]
    samples = []
    for ix in np.sort(indices):
        raw = torch.from_numpy(g["raw"][int(ix)])[None].to(device)
        values = core.features67(raw, 100, config).reshape(-1, 67)
        take = rng.choice(len(values), min(per_window, len(values)), replace=False)
        samples.append(values[take].cpu())
    return torch.cat(samples)[:count].to(device)


def prepare_controls(args):
    from paper.emg2pose.prepare_reviewer_features import LatentFSQ, sample_latents
    p = core.verify(args.root)
    scope = core.SCOPES[args.index]
    base = Path(p["storage"]) / f"encoder_260905/{scope}.h5"
    output = args.root / "controls" / scope
    output.mkdir(parents=True, exist_ok=True)
    if (output / "transforms.pt").exists():
        raise FileExistsError("controls already started")
    torch.manual_seed(260950)
    with h5py.File(base, "r") as source:
        parent = json.loads(source.attrs["manifest"])
        train = sample_latents(source, "train", 131072, 260950).to(args.device)
        val = sample_latents(source, "val", 16384, 260951).to(args.device)
        mean = train.mean(0)
        centered = train - mean
        _, eigenvectors = torch.linalg.eigh(centered.T @ centered / len(train))
        basis = eigenvectors.flip(1)[:, :4]
        coordinates = centered @ basis
        lo, hi = torch.quantile(coordinates, torch.tensor([.001, .999], device=args.device), dim=0)
        scale = centered.square().mean(0).sqrt().clamp_min(1e-4)
        fsq = LatentFSQ(256).to(args.device)
        opt = torch.optim.AdamW(fsq.parameters(), lr=3e-4, weight_decay=.01)
        best, history, selected = float("inf"), [], None
        normalized, validation = centered / scale, (val - mean) / scale
        for step in range(1, 5001):
            x = normalized[torch.randint(len(train), (1024,), device=args.device)]
            loss = (fsq(x) - x).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite FSQ loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 250 == 0:
                with torch.no_grad():
                    score = float((fsq(validation) - validation).square().mean())
                history.append({"step": step, "validation_mse": score})
                if score < best:
                    best = score
                    selected = {k: v.detach().cpu().clone() for k, v in fsq.state_dict().items()}
        fsq.load_state_dict(selected)
        fsq.eval().requires_grad_(False)
        engineered = sample_engineered(source, device=args.device)
        fm, fs = engineered.mean(0), engineered.std(0, unbiased=False).clamp_min(1e-4)
        engineered = (engineered - fm) / fs
        clusters, cluster_history = {}, {}
        for count in (256, 1024):
            centres, losses = core.lloyd_kmeans(engineered, count, 260950)
            clusters[str(count)], cluster_history[str(count)] = centres.cpu(), losses
        transforms = {
            "mean": mean.cpu(), "scale": scale.cpu(), "pca_basis": basis.cpu(),
            "pca_lo": lo.cpu(), "pca_hi": hi.cpu(), "fsq": selected, "fsq_history": history,
            "feature_mean": fm.cpu(), "feature_scale": fs.cpu(),
            "clusters": clusters, "kmeans_history": cluster_history,
            "fit_split": "train", "base_cache_id": parent["cache_id"],
            "train_identities": core.cache_identities(base, "train"),
            "scope": scope, "protocol_sha256": pose.digest(args.root / "protocol.json"),
        }
        torch.save(transforms, output / "transforms.pt")
        handles = {}
        try:
            for label in core.CONTROL_LABELS:
                # Large engineered continuous arrays use the designated storage.
                path = (Path(p["storage"]) / f"features67_{scope}.h5"
                        if label == "features67" else output / (label + ".h5"))
                out = h5py.File(path, "x")
                handles[label] = out
                out.attrs["complete"] = False
                core.compact_metadata(source, out)
                if label in core.PCA_LEVELS:
                    book = core.scalar_codebook(lo, hi, core.PCA_LEVELS[label])
                elif label == "fsq1000":
                    levels = torch.tensor([8, 5, 5, 5], device=args.device)
                    ids = torch.arange(1000, device=args.device)
                    digits = (ids[:, None] // torch.tensor([1, 8, 40, 200], device=args.device)) % levels
                    book = 2 * digits / (levels - 1) - 1
                elif label.startswith("kmeans"):
                    book = clusters[label.removeprefix("kmeans")].to(args.device)
                else:
                    book = torch.zeros(1, 67, device=args.device)
                out["codebook"] = book.cpu().numpy().astype(np.float32)
                for split in ("train", "val", "test"):
                    n, channels, frames = len(source[split]["unique_indices"]), 16, 100
                    if label == "features67":
                        out[split].create_dataset("continuous", shape=(n, channels, frames, 67), dtype="<f4")
                    else:
                        out[split].create_dataset("tokens", shape=(n, channels, frames), dtype="<u2")
            with torch.inference_mode():
                for split in ("train", "val", "test"):
                    g = source[split]
                    indices = g["unique_indices"][:]
                    for start in range(0, len(indices), 8):
                        ix = indices[start:start+8]
                        latent = torch.from_numpy(core.read_indices(g["continuous"], ix)).to(args.device)
                        raw = torch.from_numpy(core.read_indices(g["raw"], ix)).to(args.device)
                        features = (core.features67(raw, 100, parent["config"]) - fm) / fs
                        projection = (latent - mean) @ basis
                        for label, out in handles.items():
                            if label in core.PCA_LEVELS:
                                value = core.scalar_encode(projection, lo, hi, core.PCA_LEVELS[label])
                            elif label == "fsq1000":
                                value = fsq.quantize((latent - mean) / scale)[1]
                            elif label == "features67":
                                value = features
                            else:
                                value = core.nearest_centres(features, clusters[label.removeprefix("kmeans")].to(args.device)).reshape(*features.shape[:-1])
                            key = "continuous" if label == "features67" else "tokens"
                            handles[label][split][key][start:start+len(ix)] = value.cpu().numpy()
                    print(json.dumps({"control_scope": scope, "split": split}), flush=True)
            for label, out in handles.items():
                core.finish_cache(out, parent, representation=label,
                                  parent_cache_id=parent["cache_id"],
                                  transform_sha256=pose.digest(output / "transforms.pt"),
                                  completion_protocol_sha256=pose.digest(args.root / "protocol.json"))
        finally:
            for h in handles.values():
                h.close()
        # Stable expected path, tiny external-link file rather than duplicated features.
        with h5py.File(Path(p["storage"]) / f"features67_{scope}.h5", "r") as src, h5py.File(output / "features67.h5", "x") as view:
            for key in src:
                view[key] = h5py.ExternalLink(src.filename, "/" + key)
            for key, value in src.attrs.items():
                view.attrs[key] = value
    pose.write_json(output / "receipt.json", {
        "complete": True, "scope": scope, "train_identities": transforms["train_identities"],
        "protocol_sha256": pose.digest(args.root / "protocol.json"),
        "transforms_sha256": pose.digest(output / "transforms.pt"),
        "cache_sha256": {label: pose.digest(output / f"{label}.h5") for label in core.CONTROL_LABELS},
    })


def fit(args):
    p = core.verify(args.root)
    cases = [r for r in p["cases"] if r["group"] == args.group]
    row = cases[args.index]
    run = pose.parse_args([
        "fit", "--cache", row["cache"], "--arm", row["arm"],
        "--experiment-label", row["label"], "--output-dir", row["fit"],
        "--seed", str(row["head_seed"]), "--device", args.device, "--workers", "0",
    ])
    run.completion_protocol_sha256 = pose.digest(args.root / "protocol.json")
    run.case_id = row["id"]
    run.resume = args.resume
    pose.FeatureCache = core.ResidentCache
    pose.run_fit(run)


def evaluation_cases(p, group):
    new = [r for r in p["cases"] if r["group"] == group]
    references = [
        r for r in p["references"]
        if (group in ("continuous", "codec") and r["label"].startswith("vq_"))
        or (group == "controls" and r["label"] in ("vq_260905", "raw_supervised"))
        or (group == "codec" and r["label"] == "raw_supervised")
    ]
    return new + references


@torch.inference_mode()
def evaluate(args):
    p = core.verify(args.root)
    cases = evaluation_cases(p, args.group)
    directory = args.root / "evaluation" / args.group
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "results.json").exists():
        raise FileExistsError("group already evaluated")
    locks = {}
    for case in cases:
        folder = Path(case["fit"])
        summary = json.loads((folder / "fit_summary.json").read_text())
        if summary["seed"] != case["head_seed"] or summary["arm"] != case["arm"]:
            raise ValueError("incorrect fitted case")
        if case["new_fit"] and summary["signature"]["completion_protocol_sha256"] != pose.digest(args.root / "protocol.json"):
            raise ValueError("fit belongs to another completion protocol")
        if case["new_fit"] and (summary["signature"]["case_id"] != case["id"] or
                                summary["signature"]["experiment_label"] != case["label"]):
            raise ValueError("fit case identity differs")
        locks[case["id"]] = pose.digest(folder / "best.pt")
    pose.write_json(directory / "test_gate.json", {
        "protocol_sha256": pose.digest(args.root / "protocol.json"),
        "complete_fits": len(cases), "checkpoint_sha256": locks,
    })
    rows, identities, references, target_hashes = [], {}, {}, {}
    for scope in core.SCOPES:
        path = Path(p["storage"]) / f"encoder_260905/{scope}.h5"
        with h5py.File(path, "r") as h:
            ix = h["train/unique_indices"][:]
            references[scope] = core.movement_reference(
                core.read_indices(h["train/targets"], ix), core.read_indices(h["train/valid"], ix))
    for case in cases:
        state = torch.load(Path(case["fit"]) / "best.pt", map_location="cpu", weights_only=False)
        data = core.ResidentCache(case["cache"], "test", case["arm"].startswith("continuous"),
                                  raw=case["arm"] == "raw_transformer")
        if state["signature"]["cache_id"] != data.manifest["cache_id"]:
            raise ValueError("checkpoint/cache mismatch")
        identity = core.cache_identities(case["cache"], "test")
        scope = case["scope"]
        if scope in identities and identities[scope] != identity:
            raise ValueError("different assessment windows across methods")
        identities[scope] = identity
        model = pose.build_pose_model(state["model_kwargs"]).to(args.device).eval()
        model.load_state_dict(state["model"], strict=True)
        stats = {k: v.to(args.device) for k, v in state["stats"].items()}
        predictions, targets, masks = [], [], []
        loader_args = argparse.Namespace(batch_size=16, workers=0, device=args.device)
        for batch in pose.loader_for(data, loader_args):
            pred = pose.forward_batch(model, batch, data, data.codebook.to(args.device),
                                      data.descriptors.to(args.device), args.device, stats)
            predictions.append(pred.cpu().numpy())
            targets.append(batch["targets"].numpy())
            masks.append(batch["valid"].numpy())
        pred, target, valid = [np.concatenate(v) for v in (predictions, targets, masks)]
        if not np.isfinite(pred).all():
            raise ValueError("nonfinite predictions")
        th = hashlib_arrays(target, valid)
        if scope in target_hashes and target_hashes[scope] != th:
            raise ValueError("assessment targets/masks differ")
        target_hashes[scope] = th
        metadata = data.metadata()
        with h5py.File(case["cache"], "r") as h:
            stages = core.read_indices(h["test/stage"], data.indices).astype(str)
        frames = data.manifest["score_frames"]
        hold = np.broadcast_to(pred[:, -frames-1:-frames], pred[:, -frames:].shape).copy()
        subsets = {"all": np.ones(len(pred), dtype=bool)}
        if scope == "official":
            subsets.update(unseen_users=metadata["held_out_user"], unseen_stages=metadata["held_out_stage"],
                           both=metadata["held_out_user"] & metadata["held_out_stage"])
        scores = {}
        for name, mask in subsets.items():
            ix = np.flatnonzero(mask)
            if not len(ix):
                continue
            meta = {k: v[ix] for k, v in metadata.items()}
            arguments = [torch.from_numpy(v) for v in (pred[ix, -frames:], target[ix, -frames:], valid[ix, -frames:])]
            current = pose.summarize_predictions(*arguments, meta, 50)
            scores[name] = current
        saved = Path(p["storage"]) / "predictions" / args.group / (case["id"] + ".npz")
        saved.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(saved, prediction=pred, target=target, valid=valid,
                            users=metadata["user"].astype(str), stages=stages)
        motion = core.motion_rows(pred, target, valid, metadata["user"], stages, references[scope])
        pose.write_json(directory / (case["id"] + "_motion.json"), {
            "case": case, "reference": references[scope], "rows": motion,
            "predictions_sha256": pose.digest(saved),
        })
        rows.append({
            **case, "scores": scores,
            "hold": pose.summarize_predictions(torch.from_numpy(hold), torch.from_numpy(target[:, -frames:]),
                                             torch.from_numpy(valid[:, -frames:]), metadata, 50),
            "checkpoint_sha256": locks[case["id"]], "predictions": str(saved),
            "predictions_sha256": pose.digest(saved),
            "parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
        })
        if pose.digest(Path(case["fit"]) / "best.pt") != locks[case["id"]]:
            raise ValueError("checkpoint changed during evaluation")
        data.close()
        del model, data
        print(json.dumps({"evaluated": case["id"]}), flush=True)
    comparisons = {}
    for scope in core.SCOPES:
        names = ("all", "unseen_users", "unseen_stages", "both") if scope == "official" else ("all",)
        for subgroup in names:
            paired = [{"label": r["label"], "per_user": r["scores"][subgroup]["per_user"]}
                      for r in rows if r["scope"] == scope and subgroup in r["scores"]]
            labels = sorted({r["label"] for r in paired})
            if args.group == "continuous":
                pairs = [(f"vq_{s}", f"continuous_{s}") for s in core.ENCODER_SEEDS]
            else:
                pairs = [("vq_260905", label) for label in labels if label != "vq_260905"]
            for ref, other in pairs:
                comparisons[f"{scope}/{subgroup}/{ref}_vs_{other}"] = core.paired_summary(paired, ref, other)
            if args.group in ("continuous", "codec"):
                family = "continuous" if args.group == "continuous" else "biocodec6"
                families = {"vq": core.ENCODER_SEEDS, family: core.ENCODER_SEEDS if family == "continuous" else (260940, 260941, 260942)}
                collapsed = []
                for prefix, seeds in families.items():
                    for head_seed in core.HEAD_SEEDS:
                        selected = [r["scores"][subgroup]["per_user"] for r in rows
                                    if r["scope"] == scope and r["head_seed"] == head_seed
                                    and r["label"] in {f"{prefix}_{s}" for s in seeds}]
                        if len(selected) != 3 or any(set(v) != set(selected[0]) for v in selected):
                            raise ValueError("encoder-average comparison lacks matched participants")
                        collapsed.append({"label": prefix, "per_user": {
                            u: {"landmark_mae_palm_lengths": float(np.mean([v[u]["landmark_mae_palm_lengths"] for v in selected]))}
                            for u in selected[0]}})
                comparisons[f"{scope}/{subgroup}/three_encoder_mean_vq_vs_{family}"] = core.paired_summary(collapsed, "vq", family)
    pose.write_json(directory / "results.json", {
        "complete": True, "group": args.group, "rows": rows,
        "paired": comparisons, "test_identities": identities, "target_hashes": target_hashes,
        "protocol_sha256": pose.digest(args.root / "protocol.json"),
    })


def hashlib_arrays(*arrays):
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "continuous", "controls", "fit", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--storage", type=Path, default=paths.RESULTS / "paper_completion_v1")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--group", choices=("continuous", "controls", "codec"), default="continuous")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"freeze": freeze, "continuous": prepare_continuous, "controls": prepare_controls,
     "fit": fit, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
