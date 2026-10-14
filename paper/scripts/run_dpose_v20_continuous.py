"""Matched continuous-input V20 control with float32 encoder features."""

import argparse
import copy
import gc
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import paths
import run_dpose_pilot as pilot
import run_dpose_v20 as v20
import torch
from torch.nn import functional as F


@contextmanager
def backend(book):
    with v20.backend(book) as within:
        original = pilot.build

        def build(arm, p, config):
            assert arm == "continuous"
            return original("tokens", p, config)

        def predict(model, data, ix, arm, device):
            assert arm == "continuous" and data["features"].dtype == torch.float32
            return model.forward_embeddings(data["features"][ix].to(device))

        pilot.ARMS = within.ARMS = ("continuous",)
        pilot.build, pilot.predict_batch = build, predict
        yield within


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError(args.root)
    old = json.loads((args.reference / "protocol.json").read_text())
    assert old["readout"] == v20.READOUT and old["arms"] == ["tokens"]
    assert json.loads((args.reference / "results.json").read_text())["complete"]
    assert pilot.sha256(args.checkpoint) == old["checkpoint_sha256"]
    p = copy.deepcopy(old)
    p.update(
        arms=["continuous"],
        reference=str(args.reference),
        reference_protocol_sha256=pilot.sha256(args.reference / "protocol.json"),
        reference_results_sha256=pilot.sha256(args.reference / "results.json"),
        reference_selection_sha256=pilot.sha256(args.reference / "selection.json"),
        models="The identical 704700-parameter V20, seeded and trained with the original token V20 recipe; forward_embeddings receives unquantized frozen encoder features. Separate fits for nine people and three seeds.",
        precision="float32 throughout. Reencode already-normalized cached raw EMG with the same frozen tokenizer; require exact IDs and exact float16 parity to the original cache for every channel/window. Save pre-rounding float32 vectors. No repeated calibration.",
        cache="Own float32 continuous arrays; read-only external links for original IDs, codebook, descriptors, targets and window identities. Hash final cache and consumed raw input; original caches untouched.",
        evaluation="All 27 fits and input/checkpoint integrity must pass before test scoring. Primary: unfiltered paired pose error, participant-averaged decoder seeds. Secondary: matched validation-selected One Euro filtering and frozen native-noise replay.",
        engineering_noninferiority={
            "question": "Is token V20 within 5% of continuous V20 mean 3D error?",
            "maximum_error_ratio": 1.05,
            "criterion": "Upper endpoint of paired participant bootstrap 95% percentile interval for ratio of macro-person errors below 1.05; 10000 draws, seed 260930.",
            "limitation": "Exploratory engineering tolerance chosen before this continuous test; not clinical equivalence. Prior token test results were already inspected.",
        },
    )
    p.pop("reference_predictions_sha256", None)
    # The library (including the vendored V20 module) is hashed through src/.
    sources = [
        str(x.relative_to(paths.REPO)) for x in (paths.REPO / "src").rglob("*.py")
    ] + [
        "scripts/run_dpose_v20_continuous.py",
        "scripts/run_dpose_v20.py",
        "scripts/run_dpose_pilot.py",
        "scripts/run_dpose_within.py",
        "src/myoji/_vendor/LICENSE-GPL-3.0",
    ]
    p["source_sha256"] = {
        x: pilot.sha256(pilot.source_path(x)) for x in sorted(sources)
    }
    p["token_checkpoints"] = {}
    for subject in v20.SUBJECTS:
        for seed in v20.SEEDS:
            key = f"subject_{subject:02d}/fits/tokens_{seed}"
            fit = json.loads((args.reference / key / "fit.json").read_text())
            assert fit["complete"] and fit["best_sha256"] == pilot.sha256(
                args.reference / key / "best.pt"
            )
            p["token_checkpoints"][key] = {
                "best": fit["best_sha256"],
                "prediction": pilot.sha256(
                    args.reference / key / "test_predictions.npz"
                ),
            }
    pilot.write_json(args.root / "protocol.json", p)
    selection = json.loads((args.reference / "selection.json").read_text())
    selection["protocol_sha256"] = pilot.sha256(args.root / "protocol.json")
    pilot.write_json(args.root / "selection.json", selection)
    for subject in v20.SUBJECTS:
        pilot.write_json(
            args.root / f"subject_{subject:02d}/protocol.json",
            dict(
                p,
                subject=subject,
                campaign_protocol_sha256=pilot.sha256(args.root / "protocol.json"),
            ),
        )
    print(
        json.dumps(
            {
                "frozen": True,
                "fits": 27,
                "float32_cache_gib": 864 * 320 * 100 * 256 * 4 / 2**30,
            }
        ),
        flush=True,
    )


def verify(args):
    p = json.loads((args.root / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        assert pilot.sha256(pilot.source_path(name)) == digest, name
    for name in ("protocol", "results", "selection"):
        assert (
            pilot.sha256(args.reference / f"{name}.json")
            == p[f"reference_{name}_sha256"]
        )
    assert pilot.sha256(args.checkpoint) == p["checkpoint_sha256"]
    return p


@torch.inference_mode()
def reencode(tokenizer, raw, old_ids, old_cont, device):
    """Raw is ALREADY normalized; replay at the original 32-channel batch size."""
    raw = torch.from_numpy(raw)[None].to(device)
    chunks = []
    for c in range(0, 320, 32):
        encoded = tokenizer.model(
            raw[:, c : c + 32],
            sensor="hd",
            include_spatial=False,
            channel_descriptors=torch.zeros(
                32, tokenizer.config.geometry_dim, device=device
            ),
        )
        ids = encoded.tokens[0, :, -100:].cpu().numpy()
        vectors = encoded.continuous[0, :, -100:].float().cpu().numpy()
        np.testing.assert_array_equal(ids, old_ids[c : c + 32])
        np.testing.assert_array_equal(vectors.astype(np.float16), old_cont[c : c + 32])
        assert np.isfinite(vectors).all()
        chunks.append(vectors)
    return np.concatenate(chunks)


def prepare(args, p, subject):
    folder = args.root / f"subject_{subject:02d}"
    out = folder / "features.h5"
    receipt_path = folder / "cache_receipt.json"
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        assert r["complete"] and r["protocol_sha256"] == pilot.sha256(
            folder / "protocol.json"
        )
        assert r["cache_sha256"] == pilot.sha256(out)
        return folder
    temporary = folder / "features.partial.h5"
    if out.exists() or temporary.exists():
        raise FileExistsError("Incomplete cache requires explicit recovery")
    source = Path(p["reused_caches"][str(subject)]["path"])
    assert (
        v20.cache_digest(source)
        == p["reused_caches"][str(subject)]["consumed_arrays_sha256"]
    )
    tokenizer = pilot.Tokenizer(args.checkpoint, device=args.device)
    before = pilot.state_hash(tokenizer.model)
    raw_hash, feature_hash = hashlib.sha256(), hashlib.sha256()
    with h5py.File(source, "r") as old, h5py.File(temporary, "w") as out_h5:
        out_h5.attrs["complete"] = False
        out_h5.attrs["protocol_sha256"] = pilot.sha256(folder / "protocol.json")
        for name in ("codebook", "descriptors"):
            out_h5[name] = h5py.ExternalLink(str(source), name)
        for split in ("train", "val", "test"):
            group = out_h5.create_group(split)
            for name in ("tokens", "targets", "subjects", "tasks", "starts"):
                group[name] = h5py.ExternalLink(str(source), f"{split}/{name}")
            count = len(old[f"{split}/tokens"])
            vectors = group.create_dataset(
                "continuous", (count, 320, 100, 256), dtype="f4"
            )
            for index in range(count):
                raw = old[f"{split}/raw"][index]
                raw_hash.update(raw.tobytes())
                values = reencode(
                    tokenizer,
                    raw,
                    old[f"{split}/tokens"][index],
                    old[f"{split}/continuous"][index],
                    args.device,
                )
                vectors[index] = values
                feature_hash.update(values.tobytes())
            out_h5.flush()
            print(
                json.dumps({"prepared": split, "subject": subject, "windows": count}),
                flush=True,
            )
        assert before == pilot.state_hash(tokenizer.model)
        out_h5.attrs["complete"] = True
    temporary.replace(out)
    out.chmod(0o444)
    pilot.write_json(
        receipt_path,
        {
            "complete": True,
            "protocol_sha256": pilot.sha256(folder / "protocol.json"),
            "cache_sha256": pilot.sha256(out),
            "normalized_raw_sha256": raw_hash.hexdigest(),
            "continuous_float32_sha256": feature_hash.hexdigest(),
            "tokenizer_state_sha256": before,
            "all_ids_and_float16_features_exact": True,
            "windows": 96,
            "bytes": out.stat().st_size,
            "runtime": tokenizer.metadata(),
        },
    )
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return folder


def verify_fit(folder, seed, book):
    out = folder / f"fits/continuous_{seed}"
    fit = json.loads((out / "fit.json").read_text())
    assert fit["complete"] and fit["parameters"] == 704700
    assert fit["protocol_sha256"] == pilot.sha256(folder / "protocol.json")
    assert fit["best_sha256"] == pilot.sha256(out / "best.pt")
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert state["protocol_sha256"] == fit["protocol_sha256"]
    assert state["arm"] == "continuous" and state["seed"] == seed
    assert state["protocol"] == json.loads((folder / "protocol.json").read_text())
    assert torch.equal(state["model"]["codebook"], book)
    assert all(torch.isfinite(v).all() for v in state["model"].values())
    with h5py.File(folder / "features.h5", "r") as f:
        target = torch.from_numpy(f["train/targets"][:, -50:]).reshape(-1, 60)
    assert torch.equal(state["mean"], target.mean(0))
    assert torch.equal(state["scale"], target.std(0, correction=0).clamp_min(1e-4))
    pilot.write_json(
        out / "integrity.json",
        {
            "verified": True,
            "best_sha256": fit["best_sha256"],
            "codebook_float32_sha256": v20.book_digest(book),
            "train_only_target_statistics_exact": True,
        },
    )


def smoke(args, p, book):
    tokenizer = pilot.Tokenizer(args.checkpoint, device=args.device)
    with h5py.File(p["reused_caches"]["1"]["path"], "r") as f:
        values = [
            reencode(
                tokenizer,
                f["train/raw"][i],
                f["train/tokens"][i],
                f["train/continuous"][i],
                args.device,
            )
            for i in range(2)
        ]
    model = pilot.build("continuous", p, {}).to(args.device)
    x = torch.from_numpy(np.stack(values)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(
            model.forward_embeddings(x), torch.zeros(2, 100, 60, device=args.device)
        )
        loss.backward()
        assert all(
            v.grad is not None and torch.isfinite(v.grad).all()
            for v in model.parameters()
        )
        optimizer.step()
    assert torch.equal(model.codebook.cpu(), book)
    pilot.write_json(
        args.root / "smoke.json",
        {
            "passed": True,
            "train_inputs_only": True,
            "weights_discarded": True,
            "exact_token_and_float16_replay": True,
            "finite_gradients": True,
            "protocol_sha256": pilot.sha256(args.root / "protocol.json"),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
    )


def compare(args):
    rows = []
    for subject in v20.SUBJECTS:
        for seed in v20.SEEDS:
            token = (
                args.reference
                / f"subject_{subject:02d}/fits/tokens_{seed}/test_predictions.npz"
            )
            continuous = (
                args.root
                / f"subject_{subject:02d}/fits/continuous_{seed}/test_predictions.npz"
            )
            with np.load(token) as a, np.load(continuous) as b:
                np.testing.assert_array_equal(a["target"], b["target"])
                error = lambda z: float(
                    np.linalg.norm(
                        (z["prediction"].astype(np.float64) - z["target"]).reshape(
                            -1, 20, 3
                        ),
                        axis=-1,
                    ).mean()
                )
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "token": error(a),
                        "continuous": error(b),
                    }
                )
    paired = np.array(
        [
            [
                np.mean([r[k] for r in rows if r["subject"] == s])
                for k in ("token", "continuous")
            ]
            for s in v20.SUBJECTS
        ]
    )
    sample = paired[np.random.default_rng(260930).integers(0, 9, (10000, 9))].mean(1)
    interval = np.percentile(sample[:, 0] / sample[:, 1], [2.5, 97.5]).tolist()
    pilot.write_json(
        args.root / "comparison.json",
        {
            "complete": True,
            "rows": rows,
            "macro_person_token": float(paired[:, 0].mean()),
            "macro_person_continuous": float(paired[:, 1].mean()),
            "token_over_continuous_ratio": float(
                paired[:, 0].mean() / paired[:, 1].mean()
            ),
            "participant_bootstrap_ratio_ci95": interval,
            "within_predeclared_5_percent_margin": interval[1] < 1.05,
            "inference": "Exploratory same-session comparison; no clinical equivalence or independent-session claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "smoke", "subject", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=paths.RESULTS / "dpose_v20_v1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    args.root, args.reference = args.root.resolve(), args.reference.resolve()
    torch.set_num_threads(4)
    if args.mode == "freeze":
        freeze(args)
        return
    p = verify(args)
    book = torch.load(args.checkpoint, map_location="cpu", weights_only=True)[
        "state_dict"
    ]["quantizer.codebook"].float()
    with backend(book) as within:
        if args.mode == "smoke":
            smoke(args, p, book)
            return
        receipt = json.loads((args.root / "smoke.json").read_text())
        assert receipt["passed"] and receipt["protocol_sha256"] == pilot.sha256(
            args.root / "protocol.json"
        )
        if args.mode == "subject":
            assert 0 <= args.index < 9
            folder = prepare(args, p, args.index + 1)
            subargs = copy.copy(args)
            subargs.root = folder
            for index, seed in enumerate(v20.SEEDS):
                subargs.index = index
                pilot.fit(subargs)
                verify_fit(folder, seed, book)
                gc.collect()
                torch.cuda.empty_cache()
        else:
            for subject in v20.SUBJECTS:
                folder = prepare(args, p, subject)
                assert (
                    v20.cache_digest(folder / "features.h5")
                    == p["reused_caches"][str(subject)]["consumed_arrays_sha256"]
                )
                for seed in v20.SEEDS:
                    verify_fit(folder, seed, book)
            within.evaluate(args)
            compare(args)


if __name__ == "__main__":
    main()
