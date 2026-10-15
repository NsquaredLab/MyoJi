"""Frozen EMG tokens -> causal transformer -> personalized 3D kinematics."""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

import h5py
from paper.hdemg import run_dpose_pilot as pilot
from paper.hdemg import run_dpose_within as within
import torch
from paper.common.paths import LEARNED_TOKENIZER, REPO, RESULTS

ARMS = ("tokens",)
SEEDS = (260911, 260912, 260913)
pilot.ARMS = within.ARMS = ARMS
pilot.SEEDS = within.SEEDS = SEEDS


def cache_digest(path):
    """Bind the exact token/target/geometry arrays consumed by this campaign."""
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        assert f.attrs["complete"]
        for name in ["codebook", "descriptors"] + [
            f"{s}/{k}"
            for s in ("train", "val", "test")
            for k in ("tokens", "targets", "subjects", "tasks", "starts")
        ]:
            data = f[name][:]
            h.update(name.encode())
            h.update(str(data.shape).encode())
            h.update(data.dtype.str.encode())
            h.update(data.tobytes())
    return h.hexdigest()


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError(args.root / "protocol.json")
    old = json.loads((args.reference / "protocol.json").read_text())
    assert old["readout"]["kind"] == "tcn"
    p = copy.deepcopy(old)
    p["arms"] = list(ARMS)
    p["readout"].update(kind="transformer", layers=2)
    p["scope"] = (
        "Exploratory transformer architecture follow-up on the same personalized, same-session/task chronological split. Previous TCN test results have already been inspected. Tokenizer saw these recordings during pretraining. Not unseen-person/session/task performance or an untouched confirmatory test."
    )
    p["models"] = (
        "Frozen tokenizer and codebook; learned 256-to-64 projection, fixed-montage spatial pooling into four slots, 2-layer causal temporal transformer with width 64, 4 heads, feedforward width 128 and fixed sinusoidal positions over 100 frames, then 60-coordinate output. Separate decoder per subject and seed; 27 fits. No raw EMG, pose history, task IDs or calibration metadata reach the decoder."
    )
    p["evaluation"] = (
        "All 27 fits must complete before test scoring. Compare with the frozen prior token-TCN run and personalized constant baseline on identical windows/seeds. No hyperparameter sweep or test-driven reruns. Same optimization budget, not parameter-matched architectures."
    )
    p["reference_protocol_sha256"] = pilot.sha256(args.reference / "protocol.json")
    p["reference_results_sha256"] = pilot.sha256(args.reference / "results.json")
    p["cache"] = (
        "Read-only symbolic links to completed within-person feature caches; no new encoding. Hash token IDs, targets, geometry, codebook and window identities before reuse."
    )
    base = Path(__file__).resolve().parents[1]
    p["source_sha256"] = {
        str(path.relative_to(REPO)): pilot.sha256(path)
        for path in [
            Path(__file__).resolve(),
            base / "hdemg/run_dpose_pilot.py",
            base / "hdemg/run_dpose_within.py",
            REPO / "src/myoji/decoders.py",
            base / "receivers/report_dpose_transformer.py",
        ]
    }
    p["reused_caches"] = {}
    for subject in within.SUBJECTS:
        ref = args.reference / f"subject_{subject:02d}"
        source_protocol = pilot.sha256(ref / "protocol.json")
        with h5py.File(ref / "features.h5", "r") as f:
            assert f.attrs["complete"] and f.attrs["protocol_sha256"] == source_protocol
        p["reused_caches"][str(subject)] = {
            "path": str((ref / "features.h5").resolve()),
            "source_protocol_sha256": source_protocol,
            "consumed_arrays_sha256": cache_digest(ref / "features.h5"),
        }
    pilot.write_json(args.root / "protocol.json", p)
    selection = json.loads((args.reference / "selection.json").read_text())
    selection["protocol_sha256"] = pilot.sha256(args.root / "protocol.json")
    pilot.write_json(args.root / "selection.json", selection)
    for subject in within.SUBJECTS:
        folder = args.root / f"subject_{subject:02d}"
        pilot.write_json(
            folder / "protocol.json",
            dict(
                p,
                subject=subject,
                campaign_protocol_sha256=pilot.sha256(args.root / "protocol.json"),
            ),
        )
        (folder / "features.h5").symlink_to(p["reused_caches"][str(subject)]["path"])
    print(
        json.dumps({"fits": 27, "reused_windows": 864, "readout": p["readout"]}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "subject", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=RESULTS / "dpose_within_v1")
    parser.add_argument("--checkpoint", type=Path, default=LEARNED_TOKENIZER)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "freeze":
        freeze(args)
        return
    p = json.loads((args.root / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        assert pilot.sha256(REPO / name) == digest
    assert pilot.sha256(args.checkpoint) == p["checkpoint_sha256"]
    assert (
        pilot.sha256(args.reference / "protocol.json") == p["reference_protocol_sha256"]
    )
    assert (
        pilot.sha256(args.reference / "results.json") == p["reference_results_sha256"]
    )
    if args.mode == "subject":
        subject = args.index + 1
        assert subject in within.SUBJECTS
        args.root = args.root / f"subject_{subject:02d}"
        assert (
            cache_digest(args.root / "features.h5")
            == p["reused_caches"][str(subject)]["consumed_arrays_sha256"]
        )
        for index in range(3):
            args.index = index
            pilot.fit(args)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        for subject in within.SUBJECTS:
            assert (
                cache_digest(args.root / f"subject_{subject:02d}/features.h5")
                == p["reused_caches"][str(subject)]["consumed_arrays_sha256"]
            )
        within.evaluate(args)
        from paper.receivers.report_dpose_transformer import report

        report(args.root, args.reference, args.root / "report")


if __name__ == "__main__":
    main()
