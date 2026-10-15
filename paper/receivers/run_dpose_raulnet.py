"""Matched-endpoint, personalized RaulNet V19 q32 baseline."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import h5py
import joblib
import lightning as L
import numpy as np
import torch
from paper.receivers.dpose_raulnet_model import (
    CONFIG,
    SOURCE_SHA256,
    RaulNetV19,
    configure_eager,
    dual_input,
    native_windows,
    source_ends,
)
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from paper.common.paths import HDEMG, RESULTS
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

SUBJECTS = tuple(range(1, 10))
SEEDS = (260911, 260912, 260913)
TRAINING_DATA = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def freeze(args):
    if (args.root / "protocol.json").exists():
        raise FileExistsError(args.root / "protocol.json")
    old = json.loads((args.reference / "protocol.json").read_text())
    base = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        base / "receivers/dpose_raulnet_model.py",
        base / "vendor/raulnet_v19.py",
        base / "receivers/report_dpose_raulnet.py",
    ]
    p = {
        "scope": "Exploratory personalized same-person/session/task baseline on the previously inspected chronological token-decoder split. Same supervised targets and split, different input/context/model/training recipe. No unseen-person claim or equal-compute claim.",
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "files": old["files"],
        "reference_protocol_sha256": sha(args.reference / "protocol.json"),
        "reference_results_sha256": sha(args.reference / "results.json"),
        "transformer_results_sha256": sha(args.transformer / "results.json"),
        "source_sha256": {str(s.relative_to(base)): sha(s) for s in sources},
        "model": "Pinned unmodified RaulNetV19 q32 architecture, 7,489,436 parameters; initialized from scratch; optional torch.compile disabled only.",
        "model_source_sha256": SOURCE_SHA256,
        "model_config": CONFIG,
        "inputs": "320 native 2048-Hz samples ending at each scored token timestamp; raw plus MyoVerse-equivalent four-pass 20-Hz lowpass (Q=.707, native torchaudio clamp behavior), window-local zero state. Native per-grid/window V19 normalization. No token prefix clipping/normalization or 20-ms tokenizer FIR delay. All input samples are at or before the target endpoint and inside the original assigned 3-second window.",
        "targets": "Exact 60-coordinate instantaneous targets from final 50 frames of each reference window; no window target averaging. Stored coordinate units, physical unit unverified. No target standardization for native V19 L1 recipe.",
        "training": {
            "epochs": 150,
            "sampled_endpoints_per_epoch": 10000,
            "batch_size": 64,
            "seed_policy": "same three seeds as token heads; replacement sampler and RNG reset to seed+epoch",
            "loss": "unstandardized coordinate L1",
            "optimizer": "native V19 AdamW amsgrad, weight_decay=.32",
            "lr": 1e-4,
            "scheduler": "native OneCycle max_lr=1e-4*10**1.5; SWA from .75 with lr=1e-4 and 5 annealing epochs",
            "noise_std": 0.1,
            "precision": "16-mixed",
            "selection": "Lowest validation mean 3D point error, including final SWA candidate; earliest epoch on ties. No early stopping. Same available supervised endpoints, larger training exposure than token heads; not equal compute.",
        },
        "evaluation": "Only after all 27 fits complete and checkpoint hashes pass. Recompute every prediction error, baseline and paired token comparisons. Report 9 subject means, seed macro-mean and sample SD; no test-driven changes.",
    }
    write(args.root / "protocol.json", p)
    selection = json.loads((args.reference / "selection.json").read_text())
    selection["protocol_sha256"] = sha(args.root / "protocol.json")
    write(args.root / "selection.json", selection)
    print(
        json.dumps({"frozen": True, "fits": 27, "model_parameters": 7489436}),
        flush=True,
    )


def prepare(args, subject):
    folder = args.root / f"subject_{subject:02d}"
    folder.mkdir(exist_ok=True)
    receipt = folder / "cache_receipt.json"
    if receipt.exists():
        r = json.loads(receipt.read_text())
        assert r["complete"] and r["protocol_sha256"] == sha(
            args.root / "protocol.json"
        )
        with h5py.File(folder / "native.h5", "r") as f:
            assert f.attrs["complete"]
        return
    if (folder / "native.h5").exists():
        raise FileExistsError("Incomplete native cache requires explicit recovery")
    p = json.loads((args.root / "protocol.json").read_text())
    paths = [args.data / f"Sub{subject}_{kind}.pkl" for kind in ("emg", "kinematics")]
    for path in paths:
        assert sha(path) == p["files"][path.name]
    emg, kin = [joblib.load(path) for path in paths]
    with (
        h5py.File(args.reference / f"subject_{subject:02d}/features.h5", "r") as ref,
        h5py.File(folder / "native.h5", "w") as out,
    ):
        assert ref.attrs["complete"]
        out.attrs["complete"] = False
        out.attrs["protocol_sha256"] = sha(args.root / "protocol.json")
        counts = {}
        for split in ("train", "val", "test"):
            r = ref[split]
            assert np.all(r["subjects"][:] == subject)
            n = len(r["subjects"])
            g = out.create_group(split)
            dtype = next(iter(emg.values())).dtype
            ds = g.create_dataset("raw", (n, 50, 320, 320), dtype=dtype)
            g["targets"] = r["targets"][:, -50:]
            for key in ("subjects", "tasks", "starts"):
                g[key] = r[key][:]
            for i, (task, start) in enumerate(zip(r["tasks"][:], r["starts"][:])):
                x, y = emg[str(task)], kin[str(task)]
                assert x.dtype == dtype
                assert x.shape[0] == 320 and y.shape == (21, 3, x.shape[1])
                samples = native_windows(x, int(start))
                assert np.isfinite(samples).all()
                ds[i] = samples
                ends = source_ends(int(start))
                expected = np.stack(
                    [(y[1:, :, ix] - y[:1, :, ix]).reshape(60) for ix in ends]
                ).astype(np.float32)
                assert np.array_equal(expected, g["targets"][i])
            counts[split] = n * 50
            out.flush()
            print(
                json.dumps({"subject": subject, "cached": split, "endpoints": n * 50}),
                flush=True,
            )
        out.attrs["complete"] = True
    write(
        receipt,
        {
            "complete": True,
            "subject": subject,
            "endpoints": counts,
            "all_targets_exact": True,
            "protocol_sha256": sha(args.root / "protocol.json"),
            "native_cache_sha256": sha(folder / "native.h5"),
        },
    )


def load_data(folder, split, device):
    with h5py.File(folder / "native.h5", "r") as f:
        assert f.attrs["complete"]
        g = f[split]
        n = len(g["subjects"]) * 50
        features = torch.empty(n, 2, 320, 320, device=device)
        raw = g["raw"][:].reshape(n, 320, 320)
        for i in range(0, n, 64):
            features[i : i + 64] = dual_input(
                torch.from_numpy(raw[i : i + 64]).to(device)
            )
        target = torch.from_numpy(g["targets"][:].reshape(n, 60)).to(device)
    assert torch.isfinite(features).all() and torch.isfinite(target).all()
    return features, target


class MatchedV19(RaulNetV19):
    def training_step(self, batch, batch_idx):
        ix = batch[0]
        x = TRAINING_DATA["train"][0][ix]
        pred = self(x + torch.randn_like(x) * 0.1)
        loss = self.criterion(pred, TRAINING_DATA["train"][1][ix])
        assert torch.isfinite(loss)
        self.log("train_l1", loss, on_step=False, on_epoch=True, batch_size=len(ix))
        return loss

    def validation_step(self, batch, batch_idx):
        ix = batch[0]
        pred = self(TRAINING_DATA["val"][0][ix]).float()
        error = (
            (pred - TRAINING_DATA["val"][1][ix]).reshape(-1, 20, 3).norm(dim=-1).mean()
        )
        assert torch.isfinite(error)
        self.log("val_3d", error, on_step=False, on_epoch=True, batch_size=len(ix))


class Progress(L.Callback):
    def __init__(self, out, seed, sampler):
        self.out, self.seed, self.sampler = out, seed, sampler
        self.started = time.time()
        self.rows = (
            json.loads((out / "history.json").read_text())
            if (out / "history.json").exists()
            else []
        )

    def on_train_epoch_start(self, trainer, pl_module):
        torch.manual_seed(self.seed + trainer.current_epoch)
        self.sampler.generator.manual_seed(self.seed + trainer.current_epoch)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        row = {
            "epoch": trainer.current_epoch + 1,
            "validation_3d": float(trainer.callback_metrics["val_3d"]),
            "seconds_this_invocation": time.time() - self.started,
        }
        self.rows.append(row)
        write(self.out / "history.json", self.rows)
        print(json.dumps(row), flush=True)


@torch.inference_mode()
def predict(model, x):
    model.eval()
    parts = []
    for i in range(0, len(x), 64):
        with torch.autocast("cuda", dtype=torch.float16):
            parts.append(model(x[i : i + 64]).float().cpu())
    pred = torch.cat(parts).numpy()
    assert np.isfinite(pred).all()
    return pred


def fit(args, subject, seed, train, val):
    folder = args.root / f"subject_{subject:02d}"
    out = folder / "fits" / f"raulnet_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fit.json").exists():
        assert json.loads((out / "fit.json").read_text())["complete"]
        return
    L.seed_everything(seed, workers=True)
    model = configure_eager(MatchedV19(**CONFIG))
    TRAINING_DATA.update(train=train, val=val)
    dataset = TensorDataset(torch.arange(len(train[0])))
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=10000,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=0)
    validation = DataLoader(
        TensorDataset(torch.arange(len(val[0]))), batch_size=64, num_workers=0
    )
    ckpt = ModelCheckpoint(
        dirpath=out / "checkpoints",
        monitor="val_3d",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch{epoch:03d}",
    )
    progress = Progress(out, seed, sampler)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=150,
        callbacks=[
            ckpt,
            StochasticWeightAveraging(
                swa_lrs=1e-4, swa_epoch_start=0.75, annealing_epochs=5
            ),
            progress,
        ],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        benchmark=False,
        default_root_dir=out,
    )
    started = time.time()
    last = out / "checkpoints/last.ckpt"
    trainer.fit(
        model, loader, validation, ckpt_path=str(last) if last.exists() else None
    )
    # Native SWA weights are installed at fit end; select them only on validation.
    model.to(args.device)
    swa_pred = predict(model, val[0])
    target = val[1].cpu().numpy()
    swa_error = float(
        np.linalg.norm((swa_pred - target).reshape(-1, 20, 3), axis=-1).mean()
    )
    regular = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)
    use_swa = swa_error < float(ckpt.best_model_score)
    state = (
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if use_swa
        else regular["state_dict"]
    )
    torch.save(
        {
            "state_dict": state,
            "config": CONFIG,
            "subject": subject,
            "seed": seed,
            "selection": "final_swa" if use_swa else "validation_checkpoint",
            "protocol_sha256": sha(args.root / "protocol.json"),
        },
        out / "best.pt",
    )
    write(
        out / "fit.json",
        {
            "complete": True,
            "subject": subject,
            "seed": seed,
            "parameters": 7489436,
            "epochs": 150,
            "best_validation_3d": min(swa_error, float(ckpt.best_model_score)),
            "final_swa_validation_3d": swa_error,
            "selection": "final_swa" if use_swa else "validation_checkpoint",
            "seconds_this_invocation": time.time() - started,
            "protocol_sha256": sha(args.root / "protocol.json"),
            "best_sha256": sha(out / "best.pt"),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
    )
    del trainer, model, state, regular
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(args):
    for subject in SUBJECTS:
        for seed in SEEDS:
            folder = args.root / f"subject_{subject:02d}/fits/raulnet_{seed}"
            r = json.loads((folder / "fit.json").read_text())
            assert (
                r["complete"]
                and r["protocol_sha256"] == sha(args.root / "protocol.json")
                and r["best_sha256"] == sha(folder / "best.pt")
            )
    for subject in SUBJECTS:
        folder = args.root / f"subject_{subject:02d}"
        x, y = load_data(folder, "test", args.device)
        with h5py.File(folder / "native.h5", "r") as f:
            np.savez_compressed(
                folder / "baseline_training_targets.npz", target=f["train/targets"][:]
            )
        for seed in SEEDS:
            out = folder / "fits" / f"raulnet_{seed}"
            model = configure_eager(MatchedV19(**CONFIG)).to(args.device)
            state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
            pred = predict(model, x).reshape(-1, 50, 60)
            np.savez_compressed(
                out / "test_predictions.npz",
                prediction=pred,
                target=y.cpu().numpy().reshape(-1, 50, 60),
                subject=np.full(len(pred), subject),
            )
            del model
        del x, y
        torch.cuda.empty_cache()
    from paper.receivers.report_dpose_raulnet import report

    report(args.root, args.reference, args.transformer, args.root / "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "subject", "evaluate", "smoke"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=RESULTS / "dpose_within_v1")
    parser.add_argument("--transformer", type=Path, default=RESULTS / "dpose_transformer_v1")
    parser.add_argument("--data", type=Path, default=HDEMG)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "smoke":
        folder = args.root / "smoke"
        folder.mkdir(parents=True, exist_ok=True)
        L.seed_everything(42)
        x = dual_input(torch.randn(8, 320, 320, device=args.device))
        y = torch.randn(8, 60, device=args.device)
        TRAINING_DATA.update(train=(x, y), val=(x, y))
        model = configure_eager(MatchedV19(**CONFIG))
        dataset = TensorDataset(torch.arange(8))
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=8,
            generator=torch.Generator().manual_seed(42),
        )
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)
        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            max_epochs=4,
            limit_train_batches=2,
            callbacks=[
                StochasticWeightAveraging(
                    swa_lrs=1e-4, swa_epoch_start=0.5, annealing_epochs=1
                ),
                Progress(folder, 42, sampler),
            ],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, loader, DataLoader(dataset, batch_size=2))
        assert np.isfinite(predict(model.to(args.device), x)).all()
        write(
            folder / "receipt.json",
            {
                "passed": True,
                "synthetic_data_only": True,
                "checks": [
                    "native fused optimizer",
                    "AMP training",
                    "OneCycle and SWA",
                    "validation callback",
                    "finite inference",
                ],
                "torch": torch.__version__,
            },
        )
        return
    if args.mode == "freeze":
        freeze(args)
        return
    p = json.loads((args.root / "protocol.json").read_text())
    for name, h in p["source_sha256"].items():
        assert sha(Path(__file__).resolve().parents[1] / name) == h
    assert sha(args.reference / "protocol.json") == p["reference_protocol_sha256"]
    assert sha(args.reference / "results.json") == p["reference_results_sha256"]
    assert sha(args.transformer / "results.json") == p["transformer_results_sha256"]
    if args.mode == "subject":
        subject = args.index + 1
        assert subject in SUBJECTS
        prepare(args, subject)
        folder = args.root / f"subject_{subject:02d}"
        train, val = (
            load_data(folder, "train", args.device),
            load_data(folder, "val", args.device),
        )
        for seed in SEEDS:
            fit(args, subject, seed, train, val)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
