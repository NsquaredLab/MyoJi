"""Source-pinned published CNN retrained on the existing DPose endpoints."""

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import signal
import time

import h5py
import numpy as np
import torch

from paper.receivers.dpose_tbme2024 import (
    CONFIG, EFFECTIVE_BATCH, EPOCHS, MICRO_BATCH, SEEDS, SUBJECTS,
    augment, build, error_mm, optimizer, predict, preprocess, sha,
    target_stats, train_step, write,
)
from paper.common.paths import RESULTS


def now():
    return datetime.now(timezone.utc).isoformat()


def cache_protocol_hash(args):
    """Keep already verified data immutable across a documented runtime repair."""
    p = json.loads((args.root / "protocol.json").read_text())
    return p.get("cache_protocol_sha256", sha(args.root / "protocol.json"))


def sources(base):
    return [base / s for s in (
        "receivers/dpose_tbme2024.py", "receivers/run_dpose_tbme2024.py",
    )] + sorted((base / "vendor/tbme2024").glob("*.py")) + sorted(
        (base / "vendor/tbme2024").glob("*.txt")) + [
        base / "vendor/tbme2024/source_receipt.json"]


def freeze(args):
    path = args.root / "protocol.json"
    if path.exists():
        raise FileExistsError(path)
    base = Path(__file__).resolve().parents[1]
    native = json.loads((args.native / "protocol.json").read_text())
    reference = json.loads((args.reference / "protocol.json").read_text())
    if native["files"] != reference["files"]:
        raise ValueError("Native cache and tokenizer protocol source data differ")
    p = dict(
        created_at=now(), model_label="Published CNN (TBME 2024, retrained)",
        scope="Exploratory same-person/session/task system comparison on a previously inspected split; architecture and native input retained, training/evaluation adapted. Not a reproduction of the published 7.54 mm score or an equal-compute comparison. Tokenizer pretraining included these recordings without pose labels.",
        cells=[[s, seed] for s in SUBJECTS for seed in SEEDS], subjects=list(SUBJECTS), seeds=list(SEEDS),
        model_config=CONFIG, initialized_from="scratch, separately for every participant and seed",
        source_sha256={str(p.relative_to(base)): sha(p) for p in sources(base)},
        archived_source=json.loads((base / "vendor/tbme2024/source_receipt.json").read_text()),
        reference_protocol_sha256=sha(args.reference / "protocol.json"),
        reference_selection_sha256=sha(args.reference / "selection.json"),
        native_protocol_sha256=sha(args.native / "protocol.json"),
        files=reference["files"],
        native_caches={str(s): json.loads((args.native / f"subject_{s:02d}/cache_receipt.json").read_text()) for s in SUBJECTS},
        inputs="Last 192 samples of each existing native 320-sample endpoint window, 2048 Hz, 320 electrodes. Archived 50 Hz Q=75 notch filtfilt(padtype=None), then raw+fourth-order 20 Hz Butterworth sosfiltfilt with default padding. All filtering is confined to samples at or before the target endpoint. Original per-representation/per-grid/window minmax normalization inside the model.",
        targets="Exact final 50 instantaneous reference targets per assigned window, 20 wrist-relative landmarks x 3 coordinates in millimetres (author confirmed). Original output dimensions retained, with current landmark ordering. No window averaging, prediction smoothing, clipping or test-time fitting.",
        target_normalization="Coordinate-wise min/max fitted only on unaugmented training targets, mapped to [-1,1], inverted before validation/test error. Constant coordinate span set to 1.",
        storage="Prepared float32 inputs on shared Vault storage, linked from the run directory. Checkpoints on Work; discard only this campaign's temporary resume checkpoint after its selected model and completion receipt are durable.",
        training=dict(
            epochs=EPOCHS, original_training_endpoints=3200, augmented_endpoints=9600,
            batch=EFFECTIVE_BATCH, physical_batch=MICRO_BATCH,
            accumulation="Eight physical batches of 64 per 512-example update; final update uses 384 endpoints. Original BatchNorm layers retained. Single-device running statistics differ from the historical multi-GPU execution.",
            augmentation="One original, one 5 dB Gaussian-noise, one magnitude-warped copy per training endpoint. Archived additive-noise int16 conversion retained; six cubic-spline control points per grid drawn from N(1,.35), shared across both input representations. Fixed per-participant cache seed 260911 + participant, shared across fit seeds. Explicit RNG replaces archived unseeded magnitude-warp RNG; all endpoints augmented, no sentinel/dropped remainder rows.",
            ordering="All 9600 endpoints shuffled without replacement each epoch by SeedSequence([seed,epoch,2024]); no dropped endpoint.",
            loss="Coordinate L1 on training-minmax-normalized targets",
            optimizer="AdamW, AMSGrad, weight_decay=.01, gradient norm clipped at 1",
            scheduler="OneCycle: initial 1e-4, maximum 10^-2.5, final 1e-7; 50 x 19 optimizer updates",
            precision="CUDA float16 mixed precision; retry overflows without advancing data, schedule or BatchNorm state",
            selection="Lowest validation mean 3D error in mm over 50 completed epochs, earliest on ties; no early stopping or SWA",
            differences_from_archive="Published 50 epochs used (archived launcher defaults to 25). Native batch512 is accumulated on one GPU. Split, instantaneous/wrist-relative target semantics, training-only output normalization, seeded augmentation and validation checkpoint selection adapt the published recipe to current DPose.",
        ),
        evaluation="All 27 fits and selected-checkpoint hashes must pass a global gate before any new test inference. Exact target equality checked against token TCN predictions. Report participant means after averaging three seeds, between-participant sample SD, and per-seed macro means. No tuning from new test scores.",
    )
    write(path, p)
    write(args.root / "selection.json", json.loads((args.reference / "selection.json").read_text()))
    print(json.dumps(dict(frozen=True, fits=27, protocol_sha256=sha(path))), flush=True)


def verify(args):
    p = json.loads((args.root / "protocol.json").read_text())
    base = Path(__file__).resolve().parents[1]
    for name, digest in p["source_sha256"].items():
        if sha(base / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    for file, key in [(args.reference / "protocol.json", "reference_protocol_sha256"),
                      (args.reference / "selection.json", "reference_selection_sha256"),
                      (args.native / "protocol.json", "native_protocol_sha256")]:
        if sha(file) != p[key]:
            raise ValueError(f"Reference provenance changed: {file}")
    return p


def prepare(args, subject):
    p = verify(args)
    data_protocol_hash = cache_protocol_hash(args)
    folder = args.root / f"subject_{subject:02d}"
    receipt = folder / "cache_receipt.json"
    if receipt.exists():
        check_cache(args, subject)
        return
    folder.mkdir(exist_ok=True)
    source = args.native / f"subject_{subject:02d}/native.h5"
    original = p["native_caches"][str(subject)]
    if not original["complete"] or sha(source) != original["native_cache_sha256"]:
        raise ValueError("Native endpoint cache failed checksum")
    storage = (args.cache_root / folder.name) if args.cache_root else folder
    storage.mkdir(parents=True, exist_ok=True)
    temp = storage / "prepared.h5.tmp"
    rng = np.random.default_rng(260911 + subject)
    with h5py.File(source, "r") as native, h5py.File(args.reference / f"subject_{subject:02d}/features.h5", "r") as ref, h5py.File(temp, "w") as out:
        if not native.attrs["complete"] or not ref.attrs["complete"]:
            raise ValueError("Incomplete source cache")
        out.attrs["complete"] = False
        out.attrs["protocol_sha256"] = data_protocol_hash
        for split in ("train", "val", "test"):
            old = native[split]
            targets = old["targets"][:]
            if not np.array_equal(targets, ref[split]["targets"][:, -50:]):
                raise ValueError("Endpoint targets do not match tokenizer cache")
            for key in ("subjects", "tasks", "starts"):
                if not np.array_equal(old[key][:], ref[split][key][:]):
                    raise ValueError(f"Endpoint metadata differ: {key}")
            n = targets.shape[0] * 50
            copies = 3 if split == "train" else 1
            g = out.create_group(split)
            ds = g.create_dataset("inputs", (n * copies, 2, 320, 192), dtype=np.float32)
            g["targets"] = np.tile(targets.reshape(n, 60), (copies, 1))
            for key in ("subjects", "tasks", "starts"):
                g[key] = old[key][:]
            for i in range(len(targets)):
                x = preprocess(old["raw"][i, :, :, -192:])
                ds[i * 50:(i + 1) * 50] = x
                if split == "train":
                    noisy, warped = augment(x, rng)
                    ds[n + i * 50:n + (i + 1) * 50] = noisy
                    ds[2 * n + i * 50:2 * n + (i + 1) * 50] = warped
            if split == "train":
                minimum, span = target_stats(targets)
                out["target_minimum"], out["target_span"] = minimum, span
            out.flush()
            print(json.dumps(dict(subject=subject, prepared=split, endpoints=n * copies)), flush=True)
        out.attrs["complete"] = True
    path = storage / "prepared.h5"
    temp.replace(path)
    if storage != folder:
        link = folder / "prepared.h5"
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != path.resolve():
                raise FileExistsError(link)
        else:
            link.symlink_to(path)
    write(receipt, dict(complete=True, subject=subject, protocol_sha256=data_protocol_hash,
                        prepared_sha256=sha(path), native_sha256=sha(source), all_targets_exact=True,
                        counts=dict(train=9600, val=800, test=800), bytes=path.stat().st_size))


def check_cache(args, subject, full=False):
    folder = args.root / f"subject_{subject:02d}"
    r = json.loads((folder / "cache_receipt.json").read_text())
    if not r["complete"] or r["protocol_sha256"] != cache_protocol_hash(args):
        raise ValueError("Invalid prepared-cache receipt")
    if full and sha(folder / "prepared.h5") != r["prepared_sha256"]:
        raise ValueError("Prepared-cache checksum mismatch")
    with h5py.File(folder / "prepared.h5", "r") as f:
        if not f.attrs["complete"] or f.attrs["protocol_sha256"] != r["protocol_sha256"]:
            raise ValueError("Invalid prepared-cache header")
    return r


def load_data(args, subject, splits):
    check_cache(args, subject)
    data = {}
    with h5py.File(args.root / f"subject_{subject:02d}/prepared.h5", "r") as f:
        minimum = torch.from_numpy(f["target_minimum"][:]).cuda()
        span = torch.from_numpy(f["target_span"][:]).cuda()
        for split in splits:
            g = f[split]
            x = torch.empty(g["inputs"].shape, dtype=torch.float32, device="cuda")
            for start in range(0, len(x), 64):
                x[start:start + 64] = torch.from_numpy(g["inputs"][start:start + 64]).cuda()
            y = torch.from_numpy(g["targets"][:]).cuda()
            data[split] = (x, y)
    return data, minimum, span


def cell_folder(root, cell):
    subject, seed = cell
    return root / f"subject_{subject:02d}/fits/tbme2024_{seed}"


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def fit(args, cell):
    subject, seed = cell
    folder = cell_folder(args.root, cell)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fit.json").exists():
        verify_fit(args.root, cell)
        return 0
    check_cache(args, subject, full=True)
    stop = False

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.manual_seed(seed)
    model = build().cuda()
    data, minimum, span = load_data(args, subject, ("train", "val"))
    x, target = data["train"]
    y = 2 * (target - minimum) / span - 1
    val_x, val_y = data["val"]
    truth = val_y.cpu().numpy()
    per_epoch = math.ceil(len(x) / EFFECTIVE_BATCH)
    opt, schedule = optimizer(model, EPOCHS * per_epoch)
    scaler = torch.amp.GradScaler("cuda")
    step, best, history = 0, float("inf"), []
    protocol_hash = sha(args.root / "protocol.json")
    last = folder / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["protocol_sha256"] != protocol_hash:
            raise ValueError("Resume checkpoint has wrong protocol")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        scaler.load_state_dict(saved["scaler"])
        step, best, history = saved["step"], saved["best"], saved["history"]
        del saved
    started = time.monotonic()

    def save_resume():
        atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), schedule=schedule.state_dict(),
                         scaler=scaler.state_dict(), step=step, best=best, history=history,
                         protocol_sha256=protocol_hash), last)

    while step < EPOCHS * per_epoch:
        epoch, offset = divmod(step, per_epoch)
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2024])).permutation(len(x))
        epoch_start = time.monotonic()
        for batch in range(offset, per_epoch):
            if stop:
                save_resume()
                return 75
            indices = torch.as_tensor(order[batch * EFFECTIVE_BATCH:(batch + 1) * EFFECTIVE_BATCH], device="cuda")
            loss, retries = train_step(model, x, y, indices, opt, scaler, seed + step)
            schedule.step()
            step += 1
            if step % 3 == 0 or batch + 1 == per_epoch:
                write(folder / "progress.json", dict(timestamp=now(), step=step, epoch=epoch,
                      loss=loss, retries=retries, seconds_this_invocation=time.monotonic() - started))
        value = error_mm(predict(model, val_x, minimum, span), truth)
        if value < best:
            best = value
            atomic_save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                             epoch=epoch, step=step, validation_error_mm=best, config=CONFIG,
                             protocol_sha256=protocol_hash), folder / "best.pt")
        history.append(dict(epoch=epoch, step=step, validation_error_mm=value,
                            seconds=time.monotonic() - epoch_start))
        write(folder / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if (epoch + 1) % 5 == 0 or stop:
            save_resume()
    write(folder / "fit.json", dict(complete=True, subject=subject, seed=seed, epochs=EPOCHS,
          steps=step, parameters=sum(p.numel() for p in model.parameters()),
          validation_error_mm=best, best_sha256=sha(folder / "best.pt"),
          protocol_sha256=protocol_hash, cache_receipt_sha256=sha(args.root / f"subject_{subject:02d}/cache_receipt.json"),
          torch=torch.__version__, gpu=torch.cuda.get_device_name(), completed_at=now()))
    verify_fit(args.root, cell)
    if last.exists():
        last.unlink()
    del model, data, x, y, val_x, val_y, target, opt, schedule
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def verify_fit(root, cell):
    folder = cell_folder(root, cell)
    r = json.loads((folder / "fit.json").read_text())
    if (not r["complete"] or r["epochs"] != EPOCHS or r["steps"] != EPOCHS * 19
            or r["protocol_sha256"] != sha(root / "protocol.json")
            or r["best_sha256"] != sha(folder / "best.pt")
            or r["cache_receipt_sha256"] != sha(root / f"subject_{cell[0]:02d}/cache_receipt.json")):
        raise ValueError(f"Invalid completed fit: {cell}")
    return r


def gate(args):
    p = verify(args)
    if p["cells"] != [[s, seed] for s in SUBJECTS for seed in SEEDS]:
        raise ValueError("Expected every participant and seed")
    fits = [verify_fit(args.root, c) for c in p["cells"]]
    for subject in SUBJECTS:
        check_cache(args, subject, full=True)
    receipt = dict(passed=True, timestamp=now(), protocol_sha256=sha(args.root / "protocol.json"),
                   fits=27, selected_checkpoints={f"{r['subject']}_{r['seed']}": r["best_sha256"] for r in fits})
    write(args.root / "test_gate.json", receipt)
    return receipt


def test(args, cell):
    g = json.loads((args.root / "test_gate.json").read_text())
    if not g["passed"] or g["fits"] != 27 or g["protocol_sha256"] != sha(args.root / "protocol.json"):
        raise ValueError("Test gate did not pass")
    r = verify_fit(args.root, cell)
    if r["best_sha256"] != g["selected_checkpoints"][f"{cell[0]}_{cell[1]}"]:
        raise ValueError("Selected checkpoint changed after test gate")
    folder = cell_folder(args.root, cell)
    if (folder / "test.json").exists():
        return
    data, minimum, span = load_data(args, cell[0], ("test",))
    x, y = data["test"]
    model = build().cuda()
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["state_dict"])
    pred = predict(model, x, minimum, span).reshape(16, 50, 60)
    target = y.cpu().numpy().reshape(16, 50, 60)
    source = args.reference / f"subject_{cell[0]:02d}/fits/tokens_{cell[1]}/test_predictions.npz"
    with np.load(source) as f:
        if not np.array_equal(f["target"], target):
            raise ValueError("Test targets differ from existing token predictions")
    path = folder / "test_predictions.npz"
    np.savez_compressed(path, prediction=pred, target=target, subject=np.full(16, cell[0]))
    write(folder / "test.json", dict(complete=True, prediction_sha256=sha(path),
          protocol_sha256=g["protocol_sha256"], selected_checkpoint_sha256=r["best_sha256"],
          error_mm=error_mm(pred, target), paired_targets_exact=True))


def report(args):
    p = verify(args)
    rows, comparisons = [], []
    for cell in p["cells"]:
        r = verify_fit(args.root, cell)
        folder = cell_folder(args.root, cell)
        receipt = json.loads((folder / "test.json").read_text())
        if not receipt["complete"] or sha(folder / "test_predictions.npz") != receipt["prediction_sha256"]:
            raise ValueError("Invalid test output")
        with np.load(folder / "test_predictions.npz") as f:
            pred, truth = f["prediction"], f["target"]
        row = dict(subject=cell[0], seed=cell[1], model="tbme2024_retrained",
                   mean_3d_error=error_mm(pred, truth), parameters=r["parameters"])
        rows.append(row)
        for name, root, prefix in [("token_tcn", args.reference, "tokens"), ("waveform_cnn", args.native, "raulnet")]:
            with np.load(root / f"subject_{cell[0]:02d}/fits/{prefix}_{cell[1]}/test_predictions.npz") as f:
                if not np.array_equal(truth, f["target"]):
                    raise ValueError("Paired comparison targets differ")
                comparisons.append(dict(subject=cell[0], seed=cell[1], model=name,
                                        mean_3d_error=error_mm(f["prediction"], truth)))
    participants = [float(np.mean([r["mean_3d_error"] for r in rows if r["subject"] == s])) for s in SUBJECTS]
    seed_means = [float(np.mean([r["mean_3d_error"] for r in rows if r["seed"] == seed])) for seed in SEEDS]
    result = dict(complete=True, verified=True, paired_targets_exact=True, units="mm",
                  protocol_sha256=sha(args.root / "protocol.json"), model_label=p["model_label"],
                  summary=dict(mean=float(np.mean(participants)), participant_sd=float(np.std(participants, ddof=1)),
                               seed_sd=float(np.std(seed_means, ddof=1))),
                  participant_means=participants, seed_macro_means=seed_means,
                  per_subject_seed=rows, paired_comparators=comparisons, scope=p["scope"])
    write(args.root / "results.json", result)
    print(json.dumps(result["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "prepare", "fit", "gate", "test", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=RESULTS / "dpose_within_v1")
    parser.add_argument("--native", type=Path, default=RESULTS / "dpose_raulnet_v1")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if args.mode == "freeze":
        freeze(args)
        return 0
    p = verify(args)
    if args.mode == "prepare":
        prepare(args, SUBJECTS[args.index])
    elif args.mode == "fit":
        return fit(args, p["cells"][args.index])
    elif args.mode == "test":
        test(args, p["cells"][args.index])
    else:
        globals()[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
