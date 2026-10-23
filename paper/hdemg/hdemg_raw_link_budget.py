"""Raw EMG from only as many electrodes as the assumed link carries, decoded by the adapted CNN.

Declared before running (2026-10-10); exploratory. At the assumed 1.4 Mbit/s application rate, int16 EMG at
2,048 Hz fits floor(1,400,000 / 32,768) = 42 channels. Montage: the first 42 electrodes of the 320-channel
layout (first 8x8 forearm array, rows 1 to 5 and the first two electrodes of row 6), i.e. a subset of the
63-electrode SPAR montage on the same array. The other 278 channels are set to zero before the raw and
20-Hz-low-pass inputs are formed, so the pinned RaulNetV19 q32 architecture (7,489,436 parameters), its
per-grid window normalization and the adapted recipe of results/hdemg_raulnet_v1 (run_hdemg_raulnet.fit:
150 epochs, 10,000 endpoints per epoch, batch 64, L1, AdamW amsgrad, OneCycle, SWA, 16-mixed, validation
selection) are unchanged, with seeds 260911-260913 and the campaign's windows and targets.

Contrasts (nine participants, seeds averaged; exact sign-flip over 512 patterns, Holm over the two;
participant bootstrap 10,000 draws, seed 260920): raw 42 minus SPAR tokens at 63 electrodes
(results/hdemg_spar_63_electrodes_v1), and raw 42 minus the adapted CNN at 320 electrodes. SPAR tokens at
320 electrodes are reported descriptively.

    uv run python -m paper.hdemg.hdemg_raw_link_budget run --seed 260911 [--subjects ...]
    ... report
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
from paper.receivers import run_hdemg_raulnet as raul  # noqa: E402
from paper.receivers.hdemg_raulnet_model import CONFIG, configure_eager, dual_input, native_windows, source_ends  # noqa: E402
from paper.common.paths import HDEMG  # noqa: E402

OUT = BASE / "results/hdemg_raw_link_budget_v1"
CAMPAIGN = BASE / "results/hdemg_spar_simple_full_v1"
CNN320 = BASE / "results/hdemg_raulnet_v1"
SPAR63 = BASE / "results/hdemg_spar_63_electrodes_v1"
DATA = HDEMG
N = 42


def split_data(subject, device):
    p = json.loads((CAMPAIGN / f"subject_{subject:02d}/protocol.json").read_text())
    emg = joblib.load(DATA / f"Sub{subject}_emg.pkl")
    kin = joblib.load(DATA / f"Sub{subject}_kinematics.pkl")
    out = {}
    for split in ("train", "val", "test"):
        rows = [r for r in p["windows"] if r["split"] == split]
        raw, targets = [], []
        for r in rows:
            key = next(k for k in emg if int(k) == int(r["task"]))
            x = emg[key]
            w = native_windows(x, int(r["start"])).astype(np.float32)        # (50, 320, 320)
            w[:, N:, :] = 0.0
            raw.append(w)
            y = kin[key]
            ends = source_ends(int(r["start"]))
            targets.append(np.stack([(y[1:, :, ix] - y[:1, :, ix]).reshape(60) for ix in ends]).astype(np.float32))
        raw = np.concatenate(raw)                                               # (n*50, 320, 320)
        feats = torch.empty(len(raw), 2, 320, 320, device=device)
        for i in range(0, len(raw), 64):
            feats[i:i + 64] = dual_input(torch.from_numpy(raw[i:i + 64]).to(device))
        assert torch.isfinite(feats).all()
        out[split] = (feats, torch.from_numpy(np.concatenate(targets)).to(device))
    return out


def run(subject, seed, device):
    folder = OUT / f"subject_{subject:02d}"
    if (folder / f"fits/raulnet_{seed}/test.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    if not (OUT / "protocol.json").exists():
        raul.write(OUT / "protocol.json", dict(name="hdemg_raw_link_budget_v1", electrodes=N, montage="first 42 of 320",
                                               recipe="run_hdemg_raulnet.fit unchanged", seeds=[260911, 260912, 260913],
                                               docstring=__doc__))
    data = split_data(subject, device)
    args = SimpleNamespace(root=OUT, device=device)
    raul.fit(args, subject, seed, data["train"], data["val"])
    out = folder / f"fits/raulnet_{seed}"
    model = configure_eager(raul.MatchedV19(**CONFIG)).to(device)
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    x, y = data["test"]
    pred = raul.predict(model, x)
    target = y.cpu().numpy()
    cnn = np.load(CNN320 / f"subject_{subject:02d}/fits/raulnet_{seed}/test_predictions.npz")
    assert np.allclose(cnn["target"].reshape(-1, 60), target), "test targets differ from the 320-electrode CNN campaign"
    err = float(np.linalg.norm((pred - target).reshape(-1, 20, 3), axis=-1).mean())
    np.savez_compressed(out / "test_predictions.npz", prediction=pred.reshape(-1, 50, 60), target=target.reshape(-1, 50, 60))
    raul.write(out / "test.json", dict(subject=subject, seed=seed, error_mm=err, selection=state["selection"]))
    print(json.dumps(dict(subject=subject, seed=seed, error_mm=err)), flush=True)
    del data, model
    torch.cuda.empty_cache()


def exact_p(d):
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))


def contrast(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return dict(mean=float(d.mean()), per_participant=d.tolist(), positive=int((d > 0).sum()),
                bootstrap95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], p_exact=exact_p(d))


def cnn320_error(subject):
    errs = []
    for seed in (260911, 260912, 260913):
        z = np.load(CNN320 / f"subject_{subject:02d}/fits/raulnet_{seed}/test_predictions.npz")
        errs.append(float(np.linalg.norm((z["prediction"] - z["target"]).reshape(-1, 20, 3), axis=-1).mean()))
    return float(np.mean(errs))


def report():
    rng = np.random.default_rng(260920)
    raw42 = [float(np.mean([json.loads((OUT / f"subject_{s:02d}/fits/raulnet_{seed}/test.json").read_text())["error_mm"]
                            for seed in (260911, 260912, 260913)])) for s in range(1, 10)]
    s63 = json.loads((SPAR63 / "summary.json").read_text())["per_participant"]
    spar63, spar320 = s63["e63"]["tokens"], s63["e320"]["tokens"]
    cnn320 = [cnn320_error(s) for s in range(1, 10)]
    tests = dict(raw42_minus_spar63_tokens=contrast(raw42, spar63, rng), raw42_minus_cnn320=contrast(raw42, cnn320, rng))
    names = sorted(tests, key=lambda k: tests[k]["p_exact"])
    running = 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, tests[k]["p_exact"] * (len(names) - i)))
        tests[k]["p_holm"] = running
    summary = dict(electrodes=N, mean_mm=dict(raw42_cnn=float(np.mean(raw42)), spar63_tokens=float(np.mean(spar63)),
                                              spar320_tokens=float(np.mean(spar320)), cnn320=float(np.mean(cnn320))),
                   per_participant=dict(raw42_cnn=raw42, spar63_tokens=spar63, spar320_tokens=spar320, cnn320=cnn320),
                   payload_kB_s=dict(raw42_int16=N * 2 * 2048 / 1000, spar63_tokens_packed=63 * 10 / 8 * 50 / 1000,
                                     spar63_tokens_uint16=63 * 2 * 50 / 1000), **tests)
    raul.write(OUT / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_participant"}, indent=1, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--seed", type=int, default=260911)
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "run":
        for subject in args.subjects:
            run(subject, args.seed, args.device)
    else:
        report()


if __name__ == "__main__":
    main()
