"""Optimized STM32N6 SPAR port on the 63-electrode montage with the refitted 63-electrode pipelines.

Declared before running (2026-10-10). Host build of the shipped optimized code (arm64, standardization constants
zeroed, as in validate.py) replays the 16 test windows of each participant on the first 63 electrodes: C
calibration and normalization, spectra and matrix logarithms in C with band weights over the 63 channels, then
each participant's refitted 63-electrode standardization and codebook (results/dpose_spar_63_electrodes_v1) and the
port's batched nearest-code search. Two montages: "board", the firmware's N-channel rule (benchmark.c montage():
an absent neighbour is replaced by the electrode itself and its coupling masked), and "reference", the
grid_layout rule of the accuracy study (mirrored neighbour substituted, coupling masked). Outputs: token agreement
with the reference-implementation tokens and the frozen 63-electrode token decoders' error on each stream.

    uv run python -m paper.firmware.stm32n6_pose_validation.replay63 \
        --data <dir with subject_XX/{windows.npy,prepare.json}>
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
from paper.firmware.stm32n6_pose_validation import validate as V  # noqa: E402
from paper.hdemg import run_dpose_spar_simple as pilot  # noqa: E402
from myoji.spar import grid_layout  # noqa: E402

N = 63
RUN63 = BASE / "results/dpose_spar_63_electrodes_v1"
OUT = BASE / "results/stm32n6_pose_validation_v1/replay63"


def montages(lib):
    full_m = np.array([[lib.members[c][j] for j in range(4)] for c in range(320)], np.uint16)
    full_k = np.array([[lib.masks[c][j] for j in range(3)] for c in range(320)], np.uint8)
    bm, bk = np.empty((N, 4), np.uint16), np.empty((N, 3), np.uint8)
    for c in range(N):
        bm[c] = [m if m < N else c for m in full_m[c]]
        bk[c] = [full_k[c][0], full_k[c][1] and full_m[c][2] < N, full_k[c][2] and full_m[c][3] < N]
    lay = grid_layout([np.r_[np.arange(N), -1].reshape(8, 8)])
    rm, rk = lay.members.astype(np.uint16), lay.mask.astype(np.uint8)
    return dict(board=(bm, bk), reference=(rm, rk))


def entries(lib, spectra, raw, normalized, end, members, masks):
    for c in range(N):
        lib.fast_spar_spectra(V.ptr(normalized[c, end - 319:end + 1]), V.ptr(spectra[c]))
    m = np.ascontiguousarray(members); k = np.ascontiguousarray(masks)
    lib.fast_spar_features(V.ptr(spectra), V.ptr(raw), N, m.ctypes.data_as(V.ct.c_void_p), k.ctypes.data_as(V.ct.c_void_p))
    return raw[:N].copy()


def exact_p(d):
    signs = np.asarray(list(product((-1., 1.), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    args = ap.parse_args()
    lib = V.load_lib(BASE / "results/stm32n6_pose_validation_v1/build_fast_darwin/host.dylib")
    mont = montages(lib)
    diff = {name: int((np.any(m != mont["reference"][0], 1) | np.any(k != mont["reference"][1], 1)).sum()) for name, (m, k) in mont.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in args.subjects:
        folder = args.data / f"subject_{subject:02d}"
        order = [tuple(x) for x in json.loads((folder / "prepare.json").read_text())["window_order"]]
        windows = np.load(folder / "windows.npy", mmap_mode="r")
        with h5py.File(RUN63 / f"subject_{subject:02d}/features.h5", "r") as h:
            mean, scale, mask = h["feature_mean"][:], h["feature_scale"][:], h["mask"][:].astype(np.float32)
            book, ref_tokens, targets = h["codebook"][:], h["test/tokens"][:], h["test/targets"][:]
        stream = V.CStream(lib)
        packed = stream.prepare_book(book)
        out = {name: np.zeros((16, 320, 100), np.int64) for name in mont}
        dev = {name: 0.0 for name in mont}
        with h5py.File(RUN63 / f"subject_{subject:02d}/features.h5", "r") as h:
            ref_cont = h["test/continuous"][:, :N]                       # (16, 63, 100, 18)
        for n, (split, wi) in enumerate(order):
            if split != "test":
                continue
            window = np.ascontiguousarray(windows[n][:N])
            norm = np.empty_like(window)
            for c in range(N):
                row = np.ascontiguousarray(window[c]); st = np.empty(2, np.float32)
                lib.calibrate_channel(V.ptr(row), V.ptr(st[:1]), V.ptr(st[1:]))
                lib.normalize_samples(V.ptr(row), V.ptr(norm[c]), window.shape[1], float(st[0]), float(st[1]))
            for name, (m, k) in mont.items():
                z = np.empty((100, N, 18), np.float32)
                for ui, end in enumerate(V.ENDS):
                    z[ui] = ((entries(lib, stream.spectra, stream.raw, norm, int(end), m, k) - mean) / scale) * mask[:N]
                dev[name] = max(dev[name], float(np.abs(z.transpose(1, 0, 2) - ref_cont[wi]).max()))
                out[name][wi, :N] = stream.quantize(z.reshape(-1, 18), packed).reshape(100, N).T
        res = dict(subject=subject)
        for name in mont:
            res[f"agreement_{name}"] = float((out[name][:, :N] == ref_tokens[:, :N]).mean())
            res[f"max_feature_deviation_{name}"] = dev[name]
        errs = {"released": [], **{name: [] for name in mont}}
        for seed in pilot.SEEDS:
            saved = torch.load(RUN63 / f"subject_{subject:02d}/fits/tokens_{seed}/best.pt", map_location="cpu", weights_only=True)
            model = pilot.SparPose(torch.from_numpy(book), torch.from_numpy(mask))
            model.load_state_dict(saved["model"])
            for name, toks in (("released", ref_tokens.astype(np.int64)), *out.items()):
                pred = pilot.prediction(model, dict(features=torch.from_numpy(toks)), "tokens", saved["mean"], saved["scale"])
                errs[name].append(pilot.error_mm(pred, targets[:, -50:]))
        res.update({f"error_{k}": float(np.mean(v)) for k, v in errs.items()})
        rows.append(res)
        print(json.dumps(res), flush=True)
    summary = dict(montage_rows_differing_from_reference=diff, rows=rows)
    for name in mont:
        d = np.array([r[f"error_{name}"] - r["error_released"] for r in rows])
        summary[name] = dict(agreement_mean=float(np.mean([r[f"agreement_{name}"] for r in rows])),
                             agreement_min=float(np.min([r[f"agreement_{name}"] for r in rows])),
                             error_mean=float(np.mean([r[f"error_{name}"] for r in rows])),
                             error_change_mean=float(d.mean()), positive=int((d > 0).sum()), p_exact=exact_p(d) if len(d) == 9 else None)
    summary["error_released_mean"] = float(np.mean([r["error_released"] for r in rows]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
