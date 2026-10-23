"""Replay recorded HD-EMG through three SPAR tokenizers and score the frozen decoders.

Streams per participant (1-9), same windows, normalization, standardization and
codebook as the nine-participant campaign (results/hdemg_spar_simple_full_v1):
  pytorch  released tokenizer (myoji.spar), regenerated here;
  exact    host build of the N6 port with STM32N6_EXACT=1 (radix-2 FFT, libm hypotf);
  fast     host build of the shipped N6 port (CMSIS rfft, sqrt hypot), scalar CMSIS.
The C builds use zeroed standardization constants so that they return raw
matrix-log entries; standardization and the batched nearest-code search then use
the participant's own statistics and frozen codebook, exactly as the firmware
would with that participant's constants compiled in.

Stages. `prepare` runs where the HD-EMG recordings (paths.HDEMG) and a GPU are:
it regenerates the campaign cache and exports the resampled windows and a
compact cache extract. `build`, `fixture`, `subject` and `report` run on an
arm64 host (tested on macOS), because the CMSIS headers of the STM32CubeN6
checkout (STM32CUBEN6_ROOT) refuse x86 hosts.

No model is fitted. Host timings are not microcontroller timings.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
from paper.common import paths  # noqa: E402
from paper.hdemg import run_hdemg_spar_simple as pilot  # noqa: E402

CAMPAIGN = BASE / "results/hdemg_spar_simple_full_v1"
OUT = BASE / "results/stm32n6_pose_validation_v1"
SEEDS = pilot.SEEDS
ENDS = pilot.ENDS
FP = ct.POINTER(ct.c_float)


def ptr(a):
    return a.ctypes.data_as(FP)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1) + "\n")


# ----------------------------------------------------------------------------- prepare (GPU host)
def prepare(subject, work, device):
    """Regenerate the campaign cache with the released tokenizer; export windows and a compact extract."""
    import h5py
    from paper.hdemg import run_hdemg_spar_full as full
    from myoji.codebook import fit_standardization
    t0 = time.monotonic()
    root = work / f"subject_{subject:02d}"
    root.mkdir(parents=True, exist_ok=True)
    source = CAMPAIGN / f"subject_{subject:02d}"
    p = json.loads((source / "protocol.json").read_text())
    if not (root / "protocol.json").exists():
        shutil.copy(source / "protocol.json", root / "protocol.json")
    args = SimpleNamespace(root=root, data=paths.HDEMG, device=device)
    # The 20-task pickles are not the 8-task export the campaign hashed, so
    # the file-hash gate is replaced by the normalized-window hash gate below.
    full.prepare(args, dict(p, data_sha256={}))
    receipt_new = json.loads((root / "cache-receipt.json").read_text())
    receipt_old = json.loads((source / "cache-receipt.json").read_text())
    with h5py.File(root / "features.h5", "r") as f:
        extract = dict(mask=f["mask"][:], codebook=f["codebook"][:], feature_mean=f["feature_mean"][:], feature_scale=f["feature_scale"][:])
        rows = {}
        for split in ("train", "val", "test"):
            extract[f"{split}_tokens"] = f[f"{split}/tokens"][:]
            extract[f"{split}_tasks"] = f[f"{split}/tasks"][:]
            extract[f"{split}_starts"] = f[f"{split}/starts"][:]
            rows[split] = (f[f"{split}/tasks"][:], f[f"{split}/starts"][:])
        extract["test_spar"] = f["test/spar"][:]
        extract["test_targets"] = f["test/targets"][:]
        mean, scale = fit_standardization(f["train/spar"][:], f["mask"][:].astype(bool))
    np.testing.assert_array_equal(mean, extract["feature_mean"])
    np.testing.assert_array_equal(scale, extract["feature_scale"])
    np.savez(root / "cache.npz", **extract)
    # Resampled (not yet normalized) 6,000-sample windows in cache order.
    emg = full.joblib.load(args.data / f"Sub{subject}_emg.pkl")
    taps = full.firwin(10241, 800, fs=256000, window=("kaiser", 8.6)) * 125
    order = [(split, i) for split in ("train", "val", "test") for i in range(len(rows[split][0]))]
    windows = np.empty((len(order), 320, 6000), np.float32)
    resampled = {}
    for n, (split, i) in enumerate(order):
        task, start = int(rows[split][0][i]), int(rows[split][1][i])
        if task not in resampled:
            sig = emg[str(task)] if str(task) in emg else emg[task]
            resampled[task] = full.upfirdn(taps, sig.astype(np.float32), up=125, down=128, axis=-1)[..., :sig.shape[-1] * 125 // 128].astype(np.float32)
        windows[n] = resampled[task][:, start:start + 6000]
    raw_hash = hashlib.sha256()
    for w in windows:
        raw_hash.update(pilot.normalize_raw(w).tobytes())
    np.save(root / "windows.npy", windows)
    info = dict(subject=subject, seconds=time.monotonic() - t0, device=device, window_order=order,
                cache_identical_to_campaign=receipt_new["sha256"] == receipt_old["sha256"],
                normalized_raw_identical_to_campaign=receipt_new["normalized_raw_sha256"] == receipt_old["normalized_raw_sha256"],
                exported_windows_normalized_sha256=raw_hash.hexdigest(), campaign_normalized_raw_sha256=receipt_old["normalized_raw_sha256"],
                exported_windows_match_campaign=raw_hash.hexdigest() == receipt_old["normalized_raw_sha256"])
    write(root / "prepare.json", info)
    print(json.dumps(info), flush=True)


# ----------------------------------------------------------------------------- builds (arm64 host)
def patched_frozen(n6build, dest):
    """Frozen artifacts with spar_mean = 0 and spar_scale = 1 (raw entries out)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("learned_weights.f32", "weight_offsets.h", "constants.c"):
        shutil.copy(n6build.FROZEN / name, dest / name)
    text = (dest / "constants.c").read_text()
    text, n1 = re.subn(r"(const float spar_mean\[18\] = \{)[^}]*(\})", lambda m: m.group(1) + ", ".join(["0.f"] * 18) + m.group(2), text)
    text, n2 = re.subn(r"(const float spar_scale\[18\] = \{)[^}]*(\})", lambda m: m.group(1) + ", ".join(["1.f"] * 18) + m.group(2), text)
    assert n1 == 1 and n2 == 1, "standardization constants not found"
    (dest / "constants.c").write_text(text)
    return dest


def build_variant(exact):
    from paper.firmware.stm32n6_optimized import build as n6build
    original = (n6build.EXACT, n6build.FROZEN, n6build.OUT)
    try:
        n6build.EXACT = exact
        n6build.FROZEN = patched_frozen(n6build, OUT / "frozen_zero_standardization")
        n6build.OUT = OUT / f"build_{'exact' if exact else 'fast'}_{sys.platform}"
        n6build.OUT.mkdir(parents=True, exist_ok=True)
        lib = n6build.build_host(n6build.generated())
    finally:
        n6build.EXACT, n6build.FROZEN, n6build.OUT = original
    return lib


def load_lib(path):
    lib = ct.CDLL(str(path))
    lib.calibrate_channel.argtypes = [FP, FP, FP]
    lib.calibrate_channel.restype = None
    lib.normalize_samples.argtypes = [FP, FP, ct.c_uint, ct.c_float, ct.c_float]
    lib.normalize_samples.restype = None
    lib.fast_spar_spectra.argtypes = [FP, FP]
    lib.fast_spar_spectra.restype = None
    lib.fast_spar_features.argtypes = [FP, FP, ct.c_uint, ct.c_void_p, ct.c_void_p]
    lib.fast_spar_features.restype = None
    lib.fast_prepare_book.argtypes = [FP, ct.c_uint, FP]
    lib.fast_prepare_book.restype = None
    lib.fast_nearest_code_batch.argtypes = [FP, ct.c_uint, ct.c_uint, FP, ct.c_uint, ct.POINTER(ct.c_uint16)]
    lib.fast_nearest_code_batch.restype = None
    lib.members = ((ct.c_uint16 * 4) * 320).in_dll(lib, "spar_members")
    lib.masks = ((ct.c_uint8 * 3) * 320).in_dll(lib, "spar_mask")
    return lib


class CStream:
    """Raw matrix-log entries from one C build, standardized and quantized per participant."""

    def __init__(self, lib):
        self.lib = lib
        self.spectra = np.empty((320, 7, 32), np.complex64)
        self.raw = np.empty((320, 18), np.float32)

    def normalize(self, window):
        """C calibration (first 1,000 samples) and normalization of a (320, n) float32 window."""
        out = np.empty_like(window)
        stats = np.empty((320, 2), np.float32)
        for c in range(320):
            row = np.ascontiguousarray(window[c])
            self.lib.calibrate_channel(ptr(row), ptr(stats[c, :1]), ptr(stats[c, 1:]))
            self.lib.normalize_samples(ptr(row), ptr(out[c]), window.shape[1], float(stats[c, 0]), float(stats[c, 1]))
        return out

    def entries(self, normalized, end):
        """Raw (320, 18) matrix-log entries for the 320-sample window ending at `end` (inclusive)."""
        lib = self.lib
        for c in range(320):
            lib.fast_spar_spectra(ptr(normalized[c, end - 319:end + 1]), ptr(self.spectra[c]))
        lib.fast_spar_features(ptr(self.spectra), ptr(self.raw), 320, lib.members, lib.masks)
        return self.raw.copy()

    def prepare_book(self, book):
        packed = np.empty_like(book)
        self.lib.fast_prepare_book(ptr(np.ascontiguousarray(book)), 18, ptr(packed))
        return packed

    def quantize(self, standardized, packed):
        """Batched nearest-code search of (N, 18) float32 vectors (bit-exact with the per-vector search)."""
        x = np.ascontiguousarray(standardized, dtype=np.float32)
        ids = np.empty(len(x), np.uint16)
        self.lib.fast_nearest_code_batch(ptr(x.ravel()), 18, len(x), ptr(packed), 18, ids.ctypes.data_as(ct.POINTER(ct.c_uint16)))
        return ids


def check_fixture(libs):
    """Frozen 320-channel fixture: the exact build must reproduce the original port's
    standardized features and both builds must reproduce all 960 released IDs."""
    from paper.firmware.stm32n6_optimized import build as n6build
    frozen = n6build.FROZEN
    ref = np.load(frozen / "reference.npz")
    previous = np.load(frozen / "host_outputs.npz")
    inputs = np.load(paths.ASSETS / "spar/codebooks-1024.npz")  # participant 1, whose codebook the fixture uses
    mean, scale, mask = inputs["subject_01/feature_mean"], inputs["subject_01/feature_scale"], inputs["subject_01/mask"].astype(np.float32)
    raw = np.fromfile(frozen / "raw_320x2120.f32", dtype="<f4").reshape(320, 2120)
    report = {}
    for name, lib in libs.items():
        stream = CStream(lib)
        norm = stream.normalize(raw)
        np.testing.assert_array_equal(norm, ref["normalized"])
        packed = stream.prepare_book(ref["spar_book"])
        ids, deviation = [], 0.0
        for f, end in enumerate([2039, 2079, 2119]):
            entries = stream.entries(norm, end)
            z = ((entries - mean) / scale) * mask
            deviation = max(deviation, float(np.abs(z - previous["spar_features"][f]).max()))
            ids.append(stream.quantize(z, packed))
        ids = np.stack(ids)
        matched = int((ids == previous["spar_ids"]).sum())
        report[name] = dict(normalized_exact=True, max_abs_feature_deviation_from_original_port=deviation,
                            released_ids_reproduced=matched, released_ids_total=int(ids.size))
        print(json.dumps({name: report[name]}), flush=True)
        assert matched == ids.size, name
        if name == "exact":
            assert deviation < 1e-5, deviation
    return report


# ----------------------------------------------------------------------------- subject (arm64 host)
def run_subject(subject, libs, data, device):
    t0 = time.monotonic()
    folder = data / f"subject_{subject:02d}"
    info = json.loads((folder / "prepare.json").read_text())
    cache = np.load(folder / "cache.npz")
    windows = np.load(folder / "windows.npy", mmap_mode="r")
    source = CAMPAIGN / f"subject_{subject:02d}"
    mask = cache["mask"].astype(np.float32)
    mean, scale = cache["feature_mean"], cache["feature_scale"]
    order = [tuple(x) for x in info["window_order"]]
    # The decoder's frozen codebook is the reference vocabulary.
    checkpoints = {seed: torch.load(source / f"fits/tokens_{seed}/best.pt", map_location="cpu", weights_only=True) for seed in SEEDS}
    frozen_book = checkpoints[SEEDS[0]]["model"]["decoder.codebook"].numpy()
    for seed in SEEDS:
        assert np.array_equal(checkpoints[seed]["model"]["decoder.codebook"].numpy(), frozen_book)
    book_identical = bool(np.array_equal(frozen_book, cache["codebook"]))
    # Released-tokenizer test tokens under the frozen codebook.
    z_test = (((cache["test_spar"] - mean) / scale) * mask[None, :, None, :]).astype(np.float32)   # (16, 320, 100, 18)
    zt = torch.from_numpy(z_test.transpose(0, 2, 1, 3).reshape(-1, 18))
    bt = torch.from_numpy(frozen_book)
    pytorch_test = (zt.square().sum(-1, keepdim=True) + bt.square().sum(-1)[None] - 2 * zt @ bt.T).clamp_min(0).argmin(-1)
    pytorch_test = pytorch_test.numpy().reshape(len(z_test), 100, 320).transpose(0, 2, 1).astype(np.uint16)
    cache_vs_frozen_book = float((pytorch_test == cache["test_tokens"]).mean())
    pytorch = {split: cache[f"{split}_tokens"] for split in ("train", "val", "test")}
    if not book_identical:
        pytorch["test"] = pytorch_test
    # C streams over every exported window; C normalization checked against the released one.
    streams = {name: CStream(lib) for name, lib in libs.items()}
    packed = {name: s.prepare_book(frozen_book) for name, s in streams.items()}
    c_ids = {name: {split: np.empty_like(pytorch[split]) for split in pytorch} for name in streams}
    c_dev = {name: 0.0 for name in streams}
    exact_vs_fast_dev = 0.0
    normalization_mismatches = 0
    test_index = {}
    for n, (split, wi) in enumerate(order):
        window = np.ascontiguousarray(windows[n])
        norm_ref = pilot.normalize_raw(window)
        per_stream_z = {}
        for name, s in streams.items():
            norm = s.normalize(window)
            normalization_mismatches += int(not np.array_equal(norm, norm_ref))
            z = np.empty((100, 320, 18), np.float32)
            for ui, end in enumerate(ENDS):
                z[ui] = ((s.entries(norm, int(end)) - mean) / scale) * mask
            per_stream_z[name] = z
            c_ids[name][split][wi] = s.quantize(z.reshape(-1, 18), packed[name]).reshape(100, 320).T
        if split == "test":
            ref_z = z_test[wi].transpose(1, 0, 2)
            for name in streams:
                c_dev[name] = max(c_dev[name], float(np.abs(per_stream_z[name] - ref_z).max()))
        exact_vs_fast_dev = max(exact_vs_fast_dev, float(np.abs(per_stream_z["fast"] - per_stream_z["exact"]).max()))
        if n % 16 == 15:
            print(json.dumps(dict(subject=subject, windows_done=n + 1, seconds=time.monotonic() - t0)), flush=True)
    agreement = {}
    for split in pytorch:
        agreement[split] = {
            "fast_vs_pytorch": float((c_ids["fast"][split] == pytorch[split]).mean()),
            "exact_vs_pytorch": float((c_ids["exact"][split] == pytorch[split]).mean()),
            "fast_vs_exact": float((c_ids["fast"][split] == c_ids["exact"][split]).mean()),
            "channel_updates": int(pytorch[split].size)}
    # Frozen token decoders on each stream (test windows only).
    target = cache["test_targets"][:, -50:]
    stored = {seed: np.load(source / f"fits/tokens_{seed}/test_predictions.npz") for seed in SEEDS}
    errors, gate = {}, {}
    data_dict = dict(targets=torch.from_numpy(cache["test_targets"]), book=torch.from_numpy(frozen_book), mask=torch.from_numpy(mask))
    for seed in SEEDS:
        model = pilot.SparPose(data_dict["book"], data_dict["mask"])
        model.load_state_dict(checkpoints[seed]["model"])
        model.to(device)
        np.testing.assert_array_equal(stored[seed]["target"], target)
        for name, ids in (("pytorch", pytorch["test"]), ("exact", c_ids["exact"]["test"]), ("fast", c_ids["fast"]["test"])):
            data_dict["features"] = torch.from_numpy(ids.astype(np.int64)).to(device)
            pred = pilot.prediction(model, data_dict, "tokens", checkpoints[seed]["mean"], checkpoints[seed]["scale"])
            errors.setdefault(name, {})[str(seed)] = pilot.error_mm(pred, target)
            if name == "pytorch":
                gate[str(seed)] = dict(max_abs_prediction_difference_mm=float(np.abs(pred - stored[seed]["prediction"]).max()),
                                       stored_error_mm=pilot.error_mm(stored[seed]["prediction"], target))
    summary = {name: float(np.mean(list(v.values()))) for name, v in errors.items()}
    result = dict(subject=subject, complete=True, seconds=time.monotonic() - t0, device=device,
                  cache_identical_to_campaign=info["cache_identical_to_campaign"],
                  normalized_raw_identical_to_campaign=info["normalized_raw_identical_to_campaign"],
                  exported_windows_match_campaign=info["exported_windows_match_campaign"],
                  codebook_identical_to_frozen=book_identical, cache_test_tokens_vs_frozen_book_tokens=cache_vs_frozen_book,
                  c_normalization_mismatches=normalization_mismatches,
                  max_abs_standardized_feature_deviation=dict(exact_vs_pytorch_test=c_dev["exact"], fast_vs_pytorch_test=c_dev["fast"], fast_vs_exact_all=exact_vs_fast_dev),
                  token_agreement=agreement, decoder_error_mm=errors, decoder_error_mean_mm=summary,
                  prediction_gate=gate, windows=dict(test=int(len(cache["test_tasks"])), all=len(order)))
    write(OUT / f"subject_{subject:02d}/result.json", result)
    np.savez_compressed(OUT / f"subject_{subject:02d}/tokens.npz", **{f"{n}_{s}": c_ids[n][s] for n in c_ids for s in c_ids[n]},
                        **{f"pytorch_{s}": pytorch[s] for s in pytorch})
    print(json.dumps(result, indent=1), flush=True)
    return result


def report():
    from paper.common.hdemg_figure_statistics import exact_paired_test
    results = [json.loads((OUT / f"subject_{s:02d}/result.json").read_text()) for s in range(1, 10) if (OUT / f"subject_{s:02d}/result.json").exists()]
    subjects = [r["subject"] for r in results]
    err = {name: np.array([r["decoder_error_mean_mm"][name] for r in results]) for name in ("pytorch", "exact", "fast")}
    rng = np.random.default_rng(260909)
    keys = ("fast_vs_pytorch", "exact_vs_pytorch", "fast_vs_exact")
    # Only the test windows carry released-tokenizer tokens under the frozen codebook; the
    # cache's train/val tokens use the codebook regenerated on this machine and are not comparable.
    out = dict(subjects=subjects, n=len(results), mean_error_mm={k: float(v.mean()) for k, v in err.items()},
               participant_sd_mm={k: float(v.std(ddof=1)) for k, v in err.items()} if len(results) > 1 else {},
               agreement_test={k: float(np.mean([r["token_agreement"]["test"][k] for r in results])) for k in keys},
               agreement_test_min={k: float(np.min([r["token_agreement"]["test"][k] for r in results])) for k in keys},
               agreement_test_channel_updates=int(sum(r["token_agreement"]["test"]["channel_updates"] for r in results)),
               agreement_fast_vs_exact_all_windows=float(np.mean([np.mean([r["token_agreement"][s]["fast_vs_exact"] for s in ("train", "val", "test")]) for r in results])),
               agreement_per_subject_test={str(r["subject"]): r["token_agreement"]["test"] for r in results},
               prediction_gate_max_abs_mm=float(max(g["max_abs_prediction_difference_mm"] for r in results for g in r["prediction_gate"].values())),
               prediction_gate_mean_error_reproduced_mm=float(max(abs(g["stored_error_mm"] - r["decoder_error_mm"]["pytorch"][seed]) for r in results for seed, g in r["prediction_gate"].items())),
               normalized_raw_identical=[r["normalized_raw_identical_to_campaign"] for r in results],
               codebook_identical=[r["codebook_identical_to_frozen"] for r in results],
               normalization_mismatches=int(sum(r["c_normalization_mismatches"] for r in results)),
               max_abs_standardized_feature_deviation={k: float(max(r["max_abs_standardized_feature_deviation"][k] for r in results)) for k in results[0]["max_abs_standardized_feature_deviation"]},
               per_subject_error_mm={str(r["subject"]): r["decoder_error_mean_mm"] for r in results}, contrasts={})
    for name in ("fast", "exact"):
        d = err[name] - err["pytorch"]
        entry = dict(mean_mm=float(d.mean()), per_participant_mm=d.tolist())
        if len(results) == 9:
            observed, pvalue, _ = exact_paired_test(err[name], err["pytorch"])
            boot = np.array([d[rng.integers(0, 9, 9)].mean() for _ in range(10000)])
            entry.update(mean_mm=float(observed), p_two_sided_exact=float(pvalue),
                         bootstrap95_mm=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])
        out["contrasts"][f"{name}_minus_pytorch"] = entry
    write(OUT / "summary.json", out)
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "fixture", "subject", "report"))
    parser.add_argument("--subjects", type=int, nargs="*", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work", type=Path, default=OUT / "prepare", help="prepare: output folder (windows.npy is about 0.7 GB per participant)")
    parser.add_argument("--data", type=Path, default=OUT / "prepare", help="subject: directory holding subject_XX/{prepare.json,cache.npz,windows.npy}")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "prepare":
        for subject in args.subjects:
            prepare(subject, args.work, args.device)
        return
    if args.stage == "build":
        libs = {name: str(build_variant(exact)) for name, exact in (("exact", True), ("fast", False))}
        write(OUT / f"builds_{sys.platform}.json", dict(libraries=libs, sha256={k: sha(v) for k, v in libs.items()}))
        print(json.dumps(libs))
        return
    if args.stage == "report":
        report()
        return
    builds = json.loads((OUT / f"builds_{sys.platform}.json").read_text())["libraries"]
    libs = {name: load_lib(path) for name, path in builds.items()}
    if args.stage == "fixture":
        write(OUT / "fixture.json", check_fixture(libs))
    else:
        for subject in args.subjects:
            run_subject(subject, libs, args.data, args.device)


if __name__ == "__main__":
    main()
