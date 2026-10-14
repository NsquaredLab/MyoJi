"""Add the 320-electrode context to a built token explorer, using saved inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

import paths


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="folder holding movement-inputs.h5, the token cache the explorer was built from")
    p.add_argument("--output", type=Path, default=paths.OUTPUT / "token-explorer", help="the built token explorer to extend")
    args = p.parse_args()
    from myoji.preprocessing import calibration_normalize
    from run_dpose_pilot import resample

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root, out = args.root, args.output
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "data-audit.json").read_text())
    data = paths.HDEMG / "Sub1_emg.pkl"
    assert sha(data) == audit["source_hashes"]["raw_emg"]
    original = joblib.load(data)
    (out / "contexts").mkdir(exist_ok=True)
    index = dict(version=1, channels=320, samples=960, sample_rate_hz=2000,
        source_raw_sha256=sha(data), manifest_sha256=sha(out / "manifest.json"), tasks={}, windows={},
        response_format="8-byte <4sHH header (EMGC, channels, samples), uint16 channel token IDs, float32 [normalized/input, channel, sample]; all little endian")
    all_examples = []
    for token in manifest["tokens"]:
        payload = json.loads((out / "tokens" / f"{token['id']}.json").read_text())
        for ei, example in enumerate(payload["examples"]):
            all_examples.append((token["id"], ei, payload, example))
    checked = 0
    with h5py.File(root / "movement-inputs.h5", "r") as cache:
        assert cache.attrs["signature"] == audit["cache_signature"]
        for task in manifest["tasks"]:
            tid = task["id"]
            windows = [w for w in audit["windows"] if w["task"] == tid]
            first = min(w["start"] for w in windows)
            last = max(w["start"] + 6000 for w in windows)
            signal = resample(original[str(tid)])[:, first:last].copy()
            path = out / "contexts" / f"task-{tid}.f32"
            signal.astype("<f4").tofile(path)
            index["tasks"][str(tid)] = dict(file=f"contexts/{path.name}", shape=list(signal.shape),
                first_sample=first, last_sample_exclusive=last, sha256=sha(path))
            for window in windows:
                wi, start = window["cache_index"], window["start"]
                raw = signal[:, start-first:start-first+6000]
                norm, center, scale = calibration_normalize(torch.from_numpy(raw)[None], 1000)
                normalized = norm[0].numpy()
                tokens = cache["tokens"][wi].astype("<u2")
                token_path = out / "contexts" / f"window-{wi}.u16"
                tokens.tofile(token_path)
                index["windows"][str(wi)] = dict(task=tid, start_sample=start,
                    calibration_center=center[0, :, 0].tolist(), calibration_scale=scale[0, :, 0].tolist(),
                    token_file=f"contexts/{token_path.name}", token_shape=list(tokens.shape), token_sha256=sha(token_path))
                for token, ei, payload, example in all_examples:
                    if example["cache_window"] != wi:
                        continue
                    channel, frame = example["channel_index"], example["cache_frame"]
                    assert int(tokens[channel, frame]) == token
                    end = 2039 + frame * 40
                    snippet = normalized[:, end-959:end+1]
                    before = raw[:, end-959:end+1]
                    assert snippet.shape == before.shape == (320, 960)
                    # This is also the server's float32 normalization recipe.
                    scaled = np.clip((before - center[0].numpy()) / scale[0].numpy(), -12, 12)
                    np.testing.assert_array_equal(scaled, snippet)
                    expected = np.memmap(out / payload["waveform_file"], dtype="<f4", mode="r", shape=tuple(payload["waveform_shape"]))[ei]
                    np.testing.assert_array_equal(snippet[channel], expected[0])
                    np.testing.assert_array_equal(before[channel], expected[1])
                    checked += 1
            print(f"Exported all 320 electrodes: {task['name']}", flush=True)
    assert checked == manifest["example_count"]
    write(out / "full-channel-index.json", index)
    result = dict(passed=True, device="cpu", cpu_threads=4, gpu_used=False,
        builder_sha256=sha(__file__), raw_source_sha256=sha(data), channels=320,
        windows=len(index["windows"]), checked_occurrences=checked,
        all_320_normalized_contexts_verified=True, selected_channel_matches_existing_examples_exactly=True,
        all_selected_ids_match_saved_assignments=True, source_split="train",
        index_sha256=sha(out / "full-channel-index.json"),
        context_file_bytes=sum(p.stat().st_size for p in (out / "contexts").iterdir()))
    write(out / "full-channel-audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
