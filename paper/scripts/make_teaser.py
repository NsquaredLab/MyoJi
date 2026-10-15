"""Draw the teaser figure: a forearm montage becomes tokens becomes a hand pose.

This is a manually specified SVG layout. Everything is read from recorded
arrays: the waveform fan is one 160 ms tokenizer input window of the frozen
320-channel fixture, the lit codebook cell is the SPAR token that window
actually produced, and the skeleton is a stored SPAR validation prediction.
Run from the repository: .venv/bin/python scripts/make_teaser.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from make_encoder_replacement_draft import render
import paths


ROOT = Path(__file__).resolve().parents[1]
OUT = paths.OUTPUT / "figures/teaser"
WIDTH, HEIGHT = 716, 250
PRINT_WIDTH_IN = 7.16
INK, GREY, LIGHT = "#2b2e4c", "#6d6f72", "#d8d9dd"
TOKEN, FEATURE = "#d80032", "#5c618e"

FIXTURE = ROOT / "results/raw_tokenizers_cortex_m4_v1/reference.npz"
RECEIPT = ROOT / "results/raw_tokenizers_cortex_m4_v1/fixture_receipt.json"
EXPLORER = ROOT / "output/token-explorer"
INDEX = EXPLORER / "full-channel-index.json"
VIDEO = paths.RESULTS / "dpose_movement_video"  # select_dpose_movement_cycles.py
SPAR = ROOT / "output/figures/tokenizer-mechanism-data.npz"
PROVENANCE = paths.OUTPUT / "figures/pose_paper/interface.json"

# The frozen fixture's third token endpoint and the 160 ms window feeding it.
END, PATCH, FRAME = 2119, 320, 2
TRACES = 64                      # every fifth channel of the 320-channel montage
CODES, SIDE = 1024, 32           # the SPAR codebook drawn at its true size

# An unwarped stack of the same recording: constant gain, constant spacing and
# a linear time axis, so the signal itself stays readable. The recording holds
# eight seconds of each demonstration movement; the frozen fixture is only the
# first 1.06 s of task 2. Task 15 has the clearest rest-to-burst contrast of the
# eight (2.46x, against 1.44-1.79), so the activity is legible.
STACK_TASK, STACK_ROWS, STACK_SECONDS = 15, 8, 4.0
STACK_W, STACK_PITCH, STACK_GAIN = 1400.0, 30.0, 13.0

# Two fist cycles with the kinematics above the EMG that produced them. The
# window is the frozen video selection: peaks of the instructed-finger signal
# are the open hand, the troughs between them are the fist.
# The montage the study records against, above the channel count a wrist
# wearable would carry. Same recording, same instants: only the count differs.
CYCLE_W = 1240.0
CYCLE_X0, CYCLE_X1 = 80.0, 1160.0
CYCLE_HAND_Y, CYCLE_HAND_H = 126.0, 152.0
CYCLE_SPREAD_Y, CYCLE_SPREAD_H = 272.0, 62.0
# (channels, top, pitch, gain, columns). Four seconds at 2 kHz is 7,921 samples
# over roughly a thousand points, so no row can resolve a waveform: drawing every
# sample piles seven of them onto each point and the overdraw reads as a jagged
# band. Both rows are therefore per-column min/max ribbons of every sample -- the
# exact extent of the signal at display resolution, never a filter or a subsample.
CYCLE_BANDS = [(320, 354.0, 0.95, 0.46, 540), (8, 688.0, 25.0, 10.0, 1080)]
# Tip-first chains, with the fixed zero wrist prepended as index 0.
CHAINS = [[0, *range(base, base - 4, -1)] for base in [4, 8, 12, 16, 20]]

# Horizontal bands: converging fan, tokenizer, token stream, pose.
FAN_X, BOARD_X, HAND_X = 258.0, 430.0, 598.0
BOARD_W = 66.0
CY = HEIGHT / 2
FAN_HALF = 92.0                  # half-height of the fan at its open end
FAN_GAIN = 4.6                   # trace amplitude there, in the same units


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standalone(elements: list[str], box: tuple[float, float, float, float]) -> str:
    """One region on its own transparent canvas, for import into a drawing tool."""
    x, y, w, h = box
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w / 100:.4f}in" height="{h / 100:.4f}in" viewBox="{x:.2f} {y:.2f} {w:.2f} {h:.2f}">\n'
            + "\n".join(elements) + "\n</svg>\n")


def build() -> tuple[str, dict, dict]:
    fixture = np.load(FIXTURE, allow_pickle=False)
    window = fixture["normalized"][:, END + 1 - PATCH: END + 1]
    ids = fixture["spar_ids"]
    hand = np.load(SPAR, allow_pickle=False)["pose_mm"]
    provenance = json.loads(PROVENANCE.read_text())
    assert window.shape == (320, PATCH) and np.isfinite(window).all()
    assert ids.shape == (3, 320) and np.all(ids < CODES)
    assert hand.shape == (21, 3) and np.isfinite(hand).all()

    channels = np.arange(0, 320, 320 // TRACES)
    assert channels.size == TRACES
    lit = int(ids[FRAME, channels[0]])
    # One gain for every trace, so relative channel amplitudes stay truthful.
    traces = window[channels] / np.percentile(np.abs(window[channels]), 99)

    parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{PRINT_WIDTH_IN}in"
     height="{PRINT_WIDTH_IN * HEIGHT / WIDTH}in" viewBox="0 0 {WIDTH} {HEIGHT}"
     role="img" aria-labelledby="title desc">
<title id="title">A worn electrode montage is tokenized on the device and decoded to hand pose</title>
<desc id="desc">Many EMG channels recorded by a forearm electrode sleeve converge
into a tokenizer running on a microcontroller. The tokenizer assigns each channel
one entry of a 1024-code book every 20 ms, and only those token identifiers leave
the device. A receiver decodes them into a 3D hand pose.</desc>
<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>''']

    # Every trace narrows linearly to the tokenizer, so the fan is a true wedge.
    fan: list[str] = []
    x = np.linspace(FAN_X, BOARD_X, PATCH)
    taper = 1.0 - (x - FAN_X) / (BOARD_X - FAN_X)
    seats = np.linspace(-FAN_HALF, FAN_HALF, TRACES)
    for seat, trace in zip(seats, traces):
        y = CY + (seat + FAN_GAIN * trace) * taper
        points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))
        fan.append(f'<polyline points="{points}" fill="none" stroke="{GREY}" '
                   f'stroke-width=".45" stroke-opacity=".75"/>')
    parts.append('<g id="channel-fan">' + "".join(fan) + '</g>')

    # The codebook at its true size: 1024 cells, with the assigned one filled.
    chip: list[str] = []
    top, pad = CY - BOARD_W / 2, 6.0
    chip.append(f'<rect x="{BOARD_X}" y="{top}" width="{BOARD_W}" height="{BOARD_W}" '
                 f'rx="3" fill="{INK}"/>')
    for i in range(7):
        pin = top + 6 + i * (BOARD_W - 12) / 6
        chip.append(f'<rect x="{BOARD_X - 1.6}" y="{pin - .8}" width="1.6" height="1.6" fill="{GREY}"/>'
                     f'<rect x="{BOARD_X + BOARD_W}" y="{pin - .8}" width="1.6" height="1.6" fill="{GREY}"/>')
    grid, cell = BOARD_W - 2 * pad, (BOARD_W - 2 * pad) / SIDE
    chip.append(f'<rect x="{BOARD_X + pad}" y="{top + pad}" width="{grid}" height="{grid}" fill="white"/>')
    for code in range(CODES):
        a = BOARD_X + pad + (code % SIDE) * cell
        b = top + pad + (code // SIDE) * cell
        fill, inset = (TOKEN, 0.0) if code == lit else (LIGHT, cell * .22)
        chip.append(f'<rect x="{a + inset:.2f}" y="{b + inset:.2f}" '
                    f'width="{cell - 2 * inset:.2f}" height="{cell - 2 * inset:.2f}" fill="{fill}"/>')
    parts.append('<g id="tokenizer">' + "".join(chip) + '</g>')

    # A stored SPAR validation prediction, projected onto the rigid palm plane
    # and turned so the wrist meets the stream and the fingers point away.
    up = hand[[5, 9, 13, 17]].mean(axis=0) - hand[0]
    up /= np.linalg.norm(up)
    right = hand[17] - hand[5]
    right -= up * np.dot(right, up)
    right /= np.linalg.norm(right)
    xy = hand @ np.stack([right, up], axis=1)
    axis = xy[[5, 9, 13, 17]].mean(axis=0) - xy[0]
    axis /= np.linalg.norm(axis)
    xy = xy @ np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])
    xy -= xy[0]
    scale = min((WIDTH - 6 - HAND_X) / xy[:, 0].max(),
                (HEIGHT / 2 - 10) / np.abs(xy[:, 1]).max())
    xy = xy * scale + [HAND_X, CY]

    # Only identifiers leave the device; no value is encoded in a single mark.
    parts.append('<g id="token-stream">')
    start_x = BOARD_X + BOARD_W + 6
    parts.append(f'<path d="M {start_x} {CY} L {xy[0, 0]:.2f} {xy[0, 1]:.2f}" stroke="{TOKEN}" stroke-width="1.1"/>')
    for step in np.linspace(.07, .88, 7):
        a, b = start_x + (xy[0, 0] - start_x) * step, CY + (xy[0, 1] - CY) * step
        parts.append(f'<rect x="{a - 2.7:.2f}" y="{b - 2.7:.2f}" width="5.4" height="5.4" fill="{TOKEN}"/>')
    parts.append('</g>')

    parts.append('<g id="decoded-pose">')
    chains = [[0, *range(start, start + 4)] for start in (1, 5, 9, 13, 17)] + [[5, 9, 13, 17]]
    for chain in chains:
        points = " ".join(f"{xy[i, 0]:.2f},{xy[i, 1]:.2f}" for i in chain)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{FEATURE}" '
                     f'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    for a, b in xy:
        parts.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="2.9" fill="{TOKEN}" stroke="white" stroke-width="1"/>')
    parts.append('</g></svg>')

    # Reproducing the frozen fixture from task 2 is the check that the layout and
    # the index's stored calibration still agree with what the tokenizer saw.
    index = json.loads(INDEX.read_text())

    def task_signal(task: int) -> np.ndarray:
        meta = index["tasks"][str(task)]
        window = next(w for w in index["windows"].values() if w["task"] == task)
        samples = np.fromfile(EXPLORER / meta["file"], dtype=np.float32).reshape(*meta["shape"])
        centre = np.asarray(window["calibration_center"], np.float32)[:, None]
        scale = np.asarray(window["calibration_scale"], np.float32)[:, None]
        return (samples - centre) / scale

    np.testing.assert_array_equal(task_signal(2)[:, :2120], fixture["normalized"])
    recording = task_signal(STACK_TASK)
    span = int(STACK_SECONDS * 2000)
    assert span <= recording.shape[1], (span, recording.shape)

    stack: list[str] = []
    rows = np.arange(0, 320, 320 // STACK_ROWS)
    block = recording[rows, :span]
    block = block / np.percentile(np.abs(block), 99)
    sx = np.linspace(0, STACK_W, span)
    for i, trace in enumerate(block):
        sy = (i + .5) * STACK_PITCH - STACK_GAIN * trace
        points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(sx, sy))
        stack.append(f'<polyline points="{points}" fill="none" stroke="{GREY}" stroke-width=".4"/>')

    # Two fist cycles: the kinematics, the movement signal, and the EMG beneath.
    selection = json.loads((VIDEO / "segment-selection.json").read_text())
    fist = next(x for x in selection["segments"] if x["task"] == STACK_TASK)
    cycles = np.load(VIDEO / "kinematic-cycle-selection.npz", allow_pickle=False)
    truth = cycles[f"task_{STACK_TASK}_truth"]
    ends = cycles[f"task_{STACK_TASK}_ends"]
    smoothed = cycles[f"task_{STACK_TASK}_selection_smoothed"]
    opens = fist["cycle_peak_frames"]
    shuts = [int(a + np.argmin(smoothed[a:b])) for a, b in zip(opens[:-1], opens[1:])]
    first, last = opens[0], opens[-1]
    assert (first, last) == (fist["display_start_frame"], fist["display_end_frame_inclusive"])
    assert len(opens) == 3 and len(shuts) == 2, (opens, shuts)

    # How far the five fingertips sit from the wrist: high is open, low is fist.
    tips = np.linalg.norm(truth[:, [0, 4, 8, 12, 16], :], axis=2).mean(axis=1)
    low, high = tips[first:last + 1].min(), tips[first:last + 1].max()
    openness = lambda frame: (tips[frame] - low) / (high - low)
    # Five equally spaced instants rather than the turning points themselves.
    # The two cycles differ in length (1.74 s and 2.22 s), so the turning points
    # are unevenly paced; sampling evenly lands within three frames of them and
    # keeps every hand above the EMG that actually produced it.
    keys = np.round(np.linspace(first, last, 5)).astype(int).tolist()
    posture = ["open" if openness(k) > .5 else "fist" for k in keys]
    assert posture == ["open", "fist", "open", "fist", "open"], list(zip(keys, posture))

    # One shared orthographic camera and one scale for all five poses.
    poses = np.concatenate([np.zeros((len(keys), 1, 3)), truth[keys]], axis=1)
    reference = poses.mean(axis=0)
    up = reference[[4, 8, 12, 16, 20]].mean(axis=0) - reference[0]
    up /= np.linalg.norm(up)
    right = reference[20] - reference[4]
    right -= up * np.dot(right, up)
    right /= np.linalg.norm(right)
    flat = poses @ np.stack([right, up], axis=1)
    flat[:, :, 1] *= -1
    plane = flat.reshape(-1, 2)
    hand_scale = CYCLE_HAND_H / np.ptp(plane[:, 1])
    # One anchor and one framing shared with the Blender renders, so a shaded
    # hand drops onto exactly the spot its stick figure occupied.
    middle = poses.reshape(-1, 3).mean(axis=0)
    view_half = 1.4 * np.linalg.norm(poses.reshape(-1, 3) - middle, axis=1).max()
    anchor = np.array([middle @ right, -(middle @ up)])
    at = lambda frame: CYCLE_X0 + (frame - first) * (CYCLE_X1 - CYCLE_X0) / (last - first)

    hands = {
        "note": "Keyframe poses for scripts/render_teaser_hands.py. Millimetres, "
                "wrist at the origin, tip-first chains with the wrist prepended.",
        "participant": 1, "task": STACK_TASK, "task_name": fist["name"],
        "frames_50hz": keys,
        "posture": posture,
        "chains": CHAINS,
        "camera_right": right.tolist(), "camera_up": up.tolist(),
        "camera_forward": np.cross(right, up).tolist(),
        "centre_mm": middle.tolist(), "view_half_mm": float(view_half),
        "poses_mm": poses.tolist(),
    }

    written = OUT / "hands.json"
    written.write_text(json.dumps(hands, indent=1) + "\n")
    shaded = [OUT / "hands" / f"hand-{k:03d}-{p}.png" for k, p in zip(keys, hands["posture"])]
    stamp = OUT / "hands" / "manifest.json"
    renders = json.loads(stamp.read_text()) if stamp.exists() else {}
    # Only trust renders built from this exact geometry; otherwise fall back.
    rendered = (all(path.exists() for path in shaded)
                and renders.get("hands_json_sha256") == sha(written))
    if shaded[0].exists() and not rendered:
        print("hands.json changed since the last Blender pass; drawing stick figures. "
              "Re-run scripts/render_teaser_hands.py.")

    cycle: list[str] = []
    for key, pose, path in zip(keys, flat, shaded):
        if rendered:
            # Square renders carry their own framing; scale so the hand itself
            # stands CYCLE_HAND_H tall, centred on the instant it belongs to.
            side = CYCLE_HAND_H * renders["frame_units"] / renders["hand_units"]
            data = base64.b64encode(path.read_bytes()).decode()
            cycle.append(f'<image x="{at(key) - side / 2:.2f}" y="{CYCLE_HAND_Y - side / 2:.2f}" '
                         f'width="{side:.2f}" height="{side:.2f}" '
                         f'href="data:image/png;base64,{data}"/>')
            continue
        xy = (pose - anchor) * hand_scale + [at(key), CYCLE_HAND_Y]
        for chain in CHAINS:
            points = " ".join(f"{xy[i, 0]:.2f},{xy[i, 1]:.2f}" for i in chain)
            cycle.append(f'<polyline points="{points}" fill="none" stroke="{FEATURE}" '
                         f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')
        for a, b in xy:
            cycle.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="2.3" fill="{TOKEN}" '
                         f'stroke="white" stroke-width=".8"/>')

    band = tips[first:last + 1]
    ty = CYCLE_SPREAD_Y + CYCLE_SPREAD_H * (high - band) / (high - low)
    points = " ".join(f"{at(first + i):.2f},{v:.2f}" for i, v in enumerate(ty))
    cycle.append(f'<polyline points="{points}" fill="none" stroke="{TOKEN}" stroke-width="1.7" '
                 f'stroke-linejoin="round"/>')
    for key in keys:
        cycle.append(f'<circle cx="{at(key):.2f}" cy="{ty[key - first]:.2f}" r="3.2" '
                     f'fill="white" stroke="{TOKEN}" stroke-width="1.7"/>')

    whole = recording[:, ends[first] - 2000: ends[last] - 2000 + 1]
    whole = whole / np.percentile(np.abs(whole), 99)
    for count, top, pitch, gain, columns in CYCLE_BANDS:
        keep = whole.shape[1] // columns * columns
        binned = whole[:, :keep].reshape(320, columns, -1)
        lows, highs = binned.min(axis=2), binned.max(axis=2)
        edges = np.linspace(CYCLE_X0, CYCLE_X1, columns)
        ink = INK if count > 64 else GREY
        for i, c in enumerate(np.arange(0, 320, 320 // count)):
            base = top + (i + .5) * pitch
            # Closed ribbon: out along the per-column maximum, back along the minimum.
            out = " ".join(f"{a:.1f},{base - gain * v:.1f}" for a, v in zip(edges, highs[c]))
            back = " ".join(f"{a:.1f},{base - gain * v:.1f}"
                            for a, v in zip(edges[::-1], lows[c][::-1]))
            cycle.append(f'<polygon points="{out} {back}" fill="{ink}" fill-opacity=".88" '
                         f'stroke="none"/>')

    source = json.loads(RECEIPT.read_text())["source"]
    receipt = {
        "scope": "Teaser: worn montage, on-device tokenizer, token stream, decoded pose",
        "manual_vector_layout": True,
        "width_inches": PRINT_WIDTH_IN,
        "height_inches": PRINT_WIDTH_IN * HEIGHT / WIDTH,
        "colors": {"token": TOKEN, "pose": FEATURE, "neutral": GREY, "board": INK},
        "channel_fan": {
            "source": str(FIXTURE.relative_to(ROOT)), "task": source["task"],
            "source_split": source["source_split"], "window_end_sample": END,
            "window_samples": PATCH, "sample_rate_hz": 2000,
            "channels_zero_based": channels.tolist(),
            "shared_gain": "single 99th-percentile scale across the drawn channels",
            "taper": "linear to zero at the tokenizer; a display transform only"},
    "unwarped_stack": {
        "file": "stack.svg", "rows": STACK_ROWS, "channels_zero_based": rows.tolist(),
        "source": str((EXPLORER / index["tasks"][str(STACK_TASK)]["file"]).relative_to(ROOT)),
        "participant": 1, "task": STACK_TASK,
        "task_chosen_for": "clearest rest-to-burst contrast of the eight demonstrations",
        "from_sample": 0, "samples": span, "duration_s": STACK_SECONDS,
        "sample_rate_hz": 2000, "recording_seconds_available": recording.shape[1] / 2000,
        "calibration": "the index's stored per-channel centre and scale for this task",
        "task_2_reproduces_fixture_prefix_exactly": True,
        "shared_gain": "single 99th-percentile scale across the drawn channels",
        "time_and_amplitude_are_linear": True, "annotations": None},
    "fist_cycles": {
        "file": "cycles.svg", "participant": 1, "task": STACK_TASK, "task_name": fist["name"],
        "source_split": fist["source_split"],
        "selection": str((VIDEO / "segment-selection.json").relative_to(ROOT)),
        "selection_policy": selection["policy"],
        "frames_50hz": [first, last], "cycles": 2,
        "window_seconds": round((last - first) / 50, 2),
        "cycle_durations_seconds": fist["cycle_durations_seconds"],
        "scale_bar_seconds": 1,
        "drawn_frames": keys, "drawn_posture": posture,
        "drawn_frames_are": "five equally spaced instants across the two cycles, so the "
                            "hands are evenly paced and each still sits above its own EMG",
        "drawn_openness_percent": [round(100 * float(openness(k))) for k in keys],
        "turning_points_open": opens, "turning_points_fist": shuts,
        "turning_points_fist_derived_as": "the minimum of the frozen smoothed selection signal between consecutive peaks",
        "largest_offset_from_a_turning_point_frames":
            max(min(abs(k - t) for t in sorted(opens + shuts)) for k in keys),
        "emg_samples_2000hz": [int(ends[first]), int(ends[last])],
        "duration_s": (last - first) / 50,
        "kinematics": "ground truth, 20 wrist-relative landmarks, zero wrist prepended",
        "camera": "one shared orthographic palm-plane projection and scale for all five poses",
        "hands_drawn_as": "shaded Cycles renders (scripts/render_teaser_hands.py)" if rendered
                          else "flat vector stick figures",
        "hand_renders": {**{k: v for k, v in renders.items() if k != "rendered"},
                         "geometry_from": "hands.json", "is_measured_data": False,
                         "note": "Spheres at the measured joints and capsules along the "
                                 "measured bones. Lighting and shading are illustration; "
                                 "no coordinate is altered."} if rendered else None,
        "curve": "mean distance of the five fingertips from the wrist; high is open, low is fist",
        "fingertip_spread_mm": [round(float(low), 1), round(float(high), 1)],
        "emg_bands": [{"channels": c,
                       "channels_zero_based": None if c == 320 else np.arange(0, 320, 320 // c).tolist(),
                       "drawn_as": "per-column min/max ribbon of every sample",
                       "columns": columns, "samples_per_column": round(whole.shape[1] / columns, 1)}
                      for c, _, _, _, columns in CYCLE_BANDS],
        "emg_samples_drawn": int(whole.shape[1]),
        "emg_sample_rate_hz": 2000,
        "emg_rate_note": "the tokenizer's input rate; the dataset export causally resamples "
                         "DPose's native 2,048 Hz to 2,000 Hz. No display decimation.",
        "emg_shared_gain": "one 99th-percentile scale over all 320 channels; the two bands "
                           "differ only in display gain, so amplitudes stay comparable",
        "eight_channel_band_is": "an evenly spaced subset of the same 320-channel recording, "
                                 "standing in for a wrist wearable's channel count, not "
                                 "data from a wearable.",
        "no_prediction_or_error_used": True},
        "codebook": {"codes": CODES, "drawn_as": f"{SIDE} by {SIDE}",
                     "lit_code": lit, "lit_code_is": "the SPAR token this window "
                     f"produced for channel {int(channels[0])}"},
        "token_stream": {"marks": 7, "encodes_a_value": False,
                         "note": "Uniform marks standing for identifiers; no mark is a specific token"},
        "pose": {**provenance["example"], "decoder_seed": provenance["decoder_seed"],
                 "kind": provenance["pose_kind"]},
        "illustrations_are_accuracy_evidence": False,
        "new_training_or_scoring": False,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p)
                          for p in [FIXTURE, INDEX, ROOT / "scripts/render_teaser_hands.py", EXPLORER / index["tasks"][str(STACK_TASK)]["file"],
                                    SPAR, PROVENANCE, Path(__file__)]},
    }
    # A one-second bar under the bands, so the cycle rate is readable without
    # an axis. The window is 3.96 s, so a second is a little over a quarter of it.
    seconds = (last - first) / 50
    bar = (CYCLE_X1 - CYCLE_X0) / seconds
    foot = CYCLE_BANDS[-1][1] + CYCLE_BANDS[-1][0] * CYCLE_BANDS[-1][2] + 20
    cycle.append(f'<path d="M {CYCLE_X0} {foot - 4:.0f} V {foot:.0f} H {CYCLE_X0 + bar:.1f} '
                 f'V {foot - 4:.0f}" fill="none" stroke="{INK}" stroke-width="1.1"/>')
    cycle.append(f'<text x="{CYCLE_X0 + bar / 2:.1f}" y="{foot + 13:.0f}" font-family="Arial" '
                 f'font-size="12" text-anchor="middle" fill="{INK}">1 s</text>')

    pieces = {
        "cone": standalone(fan, (FAN_X, CY - FAN_HALF - FAN_GAIN, BOARD_X - FAN_X,
                                 2 * (FAN_HALF + FAN_GAIN))),
        "stack": standalone(stack, (-4, -4, STACK_W + 8, STACK_ROWS * STACK_PITCH + 8)),
        "chip": standalone(chip, (BOARD_X - 5, top - 5, BOARD_W + 10, BOARD_W + 10)),
        "cycles": standalone(cycle, (0, 0, CYCLE_W, foot + 22)),
    }
    return "\n".join(parts) + "\n", receipt, pieces


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / "teaser.svg"
    contents, receipt, pieces = build()
    svg.write_text(contents)
    for name, piece in pieces.items():
        (OUT / f"{name}.svg").write_text(piece)
    render(svg, PRINT_WIDTH_IN * 96, PRINT_WIDTH_IN * HEIGHT / WIDTH * 96)
    subprocess.run(["pdftoppm", "-singlefile", "-r", "300", "-png",
                    str(svg.with_suffix(".pdf")), str(svg.with_suffix(""))], check=True)
    receipt["output_sha256"] = {ext: sha(svg.with_suffix(ext)) for ext in [".svg", ".pdf", ".png"]}
    receipt["standalone_parts"] = {f"{name}.svg": sha(OUT / f"{name}.svg") for name in pieces}
    svg.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    for ext in [".svg", ".png", ".json"]:
        dest = ROOT / "output/figures" / f"teaser{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(svg.with_suffix(ext), dest)
    shutil.copy2(svg.with_suffix(".pdf"), ROOT / "output/pdf/teaser.pdf")
    print(json.dumps({"svg": str(svg), "pdf": str(svg.with_suffix('.pdf')),
                      "lit_code": receipt["codebook"]["lit_code"],
                      "channels": len(receipt["channel_fan"]["channels_zero_based"])}, indent=2))


if __name__ == "__main__":
    main()
