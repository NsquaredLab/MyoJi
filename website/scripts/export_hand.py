"""Export the Fig. 5A hand movements for the decoded-hand animation; no fitting or new inference.

Participant 6, seed 260911: for each movement, 4 s of contiguous test-block updates (two test
windows plus two in-between windows, all inside the test block) from
tmp/decoding-continuous/p6_continuous.npz. For display, predictions pass the causal One Euro
filter of Fig. 5A with the hand viewer's steadier beta (0.003; build_hdemg_hand_viewer.py): the
responsive Fig. 5A setting leaves the decoded hands 10-20x jitterier than the measured hand.
Measured poses stay unfiltered, and the errors are unfiltered. Poses are projected with the hand
viewer's oblique orthographic camera (yaw -0.42, pitch 0.32).

    uv run python website/scripts/export_hand.py
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from hdemg_one_euro import smooth  # noqa: E402

OUT = ROOT / 'website/static/data/hand.json'
SOURCE = ROOT / 'tmp/decoding-continuous/p6_continuous.npz'
RECEIPT = json.loads((ROOT/'manuscript/figures/pose_paper/decoding_examples.json').read_text())
FILTER = {k: v for k, v in RECEIPT['display_filter'].items() if k != 'kind'} | {'beta': .003}
UNIT = .2                                   # stored coordinates are multiples of 0.2 mm
YAW, PITCH = -.42, .32                      # build_hdemg_hand_viewer.py "oblique" preset
TIPS = (0, 4, 8, 12, 16)                    # tip-first blocks of four landmarks per finger
dist = lambda w, a: np.linalg.norm(w[:, a], axis=-1)
gap = lambda w, a, b: np.linalg.norm(w[:, a] - w[:, b], axis=-1)
MOVES = {                                   # as plot_decoding_examples.py, tab10 as in the paper
    2: ('Thumb', '#1f77b4', 'Thumb tip to pinky base', lambda w: gap(w, 0, 19)),
    4: ('Index', '#ff7f0e', 'Index tip to wrist', lambda w: dist(w, 4)),
    6: ('Middle', '#2ca02c', 'Middle tip to wrist', lambda w: dist(w, 8)),
    8: ('Ring', '#d62728', 'Ring tip to wrist', lambda w: dist(w, 12)),
    10: ('Pinky', '#9467bd', 'Pinky tip to wrist', lambda w: dist(w, 16)),
    15: ('Fist', '#8c564b', 'Mean fingertip to wrist', lambda w: np.mean([dist(w, t) for t in TIPS], 0)),
    18: ('2-Pinch', '#7f7f7f', 'Thumb tip to index tip', lambda w: gap(w, 0, 4)),
    19: ('3-Pinch', '#e377c2', 'Thumb tip to index and middle tips',
         lambda w: (gap(w, 0, 4) + gap(w, 0, 8)) / 2),
}


def project(points):
    """Viewer camera: rotate about y by yaw, then about x by pitch; screen y points down."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    rx = np.cos(YAW) * x + np.sin(YAW) * z
    rz = -np.sin(YAW) * x + np.cos(YAW) * z
    return np.stack([rx, -(np.cos(PITCH) * y - np.sin(PITCH) * rz)], -1)


z = np.load(SOURCE)
poses, movements = {}, []
for task, (name, color, measure, f) in MOVES.items():
    target = z[f'target_{task}'].reshape(200, 20, 3).astype(float)
    raw = {arm: z[f'{arm}_{task}'].reshape(200, 20, 3).astype(float) for arm in ('tokens', 'continuous')}
    shown = {arm: smooth(v.reshape(1, 200, 60), **FILTER).reshape(200, 20, 3) for arm, v in raw.items()}
    panel = RECEIPT['panels'][name]
    error = {arm: float(np.linalg.norm(v - target, axis=-1).mean()) for arm, v in raw.items()}
    for arm in raw:   # same blocks and scoring as Fig. 5A
        assert abs(error[arm] - panel[f'error_mm_{arm}_unfiltered']) < 1e-4, (name, arm)
    poses[task] = {'measured': project(target), **{arm: project(v) for arm, v in shown.items()}}
    movements.append(dict(task=task, name=name, color=color, measure=measure,
                          error_mm={arm: round(v, 2) for arm, v in error.items()},
                          trace={'measured': np.round(f(target), 1).tolist(),
                                 **{arm: np.round(f(v), 1).tolist() for arm, v in shown.items()}}))

every = np.concatenate([p.reshape(-1, 2) for d in poses.values() for p in d.values()] + [np.zeros((1, 2))])
low = np.floor(every.min(0) / UNIT) - 25
size = np.ceil(every.max(0) / UNIT) + 25 - low
cell = lambda p: np.rint(p / UNIT - low).astype(int)
for m in movements:
    for kind, p in poses[m['task']].items():
        m[kind] = cell(p).reshape(-1).tolist()   # frame-major, landmark, x/y
payload = dict(
    participant=6, seed=260911, fps=50, frames=200, display_filter=dict(kind='one_euro', **FILTER),
    unit_mm=UNIT, view=[0, 0, *size.astype(int).tolist()], wrist=cell(np.zeros(2)).tolist(),
    chains=[[b + 3, b + 2, b + 1, b] for b in TIPS],   # wrist outward to the tip
    movements=movements,
    provenance='Participant 6, seed 260911, 4 s of contiguous test-block updates per movement as in '
               'Fig. 5A. Predictions are smoothed for display with a causal One Euro filter (minimum '
               'cutoff 1 Hz, beta 0.003); '
               'errors are unfiltered means over 20 landmarks. Wrist-relative landmarks, one fixed '
               'orthographic camera.')
OUT.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
print(OUT, f'{OUT.stat().st_size / 1e3:.0f} kB', payload['view'],
      {m['name']: m['error_mm'] for m in movements})
