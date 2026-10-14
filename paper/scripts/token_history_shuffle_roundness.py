"""Roundness of the shuffled-history control in the Fig. 6 UMAP (participant 1, 480 ms history).

make_token_pose_recovery_figure.py saved the control's face-on embedding next to the main one: each
update keeps its current tokens, while its four-lag history is taken from a random update of the same
repetition, then UMAP is refitted and the same projection applied. This scores both with the mean-route
roundness used for the history-length supplement (4*pi*area/perimeter^2; 1 = circle).

    uv run python scripts/token_history_shuffle_roundness.py
"""
import json
from pathlib import Path

import numpy as np

from make_token_history_supplement import NAMES, mean_route, roundness

ROOT = Path(__file__).resolve().parents[1]
UMAP = ROOT / 'tmp/token-umap/pose-recovery/umap_sub01_tokens.npz'
OUT = ROOT / 'output/figures/token-history-shuffled-control.json'


def main():
    d = np.load(UMAP)
    label, cycle, phase = d['label'], d['cycle'], d['phase']
    out = {}
    for key in ('embedding', 'embedding_shuffled_history'):
        r = [roundness(mean_route(d[key], label == c, cycle, phase)) for c in range(len(NAMES))]
        out[key] = dict(roundness=dict(zip(NAMES, r)), mean_roundness=float(np.mean(r)))
    out['source'] = dict(file=UMAP.name, settings=str(d['settings']))
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print({k: round(v['mean_roundness'], 3) for k, v in out.items() if k != 'source'})


if __name__ == '__main__':
    main()
