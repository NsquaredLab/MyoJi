"""Export saved manuscript embeddings and phase-aligned routes; never refit UMAP."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tmp/token-umap/pose-recovery'
OUT = ROOT / 'website/static/data'
names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'Fist', '2-Pinch', '3-Pinch']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f', '#e377c2']
main = np.load(SOURCE / 'umap_sub01_tokens.npz')
history = np.load(SOURCE / 'history_supplement.npz')
assert np.array_equal(main['embedding'], history['w480'])
label, cycle, phase = (main[k] for k in ('label', 'cycle', 'phase'))
assert np.all(main['history_valid'] & (main['native_purity'] >= .5))
bins = np.linspace(0, 1, 21)
windows = {}
for ms in history['windows_ms']:
    points = history[f'w{ms}']
    assert points.shape == (len(label), 2) and np.isfinite(points).all()
    routes, counts, centres = [], [], []
    for c in range(8):
        aligned = []
        for repetition in np.unique(cycle[label == c]):
            mask = (label == c) & (cycle == repetition)
            groups = [points[mask & (phase >= a) & (phase < b)] for a, b in zip(bins[:-1], bins[1:])]
            if all(len(g) for g in groups):
                aligned.append([np.median(g, axis=0) for g in groups])
        # Match the main figure's mean trajectory: per-cycle bin medians, then mean
        # across repetitions; circular Gaussian sigma .5 and periodic spline.
        mean = np.mean(aligned, axis=0)
        offsets = np.arange(-3, 4)
        weights = np.exp(-.5 * (offsets / .5) ** 2)
        weights /= weights.sum()
        smooth = sum(w * np.roll(mean, -o, axis=0) for o, w in zip(offsets, weights))
        spline = CubicSpline(bins, np.vstack([smooth, smooth[:1]]), bc_type='periodic')
        routes.append(spline(np.linspace(0, 1, 241)).round(5).tolist())
        centres.append(np.median(points[label == c], axis=0).round(5).tolist())
        counts.append(len(aligned))
    lo, hi = points.min(0), points.max(0)
    pad = (hi - lo) * .07
    windows[str(ms)] = dict(points=points.round(5).tolist(), routes=routes,
                           centres=centres, cycles=counts,
                           bounds=np.r_[lo-pad, hi+pad].round(5).tolist())
assert windows['480']['cycles'] == [19] * 8
provenance = dict(
    participant=1, updates=len(label), electrodes=320, update_ms=20,
    settings=json.loads(str(main['settings'])),
    sources={name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
             for name in ('umap_sub01_tokens.npz', 'history_supplement.npz')},
    routes='20 phase-bin medians per complete repetition, averaged across repetitions; circular Gaussian sigma=.5 bins, periodic cubic spline. The spline explicitly joins the endpoints.',
    display='Saved 3D UMAP, each movement projected into its mean-trajectory plane; groups positioned separately. Every history window is independently fitted and arranged. Full point bounds plus 7% padding, equal axis scale.',
    timing='Normalized repetition phase, played over 8 seconds; a mean trajectory, not one recorded trial or real-time playback.',
)
out = dict(names=names, colors=colors, labels=label.tolist(), windows=windows, provenance=provenance)
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'umap.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
(OUT/'umap-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(f'Exported {len(label):,} updates, {len(windows)} saved history views; 19 mean-trajectory repetitions per movement.')
