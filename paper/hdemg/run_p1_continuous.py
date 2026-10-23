"""Continuous test-block predictions for participant 1 (SPAR tokens and continuous SPAR).

Each movement's two test windows start 6000 samples apart; only their last second is
scored. Two extra 3 s windows starting 2000 and 4000 samples after the first test window
make the scored seconds tile 4 s without gaps. All four windows lie inside the test block.
The frozen pipeline (run_hdemg_spar_simple / run_hdemg_spar_full) is reused unchanged, and
the run is gated: for the original test windows it must reproduce the cached features,
tokens and the stored test predictions.

    uv run python -m paper.hdemg.run_p1_continuous
"""
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.signal import firwin, upfirdn

from paper.common import paths
from paper.hdemg import run_hdemg_spar_simple as pilot
from myoji.codebook import standardize, tokenize
from myoji.spar import grid_layout, spar_simple_dataset_batched

ROOT = Path(__file__).resolve().parents[1]
RUN = paths.RESULTS / 'hdemg_spar_simple_full_v1'

SUBJECT = int(next((a for a in sys.argv[1:] if a.isdigit()), 1))
SEED = 260911
CACHE = RUN / f'subject_{SUBJECT:02d}/features.h5'    # the campaign cache, sha-gated below
DATA = paths.HDEMG
GATE = []
DIAG = len(sys.argv) > 1 and sys.argv[1] == 'diag'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    p = json.loads((RUN / f'subject_{SUBJECT:02d}/protocol.json').read_text())
    receipt = json.loads((RUN / f'subject_{SUBJECT:02d}/cache-receipt.json').read_text())
    import hashlib
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == receipt['sha256'], 'cache is not this run\'s'
    with h5py.File(CACHE) as f:
        book, mean, scale, mask = (f[k][:] for k in ('codebook', 'feature_mean', 'feature_scale', 'mask'))
        cached = {arm: f[f'test/{arm}'][:] for arm in ('continuous', 'tokens')}
        cached_tasks, cached_starts = f['test/tasks'][:], f['test/starts'][:]
    layout = grid_layout(p['geometry']['grids'])
    emg = joblib.load(DATA / f'Sub{SUBJECT}_emg.pkl')
    kin = joblib.load(DATA / f'Sub{SUBJECT}_kinematics.pkl')
    taps = firwin(10241, 800, fs=256000, window=('kaiser', 8.6)) * 125
    indices = pilot.ENDS[:, None] - np.arange(319, -1, -1)[None]
    models = {}
    for arm in ('tokens', 'continuous'):
        saved = torch.load(RUN / f'subject_{SUBJECT:02d}/fits/{arm}_{SEED}/best.pt', map_location='cpu', weights_only=True)
        m = pilot.SparPose(torch.from_numpy(book), torch.from_numpy(mask)); m.load_state_dict(saved['model'])
        models[arm] = (m.to(DEV).eval(), saved['mean'], saved['scale'])
    stored = {arm: np.load(RUN / f'subject_{SUBJECT:02d}/fits/{arm}_{SEED}/test_predictions.npz') for arm in models}

    tests = [w for w in p['windows'] if w['split'] == 'test']
    out = {}
    for task in sorted({int(w['task']) for w in tests}):
        s1, s2 = sorted(w['start'] for w in tests if int(w['task']) == task)
        assert s2 == s1 + 6000, (task, s1, s2)
        starts = [s1, s1 + 2000, s1 + 4000, s2]          # scored seconds tile s1+4000 .. s2+6000
        sig = upfirdn(taps, emg[str(task)].astype(np.float32), up=125, down=128, axis=-1)[..., :emg[str(task)].shape[-1] * 125 // 128].astype(np.float32)
        preds = {arm: [] for arm in models}; targets = []
        for start in starts:
            raw = pilot.normalize_raw(sig[:, start:start + 6000])
            analysis = raw[:, indices].transpose(1, 2, 0).copy()
            feats = spar_simple_dataset_batched(analysis, layout=layout, channelwise=True, batch_size=16, device=DEV)
            feats = feats.transpose(1, 0, 2)[None]                       # (1, 320, 100, 18)
            cont = standardize(feats, mean, scale, mask); tok = tokenize(cont, book, device=DEV)
            ti = ((start + pilot.ENDS) * 128) // 125
            pts = kin[str(task)][..., ti].transpose(2, 0, 1).astype(np.float32)
            targets.append((pts[:, 1:] - pts[:, :1]).reshape(100, 60)[-50:])
            if start in (s1, s2):                                       # gate on the original windows
                k = int(np.flatnonzero((cached_tasks == task) & (cached_starts == start))[0])
                diff = np.abs(cont[0] - cached['continuous'][k]); agree = float((tok[0] == cached['tokens'][k]).mean())
                print(json.dumps(dict(task=task, start=start, max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                                      scale=float(np.abs(cached['continuous'][k]).mean()), token_agreement=agree)), flush=True)
                if DIAG: continue
                # GPU floating point: features agree to ~1e-4 on average and tokens to >99.9 %.
                assert diff.mean() < 1e-3 and agree > .999, ('features/tokens differ beyond float noise', task, start)
            for arm, (m, om, osc) in models.items():
                x = torch.from_numpy(tok if arm == 'tokens' else cont).to(DEV)
                x = x.long() if arm == 'tokens' else x.float()
                with torch.inference_mode():
                    y = (m(x, arm).cpu() * osc + om)[:, -50:].numpy()[0]
                preds[arm].append(y)
                if start in (s1, s2) and not DIAG:
                    k = int(np.flatnonzero((stored[arm]['tasks'] == task) & (stored[arm]['starts'] == start))[0])
                    d = np.linalg.norm((y - stored[arm]['prediction'][k]).reshape(-1, 20, 3), axis=-1)
                    GATE.append(dict(arm=arm, task=task, start=start, mean_mm=float(d.mean()), max_mm=float(d.max())))
                    assert d.mean() < .5, ('prediction differs from the stored test prediction', arm, task, start, float(d.mean()))
        if DIAG and task > 4: return
        out[task] = dict(target=np.concatenate(targets), **{arm: np.concatenate(v) for arm, v in preds.items()},
                         starts=np.array(starts))
        print(json.dumps(dict(task=task, starts=starts, gate='passed')), flush=True)
    dest = paths.RESULTS / f'decoding_continuous/p{SUBJECT}_continuous.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **{f'{k}_{t}': v for t, d in out.items() for k, v in d.items()})
    print('saved', dest)
    print(json.dumps(dict(prediction_gate_mean_mm=max(g['mean_mm'] for g in GATE), prediction_gate_max_mm=max(g['max_mm'] for g in GATE))))
    (paths.RESULTS / f'decoding_continuous/gate_p{SUBJECT}.json').write_text(json.dumps(GATE, indent=1))


if __name__ == '__main__':
    main()
