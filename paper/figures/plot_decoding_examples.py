"""Decoded hand movement on the HD-EMG test set: all eight movements of one participant.

The participant is chosen by rule, not by eye: the one with the median SPAR-token error
over the nine participants (seed 260911). Every movement is shown, both scored seconds of
its two test windows, so nothing is selected per panel. Each panel plots the quantity the
movement actually changes, computed from the 20 wrist-relative landmarks (fingertips are
landmarks 0, 4, 8, 12, 16): the moving fingertip's distance from the wrist, the mean over
all five tips for the fist, and the thumb-to-fingertip distance for the pinches. Predictions
are unfiltered 20 ms updates; nothing is smoothed, shifted or aligned.

    uv run python -m paper.figures.plot_decoding_examples
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
from paper.common import paths  # noqa: E402
from paper.common import switzer_font  # noqa: E402
from paper.hdemg.hdemg_one_euro import smooth  # noqa: E402  (causal One Euro filter)
from paper.common import hdemg_figure_statistics as stats  # noqa: E402  (exact sign-flip intervals, as in Fig. 3)

# Display filter: the causal One Euro setting selected on validation data for the
# learned-token decoder (results/hdemg_v20_one_euro_v1/selection.json), applied unchanged to
# both SPAR arms. It uses past updates only, so the decoder's real lag stays visible.
FILTER = json.loads((paths.REFERENCE / 'hdemg_v20_one_euro_v1/selection.json').read_text())['selected']['config']
assert FILTER == {'kind': 'one_euro', 'mincutoff': 1.0, 'beta': 0.1, 'dcutoff': 1.0, 'freq': 50.0}

CAMPAIGN, SEED = 'results/hdemg_spar_simple_full_v1', 260911
TIPS = {'thumb': 0, 'index': 4, 'middle': 8, 'ring': 12, 'pinky': 16}
# task id -> (title, fingertip distance from one (50, 20, 3) window); one y label for all panels
dist = lambda w, a: np.linalg.norm(w[:, a], axis=-1)
gap = lambda w, a, b: np.linalg.norm(w[:, a] - w[:, b], axis=-1)
MOVES = {
    # The thumb flexes across the palm, which barely changes its tip-wrist distance; its distance
    # to the pinky base (landmark 19, the MCP) measures that movement directly.
    2: ('Thumb', lambda w: gap(w, TIPS['thumb'], 19)),
    4: ('Index', lambda w: dist(w, TIPS['index'])),
    6: ('Middle', lambda w: dist(w, TIPS['middle'])),
    8: ('Ring', lambda w: dist(w, TIPS['ring'])),
    10: ('Pinky', lambda w: dist(w, TIPS['pinky'])),
    15: ('Fist', lambda w: np.mean([dist(w, t) for t in TIPS.values()], 0)),
    18: ('2-Pinch', lambda w: gap(w, TIPS['thumb'], TIPS['index'])),
    19: ('3-Pinch', lambda w: (gap(w, TIPS['thumb'], TIPS['index']) + gap(w, TIPS['thumb'], TIPS['middle'])) / 2),
}
TOKEN, FEATURE, MEASURED = '#d80032', '#5c618e', 'black'


def load(subject, arm):
    z = np.load(ROOT / f'{CAMPAIGN}/subject_{subject:02d}/fits/{arm}_{SEED}/test_predictions.npz')
    return z['target'].reshape(-1, 50, 20, 3).astype(float), z['prediction'].reshape(-1, 50, 20, 3).astype(float), z['tasks']


import os
SUBJECT = int(os.environ.get('FIG4_SUBJECT', 6))   # example participant shown in panel A
# 4 s of contiguous test-block predictions per movement (hdemg/run_p1_continuous.py):
# the two test windows plus two in-between windows whose scored seconds fill the gap; gated against
# the stored test predictions (gate_p<subject>.json).
CONTINUOUS = paths.RESULTS / f'decoding_continuous/p{SUBJECT}_continuous.npz'
SEEDS = (260911, 260912, 260913)


def per_movement_error():
    """Mean 3D error (20 landmarks, unfiltered) per participant x movement x arm, seeds averaged."""
    out = np.zeros((9, len(MOVES), 2))
    for s in range(1, 10):
        for j, arm in enumerate(('tokens', 'continuous')):
            per_seed = []
            for seed in SEEDS:
                z = np.load(ROOT / f'{CAMPAIGN}/subject_{s:02d}/fits/{arm}_{seed}/test_predictions.npz')
                tg, pr = (z[k].reshape(-1, 50, 20, 3).astype(float) for k in ('target', 'prediction'))
                e = np.linalg.norm(pr - tg, axis=-1).mean((1, 2))          # per window
                per_seed.append([e[z['tasks'] == task].mean() for task in MOVES])
            out[s - 1, :, j] = np.mean(per_seed, 0)
    return out


def main():
    subject = SUBJECT
    tg, tok_raw, tasks = load(subject, 'tokens')
    tg2, cont_raw, tasks2 = load(subject, 'continuous')
    assert np.array_equal(tg, tg2) and np.array_equal(tasks, tasks2)
    cfg = {k: v for k, v in FILTER.items() if k != 'kind'}
    filt = lambda x: smooth(x.reshape(len(x), 50, 60), **cfg).reshape(x.shape)   # per window, past updates only
    tok, cont = filt(tok_raw), filt(cont_raw)
    err = per_movement_error()                                     # (9, 8, 2)
    overall = err.mean(1)                                          # per participant, both arms
    rank = int((overall[:, 0] < overall[subject - 1, 0]).sum()) + 1

    plt.rcParams.update(switzer_font.rcparams())
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                         'axes.linewidth': .75, 'xtick.major.size': 3, 'ytick.major.size': 3})
    fig = plt.figure(figsize=(7.16, 4.0))
    left = fig.add_gridspec(2, 4, left=.065, right=.70, top=.93, bottom=.2, wspace=.4, hspace=.45)
    axes = np.array([[fig.add_subplot(left[r, c]) for c in range(4)] for r in range(2)])
    bar = fig.add_axes((.81, .2, .18, .73))
    t = np.arange(50) * .02
    receipt = {'display_filter': FILTER, 'participant': subject,
               'participant_rank_of_nine_spar_tokens': rank,
               'participant_error_mm': {'tokens': float(overall[subject - 1, 0]), 'continuous': float(overall[subject - 1, 1])},
               'panels': {}, 'per_movement': {}}
    cz = np.load(CONTINUOUS)
    t = np.arange(200) * .02
    for ax, (task, (title, f)) in zip(axes.ravel(), MOVES.items()):
        tg_c = cz[f'target_{task}'].reshape(200, 20, 3)
        raw = {arm: cz[f'{arm}_{task}'].reshape(200, 20, 3) for arm in ('tokens', 'continuous')}
        # Display filter over the contiguous 4 s, causal (past updates only).
        sm = {arm: smooth(v.reshape(1, 200, 60), **cfg).reshape(200, 20, 3) for arm, v in raw.items()}
        ax.plot(t, f(tg_c), color=MEASURED, lw=1.5, zorder=3)
        ax.plot(t, f(sm['continuous']), color=FEATURE, lw=1.0, ls=(0, (3, 1.5)), zorder=2)
        ax.plot(t, f(sm['tokens']), color=TOKEN, lw=1.0, zorder=2)
        y_all = np.concatenate([f(tg_c), f(sm['tokens']), f(sm['continuous'])])
        pad = .12 * np.ptp(y_all)
        ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
        ax.set_title(title, fontsize=9, loc='left', pad=3)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlim(0, 4)
        meas = f(tg_c)
        r = lambda pred: float(np.corrcoef(meas, f(pred))[0, 1])
        receipt['panels'][title] = {'r_tokens': r(sm['tokens']), 'r_continuous': r(sm['continuous']),
                                    'r_tokens_unfiltered': r(raw['tokens']), 'r_continuous_unfiltered': r(raw['continuous']),
                                    'error_mm_tokens_unfiltered': float(np.linalg.norm(raw['tokens'] - tg_c, axis=-1).mean()),
                                    'error_mm_continuous_unfiltered': float(np.linalg.norm(raw['continuous'] - tg_c, axis=-1).mean())}
    for ax in axes[:, 0]:
        ax.set_ylabel('Fingertip Distance (mm)', fontsize=8, labelpad=2)
    for ax in axes[1]:
        ax.set_xlabel('Time (s)', fontsize=8)

    # B: error per movement over all nine participants; the panel-A participant marked.
    names = [v[0] for v in MOVES.values()]
    y = np.arange(len(names))
    for j, (color, dy, name) in enumerate(((TOKEN, -.19, 'tokens'), (FEATURE, .19, 'continuous'))):
        m = err[:, :, j].mean(0)
        ci = np.array([stats.exact_location_interval(err[:, i, j]) for i in range(len(names))])
        bar.barh(y + dy, m, height=.34, color=color, zorder=2)
        bar.errorbar(m, y + dy, xerr=[m - ci[:, 0], ci[:, 1] - m], fmt='none', ecolor='black', elinewidth=.7, capsize=0, zorder=3)
        for i in range(len(names)):
            bar.scatter(err[subject - 1, i, j], y[i] + dy, s=18, marker='D', color='black',
                        edgecolor='white', linewidth=.6, zorder=5)
            receipt['per_movement'].setdefault(names[i], {})[name] = {
                'mean_mm': float(m[i]), 'ci95_exact_mm': ci[i].tolist(),
                f'participant_{subject}_mm': float(err[subject - 1, i, j])}
    # Paired tokens-minus-continuous difference per movement, as an estimate with its exact
    # sign-flip interval; no test, and no markers in the panel (the claims are in Fig. 3).
    D = err[:, :, 0] - err[:, :, 1]
    for i in range(len(names)):
        receipt['per_movement'][names[i]]['tokens_minus_continuous'] = {
            'mean_mm': float(D[:, i].mean()), 'ci95_exact_mm': stats.exact_location_interval(D[:, i]),
            'participants_tokens_higher': int((D[:, i] > 0).sum()), 'per_participant_mm': D[:, i].tolist()}
    receipt['interval'] = 'exact 95 % sign-flip inversion over the nine participants (pointwise)'
    bar.set_yticks(y, names, fontsize=8)
    bar.set_ylim(len(names) - .45, -.55)
    bar.set_xlim(0, 22)
    bar.set_xticks([0, 5, 10, 15, 20])
    bar.set_xlabel('3D Error (mm)', fontsize=8.5)
    bar.tick_params(axis='x', labelsize=7.5); bar.tick_params(axis='y', length=0)
    bar.grid(axis='x', color='#DEDEDE', lw=.55); bar.set_axisbelow(True)
    bar.spines['left'].set_visible(False)

    fig.text(.004, 1.0, 'A', fontsize=12, va='top'); fig.text(.735, 1.0, 'B', fontsize=12, va='top')
    fig.legend(handles=[Line2D([], [], color=MEASURED, lw=1.5, label='Measured'),
                        Line2D([], [], color=TOKEN, lw=2.0, label='SPAR Tokens'),
                        Line2D([], [], color=FEATURE, lw=2.0, ls=(0, (3, 1.5)), label='Continuous SPAR'),
                        Line2D([], [], color='black', marker='D', ls='', ms=4, label=f'Participant {SUBJECT} (B)')],
               loc='lower center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, 0), columnspacing=1.2)
    out = paths.OUTPUT / 'figures/pose_paper' / os.environ.get('FIG4_OUT', 'decoding_examples')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf')); fig.savefig(out.with_suffix('.png'), dpi=300)
    out.with_suffix('.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ('participant', 'participant_rank_of_nine_spar_tokens', 'participant_error_mm')}, indent=1))


if __name__ == '__main__':
    main()
