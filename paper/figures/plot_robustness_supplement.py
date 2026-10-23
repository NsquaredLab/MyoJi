"""Supplementary robustness figure: absolute pose error as the input degrades.

Each panel starts from the clean input and shows the actual 3D error of the frozen
decoders as noise is added (A) or electrodes are silenced (B): the learned route's token and
continuous-feature decoders, the adapted CNN, and (dashed) the SPAR token and continuous-SPAR
pipelines of results/hdemg_spar_corruption_v1, scored on the same corrupted windows, so the reader sees how much
worse each model gets rather than a difference of differences. Per-participant errors
average three decoder seeds and three corruption draws; whiskers are exact 95 % sign-flip
intervals over the nine participants. The paired tests are those of plot_noise_robustness.py (exact
sign-flip on the increase from clean, Holm-adjusted); the only significant one is marked.

    uv run python -m paper.figures.plot_robustness_supplement
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
from paper.emg2pose import build_pose_paper_assets as data  # noqa: E402
from paper.figures import plot_noise_robustness as nr      # noqa: E402  (loaders and paired tests)
from paper.common.hdemg_figure_statistics import exact_location_interval  # noqa: E402
from paper.common import paths                            # noqa: E402
from paper.common import switzer_font                     # noqa: E402

ARMS = [('token_v20', 'Learned Tokens', '#d80032', 'o'),
        ('continuous_v20', 'Learned Features', '#5c618e', 's'),
        ('native_v19', 'Adapted CNN', '#6d6f72', 'D')]
SPAR_ARMS = [('tokens', 'SPAR Tokens', '#d80032', '^'), ('continuous', 'Continuous SPAR', '#5c618e', 'v')]
SPAR = ROOT / 'results/hdemg_spar_corruption_v1'


def spar_level(arm, kind, lev):
    """Per-participant SPAR error under one condition (decoder seeds and corruption draws averaged)."""
    out = []
    for f in sorted(SPAR.glob('subject_*.json')):
        rows = [r for r in json.loads(f.read_text())['rows'] if r['kind'] == kind and r['level'] == lev]
        assert rows, (f, kind, lev)
        out.append(np.mean([r['errors'][arm] for r in rows]))
    assert len(out) == 9
    return np.array(out)
NOISE_X = [None, 40, 30, 20, 10]                       # SNR in dB; None = clean
SCATTER = [('fraction', .05, '16'), ('fraction', .10, '32'), ('fraction', .25, '80')]
ARRAYS = [('grids', 1, '1 array\n(64)'), ('grids', 2, '2 arrays\n(128)')]


def dropout_level(drop, arm, kind, lev):
    """Per-participant absolute error under one dropout condition (seeds and draws averaged)."""
    rows = [v for v in drop['rows'] if v['arm'] == arm and v['kind'] == kind and v['level'] == lev
            and v['filtering'] == 'unfiltered']
    subjects = sorted({v['subject'] for v in rows})
    return np.array([np.mean([v['mean_3d_point_error'] for v in rows if v['subject'] == s]) for s in subjects])


def main():
    noise, drop = data.read(nr.NOISE), data.read(nr.DROPOUT)
    plt.rcParams.update(switzer_font.rcparams())
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                         'axes.linewidth': .75, 'xtick.major.size': 3, 'ytick.major.size': 3})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=True,
                               gridspec_kw={'width_ratios': [1, 1.15]})
    fig.subplots_adjust(left=.08, right=.99, top=.93, bottom=.33, wspace=.12)
    off = {'token_v20': -.16, 'continuous_v20': -.08, 'native_v19': 0, 'tokens': .08, 'continuous': .16}   # display-only dodge
    receipt = {'noise': {}, 'dropout': {}}
    for arm, name, color, marker in ARMS:
        clean = nr.level(noise, arm, None)
        # A: noise
        ys = [nr.level(noise, arm, s) for s in NOISE_X]
        x = np.arange(len(NOISE_X)) + off[arm]
        m = np.array([v.mean() for v in ys]); ci = np.array([exact_location_interval(v) for v in ys])
        a.plot(x, m, color=color, lw=1.4, zorder=3)
        a.errorbar(x, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt=marker, color=color, ms=4.5,
                   elinewidth=.9, capsize=2, markeredgecolor='white', markeredgewidth=.5, zorder=4)
        receipt['noise'][name] = dict(zip(['clean', '1%', '3.2%', '10%', '32%'], m.round(3).tolist()))
        # B: electrode loss, scattered then whole arrays, both starting from clean
        sc = [clean] + [dropout_level(drop, arm, k, l) for k, l, _ in SCATTER]
        ar = [dropout_level(drop, arm, k, l) for k, l, _ in ARRAYS]
        for xs, vals, filled in ((np.arange(4), sc, True), (np.array([4.6, 5.6]), ar, False)):
            mm = np.array([v.mean() for v in vals]); cc = np.array([exact_location_interval(v) for v in vals])
            b.plot(xs + off[arm], mm, color=color, lw=1.4, zorder=3)
            b.errorbar(xs + off[arm], mm, yerr=[mm - cc[:, 0], cc[:, 1] - mm], fmt=marker, color=color,
                       ms=4.5, elinewidth=.9, capsize=2, mfc=color if filled else 'white',
                       markeredgecolor=color if not filled else 'white', markeredgewidth=.8, zorder=4)
        receipt['dropout'][name] = {'scattered': [float(v.mean()) for v in sc], 'arrays': [float(v.mean()) for v in ar]}
    for arm, name, color, marker in SPAR_ARMS:
        ys = [spar_level(arm, None, None) if s is None else spar_level(arm, 'noise', s) for s in NOISE_X]
        x = np.arange(len(NOISE_X)) + off[arm]
        m = np.array([v.mean() for v in ys]); ci = np.array([exact_location_interval(v) for v in ys])
        a.plot(x, m, color=color, lw=1.2, ls=(0, (4, 2)), zorder=3)
        a.errorbar(x, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt=marker, color=color, ms=5, elinewidth=.9, capsize=2,
                   markeredgecolor='white', markeredgewidth=.5, zorder=4)
        receipt['noise'][name] = dict(zip(['clean', '1%', '3.2%', '10%', '32%'], m.round(3).tolist()))
        sc = [ys[0]] + [spar_level(arm, k, l) for k, l, _ in SCATTER]
        ar = [spar_level(arm, k, l) for k, l, _ in ARRAYS]
        for xs, vals, filled in ((np.arange(4), sc, True), (np.array([4.6, 5.6]), ar, False)):
            mm = np.array([v.mean() for v in vals]); cc = np.array([exact_location_interval(v) for v in vals])
            b.plot(xs + off[arm], mm, color=color, lw=1.2, ls=(0, (4, 2)), zorder=3)
            b.errorbar(xs + off[arm], mm, yerr=[mm - cc[:, 0], cc[:, 1] - mm], fmt=marker, color=color,
                       ms=5, elinewidth=.9, capsize=2, mfc=color if filled else 'white',
                       markeredgecolor=color if not filled else 'white', markeredgewidth=.8, zorder=4)
        receipt['dropout'][name] = {'scattered': [float(v.mean()) for v in sc], 'arrays': [float(v.mean()) for v in ar]}

    # The one significant paired contrast: at 32 % noise the token decoder's increase from
    # clean is smaller than the continuous decoder's (Holm-adjusted exact sign-flip test).
    tests = nr.contrasts(noise, nr.SNRS)
    sig = [t for t in tests if t['significant']]
    assert len(sig) == 1 and sig[0]['snr_db'] == 10 and sig[0]['reference'] == 'continuous_v20'
    # Plotted as absolute error the two arms overlap at 32 %, so the contrast is stated in the caption, not drawn.

    a.set_xticks(range(5), ['Clean', '1 %', '3.2 %', '10 %', '32 %'])
    a.set_xlim(-.4, 4.6)
    a.set_xlabel('Added Noise (% of EMG RMS)', fontsize=9.5, labelpad=4)
    a.set_ylabel('3D Error (mm)', fontsize=9.5, labelpad=4)
    b.set_xticks([0, 1, 2, 3, 4.6, 5.6], ['None', '16', '32', '80'] + [l for *_, l in ARRAYS])
    b.set_xlim(-.4, 6.0)
    b.axvline(4.05, color='#DEDEDE', lw=.8, zorder=0)
    b.text(1.5, -.30, 'Scattered Electrodes', transform=b.get_xaxis_transform(), ha='center', fontsize=9.5)
    b.text(5.1, -.30, 'Whole Arrays', transform=b.get_xaxis_transform(), ha='center', fontsize=9.5)
    for ax in (a, b):
        ax.grid(axis='y', color='#DEDEDE', lw=.55); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
    fig.text(.004, 1.0, 'A', fontsize=12, va='top'); fig.text(.475, 1.0, 'B', fontsize=12, va='top')
    fig.legend(handles=[Line2D([], [], color=c, marker=m, lw=1.4, ms=5, label=n) for _, n, c, m in ARMS]
               + [Line2D([], [], color=c, marker=m, lw=1.2, ls=(0, (4, 2)), ms=5, label=n) for _, n, c, m in SPAR_ARMS],
               loc='lower center', bbox_to_anchor=(.53, 0.0), ncol=5, frameon=False, fontsize=8.5,
               handlelength=2.2, columnspacing=1.2)
    out = paths.OUTPUT / 'figures/pose_paper/robustness_supplement'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf')); fig.savefig(out.with_suffix('.png'), dpi=300)
    receipt['significant'] = sig
    out.with_suffix('.json').write_text(json.dumps(receipt, indent=1, default=float))
    print(json.dumps(receipt, indent=1, default=float)[:1500])


if __name__ == '__main__':
    data.configure()
    main()
