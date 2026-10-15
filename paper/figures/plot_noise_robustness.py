"""Clean accuracy, and the paired effect of added noise and of electrode loss.

What the reader should see: quantizing the learned features does not make the
decoder more fragile. Under added noise the token decoder degrades no more than
its continuous counterpart, and less at the strongest level; under electrode
loss the two cannot be told apart. The adapted CNN is more accurate clean.

Panel B carries a test, not just overlapping whiskers: at each noise level the
token and continuous arms are compared on their degradation from clean, paired
within participant, by the exact sign-flip permutation over all 2^9 patterns,
with Holm across the eight token-versus-feature and token-versus-CNN contrasts.

    uv run python -m paper.figures.plot_noise_robustness
"""
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from paper.emg2pose import build_pose_paper_assets as data
from paper.common import tokenizer_diagnostic_style as style

from paper.common import figstyle as sf

DROPOUT = 'results/dpose_dropout_v1/results.json'
NOISE = 'results/dpose_v20_noise_followup_v2/results.json'
# Electrodes silenced by each level, so scattered and array loss share one axis.
# 320 channels in five 64-channel arrays; the three draws per level are averaged.
# No count is shared by both kinds: never call the axis "matched".
DROP_X = {('fraction', 0.05): 16, ('fraction', 0.10): 32, ('fraction', 0.25): 80,
          ('grids', 1): 64, ('grids', 2): 128}
SNRS = (40, 30, 20, 10)
BOOT_DRAWS, BOOT_SEED = 10_000, 260930    # the noise protocol's own bootstrap
# Figure 3 house style.
GRID_COLOR, GRID_WIDTH = '#DEDEDE', 0.55
INK = '#303030'


def amplitude(snr_db):
    """Noise RMS as a percentage of the channel's EMG RMS.

    dpose_noise.perturb_native scales unit Gaussian noise by
    sqrt(signal power) * 10**(-SNR/20), so this is exact, not a conversion
    convention: 10 dB is noise at 31.6 % of the signal's amplitude. It runs
    left to right with severity, which the reversed dB axis did not.
    """
    return 100 * 10 ** (-snr_db / 20)


def boot_ci(d):
    """Participant bootstrap, as the noise campaign computed its stored intervals."""
    d = np.asarray(d, float)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(d), (BOOT_DRAWS, len(d)))
    return [float(v) for v in np.percentile(d[idx].mean(1), [2.5, 97.5])]


def level(r, arm, snr):
    """Per-participant unfiltered error, averaging decoder seeds and noise seeds."""
    rows = [v for v in r['rows'] if v['filtering'] == 'unfiltered' and v['arm'] == arm]
    subjects = sorted({v['subject'] for v in rows})
    return np.array([np.mean([v['mean_3d_error'] for v in rows
                              if v['subject'] == s and v['snr_db'] == snr]) for s in subjects])


def degradation(r, arm, snr):
    """Per-participant error increase from clean."""
    return level(r, arm, snr) - level(r, arm, None)


def sign_flip_p(d):
    """Exact paired permutation over all 2^n sign patterns; n is 9 participants."""
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def holm(out):
    raw = np.array([v['p_raw'] for v in out]); order = np.argsort(raw)
    adj = np.minimum(np.maximum.accumulate(raw[order] * (len(raw) - np.arange(len(raw)))), 1)
    for rank, i in enumerate(order):
        out[i]['p_holm'] = float(adj[rank])
        out[i]['significant'] = bool(adj[rank] < .05)
    return out


def contrasts(r, levels):
    """Token minus each reference arm at every SNR, Holm-adjusted as one family."""
    out = []
    for reference in ('continuous_v20', 'native_v19'):
        for snr in levels:
            d = degradation(r, 'token_v20', snr) - degradation(r, reference, snr)
            out.append({'reference': reference, 'snr_db': snr, 'n_participants': len(d),
                        'mean_gap_mm': float(d.mean()), 'ci95': boot_ci(d),
                        'participants_token_less': int((d < 0).sum()),
                        'p_raw': sign_flip_p(d)})
    return holm(out)


def drop_delta(drop, arm, kind, lev):
    """Per-participant increase from clean, averaging the three dropout draws."""
    rows = [v['per_subject_delta'] for v in drop['summary']
            if v['arm'] == arm and v['kind'] == kind and v['level'] == lev]
    assert len(rows) == 3, (arm, kind, lev)
    return np.mean(np.array(rows), axis=0)


def dropout_contrasts(drop):
    """Tokens minus continuous under dropout, paired within participant."""
    out = []
    for kind, lev in sorted(DROP_X, key=DROP_X.get):
        d = drop_delta(drop, 'token_v20', kind, lev) - drop_delta(drop, 'continuous_v20', kind, lev)
        out.append({'electrodes': DROP_X[(kind, lev)], 'kind': kind, 'level': lev,
                    'mean_gap_mm': float(d.mean()), 'ci95': boot_ci(d),
                    'p_raw': sign_flip_p(d)})
    return holm(out)


def house_axis(ax, grid):
    """Figure 3: light grid carries the scale, no minor ticks, trimmed bottom spine."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid, which='major', color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.grid(axis='x' if grid == 'y' else 'y', visible=False)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.tick_params(axis='both', labelsize=9, length=3, color='black')
    ax.tick_params(axis='y', length=0, pad=4)
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    sf.trim_spines(ax, which=('bottom',), offset=0)


def draw(r, drop):
    style.apply()
    # Black text, ticks and axes, as in Figures 3 and 6 (the shared style greys them).
    plt.rcParams.update({k: 'black' for k in ('text.color', 'axes.labelcolor', 'axes.edgecolor',
                                              'xtick.color', 'ytick.color', 'xtick.labelcolor', 'ytick.labelcolor')})
    # Cosmetic horizontal dodge (B: factor on the log axis, C: electrodes) so the
    # token and feature whiskers do not print on top of each other.
    dodge = {'native_v19': 0, 'token_v20': -1, 'continuous_v20': 1}
    choices = [('native_v19', 'Adapted CNN', style.GRAY, 'D'),
               ('token_v20', 'Tokens', style.TOKEN, 'o'),
               ('continuous_v20', 'Continuous Features', style.FEATURE, 's')]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75),
                             gridspec_kw={'width_ratios': [.78, 1.15, 1.15]})
    fig.subplots_adjust(left=.2, right=.985, bottom=.2, top=.79, wspace=.42)
    a, b, c = axes
    details = []
    for y, (arm, name, color, marker) in enumerate(choices):
        rows = [next(v for v in r['summary'] if v['arm'] == arm and v['filtering'] == 'unfiltered'
                     and v['snr_db'] == snr) for snr in (None, *SNRS)]
        clean = rows[0]
        a.plot(clean['participant_ci95'], [y, y], color=color, lw=1.4)
        a.scatter(clean['mean_3d_error'], y, marker=marker, color=color, s=30, zorder=3,
                  edgecolor='white', linewidth=.5)
        x = amplitude(np.array([v['snr_db'] for v in rows[1:]], float)) * 1.035 ** dodge[arm]
        effect = np.array([v['clean_delta'] for v in rows[1:]])
        ci = np.array([v['paired_clean_delta_ci95'] for v in rows[1:]])
        # Tokens drawn over features; whiskers at measured levels only.
        b.plot(x, effect, color=color, lw=1.3, zorder=3 + (arm == 'token_v20'))
        b.errorbar(x, effect, yerr=np.stack((effect - ci[:, 0], ci[:, 1] - effect)),
                   fmt=marker, color=color, ms=4, elinewidth=.8, capsize=2,
                   markeredgecolor='white', markeredgewidth=.4,
                   zorder=3 + (arm == 'token_v20'))
        details += rows
    tests = contrasts(r, SNRS)
    drop_tests = dropout_contrasts(drop)

    # The one surviving contrast, drawn just left of the strongest-noise markers
    # and spanning exactly the two means it compares, so the p value cannot be
    # read as belonging to one curve. Horizontal, at the house 8.5 pt.
    for t in (v for v in tests if v['significant']):
        lo, hi = sorted(float(degradation(r, arm, t['snr_db']).mean())
                        for arm in ('token_v20', t['reference']))
        at = amplitude(t['snr_db']) / 1.15
        b.plot([at * 1.045, at, at, at * 1.045], [lo, lo, hi, hi], color=INK, lw=.7,
               zorder=5, solid_capstyle='butt')
        q = t['p_holm']    # asterisks as in Figures 3 and 6; the value is in the caption
        b.text(at / 1.04, (lo + hi) / 2, '***' if q < .001 else '**' if q < .01 else '*', ha='right',
               va='center', fontsize=11, color='black')

    a.set_yticks(range(3), [v[1] for v in choices])
    a.set(ylim=(2.6, -.6), xlim=(9.2, 16.2))
    a.set_xticks([10, 12, 14, 16])
    a.set_xlabel('Clean 3D Error (mm)', labelpad=5, fontsize=9.5)
    b.set_xscale('log')
    b.set(xlim=(.78, 40), ylim=(-.05, 1.75))
    b.xaxis.set_major_locator(FixedLocator(amplitude(np.array(SNRS, float))))
    b.set_yticks([0, .5, 1, 1.5])
    b.set_xlabel('Noise Amplitude (% of EMG RMS)', labelpad=5, fontsize=9.5)
    b.set_ylabel('Error Increase (mm)', labelpad=5, fontsize=9.5)

    # C: electrode dropout. Scattered and whole-array loss share one axis of
    # electrodes silenced; the counts differ (16/32/80 vs 64/128), so the two
    # kinds are read along the axis, not at matched points.
    drops = []
    for arm, name, color, marker in choices:
        for kind, ls, fill in (('fraction', '-', True), ('grids', ':', False)):
            keys = sorted((k for k in DROP_X if k[0] == kind), key=DROP_X.get)
            xs = np.array([DROP_X[k] for k in keys], float)
            shown = xs + 1.3 * dodge[arm]
            per = [drop_delta(drop, arm, *k) for k in keys]
            ys = np.array([p.mean() for p in per])
            ci = np.array([boot_ci(p) for p in per])
            # The summary's per-draw means must agree with the per-participant rows.
            assert np.allclose(ys, [np.mean([v['clean_delta'] for v in drop['summary']
                                             if v['arm'] == arm and v['kind'] == k[0]
                                             and v['level'] == k[1]]) for k in keys])
            c.errorbar(shown, ys, yerr=np.stack((ys - ci[:, 0], ci[:, 1] - ys)), color=color,
                       ls=ls, lw=1.2, marker=marker, ms=4,
                       markerfacecolor=color if fill else 'white', markeredgecolor=color,
                       markeredgewidth=.8, elinewidth=.7, capsize=1.6,
                       zorder=3 + (arm == 'token_v20'))
            drops += [{'arm': arm, 'kind': kind, 'electrodes': int(x), 'clean_delta': float(y),
                       'paired_ci95': [float(lo), float(hi)], 'draws_averaged': 3}
                      for x, y, (lo, hi) in zip(xs, ys, ci, strict=True)]
    assert max(v['paired_ci95'][1] for v in drops) < 8.3, 'a whisker leaves C'
    c.set(xlim=(0, 136), ylim=(-.2, 8.4))
    # Ticks where conditions were measured, not at round numbers between them.
    c.set_xticks(sorted(DROP_X.values())); c.set_yticks([0, 2, 4, 6, 8])
    c.set_xlabel('Silenced Electrodes (of 320)', labelpad=5, fontsize=9.5)
    c.set_ylabel('Error Increase (mm)', labelpad=5, fontsize=9.5)
    c.legend(handles=[
        Line2D([], [], color=style.GRAY, ls='-', marker='o', ms=3.6, lw=1.1, label='Scattered'),
        Line2D([], [], color=style.GRAY, ls=':', marker='o', ms=3.6, lw=1.1,
               markerfacecolor='white', label='Whole Arrays')],
        loc='upper left', frameon=False, handlelength=1.8, borderaxespad=.1,
        labelspacing=.3, fontsize=8.5)

    house_axis(a, 'x'); a.tick_params(axis='y', labelsize=9.5, pad=7)
    house_axis(b, 'y'); house_axis(c, 'y')
    # AFTER trim_spines, which regenerates tick labels.
    b.set_xticklabels(['1', '3.2', '10', '32'])
    fig.legend(handles=[Line2D([], [], marker=m, color=col, lw=1.3, ms=4.5,
                               markeredgecolor='white', markeredgewidth=.4, label=name)
                        for _, name, col, m in choices],
               loc='upper right', bbox_to_anchor=(.985, .935), ncol=3, frameon=False,
               columnspacing=1.4, handlelength=1.8, borderaxespad=0, fontsize=9)
    # Letters as in the Affinity-finished figures: Switzer Regular 12 pt at the top edge.
    fig.text(.004, 1.0, 'A', fontsize=12, va='top')
    for ax, letter in ((b, 'B'), (c, 'C')):
        fig.text(ax.get_position().x0 - .065, 1.0, letter, fontsize=12, va='top')

    assert any(v['significant'] for v in tests), 'nothing to annotate; check the family'
    assert not any(v['significant'] for v in drop_tests), (
        'dropout separated the two routes; the caption claims it does not')
    assert all(not ax.get_title() for ax in fig.axes) and fig._suptitle is None
    assert not sf.check_font_sizes(fig, hi=12), sf.check_font_sizes(fig, hi=12)

    # Every number the caption quotes, computed here.
    T = {s: level(r, 'token_v20', s) for s in (None, *SNRS)}
    C = {s: level(r, 'continuous_v20', s) for s in (None, *SNRS)}
    N = {s: level(r, 'native_v19', s) for s in (None, *SNRS)}
    clean_gap = T[None] - C[None]
    abs10 = T[10] - C[10]
    caption_numbers = {
        'noise_amplitude_percent': {s: amplitude(s) for s in SNRS},
        'clean_mean_mm': {'cnn': N[None].mean(), 'tokens': T[None].mean(),
                          'continuous': C[None].mean()},
        'clean_gap_token_minus_continuous': {'mean': clean_gap.mean(), 'ci95': boot_ci(clean_gap),
                                             'participants_token_worse': int((clean_gap > 0).sum())},
        'abs_gap_10db_token_minus_continuous': {'mean': abs10.mean(), 'ci95': boot_ci(abs10)},
        'percent_degradation_10db': {k: 100 * (v[10].mean() / v[None].mean() - 1)
                                     for k, v in (('tokens', T), ('continuous', C), ('cnn', N))},
        'abs_mean_10db_mm': {'tokens': T[10].mean(), 'continuous': C[10].mean(),
                             'cnn': N[10].mean()},
        'dropout_largest_gap_mm': max(abs(v['mean_gap_mm']) for v in drop_tests),
        'dropout_gap_ci_range_mm': [min(v['ci95'][0] for v in drop_tests),
                                    max(v['ci95'][1] for v in drop_tests)],
        'dropout_min_p_holm': min(v['p_holm'] for v in drop_tests),
    }
    receipt = {
        'rows': details, 'contrasts': tests, 'dropout': drops,
        'dropout_contrasts': drop_tests, 'caption_numbers': caption_numbers,
        'sources': {NOISE: data.SOURCES.get(NOISE), DROPOUT: data.SOURCES.get(DROPOUT)},
        'test': 'paired exact sign-flip permutation over all 2^9 patterns on the '
                'per-participant degradation from clean, Holm-adjusted across the '
                'eight token-versus-reference contrasts (noise) and the five '
                'token-versus-continuous contrasts (dropout)',
        'new_statistics': 'paired contrasts and the dropout intervals are new analysis on '
                          'frozen results (bootstrap 10,000 draws, seed 260930); the '
                          'noise means and intervals are the campaign\'s stored ones',
        'new_training': False}
    data.save(fig, 'noise_robustness', receipt)
    (data.OUT / 'noise_robustness.json').write_text(json.dumps(receipt, indent=1, default=float) + '\n')
    print(json.dumps({'contrasts': tests, 'dropout_contrasts': drop_tests,
                      'caption_numbers': caption_numbers}, indent=1, default=float))


if __name__ == '__main__':
    data.configure(); draw(data.read(NOISE), data.read(DROPOUT))
