"""Supplementary panels in the main-text style: electrode coverage versus payload, quantization
transfer and emg2pose cohorts, and the emg2pose comparison of quantizers (three panels whose
rows line up across the figure).

House encoding, as in Figs. 3-4 and the noise supplement: Switzer; tokens #d80032 circles,
continuous features #5c618e squares, raw EMG #6d6f72 diamonds; differences as black diamonds
with black interval lines and a dashed zero line (Fig. 3C). Values come from the Supplementary
Data each caption cites (data/supplementary SD1, SD4, SD5). Every PDF keeps the page size of the
panel it replaced, so the supplement layout does not move.

    uv run python -m paper.figures.plot_supp_figures
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
from paper.common import paths  # noqa: E402
from paper.common import switzer_font  # noqa: E402

DATA = paths.SUPPLEMENTARY
OUT = paths.OUTPUT / 'figures/pose_paper'
TOKEN, FEATURE, RAW, RULE, GRID = '#d80032', '#5c618e', '#6d6f72', '#8a8a8a', '#DEDEDE'
STYLE = {'tokens': (TOKEN, 'o'), 'continuous': (FEATURE, 's'), 'raw': (RAW, 'D')}
# emg2pose comparison rows, top to bottom, identical in all three panels.
ALTERNATIVES = [('reference', 'Learned tokens (10 bits)', 'tokens'),
                ('EP01', 'Continuous features', 'continuous'),
                ('EP03', 'FSQ (10 bits)', 'tokens'),
                ('EP05', 'Quantized PCA (10 bits)', 'tokens'),
                ('EP06', 'Engineered k-means (10 bits)', 'tokens'),
                ('EP04', 'BioCodec (48 bits)', 'tokens')]
ROWS = (.22, .64)                                  # shared axes bottom and height: rows line up
RAW_FP32_KB_S = (16 * 2000 * 4 + 9 * 50) / 1000    # 16 channels at 2 kHz, FP32, 9-byte headers at 50 Hz
LINK_KB_S = 175                                    # assumed application throughput (caption)


def table(name, key):
    with (DATA / name).open() as f:
        return {r[key]: r for r in csv.DictReader(f)}


def num(s):
    return float(s.replace('+', ''))


def panel(w_pt, h_pt, rect):
    fig = plt.figure(figsize=(w_pt / 72, h_pt / 72))
    return fig, fig.add_axes(rect)


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def rule(ax, x, label, y, side):
    """Dashed reference line labelled along the line, as in Fig. 3."""
    ax.axvline(x, color=RULE, ls=(0, (4, 3)), lw=.75, zorder=1)
    ax.annotate(label, (x, y), xytext=(-6 if side == 'left' else 6, 0), textcoords='offset points',
                rotation=90, ha='center', va='center', fontsize=7.5, color=RULE)


def rows_only(ax):
    """Value panels beside a labelled panel: no y axis, vertical grid."""
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_ylim(len(ALTERNATIVES) - .5, -.5)
    ax.grid(axis='x', color=GRID, lw=.55); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def differences(ax, est, labels, xlim, xlabel):
    """Fig. 3C style: black diamond and interval per row, dashed zero line."""
    ax.axvline(0, color='black', ls=(0, (4, 3)), lw=.75, zorder=1)
    for y, (m, lo, hi) in enumerate(est):
        if lo is not None:
            ax.plot([lo, hi], [y, y], color='black', lw=1.1, zorder=3)
        ax.scatter(m, y, s=22, marker='D', color='black', zorder=4)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=9.5, labelpad=2)
    ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', labelsize=8)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=.55); ax.set_axisbelow(True)


def transfer_and_cohorts():
    sd4 = table('SD4_transfer_and_cohorts.csv', 'comparison')
    est = lambda k: tuple(num(sd4[k][c]) for c in ('token_minus_continuous_mm', 'ci95_lower_mm', 'ci95_upper_mm'))
    cohorts = ['Official test', 'Unseen participants', 'Held-out sessions']
    for name, w, left, keys, labels in (
            ('supp_transfer', 248.4, .2, ['SPAR: separate participants', 'Learned: separate participants'],
             ['SPAR', 'Learned']),
            ('supp_cohorts', 262.8, .46, cohorts, [f"{k} ({sd4[k]['participants']})" for k in cohorts])):
        fig, ax = panel(w, 126, (left, .3, .95 - left, .66))
        differences(ax, [est(k) for k in keys], labels, (-.7, .7), 'Tokens − Continuous (mm)')
        ax.set_xticks([-.5, 0, .5])
        save(fig, name)


def mark(p):
    p = 0. if p.startswith('<') else float(p)
    return '**' if p < .01 else '*' if p < .05 else 'n.s.'


def alternatives():
    sd5 = table('SD5_emg2pose_12_comparisons.csv', 'comparison_id')
    ids, labels, kinds = zip(*ALTERNATIVES)
    marks = ['ref.'] + [mark(sd5[i]['p_maxT_12']) for i in ids[1:]]
    assert marks == ['ref.', '*', 'n.s.', '**', '**', 'n.s.'], marks   # as stated in the caption

    fig, ax = panel(270, 208.8, (.52, ROWS[0], .33, ROWS[1]))
    est = [(0., None, None)] + [tuple(num(sd5[i][c]) for c in ('alternative_minus_learned_mm', 'ci95_lower_mm',
                                                                'ci95_upper_mm')) for i in ids[1:]]
    differences(ax, est, labels, (-.15, .75), 'Error Difference (mm)')
    ax.set_xticks([0, .3, .6])
    for y, m in enumerate(marks):
        ax.text(1.05, y, m, transform=ax.get_yaxis_transform(), va='center', fontsize=8.5)
    save(fig, 'supp_alternatives_error')

    fig, ax = panel(97.92, 208.8, (.12, ROWS[0], .8, ROWS[1]))
    for y, (i, kind) in enumerate(zip(ids, kinds)):
        color, marker = STYLE[kind]
        ax.scatter(num(sd5[i]['packet_rate_kB_s']), y, s=22, marker=marker, color=color, zorder=3)
    ax.set_xscale('log'); ax.set_xlim(.7, 2500)
    ax.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1k']); ax.minorticks_off()
    rule(ax, RAW_FP32_KB_S, 'Raw EMG', 4, 'left')
    rows_only(ax)
    ax.set_xlabel('Packet Rate\n(kB/s)', fontsize=9.5, labelpad=2)
    save(fig, 'supp_alternatives_rate')

    fig, ax = panel(140.4, 208.8, (.08, ROWS[0], .86, ROWS[1]))
    for y, (i, kind) in enumerate(zip(ids, kinds)):
        color, marker = STYLE[kind]
        ax.scatter(num(sd5[i]['cpu_latency_median_ms']), y, s=22, marker=marker, color=color, zorder=3)
        ax.scatter(num(sd5[i]['v100_latency_median_ms']), y, s=22, marker=marker, facecolor='white',
                   edgecolor=color, linewidth=1, zorder=3)
    ax.set_xscale('log'); ax.set_xlim(2.2, 160)
    ax.set_xticks([3, 10, 30, 100], ['3', '10', '30', '100']); ax.minorticks_off()
    rule(ax, 20, '20 ms', 2.5, 'right')
    rows_only(ax)
    ax.set_xlabel('Replay Latency\n(ms)', fontsize=9.5, labelpad=2)
    fig.legend(handles=[Line2D([], [], ls='', marker='o', ms=5, color=RAW, label='CPU'),
                        Line2D([], [], ls='', marker='o', ms=5, mfc='white', mec=RAW, label='V100')],
               loc='upper center', bbox_to_anchor=(.51, 1.0), ncol=2, frameon=False, fontsize=8.5,
               handletextpad=.2, columnspacing=1.)
    save(fig, 'supp_alternatives_latency')


def operating_points():
    with (DATA / 'SD1_operating_points.csv').open() as f:
        sd1 = {(r['channels'], r['representation']): r for r in csv.DictReader(f)}
    # (legend, kind, [(electrodes, SD1 representation, payload column, label offset in points)])
    series = [('SPAR Tokens', 'tokens', [('320', 'SPAR (1024 codes)', 'packed_token_kB_s', (5, 4)),
                                         ('63', 'SPAR tokens', 'packed_token_kB_s', (5, 4))]),
              ('Continuous SPAR', 'continuous', [('320', 'Continuous SPAR', 'stored_array_kB_s', (-5, 4)),
                                                 ('63', 'Continuous SPAR', 'stored_array_kB_s', (5, 4))]),
              ('Raw EMG + CNN', 'raw', [('42', 'Raw int16 / adapted CNN', 'stored_array_kB_s', (-5, 4))])]
    fig, ax = panel(250.56, 223.2, (.17, .19, .79, .78))
    for legend, kind, points in series:
        color, marker = STYLE[kind]
        for channels, rep, column, (dx, dy) in points:
            r = sd1[(channels, rep)]
            x, y = float(r[column]), float(r['mean_error_mm'])
            ax.scatter(x, y, s=28, marker=marker, color=color, zorder=3)
            ax.annotate(channels, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=8.5,
                        ha='left' if dx > 0 else 'right', va='bottom')
    ax.set_xscale('log'); ax.set_xlim(2.5, 2500)
    ax.set_xticks([10, 100, 1000], ['10', '100', '1,000']); ax.minorticks_off()
    ax.set_ylim(10.4, 15.7); ax.set_yticks([11, 13, 15])
    rule(ax, LINK_KB_S, 'Link Budget', 13.3, 'right')
    ax.set_xlabel('Payload (kB/s)', fontsize=9.5, labelpad=2)
    ax.set_ylabel('3D Error (mm)', fontsize=9.5, labelpad=4)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', color=GRID, lw=.55); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], ls='', marker=STYLE[k][1], color=STYLE[k][0], ms=5, label=n)
                       for n, k, _ in series], loc='upper left', frameon=False, fontsize=8.5,
              handletextpad=.3, borderaxespad=.2)
    save(fig, 'supp_operating_points')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(switzer_font.rcparams())
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                         'axes.linewidth': .75, 'xtick.major.size': 3, 'ytick.major.size': 3})
    transfer_and_cohorts()
    alternatives()
    operating_points()


if __name__ == '__main__':
    main()
