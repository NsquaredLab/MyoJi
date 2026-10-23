"""Fig. 3C of the TNSRE version: signed effect of quantization for each decoder.

Token error minus continuous-feature error, per participant (circles) and as the mean with
its exact 95 % sign-flip interval, for the six paired decoders grouped by tokenizer, each
group followed by its pooled estimate (claims C1 and C2). The grey dashed rule is the 1 mm
non-inferiority margin; an interval entirely left of it is non-inferior. Values are read
from results/hdemg_claim_statistics.json, so nothing is recomputed. Drawn at the exact size
of the Affinity panel it replaces (1,079 x 467 px at 300 dpi).

    uv run python -m paper.figures.plot_fig3_signed_panel
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper.common import paths
from paper.common import switzer_font

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'results/hdemg_claim_statistics.json'
# (row label, contrast id or claim id); groups are separated by a gap.
GROUPS = [[('TCN', 'matched_tcn_tokens_vs_features'), ('Transformer', 'matched_transformer_tokens_vs_features'),
           ('Qwen3', 'pretrained_qwen_tokens_vs_features'), ('Proposed (Learned)', 'proposed_tokens_vs_features'),
           ('Pooled (Learned)', 'C2')],
          [('TabPFN 3.5', 'tabpfn_extension_08'), ('Proposed (SPAR)', 'spar_extension_06'),
           ('Pooled (SPAR)', 'C1')]]
WIDTH_PX, HEIGHT_PX, DPI = 1079, 467, 300
GAP = .55


def rows():
    record = json.loads(RECORD.read_text())
    found = {r['contrast_id']: r for r in record['receivers']}
    found.update({c['id']: c for c in record['claims']['margin_1mm']})
    out, y = [], 0.
    for group in GROUPS:
        for label, key in group:
            r = found[key]
            d = np.array(list(r['participant_differences_mm'].values()))
            assert len(d) == 9 and np.isclose(d.mean(), r['effect_mm'])
            out.append(dict(label=label, y=y, d=d, mean=r['effect_mm'], ci=r['ci95_mm'], pooled=key.startswith('C')))
            y += 1
        y += GAP
    return out, record['margin_mm']


def main():
    data, margin = rows()
    plt.rcParams.update(switzer_font.rcparams())
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
                         'axes.linewidth': .75, 'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'})
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI))
    ax = fig.add_axes((.37, .27, .53, .62))
    ax.axvline(0, color='black', ls=(0, (4, 3)), lw=.75, zorder=1)
    ax.axvline(margin, color='#8a8a8a', ls=(0, (4, 3)), lw=.75, zorder=1)
    rng = np.random.default_rng(0)                       # display-only vertical jitter
    for r in data:
        ax.scatter(r['d'], r['y'] + rng.uniform(-.13, .13, 9), s=6, facecolor='white', edgecolor='#6d6f72',
                   linewidth=.5, zorder=2)
        ax.plot(r['ci'], [r['y']] * 2, color='black', lw=1.6 if r['pooled'] else 1.1, zorder=3)
        ax.scatter(r['mean'], r['y'], s=30 if r['pooled'] else 22, marker='D', color='black', zorder=4)
    ax.set_yticks([r['y'] for r in data], [r['label'] for r in data], fontsize=9)
    for tick, r in zip(ax.get_yticklabels(), data):
        tick.set_fontweight('bold' if r['pooled'] else 'normal')
    ax.tick_params(axis='y', length=0, pad=6)
    ax.set_ylim(data[-1]['y'] + .6, -.6)
    lim = max(abs(np.concatenate([r['d'] for r in data]))) * 1.08
    ax.set_xlim(-lim, lim)
    ax.text(margin + .06, data[-1]['y'] + .35, f'{margin:g} mm Bound', color='#6d6f72', fontsize=8,
            ha='left', va='bottom')
    ax.tick_params(axis='x', labelsize=9, length=3)
    ax.set_xlabel('Error Difference, Tokens − Features (mm)', fontsize=9.5, labelpad=2)
    ax.text(-.08, 1.0, r'$\leftarrow$ Tokens better', transform=ax.get_xaxis_transform(), ha='right', va='bottom', fontsize=8)
    ax.text(.08, 1.0, r'Features better $\rightarrow$', transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=8)
    ax.grid(axis='x', color='#DEDEDE', lw=.55); ax.set_axisbelow(True)
    out = paths.OUTPUT / 'figures/pose_paper/fig3_signed_panel'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=DPI * 2, transparent=True)
    fig.savefig(out.with_suffix('.pdf'), transparent=True)
    print(json.dumps([{k: v for k, v in r.items() if k != 'd'} for r in data], indent=1))


if __name__ == '__main__':
    main()
