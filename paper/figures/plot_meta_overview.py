"""Meta: accuracy, transmitted payload and runtime cost on one set of rows.

Replaces meta_transfer, compact_controls and streaming_cost, which reported the
same benchmark over the same arms in three separate floats. Panels A-C share one
row vocabulary and one label column. The cohort-level effect (three cohorts,
three-encoder mean) is in Table tab:meta and the text, not drawn.

Two bases coexist here on purpose, because the stored campaign offers no single
one. Runtime rows exist only for the *_260905 arms, and every arm in A is paired
against the seed-260905 learned VQ (BioCodec has its own codec seed, raw
supervised none). Only the continuous contrast exists for all three encoder
seeds: +0.230 mm for seed 260905, +0.043 mm averaged over three encoders. The
average is quoted in the caption, not drawn, so A stays on one basis. Seed 260905 is also the encoder whose
own contrast is most favourable to VQ (+0.230 against +0.032 and -0.133 mm).

Sign: in A, right of zero favours learned VQ.

No equivalence band or verdict: any margin would be chosen after the fact. The
receipt records the largest mean change each displayed 95% interval allows.

    uv run python -m paper.figures.plot_meta_overview
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from matplotlib.textpath import TextPath
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from paper.emg2pose import build_pose_paper_assets as data
from paper.common import tokenizer_diagnostic_style as style
from paper.common import switzer_font

from paper.common import figstyle as sf
from paper.common import dpose_figure_statistics as stats

# Groups say why each row exists; continuous -> token pairs share source features.
# Bits are payload bits per channel per 20 ms update, before the 9-byte header.
GROUPS = [
    ('Quantizing engineered features', [
        ('features67', 'Engineered features (67D, FP32)', style.FEATURE),
        ('kmeans1024', 'K-means (1,024 codes; 10 bits)', style.TOKEN)]),
    ('Changing quantizers, same encoder', [
        ('fsq1000', 'FSQ (1,000 codes; 10 bits)', style.TOKEN),
        ('pca4q10', 'Quantized PCA (4D; 10 bits)', style.TOKEN)]),
    ('BioCodec: codebook count', [
        ('biocodec1_260940', 'BioCodec, 1 codebook (8 bits)', style.TOKEN),
        ('biocodec6_260940', 'BioCodec, 6 codebooks (48 bits)', style.TOKEN)]),
    ('Joint training on pose labels', [
        ('raw_supervised', 'Pose-supervised features (256D, FP32)', style.FEATURE)]),
    ('Quantizing learned features', [            # last, as in the HD-EMG figure
        ('continuous_260905', 'Continuous features (256D, FP32)', style.FEATURE),
        ('vq_260905', 'Learned tokens (1,024 codes; 10 bits)', style.TOKEN)]),
]
# Sweep variants: in the max-T family and the appendix table, not drawn.
APPENDIX = [('kmeans256', 'K-means (256 codes; 8 bits)', style.TOKEN),
            ('pca4q8', 'Quantized PCA (4D; 8 bits)', style.TOKEN),
            ('pca4q16', 'Quantized PCA (4D; 16 bits)', style.TOKEN),
            ('biocodec3_260940', 'BioCodec, 3 codebooks (24 bits)', style.TOKEN)]
ARMS, HEADERS, _y = [], [], 0.0      # (results label, row label, y or None, colour); (header, y)
for _header, _rows in GROUPS:
    HEADERS.append((_header, _y)); _y += 1
    for _label, _name, _color in _rows:
        ARMS.append((_label, _name, _y, _color)); _y += 1
    _y += .35
ARMS += [(_label, _name, None, _color) for _label, _name, _color in APPENDIX]
COHORTS = [('official', 'all', 'Official Test (176)', 176),
           ('official', 'unseen_users', 'Unseen Subset (20)', 20),
           ('session', 'all', 'Held-Out Sessions (10)', 10)]
REFERENCE = 'vq_260905'
DEADLINE_MS = 20.0                 # the 20 ms token interval the pipeline must meet
WIDTH_IN, HEIGHT_IN = 7.16, 3.6
LABEL_RIGHT = 0.337                # figure-x the shared row labels are flushed to
TOP = dict(bottom=.15, height=.75)
COLUMNS = {'error': (.343, .232), 'packet': (.660, .150), 'latency': (.838, .150)}
SIG_X = .622                        # figure-x of the significance column
# Figure 3 house style.
REF_COLOR, REF_DASHES, REF_LW = 'black', (0, (4, 3)), .75   # black as in Figure 3
GRID_COLOR, GRID_WIDTH = '#DEDEDE', .55
LABEL = dict(fontsize=9.5, labelpad=5)
CUE = dict(fontsize=8, color=REF_COLOR, ha='right', va='bottom')


def reference_rule(ax, x):
    ax.axvline(x, color=REF_COLOR, ls=REF_DASHES, lw=REF_LW, zorder=1)


def draw(r, mm_per_palm):
    style.apply()
    plt.rcParams.update({'grid.color': GRID_COLOR, 'grid.linewidth': GRID_WIDTH})
    plt.rcParams.update(switzer_font.rcparams())
    # Black text, ticks and axes, as in Figure 3 (the shared style greys them).
    plt.rcParams.update({k: 'black' for k in ('text.color', 'axes.labelcolor', 'axes.edgecolor',
                                              'xtick.color', 'ytick.color', 'xtick.labelcolor', 'ytick.labelcolor')})
    runtimes = {d: {v['case']['label']: v for v in r['runtime'][d]['result']['rows']}
                for d in ('cpu', 'cuda')}
    raw_reference = r['runtime']['cuda']['result'][
        'native_raw_float32_reference_packet_bytes_per_second'] / 1000

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    top = {k: fig.add_axes((left, TOP['bottom'], width, TOP['height']))
           for k, (left, width) in COLUMNS.items()}

    arms, ylim = [], (max(a[2] for a in ARMS if a[2] is not None) + .6, -.75)
    for label, name, y, color in ARMS:
        run_cpu, run_gpu = runtimes['cpu'][label], runtimes['cuda'][label]
        row = {'label': label, 'row_label': name, 'y': y,
               'packet_kB_s': run_gpu['packet_bytes_per_second'] / 1000,
               'cpu_p50_ms': run_cpu['latency']['p50_ms'],
               'cpu_p95_ms': run_cpu['latency']['p95_ms'],
               'cpu_max_ms': run_cpu['latency']['max_ms'],
               'cpu_calls': run_cpu['latency']['count'],
               'cpu_fraction_over_20ms': run_cpu['latency']['fraction_over_20ms'],
               'v100_p50_ms': run_gpu['latency']['p50_ms'],
               'v100_p95_ms': run_gpu['latency']['p95_ms'],
               'v100_max_ms': run_gpu['latency']['max_ms'],
               'v100_fraction_over_20ms': run_gpu['latency']['fraction_over_20ms'],
               'encoder_seed': run_gpu['case'].get('encoder_seed'),
               'codebook_kB': run_gpu['lookup_codebook_bytes'] / 1000,
               'tensors_MB': run_gpu['sum_component_tensor_bytes'] / 1e6,
               'source_unit': 'palm lengths', 'display_unit': 'mm; reference hand'}

        # A: every arm minus the learned-VQ reference, so right of zero is worse
        # than VQ. The stored pair is vq - arm, hence the display negation.
        draw_row = y is not None
        if label == REFERENCE:
            top['error'].scatter(0, y, s=26, color=color, zorder=4)
            row.update(delta_from_vq_mm=0.0, ci95_mm=None, comparison=None,
                       is_reference=True)
        else:
            p = data.pair(r, 'official', 'all', f'{REFERENCE}_vs_{label}')
            mean = -p['mean_difference'] * mm_per_palm
            ci = -np.array(p['difference_ci95'])[::-1] * mm_per_palm
            if draw_row:
                top['error'].plot(ci, [y, y], color=color, lw=1.25)
                top['error'].scatter(mean, y, s=22, color=color, zorder=3)
            row.update(delta_from_vq=-p['mean_difference'],
                       ci95=(-np.array(p['difference_ci95'])[::-1]).tolist(),
                       delta_from_vq_mm=mean, ci95_mm=ci.tolist(),
                       comparison=f'{REFERENCE}_vs_{label}', is_reference=False)

        # Medians only: the p50-p95 spread is under 1.5x for every arm and is
        # invisible on a log axis; the deadline claim rests on max_ms (receipt).
        if draw_row:
            top['latency'].scatter(row['v100_p50_ms'], y, marker='s', s=20, color=color,
                                   edgecolor='white', linewidth=.3, zorder=3)
            top['latency'].scatter(row['cpu_p50_ms'], y, marker='o', s=22, color=color,
                                   edgecolor='white', linewidth=.3, zorder=3)
        arms.append(row)

    labels = blended_transform_factory(fig.transFigure, top['error'].transData)
    for _, name, y, _ in ARMS:
        if y is not None:
            fig.text(LABEL_RIGHT, y, name, transform=labels, ha='right', va='center', fontsize=9)
    for header, y in HEADERS:
        fig.text(.015, y, header, transform=labels, ha='left', va='center',
                 fontsize=9, weight='bold')
    fig.text(.015, -1.0, 'Representation', transform=labels, ha='left', va='center', fontsize=9.5)

    error, packet, latency = top['error'], top['packet'], top['latency']
    error.set_xlim(-.42, 1.55)
    error.set_xlabel('Error Difference (mm)', **LABEL)
    error.set_xticks([-.4, 0, .4, .8, 1.2], ['−0.4', '0', '0.4', '0.8', '1.2'])
    reference_rule(error, 0)
    error.text(1, 1.01, r'Favours Learned Tokens $\rightarrow$', transform=error.transAxes, **CUE)

    # B: packet rate per drawn row on a log axis, with the raw float32 EMG stream.
    for a in arms:
        if a['y'] is not None:
            color = next(c for l, _, _, c in ARMS if l == a['label'])
            packet.scatter(a['packet_kB_s'], a['y'], s=22, color=color, zorder=3)
    packet.set(xscale='log', xlim=(.8, 1600))
    packet.set_xlabel('Packet Rate (kB/s)', **LABEL)
    packet.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1,000'])
    reference_rule(packet, raw_reference)
    packet.text(raw_reference * .88, .015, 'Raw float32 EMG', rotation=90, fontsize=8.5,
                color=REF_COLOR, ha='right', va='bottom', transform=packet.get_xaxis_transform())
    latency.set(xscale='log', xlim=(2, 260))
    latency.set_xlabel('Pipeline Latency (ms)', **LABEL)
    latency.set_xticks([3, 10, 30, 100], ['3', '10', '30', '100'])
    reference_rule(latency, DEADLINE_MS)
    latency.text(DEADLINE_MS * 1.08, .015, '20 ms', rotation=90, fontsize=8.5,
                 color=REF_COLOR, ha='left', va='bottom',
                 transform=latency.get_xaxis_transform())
    # Device key above C (the reference row is last, so no row carries it).
    latency.legend(handles=[Line2D([], [], marker='s', ls='', ms=4.5, color='black', label='V100 GPU'),
                            Line2D([], [], marker='o', ls='', ms=4.5, color='black', label='CPU')],
                   loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8.5,
                   handletextpad=.2, columnspacing=.8, borderaxespad=.1, borderpad=0, labelcolor='black')
    for ax in top.values():
        ax.set_ylim(*ylim)
        ax.set_yticks([])

    # D/E: the cohort effect, on the three-encoder mean rather than one seed.
    # Absolute official-test error of every single fit, so the caption can show
    # that seed 260905 pairs the best VQ with the worst continuous fit.
    single_fit_mm = {f'{k}_{x}': float(data.vector(r, [f'{k}_{x}'], 'official', 'all')[1].mean()
                                       * mm_per_palm)
                     for k in ('vq', 'continuous') for x in data.SEEDS}
    # Refitting the same encoder under another seed, in display sign (other - 260905).
    refits = {x: -data.pair(r, 'official', 'all', f'{REFERENCE}_vs_vq_{x}')
              ['mean_difference'] * mm_per_palm for x in data.SEEDS[1:]}
    details, participants, bounds = [], [], []
    for row_index, (scope, subgroup, name, n) in enumerate(COHORTS):
        people, cont = data.vector(r, [f'continuous_{s}' for s in data.SEEDS], scope, subgroup)
        other, token = data.vector(r, [f'vq_{s}' for s in data.SEEDS], scope, subgroup)
        assert people == other and len(people) == n
        delta = (cont - token) * mm_per_palm      # positive favours learned VQ
        participants.append({'scope': scope, 'subgroup': subgroup, 'users': people,
                             'continuous_minus_token_mm': delta.tolist()})
        for seed in [*data.SEEDS, None]:
            key = f'vq_{seed}_vs_continuous_{seed}' if seed else 'three_encoder_mean_vq_vs_continuous'
            p = data.pair(r, scope, subgroup, key)
            mean = -p['mean_difference'] * mm_per_palm
            ci = -np.array(p['difference_ci95'])[::-1] * mm_per_palm
            if seed is None:
                bounds.append({'scope': scope, 'subgroup': subgroup, 'ci95_mm': ci.tolist(),
                               'largest_change_allowed_mm': float(np.abs(ci).max())})
            details.append({'scope': scope, 'subgroup': subgroup, 'comparison': key,
                            'continuous_minus_token': -p['mean_difference'],
                            'ci95': (-np.array(p['difference_ci95'])[::-1]).tolist(),
                            'continuous_minus_token_mm': mean, 'ci95_mm': ci.tolist(),
                            'source_unit': 'palm lengths', 'display_unit': 'mm; reference hand',
                            'interval_displayed': seed is None})

    for ax in top.values():
        style.finish(ax, categorical=True)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='x', labelsize=9, length=3)
        ax.tick_params(axis='y', labelsize=9.5, length=0, pad=7)

    # Significance, as in the HD-EMG figure: studentized paired sign-flip tests
    # of every arm against learned tokens, one max-T family over the 12
    # contrasts with synchronized participant signs. 176 participants rule out
    # enumeration, so 9,999 random sign patterns plus the observed one are used.
    contrasts = [a for a in arms if not a['is_reference']]
    per_user = [data.pair(r, 'official', 'all', a['comparison'])['per_user_difference'] for a in contrasts]
    people = sorted(per_user[0])
    assert len(people) == 176 and all(sorted(d) == people for d in per_user), 'contrasts must share participants'
    D = np.array([[d[u] for d in per_user] for u in people])           # participants x contrasts
    rng = np.random.default_rng(20261008)
    signs = np.vstack([np.ones(len(D)), rng.choice([-1., 1.], (9999, len(D)))])
    observed = stats.studentized_mean(D)
    null_t = np.abs(np.stack([stats.studentized_mean(sg[:, None] * D) for sg in signs]))
    raw_p = stats.inclusive_tail(null_t, observed)
    adj_p = stats.inclusive_tail(null_t.max(1)[:, None], observed)
    symbol = lambda q: '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else 'n.s.'
    # Bracket as in the HD-EMG figure: one stem, a cap at the reference row and a
    # tick per drawn contrast, with its max-T mark beside the tick.
    col = blended_transform_factory(fig.transFigure, error.transData)
    x_stem, cap = SIG_X - .03, .007
    drawn = [a['y'] for a in arms if a['y'] is not None]
    y_ref = next(a['y'] for a in arms if a['is_reference'])
    error.add_line(Line2D([x_stem, x_stem], [min(drawn), max(drawn)], transform=col, clip_on=False,
                          color='black', lw=.7))
    error.add_line(Line2D([x_stem - cap, x_stem], [y_ref, y_ref], transform=col, clip_on=False,
                          color='black', lw=.7))
    for a, t, q, q_raw in zip(contrasts, observed, adj_p, raw_p):
        mark = symbol(q)
        if a['y'] is not None:
            error.add_line(Line2D([x_stem, x_stem + cap], [a['y'], a['y']], transform=col,
                                  clip_on=False, color='black', lw=.7))
            label = error.text(x_stem + cap + .003, a['y'], mark, transform=col, ha='left', va='center',
                               clip_on=False, fontsize=11 if mark != 'n.s.' else 8, color='black')
            # Centre the ink, not the font box, on the tick: asterisks sit high and
            # 'n.s.' low in their boxes.
            ink = TextPath((0, 0), mark, prop=label.get_fontproperties()).get_extents()
            label.set_verticalalignment('baseline')
            label.set_transform(col + ScaledTranslation(0, -(ink.y0 + ink.y1) / 144, fig.dpi_scale_trans))
        a.update(t_statistic=float(t), p_raw=float(q_raw), p_maxT=float(q), significance=mark)

    # Panel letters at the figure edge (A, D) or each column's left edge.
    # Same letters as the Affinity-finished figures: Switzer Regular 12 pt at the top edge.
    for x, y, letter in ((.004, 1.0, 'A'), (COLUMNS['packet'][0] - .03, 1.0, 'B'), (COLUMNS['latency'][0] - .03, 1.0, 'C')):
        fig.text(x, y, letter, fontsize=12, va='top')
    # Colour key with the HD-EMG figure's meaning: red discrete codes, blue continuous values.
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=5, color=c, label=n)
                        for c, n in ((style.TOKEN, 'Tokens'), (style.FEATURE, 'Continuous Features'))],
               loc='lower left', ncol=1, frameon=False, fontsize=9, handletextpad=.3, borderaxespad=0,
               labelspacing=.3, bbox_to_anchor=(.012, .012))   # empty corner under the row labels

    assert all(not ax.get_title() for ax in fig.axes) and fig._suptitle is None
    assert not sf.check_font_sizes(fig, hi=12), sf.check_font_sizes(fig, hi=12)  # 12 pt panel letters match the Affinity figures
    assert not sf.check_label_case(fig), sf.check_label_case(fig)
    for a in arms:
        lo, hi = (a['ci95_mm'] or [0, 0])
        assert -.42 < lo and hi < 1.55, a['label']
        assert .8 < a['packet_kB_s'] < 1600 and 2 < a['v100_p50_ms'] and a['cpu_p50_ms'] < 260
    gate(arms, details, mm_per_palm)
    receipt = {
        'significance': {'test': 'two-sided studentized paired sign-flip test of each arm vs learned tokens '
                                  '(seed 260905), official test, 176 participants',
                          'family': 'max-T over the 12 panel-A contrasts, synchronized participant signs',
                          'sign_patterns': '9,999 random (seed 20261008) plus the observed pattern',
                          'symbols': '* p<0.05, ** p<0.01, *** p<0.001 on the max-T adjusted p'},
        'arms': arms, 'paired_estimates': details, 'participants': participants,
        'reference_arm': REFERENCE, 'raw_float32_payload_kB_s': raw_reference,
        'cohort_bounds': {'reading': 'no margin or verdict; largest_change_allowed_mm is '
                                     'the largest |mean change| inside the displayed 95% '
                                     'interval of the three-encoder mean',
                          'cohorts': bounds},
        'single_fit_official_error_mm': single_fit_mm,
        'vq_refit_minus_reference_mm': refits,
        'deadline_ms': DEADLINE_MS,
        'slowest_non_biocodec_cpu_ms': max(a['cpu_max_ms'] for a in arms
                                           if not a['label'].startswith('biocodec')),
        'seed_basis': {'A-C': 'every arm paired against the seed-260905 learned VQ; '
                              'BioCodec uses codec seed 260940, raw supervised has no '
                              'encoder seed; runtime rows exist for no other VQ seed',
                       'caption only': 'continuous contrast averaged over encoder seeds 260905, 260906, 260907'},
        'sign_convention': 'A displayed as alternative minus learned VQ; the stored '
                           'pairs are vq minus alternative, negated for display only',
        'intervals': 'pointwise 10,000-draw participant bootstrap; no multiplicity correction',
        'latency_scope': 'median of a fixed 100-call software replay; not a deadline guarantee',
        'unseen_is_subset_of_official': True, 'all_participants_displayed': False,
        'new_training': False,
        'replaces': ['meta_transfer', 'compact_controls', 'streaming_cost']}
    data.save(fig, 'meta_overview', receipt)
    (data.OUT / 'meta_overview.json').write_text(json.dumps(receipt, indent=1, default=float))
    write_table(arms)


def write_table(arms):
    """Appendix companion: every emg2pose contrast, including the sweep variants not drawn."""
    names = {l: n for l, n, _, _ in ARMS}
    lines = [r'\begin{tabular}{lrrrrr}\toprule',
             r'Representation & Packet rate (kB/s) & $\Delta$ (mm) & 95\,\% interval (mm) & $p_{\max T}$ & CPU / V100 (ms)\\\midrule']
    for a in sorted(arms, key=lambda a: (a['is_reference'], a['label'] != 'continuous_260905', a['delta_from_vq_mm'])):
        n = names[a['label']] + ('' if a['y'] is not None else r'$^{\dagger}$')
        rate = f"{a['packet_kB_s']:.2f}" if a['packet_kB_s'] < 10 else f"{a['packet_kB_s']:.1f}"
        lat = f"{a['cpu_p50_ms']:.1f} / {a['v100_p50_ms']:.1f}"
        if a['is_reference']:
            lines.append(f'{n} & {rate} & ref. & -- & -- & {lat}\\\\')
            continue
        lo, hi = a['ci95_mm']
        pv = '<0.001' if a['p_maxT'] < .001 else f"{a['p_maxT']:.3f}"
        m = lambda v: f'{v:+.3f}'.replace('-', '$-$')       # typographic minus
        lines.append(f"{n} & {rate} & {m(a['delta_from_vq_mm'])} & [{m(lo)}, {m(hi)}] & ${pv}$ & {lat}\\\\")
    lines.append(r'\bottomrule\end{tabular}')
    (data.OUT / 'emg2pose_contrasts.tex').write_text('\n'.join(lines) + '\n')


def gate(arms, details, mm_per_palm):
    """The merge must reproduce what the three replaced figures showed."""
    by_label = {a['label']: a for a in arms}
    single = by_label['continuous_260905']['delta_from_vq_mm']
    pooled = next(d['continuous_minus_token_mm'] for d in details
                  if d['scope'] == 'official' and d['subgroup'] == 'all'
                  and d['comparison'] == 'three_encoder_mean_vq_vs_continuous')
    assert abs(single - 0.2304) < 1e-3, single      # compact_controls, one seed
    assert abs(pooled - 0.0432) < 1e-3, pooled      # meta_transfer, three seeds
    assert all(by_label[f'biocodec{n}_260940']['cpu_fraction_over_20ms'] == 1.0
               for n in (1, 3, 6)), 'BioCodec should miss every CPU deadline'
    assert all(a['cpu_fraction_over_20ms'] == 0.0 for a in arms
               if not a['label'].startswith('biocodec')), 'only BioCodec should miss'
    assert all(a['v100_fraction_over_20ms'] == 0.0 for a in arms), 'V100 meets every deadline'
    print(f'gate ok: continuous is {single:+.3f} mm (seed 260905) and {pooled:+.3f} mm (three encoders)')


if __name__ == '__main__':
    data.configure()
    draw(data.read('results/paper_completion_v1/results.json'),
         data.read('data/reference/meta_millimetre_conversion.json')['mm_per_palm_length'])
