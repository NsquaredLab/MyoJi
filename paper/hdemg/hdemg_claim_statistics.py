"""Claim-aligned HD-EMG statistics for the TNSRE revision.

Four headline claims form one Holm family; every other contrast is an estimate with
its exact 95 % sign-flip interval. Non-inferiority (margin 1 mm, sensitivity 0.5 mm)
is a one-sided exact sign-flip test of H0: mean >= margin, entered into Holm as twice
its one-sided p; it holds exactly when the upper limit of the two-sided 95 % interval
lies below the margin. Inputs are the frozen 76-contrast record and the figure receipt;
the old max-T adjusted p-values stay as a sensitivity column. Nothing is refitted.

    uv run python -m paper.hdemg.hdemg_claim_statistics
"""
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from paper.common import paths
from paper.common.hdemg_figure_statistics import exact_location_interval, exact_paired_test, holm_adjust

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'results/hdemg_tabpfn_figure_analysis.json'
FIGURE = ROOT / 'output/figures/hdemg-error-and-resources.json'
LEGACY_COMPARISONS = paths.REFERENCE / 'hd_emg_76_comparisons.csv'
MOVEMENTS = paths.OUTPUT / 'figures/pose_paper/decoding_examples.json'   # Fig. 4B receipt
OUT_JSON = ROOT / 'results/hdemg_claim_statistics.json'
OUT_COMPARISONS = paths.SUPPLEMENTARY / 'SD2_hd_emg_comparisons.csv'
ALPHA, MARGIN, SENSITIVITY_MARGIN = .05, 1.0, .5
PARTICIPANTS = [str(i) for i in range(1, 10)]
LEARNED = ['proposed_tokens_vs_features', 'matched_tcn_tokens_vs_features',
           'matched_transformer_tokens_vs_features', 'pretrained_qwen_tokens_vs_features']
SPAR = ['spar_extension_06', 'tabpfn_extension_08']
SPAR_VS_LEARNED = 'spar_extension_07'
SPAR_TOKENS, ADAPTED_CNN = ('hdemg_spar_simple_full_v1', 'tokens'), ('hdemg_raulnet_v1', 'raulnet_v19_q32')
NAMES = {('hdemg_v20_v1', 'tokens'): 'Proposed / learned tokens',
         ('hdemg_v20_continuous_v1', 'continuous'): 'Proposed / learned features',
         ('hdemg_matched_v1', 'tcn_tokens'): 'TCN / tokens (matched)',
         ('hdemg_matched_v1', 'tcn_continuous'): 'TCN / features (matched)',
         ('hdemg_matched_v1', 'transformer_tokens'): 'Transformer / tokens (matched)',
         ('hdemg_matched_v1', 'transformer_continuous'): 'Transformer / features (matched)',
         ('hdemg_llm_full_v1', 'tokens_pretrained'): 'Qwen / pretrained tokens',
         ('hdemg_llm_full_v1', 'continuous_pretrained'): 'Qwen / pretrained features'}
# Pipeline contrasts the text describes but the 76-family never contained (alternative minus reference).
PIPELINES = [(('hdemg_v20_v1', 'tokens'), ('hdemg_matched_v1', 'tcn_tokens')),
             (('hdemg_v20_continuous_v1', 'continuous'), ('hdemg_matched_v1', 'tcn_continuous')),
             (('hdemg_v20_v1', 'tokens'), ('hdemg_matched_v1', 'transformer_tokens')),
             (('hdemg_v20_continuous_v1', 'continuous'), ('hdemg_matched_v1', 'transformer_continuous')),
             (('hdemg_llm_full_v1', 'tokens_pretrained'), ('hdemg_v20_v1', 'tokens')),
             (('hdemg_llm_full_v1', 'continuous_pretrained'), ('hdemg_v20_continuous_v1', 'continuous'))]
METHOD = (
    'Participants are the unit of analysis (n = 9); decoder seeds, windows and tasks are averaged within '
    'each participant first. Four claims form one Holm family at familywise alpha 0.05: C1 and C2 are '
    'non-inferiority tests of the token-minus-feature difference pooled over receivers (equal weight per '
    'receiver, averaged within participant), C3 is a two-sided test of SPAR versus learned tokens with the '
    'proposed decoder, and C4 is a non-inferiority test of SPAR tokens against the adapted raw-EMG CNN. '
    'Non-inferiority is a one-sided exact sign-flip test (512 patterns) of H0: mean difference >= margin, '
    'entered into Holm as twice its one-sided p; it holds exactly when the upper limit of the exact two-sided '
    '95 % interval lies below the margin. The margin (1 mm; 0.5 mm as sensitivity) was fixed during revision '
    'after the results had been inspected, so the analysis is exploratory. The smallest attainable '
    'Holm-adjusted p is 4 x 2/512 = 0.0156. All other contrasts are estimates with exact 95 % intervals that '
    'invert the two-sided sign-flip test (assuming differences symmetric about their mean); no other '
    'contrast is tested. The single-step max-T adjustment over the earlier 76-contrast family is retained '
    'only as a sensitivity column.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_one_sided_p(x, delta):
    """Exact sign-flip p of H0: mean >= delta against mean < delta (non-inferiority)."""
    y = np.asarray(x, dtype=np.float64) - delta
    null = (np.asarray(list(product((-1., 1.), repeat=len(y)))) * y).mean(axis=1)
    return float(np.mean(null <= y.mean() + 100 * np.finfo(np.float64).eps * max(1., abs(y.mean()))))


def vector(row):
    return np.array([row['participant_differences'][p] for p in PARTICIPANTS], dtype=np.float64)


def summary(x):
    lower, upper = exact_location_interval(x)
    return {'effect_mm': float(np.mean(x)), 'ci95_mm': [lower, upper],
            'participants_alternative_higher': int(np.sum(x > 0)),
            'p_vs_zero': exact_paired_test(x, np.zeros_like(x))[1],
            'participant_differences_mm': dict(zip(PARTICIPANTS, map(float, x)))}


def claims(rows, margin):
    by_id = {r.get('contrast_id'): r for r in rows}
    c4 = [r for r in rows if tuple(r['alternative']) == SPAR_TOKENS and tuple(r['reference']) == ADAPTED_CNN]
    assert len(c4) == 1
    spec = [('C1', 'SPAR tokens non-inferior to SPAR features (pooled over the proposed decoder and TabPFN 3.5)',
             'noninferiority', np.mean([vector(by_id[i]) for i in SPAR], axis=0)),
            ('C2', 'Learned tokens non-inferior to learned features (pooled over the proposed decoder, TCN, '
                   'Transformer and Qwen3)', 'noninferiority', np.mean([vector(by_id[i]) for i in LEARNED], axis=0)),
            ('C3', 'SPAR tokens differ from learned tokens with the proposed decoder', 'two-sided',
             vector(by_id[SPAR_VS_LEARNED])),
            ('C4', 'SPAR tokens (proposed decoder) non-inferior to the adapted raw-EMG CNN', 'noninferiority',
             vector(c4[0]))]
    out = []
    for cid, claim, test, x in spec:
        s = summary(x)
        noninferiority = test == 'noninferiority'
        p = min(1., 2 * exact_one_sided_p(x, margin)) if noninferiority else s['p_vs_zero']
        out.append(dict(id=cid, claim=claim, test=test, margin_mm=margin if noninferiority else None,
                        p_test=p, upper_below_margin=(s['ci95_mm'][1] < margin) if noninferiority else None, **s))
    for c, adjusted in zip(out, holm_adjust([c['p_test'] for c in out])):
        c.update(p_holm=adjusted, holds=adjusted < ALPHA)
    return out


def check(rows, x_claims):
    """Fails loudly if the tests, intervals or stored record disagree."""
    rng = np.random.default_rng(260912)
    for x in [*x_claims, *rng.normal(.3, .4, (20, 9))]:
        for delta in (MARGIN, SENSITIVITY_MARGIN, float(np.mean(x)) + .2):
            if np.mean(x) < delta:   # one-sided p is half the two-sided exact test at the shifted null
                assert np.isclose(2 * exact_one_sided_p(x, delta), exact_paired_test(x - delta, np.zeros(9))[1])
        upper = exact_location_interval(x)[1]
        assert 2 * exact_one_sided_p(x, upper + 1e-7) < ALPHA <= 2 * exact_one_sided_p(x, upper - 1e-7)
    for r in rows:
        x = vector(r)
        np.testing.assert_allclose(exact_location_interval(x), r['effect_ci95_pointwise_mm'], rtol=0, atol=1e-12)
        assert np.isclose(exact_paired_test(x, np.zeros(9))[1], r['p_raw'], rtol=0, atol=1e-12)


def main():
    record = json.loads(RECORD.read_text())
    figure = json.loads(FIGURE.read_text())
    rows = record['groups'][0]['comparisons']
    assert len(rows) == record['family_size'] == 76
    # The figure receipt and the frozen record must hold the same participant-level inputs.
    assert [r['participant_differences'] for r in rows] == \
           [r['participant_differences'] for r in figure['statistics']['groups'][0]['comparisons']]
    labels = list(csv.DictReader(LEGACY_COMPARISONS.open()))
    assert len(labels) == 76 and all(abs(float(s['effect_mm']) - r['effect_alternative_minus_reference_mm']) < 5e-4
                                     for s, r in zip(labels, rows))

    primary, sensitivity = claims(rows, MARGIN), claims(rows, SENSITIVITY_MARGIN)
    check(rows, [np.array(list(c['participant_differences_mm'].values())) for c in primary])
    expected = {'C1': (+.01, -.12, +.14), 'C2': (+.45, +.10, +.82), 'C3': (-2.40, -3.06, -1.69), 'C4': (-.51, -1.42, +.09)}
    for c in primary:
        assert np.allclose([c['effect_mm'], *c['ci95_mm']], expected[c['id']], atol=.005), c['id']
        assert c['holds'] and np.isclose(c['p_holm'], 4 * 2 / 512)
    assert [c['holds'] for c in sensitivity] == [True, False, True, True]

    by_id = {r.get('contrast_id'): r for r in rows}
    receivers = [dict(contrast_id=i, label=by_id[i]['label_a'], tokenizer='learned' if i in LEARNED else 'SPAR',
                      upper_below_margin=exact_location_interval(vector(by_id[i]))[1] < MARGIN,
                      upper_below_sensitivity_margin=exact_location_interval(vector(by_id[i]))[1] < SENSITIVITY_MARGIN,
                      **summary(vector(by_id[i]))) for i in LEARNED + SPAR]
    decoders = {(r['campaign'], r['arm']): r for r in figure['decoder_rows']}
    means = {k: np.array([r['error_participant_means'][p] for p in PARTICIPANTS]) for k, r in decoders.items()}
    pipelines = [dict(alternative=NAMES[a], reference=NAMES[b], **summary(means[a] - means[b])) for a, b in PIPELINES]
    seed_spread = {r['label']: float(np.mean([np.mean([abs(u - w) for i, u in enumerate(v) for w in v[i + 1:]])
                                              for v in (list(s.values()) for s in r['error_participant_seed_values'].values())]))
                   for r in figure['decoder_rows']}

    claim_index = {next(i for i, r in enumerate(rows) if r.get('contrast_id') == SPAR_VS_LEARNED): 'C3',
                   next(i for i, r in enumerate(rows) if tuple(r['alternative']) == SPAR_TOKENS
                        and tuple(r['reference']) == ADAPTED_CNN): 'C4'}
    by_claim = {c['id']: (c, s) for c, s in zip(primary, sensitivity)}
    table = []
    for i, (r, s) in enumerate(zip(rows, labels)):
        cid = claim_index.get(i)
        table.append(dict(comparison_id=f'HD{i + 1:02d}', role=f'claim {cid}' if cid else 'descriptive',
                          alternative=s['alternative'], reference=s['reference'], outcome=s['outcome'],
                          claim=by_claim.get(cid), p_maxT76=r['p_maxT'], **summary(vector(r))))
    extra = [('C1', 'SPAR tokens, pooled (Proposed, TabPFN 3.5)', 'SPAR features, pooled (same receivers)'),
             ('C2', 'Learned tokens, pooled (Proposed, TCN, Transformer, Qwen)', 'Learned features, pooled (same receivers)')]
    for cid, alternative, reference in extra:
        c = by_claim[cid][0]
        table.append(dict(comparison_id=f'HD{len(table) + 1:02d}', role=f'claim {cid}', alternative=alternative,
                          reference=reference, outcome='pooled_quantization_effect', claim=by_claim[cid], p_maxT76=None,
                          **{k: c[k] for k in ('effect_mm', 'ci95_mm', 'participants_alternative_higher', 'p_vs_zero',
                                               'participant_differences_mm')}))
    for p in pipelines:
        table.append(dict(comparison_id=f'HD{len(table) + 1:02d}', role='descriptive', outcome='mean_pose_error',
                          claim=None, p_maxT76=None, **p))
    for name, m in json.loads(MOVEMENTS.read_text())['per_movement'].items():
        x = np.array(m['tokens_minus_continuous']['per_participant_mm'])
        assert np.allclose(exact_location_interval(x), m['tokens_minus_continuous']['ci95_exact_mm'])
        table.append(dict(comparison_id=f'HD{len(table) + 1:02d}', role='descriptive',
                          alternative=f'Proposed / SPAR tokens, {name}', reference=f'Proposed / SPAR features, {name}',
                          outcome='movement_pose_error', claim=None, p_maxT76=None, **summary(x)))

    fmt = lambda v, d=3: '' if v is None else f'{v:+.{d}f}'
    with OUT_COMPARISONS.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['comparison_id', 'role', 'alternative', 'reference', 'outcome', 'effect_mm', 'ci95_lower_mm',
                    'ci95_upper_mm', 'participants_alternative_higher', 'n_participants', 'p_vs_zero_unadjusted',
                    'claim_test', 'p_claim_holm4_margin_1mm', 'p_claim_holm4_margin_0p5mm', 'p_maxT76_sensitivity'])
        for t in table:
            c = t['claim']
            test = '' if c is None else 'non-inferiority' if c[0]['test'] == 'noninferiority' else 'two-sided'
            w.writerow([t['comparison_id'], t['role'], t['alternative'], t['reference'], t['outcome'],
                        fmt(t['effect_mm']), fmt(t['ci95_mm'][0]), fmt(t['ci95_mm'][1]),
                        t['participants_alternative_higher'], 9, f"{t['p_vs_zero']:.4f}", test,
                        '' if c is None else f"{c[0]['p_holm']:.4f}", '' if c is None else f"{c[1]['p_holm']:.4f}",
                        '' if t['p_maxT76'] is None else f"{t['p_maxT76']:.4f}"])

    OUT_JSON.write_text(json.dumps({
        'method': METHOD, 'alpha': ALPHA, 'margin_mm': MARGIN, 'sensitivity_margin_mm': SENSITIVITY_MARGIN,
        'minimum_attainable_holm_p': 4 * 2 / 512, 'replication_unit': 'participant', 'n_participants': 9,
        'sources': {str(p.relative_to(ROOT)): sha256(p) for p in (RECORD, FIGURE, LEGACY_COMPARISONS, MOVEMENTS, Path(__file__))},
        'claims': {'margin_1mm': primary, 'margin_0p5mm': sensitivity}, 'receivers': receivers,
        'pipelines': pipelines, 'seed_spread_mean_abs_mm': seed_spread,
        'comparisons': [{k: (v[0]['id'] if k == 'claim' and v else v) for k, v in t.items()} for t in table],
        'outputs': [str(OUT_COMPARISONS.relative_to(ROOT))],
    }, indent=1, ensure_ascii=False) + '\n')

    print(f'{"claim":4s} {"effect":>7s} {"95% interval":>17s} {"Holm p (1 mm)":>14s} {"Holm p (0.5 mm)":>16s}')
    for c, s in zip(primary, sensitivity):
        print(f"{c['id']:4s} {c['effect_mm']:+7.2f} [{c['ci95_mm'][0]:+.2f}, {c['ci95_mm'][1]:+.2f}] "
              f"{c['p_holm']:14.4f} {s['p_holm']:16.4f}")
    for r in receivers:
        print(f"  {r['tokenizer']:7s} {r['label'][:44]:44s} {r['effect_mm']:+.2f} "
              f"[{r['ci95_mm'][0]:+.2f}, {r['ci95_mm'][1]:+.2f}] {r['participants_alternative_higher']}/9 higher")
    for p in pipelines:
        print(f"  {p['alternative']:30s} - {p['reference']:32s} {p['effect_mm']:+.2f} "
              f"[{p['ci95_mm'][0]:+.2f}, {p['ci95_mm'][1]:+.2f}]")
    print(f'wrote {OUT_JSON.relative_to(ROOT)} and {OUT_COMPARISONS.relative_to(ROOT)} ({len(table)} rows)')


if __name__ == '__main__':
    main()
