"""SPAR feature ablation and held-out pose retrieval: statistics and Supplementary Data 9.

Ablation: results/hdemg_spar_ablation_v1/{logpower,diagonal} against the published full-SPAR campaign
results/hdemg_spar_simple_full_v1 (same windows, 1,024-code codebooks, seeds, decoder and gates).
Per descriptor: mean error with tokens and with continuous features and their paired difference;
between descriptors: each six-value descriptor minus full SPAR. Exact sign-flip over all 512
participant patterns with exact 95 % intervals; the two token contrasts against full SPAR form one Holm
family, the rest are estimates. Retrieval rows come from hdemg/pose_recovery_heldout_statistics.py.

    uv run python -m paper.hdemg.spar_ablation_statistics
"""
import csv
import json
from pathlib import Path

import numpy as np

from paper.common import paths
from paper.common.hdemg_figure_statistics import exact_location_interval, exact_paired_test, holm_adjust

ROOT = Path(__file__).resolve().parents[1]
RUNGS = {'logpower': 'results/hdemg_spar_ablation_v1/logpower/results.json',
         'diagonal': 'results/hdemg_spar_ablation_v1/diagonal/results.json',
         'full': 'results/hdemg_spar_simple_full_v1/results.json'}
NAMES = {'logpower': 'Centre log band power (6 values)', 'diagonal': 'Matrix-log diagonal (6 values)',
         'full': 'Full SPAR (18 values)'}
RETRIEVAL = ROOT / 'output/figures/token-pose-recovery-heldout.json'
OUT = ROOT / 'results/spar_ablation_statistics.json'
SD9 = paths.SUPPLEMENTARY / 'SD9_spar_ablation_and_heldout_retrieval.csv'
ARMS = ('tokens', 'continuous')


def participant_means(path, arm):
    r = json.loads((ROOT / path).read_text())
    assert r['complete'] and r['fits'] == 54 and r['subjects'] == list(range(1, 10))
    return np.asarray(r['summary'][f'spar_{arm}']['participant_means_mm'])


def contrast(section, alternative, reference, a, b, family=''):
    effect, p, d = exact_paired_test(a, b)
    low, high = exact_location_interval(np.asarray(d))
    return dict(section=section, alternative=alternative, reference=reference, mean_alternative_mm=float(np.mean(a)),
                mean_reference_mm=float(np.mean(b)), difference_mm=effect, ci95_lower_mm=low, ci95_upper_mm=high,
                participants_alternative_higher=int((np.asarray(d) > 0).sum()), n_participants=len(d),
                p_unadjusted=p, p_holm=None, holm_family=family)


def main():
    m = {(rung, arm): participant_means(path, arm) for rung, path in RUNGS.items() for arm in ARMS}
    # The full-SPAR rung must be the published campaign (Supplementary Data 1).
    assert round(m['full', 'tokens'].mean(), 3) == 10.812 and round(m['full', 'continuous'].mean(), 3) == 10.946
    rows = [contrast('ablation', f'{NAMES[r]}, tokens', f'{NAMES[r]}, continuous', m[r, 'tokens'], m[r, 'continuous'])
            for r in RUNGS]
    family = []
    for arm in ARMS:
        for r in ('logpower', 'diagonal'):
            row = contrast('ablation', f'{NAMES[r]}, {arm}', f"{NAMES['full']}, {arm}", m[r, arm], m['full', arm],
                           'two token descriptors vs full SPAR' if arm == 'tokens' else '')
            rows.append(row)
            if arm == 'tokens':
                family.append(row)
    for row, p in zip(family, holm_adjust([row['p_unadjusted'] for row in family])):
        row['p_holm'] = p
    if RETRIEVAL.exists():
        heldout = json.loads(RETRIEVAL.read_text())
        names = {'tokens': 'Tokens (1,024 codes)', 'tokens_64': 'Tokens (64 codes)', 'tokens_hamming': 'Token IDs (Hamming)'}
        for rep, c in heldout['contrasts'].items():
            rows.append(dict(section='held-out pose retrieval', alternative=names[rep], reference='Continuous SPAR',
                             mean_alternative_mm=heldout['balanced_mean_mm'][rep],
                             mean_reference_mm=heldout['balanced_mean_mm']['continuous'],
                             difference_mm=c['minus_continuous_mm'], ci95_lower_mm=c['exact_ci95_mm'][0],
                             ci95_upper_mm=c['exact_ci95_mm'][1], participants_alternative_higher=c['positive'],
                             n_participants=len(c['per_participant_mm']), p_unadjusted=c['p_raw'], p_holm=c['p_holm'],
                             holm_family='three token representations vs continuous'))
    OUT.write_text(json.dumps(dict(rows=rows, participant_means_mm={f'{r}_{a}': m[r, a].tolist() for r, a in m},
                                   sources=RUNGS, retrieval=str(RETRIEVAL.relative_to(ROOT))), indent=1) + '\n')
    with SD9.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f'{v:+.3f}' if k in ('difference_mm', 'ci95_lower_mm', 'ci95_upper_mm') else
                            f'{v:.3f}' if k.startswith('mean_') else
                            f'{v:.4f}' if k.startswith('p_') and v is not None else '' if v is None else v)
                        for k, v in row.items()})
    for row in rows:
        holm = '' if row['p_holm'] is None else f"  Holm p={row['p_holm']:.3f}"
        print(f"{row['alternative']:42s} - {row['reference']:34s} {row['difference_mm']:+.3f} "
              f"[{row['ci95_lower_mm']:+.3f}, {row['ci95_upper_mm']:+.3f}] p={row['p_unadjusted']:.3f}{holm}")


if __name__ == '__main__':
    main()
