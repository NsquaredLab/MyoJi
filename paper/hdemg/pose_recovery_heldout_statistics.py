"""Statistics for pose recovery on held-out data (test-window queries, held-out candidate cycles).

Reads results/token_umap/pose_recovery/pose_recovery_heldout_sub*.json, written on the workstation by
pose_recovery_heldout.py. First reproduces the published per-participant balanced errors from the
saved pose_recovery_sub*.npz, then tests the three token representations against continuous SPAR as in
main Fig. 6D: exact sign-flip over all 512 patterns, Holm across the three, exact 95 % intervals.

    uv run python -m paper.hdemg.pose_recovery_heldout_statistics
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from paper.hdemg.pose_recovery.aggregate import balanced  # noqa: E402
from paper.common import paths  # noqa: E402

RECOVERY = paths.RESULTS / 'token_umap/pose_recovery'
from paper.common.hdemg_figure_statistics import exact_location_interval, exact_paired_test, holm_adjust  # noqa: E402

OUT = ROOT / 'output/figures/token-pose-recovery-heldout.json'
SUBJECTS = range(1, 10)
TOKENS = ('tokens', 'tokens_64', 'tokens_hamming')


def published_check():
    """The saved full-data searches must reproduce the published Fig. 6D values."""
    published = json.loads((ROOT / 'output/figures/token-pose-recovery.json').read_text())['balanced_mm']
    for rep in ('continuous', *TOKENS):
        mine = []
        for s in SUBJECTS:
            d = np.load(RECOVERY / f'pose_recovery_sub{s:02d}.npz')
            mine.append(balanced(d[f'err_{rep}'], d['label'], d['cycle'], d['phase'])[0])
        np.testing.assert_allclose(mine, published[rep]['per_participant'], rtol=0, atol=1e-9)
    return {rep: float(np.mean(published[rep]['per_participant'])) for rep in ('continuous', *TOKENS)}


def main():
    full = published_check()
    rows = [json.loads((RECOVERY / f'pose_recovery_heldout_sub{s:02d}.json').read_text()) for s in SUBJECTS]
    assert [r['subject'] for r in rows] == list(SUBJECTS) and all(r['token_id_agreement_with_saved'] > .999 for r in rows)
    values = {rep: np.array([r['balanced_mm'][rep] for r in rows]) for rep in ('continuous', *TOKENS, 'pose_oracle')}
    tests = [exact_paired_test(values[rep].tolist(), values['continuous'].tolist()) for rep in TOKENS]
    holm = holm_adjust([t[1] for t in tests])
    contrasts = {}
    for rep, (effect, p_raw, d), p_holm in zip(TOKENS, tests, holm):
        low, high = exact_location_interval(np.asarray(d))
        contrasts[rep] = dict(minus_continuous_mm=effect, exact_ci95_mm=[float(low), float(high)], p_raw=p_raw,
                              p_holm=p_holm, positive=int((np.asarray(d) > 0).sum()), per_participant_mm=list(d),
                              full_data_minus_continuous_mm=full[rep] - full['continuous'])
    out = dict(
        rule=rows[0]['rule'],
        queries_per_participant=[r['queries'] for r in rows],
        query_share=[r['query_share'] for r in rows],
        heldout_cycles_per_participant=[r['heldout_cycles'] for r in rows],
        cycles_per_participant=[r['cycles'] for r in rows],
        balanced_mean_mm={rep: float(v.mean()) for rep, v in values.items()},
        balanced_per_participant_mm={rep: v.tolist() for rep, v in values.items()},
        contrasts=contrasts,
        tokens_phase_shift_mean_of_medians_mm=float(np.mean([r['tokens_phase_shift_median_mm'][0] for r in rows])),
        statistics='Two-sided exact sign-flip of the mean difference over all 512 participant sign patterns; '
                   'Holm across the three token representations; exact 95 % intervals by test inversion.',
        sources={f'pose_recovery_heldout_sub{s:02d}.json':
                 hashlib.sha256((RECOVERY / f'pose_recovery_heldout_sub{s:02d}.json').read_bytes()).hexdigest() for s in SUBJECTS})
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    for rep, c in contrasts.items():
        print(f"{rep:15s} {c['minus_continuous_mm']:+.3f} mm [{c['exact_ci95_mm'][0]:+.3f}, {c['exact_ci95_mm'][1]:+.3f}] "
              f"Holm p={c['p_holm']:.3f}  (all data {c['full_data_minus_continuous_mm']:+.3f})")
    print({k: round(v, 2) for k, v in out['balanced_mean_mm'].items()}, 'shift', round(out['tokens_phase_shift_mean_of_medians_mm'], 2))


if __name__ == '__main__':
    main()
