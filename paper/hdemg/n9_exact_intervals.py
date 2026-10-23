"""Exact sign-flip intervals for every nine-participant HD-EMG contrast outside the main family.

The experiment receipts carry participant-bootstrap intervals, which cover about 90 % at
n = 9 and can disagree with their own exact p-values. This ledger recomputes each
contrast's exact 95 % interval (inversion of the two-sided sign-flip test, as in
`hdemg_figure_statistics.exact_location_interval`) from the stored per-participant values,
asserts that the exact p-values and Holm adjustments reproduce the receipts, and
supersedes the bootstrap intervals. It also rewrites the HD-EMG rows of Supplementary
Data 6 and 8. Nothing is refitted.

    uv run python -m paper.hdemg.n9_exact_intervals
"""
import csv
import json
from pathlib import Path

import numpy as np

from paper.common.hdemg_figure_statistics import exact_location_interval, exact_paired_test, holm_adjust
from paper.common import paths
from paper.figures import plot_noise_robustness as nr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/n9_exact_intervals.json'
DATA = paths.SUPPLEMENTARY
SOURCES = {
    'transfer': 'results/hdemg_tokenizer_reuse_v1/results.json',
    'spar_corruption': 'results/hdemg_spar_corruption_v1/summary.json',
    'learned_noise': nr.NOISE,
    'learned_dropout': nr.DROPOUT,
    'onset_calibration': 'results/hdemg_spar_onset_calibration_v1/summary.json',
    'replays': 'results/hdemg_spar_transfer_checks_v1/summary.json',
    'electrodes_63': 'results/hdemg_spar_63_electrodes_v1/summary.json',
    'raw_link_budget': 'results/hdemg_raw_link_budget_v1/summary.json',
    'spar_64_codes': 'results/hdemg_spar_simple_full_v64/results.json',
    'pose_recovery': 'output/figures/token-pose-recovery.json',
    'direction': 'output/figures/direction-at-matched-closure.json',
    'direction_continuous': 'results/direction_continuous_spar_v1/summary.json',
    'port_validation': 'results/stm32n6_pose_validation_v1/summary.json',
}


def load(key):
    return json.loads((ROOT / SOURCES[key]).read_text())


def item(label, d, p=None, holm=None, ci=None):
    return dict(label=label, d=np.asarray(d, dtype=np.float64), stored_p=p, stored_holm=holm, bootstrap_ci=ci)


def families():
    """(name, source key, correction, items); correction is 'holm', 'none' (unadjusted) or 'descriptive'."""
    out = []
    r = load('transfer')
    out.append(('transfer', 'transfer', 'holm', [
        item(f"{c['route']}: tokens minus continuous", c['participant_differences_mm'], c['p_raw'], c['p_holm'],
             c['bootstrap95_mm']) for c in r['comparisons']]))

    r = load('spar_corruption')
    for fam in ('noise', 'dropout'):
        out.append((f'spar_corruption_{fam}', 'spar_corruption', 'holm', [
            item(f"{c['kind']} {c['level']}: tokens minus continuous increase", c['per_participant'], c['p_raw'],
                 c['p_holm'], c['ci95']) for c in r['families'][fam]]))

    noise, drop = load('learned_noise'), load('learned_dropout')
    stored = nr.contrasts(noise, nr.SNRS)
    out.append(('learned_corruption_noise', 'learned_noise', 'holm', [
        item(f"{s['snr_db']} dB: tokens minus {s['reference']} increase",
             nr.degradation(noise, 'token_v20', s['snr_db']) - nr.degradation(noise, s['reference'], s['snr_db']),
             s['p_raw'], s['p_holm'], s['ci95']) for s in stored]))
    out.append(('learned_corruption_dropout', 'learned_dropout', 'holm', [
        item(f"{s['kind']} {s['level']}: tokens minus continuous increase",
             nr.drop_delta(drop, 'token_v20', s['kind'], s['level'])
             - nr.drop_delta(drop, 'continuous_v20', s['kind'], s['level']),
             s['p_raw'], s['p_holm'], s['ci95']) for s in nr.dropout_contrasts(drop)]))
    T = {s: nr.level(noise, 'token_v20', s) for s in (None, 10)}
    C = {s: nr.level(noise, 'continuous_v20', s) for s in (None, 10)}
    out.append(('learned_corruption_levels', 'learned_noise', 'descriptive', [
        item('clean: tokens minus continuous error', T[None] - C[None], ci=nr.boot_ci(T[None] - C[None])),
        item('10 dB: tokens minus continuous error', T[10] - C[10], ci=nr.boot_ci(T[10] - C[10]))]))

    r = load('onset_calibration')
    out.append(('onset_calibration', 'onset_calibration', 'holm', [
        item(k.replace('_', ' '), r[k]['per_participant'], r[k]['p_exact'], r[k]['p_holm'], r[k]['bootstrap95'])
        for k in ('tokens_minus_continuous_onset', 'tokens_onset_minus_window', 'continuous_onset_minus_window')]))

    r = load('replays')
    holm2 = r['fast_digits_pooled']['holm_over_two_new_token_contrasts']['adjusted']
    fast, fixed = r['fast_digits_pooled']['tokens_minus_continuous'], r['calibration']['tokens_minus_continuous_fixed_startup']
    out.append(('replay_token_contrasts', 'replays', 'holm', [
        item('1.5 Hz digits: tokens minus continuous', fast['per_participant'], fast['p_exact'], holm2['fast_digits'],
             fast['bootstrap95']),
        item('fixed startup: tokens minus continuous', fixed['per_participant'], fixed['p_exact'],
             holm2['fixed_startup'], fixed['bootstrap95'])]))
    out.append(('replay_calibration_change', 'replays', 'none', [
        item(f'{arm}: fixed startup minus per-window', c['per_participant'], c['p_exact'], ci=c['bootstrap95'])
        for arm, c in ((a, r['calibration'][a]['fixed_minus_per_window']) for a in ('tokens', 'continuous'))]))
    out.append(('replay_untrained_tasks', 'replays', 'descriptive', [
        item(f'task {task}: tokens minus continuous', c['tokens_minus_continuous']['per_participant'],
             ci=c['tokens_minus_continuous']['bootstrap95']) for task, c in r['transfer'].items()]))

    r = load('electrodes_63')
    out.append(('electrodes_63', 'electrodes_63', 'holm', [
        item(k.replace('_', ' '), r[k]['per_participant'], r[k]['p_exact'], ci=r[k]['bootstrap95'])
        for k in ('tokens_minus_continuous_63', 'tokens_63_minus_320', 'continuous_63_minus_320')]))

    r = load('raw_link_budget')
    out.append(('raw_link_budget', 'raw_link_budget', 'holm', [
        item(k.replace('_', ' '), r[k]['per_participant'], r[k]['p_exact'], r[k]['p_holm'], r[k]['bootstrap95'])
        for k in ('raw42_minus_spar63_tokens', 'raw42_minus_cnn320')]))

    r, panel = load('spar_64_codes'), load('pose_recovery')['panel_d']['paired_change_vs_continuous']['64']['ci95']
    out.append(('spar_64_codes', 'spar_64_codes', 'holm', [
        item(f"64 codes: {c['alternative']} minus {c['reference']}", c['participant_differences_mm'], c['p_raw'],
             c['p_holm'], panel if c['reference'] == 'spar_continuous' else None) for c in r['comparisons']]))

    r = load('pose_recovery')
    balanced, paired = r['balanced_mm'], r['panel_c']['paired_vs_continuous']
    cont = np.array(balanced['continuous']['per_participant'])
    out.append(('pose_recovery', 'pose_recovery', 'holm', [
        item(f'recovery: {k} minus continuous', np.array(balanced[k]['per_participant']) - cont,
             paired[k]['p_raw'], paired[k]['p_holm'], paired[k]['ci95']) for k in ('tokens', 'tokens_64', 'tokens_hamming')]))

    r = load('direction')
    out.append(('direction', 'direction', 'holm', [
        item(f"direction: {k.replace('_', ' ')}", c['per_participant'], c['p_raw'], c['p_holm'], c['bootstrap95'])
        for k, c in r['primary']['contrasts'].items()]))
    r = load('direction_continuous')
    out.append(('direction_continuous', 'direction_continuous', 'none', [
        item('direction: tokens minus continuous SPAR', r['tokens_minus_continuous']['per_participant'],
             r['tokens_minus_continuous']['p_exact'], ci=r['tokens_minus_continuous']['bootstrap95'])]))

    r = load('port_validation')
    out.append(('port_validation', 'port_validation', 'none', [
        item(k.replace('_', ' '), c['per_participant_mm'], c['p_two_sided_exact'], ci=c['bootstrap95_mm'])
        for k, c in r['contrasts'].items()]))
    return out


def excludes_zero(ci):
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def main():
    ledger = []
    for name, source, correction, items in families():
        assert all(len(i['d']) == 9 and np.isfinite(i['d']).all() for i in items), name
        p_exact = [exact_paired_test(i['d'], np.zeros(9))[1] for i in items]
        p_holm = holm_adjust(p_exact) if correction == 'holm' else [None] * len(items)
        rows = []
        for i, p, q in zip(items, p_exact, p_holm):
            if i['stored_p'] is not None:
                assert np.isclose(p, i['stored_p'], rtol=0, atol=1e-12), (name, i['label'], p, i['stored_p'])
            if i['stored_holm'] is not None:
                assert np.isclose(q, i['stored_holm'], rtol=0, atol=1e-12), (name, i['label'], q, i['stored_holm'])
            ci = exact_location_interval(i['d'])
            rows.append(dict(label=i['label'], mean_mm=float(i['d'].mean()), ci95_exact_mm=ci,
                             ci95_bootstrap_superseded_mm=i['bootstrap_ci'],
                             p_exact=None if correction == 'descriptive' else p, p_holm=q,
                             participants_positive=int((i['d'] > 0).sum()),
                             zero_exclusion_changed=(i['bootstrap_ci'] is not None
                                                     and excludes_zero(ci) != excludes_zero(i['bootstrap_ci'])),
                             per_participant=i['d'].tolist()))
        ledger.append(dict(family=name, source=SOURCES[source], correction=correction, contrasts=rows))

    by = {f['family']: f['contrasts'] for f in ledger}
    write_corruption_data(by)
    write_transfer_data(by)
    OUT.write_text(json.dumps({
        'method': ('Exact 95 % intervals invert the two-sided sign-flip test over all 512 participant sign '
                   'patterns (differences assumed symmetric about their mean); they replace the participant-'
                   'bootstrap intervals stored in the receipts. p-values are the receipts\' exact sign-flip '
                   'values, Holm-adjusted within each declared family; descriptive rows carry no test.'),
        'families': ledger}, indent=1) + '\n')

    for f in ledger:
        print(f"{f['family']} ({f['correction']})")
        for c in f['contrasts']:
            old = c['ci95_bootstrap_superseded_mm']
            flag = '  <-- zero exclusion changed' if c['zero_exclusion_changed'] else ''
            print(f"  {c['label'][:52]:52s} {c['mean_mm']:+8.3f} exact [{c['ci95_exact_mm'][0]:+.3f}, "
                  f"{c['ci95_exact_mm'][1]:+.3f}]" + (f" boot [{old[0]:+.3f}, {old[1]:+.3f}]" if old else '')
                  + (f" p={c['p_exact']:.4f}" if c['p_exact'] is not None else '')
                  + (f" holm={c['p_holm']:.4f}" if c['p_holm'] is not None else '') + flag)
    print(f'wrote {OUT.relative_to(ROOT)}, SD4 and SD8')


def write_corruption_data(by):
    summary = load('spar_corruption')
    names = {('noise', 40): 'Noise, 1% RMS', ('noise', 30): 'Noise, 3.2% RMS', ('noise', 20): 'Noise, 10% RMS',
             ('noise', 10): 'Noise, 32% RMS', ('fraction', .05): '5% of electrodes',
             ('fraction', .1): '10% of electrodes', ('fraction', .25): '25% of electrodes',
             ('grids', 1): 'One array', ('grids', 2): 'Two arrays'}
    contrasts = by['spar_corruption_noise'] + by['spar_corruption_dropout']
    mm = lambda v: f'{round(v, 2) + 0.:.2f}'              # no negative zero
    with (DATA / 'SD8_spar_corruption.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'token_error_mm', 'continuous_error_mm', 'difference_in_increase_mm',
                    'ci95_lower_mm', 'ci95_upper_mm', 'p_holm'])
        w.writerow(['Clean', f"{summary['clean']['tokens']:.2f}", f"{summary['clean']['continuous']:.2f}", '', '', '', ''])
        for s, c in zip(summary['summary'], contrasts):
            assert c['label'].startswith(f"{s['kind']} {s['level']}:")
            w.writerow([names[(s['kind'], s['level'])], mm(s['tokens_error']), mm(s['continuous_error']),
                        mm(c['mean_mm']), mm(c['ci95_exact_mm'][0]), mm(c['ci95_exact_mm'][1]), f"{c['p_holm']:.3f}"])


def write_transfer_data(by):
    path = DATA / 'SD4_transfer_and_cohorts.csv'
    rows = list(csv.reader(path.open()))
    header, emg2pose = rows[0], [r for r in rows[1:] if r[1] != 'HD-EMG']
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for c, name in zip(by['transfer'], ('SPAR: separate participants', 'Learned: separate participants')):
            assert c['label'].startswith(name.split(':')[0].lower())
            w.writerow([name, 'HD-EMG', 9, f"{c['mean_mm']:.3f}", f"{c['ci95_exact_mm'][0]:.3f}",
                        f"{c['ci95_exact_mm'][1]:.3f}", f"{c['p_holm']:.3f}", '2-contrast Holm', 'exact sign-flip interval'])
        w.writerows(emg2pose)


if __name__ == '__main__':
    main()
