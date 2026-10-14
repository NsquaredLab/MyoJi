"""Meta: individual quantization effects and uncertainty in their mean."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_pose_paper_assets as data
import tokenizer_diagnostic_style as style


def draw(r, mm_per_palm):
    style.apply()
    cohorts = [('official', 'all', 'Official test\n176 participants', 176),
               ('official', 'unseen_users', 'Unseen subset\n20 of 176 participants', 20),
               ('session', 'all', 'Held-out sessions\n10 participants', 10)]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.95), sharey=True,
                             gridspec_kw={'width_ratios': [1.5, 1]})
    fig.subplots_adjust(left=.235, right=.98, bottom=.26, top=.81, wspace=.23)
    details, participants = [], []
    rng = np.random.default_rng(20260913)  # Vertical display jitter only.
    for row, (scope, subgroup, name, n) in enumerate(cohorts):
        people, cont = data.vector(r, [f'continuous_{s}' for s in data.SEEDS], scope, subgroup)
        other, token = data.vector(r, [f'vq_{s}' for s in data.SEEDS], scope, subgroup)
        assert people == other and len(people) == n
        delta = (token - cont) * mm_per_palm
        axes[0].scatter(delta, row + rng.uniform(-.21, .21, n), s=12,
                        color=style.POINT, alpha=.7, edgecolors='white', linewidths=.25)
        participants.append({'scope': scope, 'subgroup': subgroup,
                             'users': people, 'token_minus_continuous_mm': delta.tolist()})
        for seed in [*data.SEEDS, None]:
            key = f'vq_{seed}_vs_continuous_{seed}' if seed else 'three_encoder_mean_vq_vs_continuous'
            p = data.pair(r, scope, subgroup, key)
            mean = p['mean_difference'] * mm_per_palm
            ci = np.array(p['difference_ci95']) * mm_per_palm
            if seed is None:
                axes[1].plot(ci, [row+.13]*2, color=style.TOKEN, lw=1.7)
                axes[1].scatter(mean, row+.13, marker='D', s=31, color=style.TOKEN, zorder=4)
            else:
                # Individual encoder means are diagnostic points, not three
                # extra participants. Only the pooled mean receives a CI here.
                offset = (-.025, 0, .025)[data.SEEDS.index(seed)]
                axes[1].scatter(mean, row-.16+offset, color=style.GRAY,
                                edgecolor='white', linewidth=.4, s=18, zorder=3)
            details.append({'scope': scope, 'subgroup': subgroup, 'comparison': key,
                            'difference': p['mean_difference'], 'ci95': p['difference_ci95'],
                            'difference_mm': mean, 'ci95_mm': ci.tolist(),
                            'source_unit': 'palm lengths', 'display_unit': 'mm; reference hand',
                            'interval_displayed': seed is None})
    axes[0].set_yticks(range(3), [c[2] for c in cohorts])
    axes[0].set_ylim(2.48, -.48)
    axes[0].set(xlim=(-3.45, 3.45), xlabel='Participant error change (mm)')
    axes[0].set_xticks([-3, -2, -1, 0, 1, 2, 3])
    axes[1].set(xlim=(-.72, .72), xlabel='Mean error change (mm)')
    axes[1].set_xticks([-.6, -.3, 0, .3, .6])
    for i, ax in enumerate(axes):
        ax.axvline(0, color=style.GRAY, ls='--', lw=.85, zorder=1)
        style.finish(ax, categorical=True); style.panel(ax, 'AB'[i])
    axes[1].legend(handles=[
        Line2D([], [], marker='o', ms=4, color=style.GRAY, ls='', label='Individual encoder mean'),
        Line2D([], [], marker='D', ms=4, color=style.TOKEN, lw=1.5, label='Cohort mean, 95% CI')],
        loc='lower center', bbox_to_anchor=(.52, 1.125), ncol=1, frameon=False,
        handlelength=1.4, borderaxespad=0, labelspacing=.3)
    fig.text(.595, .035, 'Change = tokens - continuous features; negative favors tokens',
             ha='center', fontsize=8.5)
    data.save(fig, 'meta_transfer', {'paired_estimates': details, 'participants': participants,
        'display': 'Participant paired differences at left; three encoder means and pooled participant-bootstrap CI at right.',
        'unseen_is_subset_of_official': True, 'all_participants_displayed': True,
        'new_statistics': False})


if __name__ == '__main__':
    data.configure()
    draw(data.read('results/paper_completion_v1/results.json'),
         data.read('data/reference/meta_millimetre_conversion.json')['mm_per_palm_length'])
