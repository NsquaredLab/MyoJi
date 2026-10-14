"""Meta compact controls: paired accuracy and packet cost on aligned rows."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import build_pose_paper_assets as data
import tokenizer_diagnostic_style as style


def draw(r, mm_per_palm):
    style.apply()
    choices = [
        ('continuous_260905', 'Continuous features (256D)', 0, style.FEATURE),
        ('fsq1000', 'Post-hoc FSQ (1,000)', 1.15, style.GRAY),
        ('pca4q8', 'PCA (4D; 8-bit)', 2.7, style.GRAY),
        ('pca4q10', 'PCA (4D; 10-bit)', 3.7, style.GRAY),
        ('pca4q16', 'PCA (4D; 16-bit)', 4.7, style.GRAY),
        ('kmeans256', 'Feature K-means (256)', 6.25, style.GRAY),
        ('kmeans1024', 'Feature K-means (1,024)', 7.25, style.GRAY),
        ('biocodec1_260940', 'BioCodec (1 book)', 8.8, style.GRAY),
        ('biocodec3_260940', 'BioCodec (3 books)', 9.8, style.GRAY),
        ('biocodec6_260940', 'BioCodec (6 books)', 10.8, style.GRAY),
    ]
    rt = {v['case']['label']: v for v in r['runtime']['cuda']['result']['rows']}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.75), sharey=True,
                             gridspec_kw={'width_ratios': [1.6, 1]})
    fig.subplots_adjust(left=.285, right=.97, bottom=.16, top=.91, wspace=.25)
    details = []
    for label, name, y, color in choices:
        p = data.pair(r, 'official', 'all', 'vq_260905_vs_' + label)
        mean = -p['mean_difference'] * mm_per_palm
        ci = -np.array(p['difference_ci95'])[::-1] * mm_per_palm
        axes[0].plot(ci, [y, y], color=color, lw=1.25)
        axes[0].scatter(mean, y, s=22, color=color, zorder=3)
        rate = rt[label]['packet_bytes_per_second'] / 1000
        axes[1].scatter(rate, y, s=22, color=color, zorder=3)
        details.append({'label': label, 'delta_from_vq': -p['mean_difference'],
            'ci95': (-np.array(p['difference_ci95'])[::-1]).tolist(),
            'delta_from_vq_mm': mean, 'ci95_mm': ci.tolist(),
            'source_unit': 'palm lengths', 'display_unit': 'mm; reference hand', 'packet_kB_s': rate})
    axes[0].set_yticks([v[2] for v in choices], [v[1] for v in choices])
    axes[0].set_ylim(11.45, -.7)
    axes[0].set(xlim=(-.1, 1.45), xlabel='Error relative to learned VQ (mm)')
    axes[0].set_xticks([0, .4, .8, 1.2])
    axes[1].set(xscale='log', xlim=(.8, 1250), xlabel='Packet rate (kB/s)')
    axes[1].set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1,000'])
    axes[0].axvline(0, color=style.TOKEN, ls='--', lw=.9)
    axes[1].axvline(rt['vq_260905']['packet_bytes_per_second']/1000,
                    color=style.TOKEN, ls='--', lw=.9)
    for i, ax in enumerate(axes):
        style.finish(ax, categorical=True); style.panel(ax, 'AB'[i])
    fig.legend(handles=[Line2D([], [], color=style.TOKEN, ls='--', lw=.9,
                              label='Learned VQ reference')], frameon=False,
               loc='upper right', bbox_to_anchor=(.98, 1.01), borderaxespad=0)
    data.save(fig, 'compact_controls', details)


if __name__ == '__main__':
    data.configure()
    draw(data.read('results/paper_completion_v1/results.json'),
         data.read('data/reference/meta_millimetre_conversion.json')['mm_per_palm_length'])
