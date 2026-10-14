"""Meta full-pipeline latency, separating hardware from representation color."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_pose_paper_assets as data
import tokenizer_diagnostic_style as style


def draw(r):
    style.apply()
    labels = [('vq_260905', 'Learned VQ (1,024)', style.TOKEN),
              ('continuous_260905', 'Continuous features (256D)', style.FEATURE),
              ('fsq1000', 'Post-hoc FSQ (1,000)', style.GRAY),
              ('pca4q10', 'PCA (4D; 10-bit)', style.GRAY),
              ('kmeans1024', 'Feature K-means (1,024)', style.GRAY),
              ('biocodec6_260940', 'BioCodec (6 books)', style.GRAY)]
    runtimes = {d: {v['case']['label']: v for v in r['runtime'][d]['result']['rows']}
                for d in ('cpu', 'cuda')}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True)
    fig.subplots_adjust(left=.285, right=.98, bottom=.19, top=.82, wspace=.2)
    details = []
    for ax, device, xlabel in zip(axes, ('cpu', 'cuda'), ('CPU latency (ms)', 'V100 latency (ms)'), strict=True):
        for y, (label, name, color) in enumerate(labels):
            v = runtimes[device][label]
            med, p95 = v['latency']['p50_ms'], v['latency']['p95_ms']
            ax.plot([med, p95], [y, y], color=color, lw=1.2)
            ax.scatter(med, y, marker='s' if 'continuous' in label else 'o',
                       s=24, color=color, zorder=3)
            ax.plot(p95, y, marker='|', color=color, ms=8, mew=1, zorder=4)
            details.append({'label': label, 'device': device, 'median_ms': med, 'p95_ms': p95,
                            'tensor_MB_in_companion_table': v['sum_component_tensor_bytes']/1e6})
        ax.set(xscale='log', xlim=(2, 140), xlabel=xlabel)
        ax.set_xticks([2, 5, 10, 20, 50, 100], ['2', '5', '10', '20', '50', '100'])
        ax.axvline(20, color=style.GRAY, ls='--', lw=.85)
        ax.set_ylim(len(labels)-.5, -.6)
        style.finish(ax, categorical=True)
    axes[0].set_yticks(range(len(labels)), [v[1] for v in labels])
    for i, ax in enumerate(axes): style.panel(ax, 'AB'[i])
    fig.legend(handles=[
        Line2D([], [], marker='o', ms=4, ls='', color=style.GRAY, label='Median'),
        Line2D([], [], marker='|', ms=6, lw=1, color=style.GRAY, label='95th percentile'),
        Line2D([], [], ls='--', lw=.85, color=style.GRAY, label='20 ms output interval')],
        loc='upper right', bbox_to_anchor=(.985, 1), ncol=3, frameon=False,
        handlelength=1.2, columnspacing=1.25, borderaxespad=0)
    data.save(fig, 'streaming_cost', details)


if __name__ == '__main__':
    data.configure(); draw(data.read('results/paper_completion_v1/results.json'))
