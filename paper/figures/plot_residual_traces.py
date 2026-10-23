"""Three readable temporal feature traces for the encoder architecture insert."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from paper.figures.plot_encoder_examples import (
    ROOT, OUT, PLOT_WIDTH_IN, PLOT_HEIGHT_IN, INSET_WIDTH_IN, INSET_HEIGHT_IN,
    SPINE_OFFSET_PT, apply_style, export, digest,
)
from paper.figures.export_encoder_examples_affinity import simple_svg, outline_text, write
from paper.figures.make_encoder_replacement_draft import render

STEM = 'residual-block2-traces'
FEATURES = np.array([1,128,256])
OFFSETS = np.array([0.,8.,16.])
HEIGHT_MULTIPLIER = 1.75
TRACE_HEIGHT_IN = PLOT_HEIGHT_IN * HEIGHT_MULTIPLIER
TRACE_INSET_HEIGHT_IN = INSET_HEIGHT_IN * HEIGHT_MULTIPLIER


def main():
    source = OUT/'residual-block2-featuremap-data.npz'
    receipt = OUT/'residual-block2-featuremap.json'
    source_meta = json.loads(receipt.read_text())
    assert digest(source) == source_meta['outputs'][source.name]
    data = np.load(source)
    times = data['time_ms']
    activation = data['activation'][:,FEATURES-1].astype(float)
    changes = activation - activation[0]
    positions = changes + OFFSETS
    assert activation.shape == (9,3) and np.isfinite(activation).all()
    assert np.max(np.abs(changes)) < 4
    apply_style()

    def draw(ax):
        for column, baseline in enumerate(OFFSETS):
            ax.plot([baseline,baseline],[times[0],times[-1]],
                color='#d4d4d4',lw=.45,zorder=0)
            ax.plot(positions[:,column],times,color='#5c618e',lw=.65,
                marker='o',markersize=1.1,markeredgewidth=0,
                solid_capstyle='round',solid_joinstyle='round')
        ax.set(xlim=(-4,20),ylim=(5,-165))

    fig = plt.figure(figsize=(PLOT_WIDTH_IN,TRACE_HEIGHT_IN))
    ax = fig.add_axes([.48,.36,.50,.55])
    draw(ax)
    ax.set(yticks=[-160,-80,0],xticks=OFFSETS,
        xticklabels=[str(i) for i in FEATURES],ylabel='Time (ms)',xlabel='Example\nfeatures')
    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 1
    ax.tick_params(length=2,pad=1)
    ax.tick_params(axis='x',length=0,labelsize=6.5)
    # Categorical trace centres have no continuous x axis. The retained time
    # spine uses the same trim and outward offset as the existing inserts.
    sns.despine(ax=ax,bottom=True,trim=True,offset=SPINE_OFFSET_PT)
    export(fig,STEM)

    fig = plt.figure(figsize=(INSET_WIDTH_IN,TRACE_INSET_HEIGHT_IN))
    ax = fig.add_axes([.01,.04,.98,.92])
    draw(ax)
    ax.axis('off')
    export(fig,STEM+'-inset')

    for stem in [STEM,STEM+'-inset']:
        tree,_ = simple_svg(OUT/f'{stem}.svg')
        target = OUT/f'{stem}-affinity.svg'
        write(tree,target)
        outlined = OUT/f'{stem}-affinity-outlined.svg'
        write(outline_text(tree),outlined)
        width,height = map(float,tree.get('viewBox').split()[2:])
        render(outlined,width=width*96/72,height=height*96/72)
        for ext in ['.pdf','.png']:
            outlined.with_suffix(ext).replace(target.with_suffix(ext))
        assert not ET.parse(target).getroot().findall('.//{http://www.w3.org/2000/svg}image')

    np.savez_compressed(OUT/f'{STEM}-data.npz',time_ms=times,feature_indices=FEATURES,
        activation=activation,change_from_first=changes,display_offsets=OFFSETS)
    rows = np.column_stack([np.repeat(times,3),np.tile(FEATURES,len(times)),
        activation.ravel(),changes.ravel()])
    np.savetxt(OUT/f'{STEM}.csv',rows,delimiter=',',
        header='time_ms,feature_index_1based,activation,change_from_first',comments='',
        fmt=['%.1f','%d','%.9g','%.9g'])
    metadata = dict(
        dataset=source_meta['dataset'],participant=source_meta['participant'],
        electrode_one_based=source_meta['electrode_one_based'],source_split=source_meta['source_split'],
        stage=source_meta['stage'],selected_feature_indices_one_based=FEATURES.tolist(),
        selection='First, middle, and last feature by index, independent of activation values.',
        time_ms_relative_to_final_output=times.tolist(),
        display='Each trace starts at its own faint baseline; horizontal deviation is activation minus its first value.',
        common_amplitude_scale=True,feature_wise_gain=False,
        subtract_initial_activation=True,display_offsets=OFFSETS.tolist(),
        line_segments='Straight connections between the nine actual 20 ms output positions; no smoothing.',
        label_note='Feature numbers identify separate traces; their spacing does not encode feature similarity.',
        labelled_size_inches=[PLOT_WIDTH_IN,TRACE_HEIGHT_IN],
        inset_size_inches=[INSET_WIDTH_IN,TRACE_INSET_HEIGHT_IN],
        height_multiplier=HEIGHT_MULTIPLIER,
        time_spine_trim=True,time_spine_offset_points=SPINE_OFFSET_PT,
        sources={str(p.relative_to(ROOT)):digest(p) for p in [source,receipt,Path(__file__),
            ROOT/'figures/plot_encoder_examples.py',ROOT/'figures/export_encoder_examples_affinity.py']},
        outputs={p.name:digest(p) for p in sorted(OUT.glob(STEM+'*'))
            if p.suffix in {'.svg','.pdf','.png','.csv','.npz'}},
    )
    (OUT/f'{STEM}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (OUT/f'{STEM}.md').write_text(
        '# Example temporal features after residual block 2\n\n'
        'A sparse alternative to the full heatmap for the architecture diagram. '
        'Three navy traces show features 1, 128, and 256 from the same recorded example; '
        'these indices were fixed without choosing by activation size or appearance. '
        'Place beside the output of the second residual addition, before block 3.\n\n'
        'Time runs downward from −160 to 0 ms relative to the final illustrated vector. '
        'Each trace starts at its faint baseline. Horizontal displacement shows the change '
        'from that feature’s first activation, on one shared amplitude scale. The traces '
        'are offset only for readability; the spacing does not encode feature similarity. '
        'No feature-specific scaling or smoothing is applied. Dots mark the nine real '
        'outputs at 20 ms intervals and straight lines connect them. All outputs have '
        'full recorded context, as documented in the source heatmap provenance.\n\n'
        'Use `residual-block2-traces-inset-affinity.svg` for the clean diagram insert, '
        'or `residual-block2-traces-affinity.svg` for feature IDs and times. '
        'The intended use is a qualitative architecture illustration, not an amplitude-comparison result. '
        'The CSV/NPZ preserve both raw activations and displayed changes.\n\n'
        'Both versions are 2.45 inches tall (1.75 times their previous height). '
        'Widths remain 1.05 inches for the labelled plot and 0.425 inches for the clean insert. '
        'Data, font sizes, line weights, and horizontal positions are unchanged.\n\n'
        'Suggested caption: “Example temporal features after residual block 2. '
        'Three feature dimensions are shown as changes from their initial activation '
        'with a shared scale and separate baselines.”\n\n'
        'Reproduce: `uv run python -m paper.figures.plot_residual_traces`.\n')
    print(json.dumps(dict(features=FEATURES.tolist(),positions=len(times),
        common_scale=True,display_change_range=[float(changes.min()),float(changes.max())]),indent=2))


if __name__=='__main__':
    main()
