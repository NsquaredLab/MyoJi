"""Show the real block-2 temporal feature map for the recorded encoder example."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import torch

from paper.figures.plot_encoder_examples import (
    ROOT, OUT, PLOT_WIDTH_IN, PLOT_HEIGHT_IN, INSET_WIDTH_IN, INSET_HEIGHT_IN,
    SPINE_OFFSET_PT, apply_style, standard_axes, export, digest, rel,
)
from paper.figures.export_encoder_examples_affinity import simple_svg, outline_text, write, NS
from paper.figures.make_encoder_replacement_draft import render
from myoji.learned.stream import Tokenizer
from paper.common import paths

STEM = 'residual-block2-featuremap'


@torch.inference_mode()
def extract():
    torch.set_num_threads(2)
    source = OUT/'example-data.npz'
    provenance_path = OUT/'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    checkpoint = paths.LEARNED_TOKENIZER
    assert digest(source) == provenance['outputs'][source.name]
    assert digest(checkpoint) == provenance['sources'][rel(checkpoint)]
    data = np.load(source)
    tokenizer = Tokenizer(checkpoint, device='cpu')
    model = tokenizer.model
    assert not model.training and len(model.temporal_encoder) == 4
    captured = []
    handle = model.temporal_encoder[1].register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().clone()))
    channel = provenance['channel_zero_based']
    try:
        output = model(torch.from_numpy(data['normalized_emg'].copy())[None,None,:],
            sensor='hd', channel_descriptors=model.sensor_positions('hd')[channel:channel+1],
            include_spatial=False)
    finally:
        handle.remove()
    assert len(captured) == 1 and captured[0].shape == (1,17,256)
    final = output.continuous[0,0,-1].cpu().numpy()
    np.testing.assert_allclose(final, data['prequantization_vector'], rtol=2e-5, atol=2e-5)
    assert int(output.tokens[0,0,-1]) == provenance['token_id']
    # The first eight block-2 outputs use causal padding rather than a full
    # recorded history. Retain only the nine fully supported output positions.
    warmup = sum(block.left_padding for block in model.temporal_encoder[:2])
    activation = captured[0][0,warmup:].cpu().numpy().copy()
    hop_ms = 1000*tokenizer.config.hop_samples/tokenizer.config.sample_rate_hz
    time_ms = np.arange(1-len(activation),1,dtype=float)*hop_ms
    np.testing.assert_array_equal(activation[-1],
        np.load(OUT/'residual-block2-vector-data.npz')['activation'])
    assert activation.shape == (9,256) and np.isfinite(activation).all()
    metadata = dict(
        dataset=provenance['dataset'], participant=provenance['participant'],
        source_split=provenance['source_split'], electrode_one_based=channel+1,
        task=provenance['task'], token_id=provenance['token_id'],
        recording_sample_endpoint_inclusive=provenance['recording_sample_endpoint_inclusive'],
        stage='Output of residual block 2, after its + and before block 3/final LayerNorm.',
        hook_module='temporal_encoder.1', activation_shape_time_by_feature=list(activation.shape),
        output_step_ms=hop_ms, output_times_relative_to_final_ms=time_ms.tolist(),
        feature_order='Original model order, displayed as 1–256 from top to bottom.',
        warmup_positions_omitted=int(warmup), full_recorded_history_per_output_ms=320,
        time_axis='Output times relative to the final illustrated vector; 0 is its endpoint.',
        final_feature_vector_max_abs_difference=float(np.max(np.abs(final-data['prequantization_vector']))),
        final_block2_vector_exact_agreement=True,
        activation_min=float(activation.min()), activation_max=float(activation.max()),
        smoothing=False, interpolation=False, feature_rescaling=False,
        new_training=False, synthetic_data=False,
        sources={rel(p):digest(p) for p in [source, provenance_path,
            OUT/'residual-block2-vector-data.npz', checkpoint,
            paths.REPO/'src/myoji/learned/encoder.py',
            ROOT/'figures/plot_encoder_examples.py',
            ROOT/'figures/export_encoder_examples_affinity.py', Path(__file__)]},
    )
    return time_ms, activation, metadata


def affinity_export(stem, expected_cells):
    tree, _ = simple_svg(OUT/f'{stem}.svg')
    mesh = next(e for e in tree.iter() if e.get('id') == 'QuadMesh_1')
    assert len(mesh.findall(f'{{{NS}}}path')) == expected_cells
    mesh.set('shape-rendering','crispEdges')
    target = OUT/f'{stem}-affinity.svg'
    write(tree,target)
    outlined = OUT/f'{stem}-affinity-outlined.svg'
    write(outline_text(tree),outlined)
    width,height = map(float,tree.get('viewBox').split()[2:])
    render(outlined,width=width*96/72,height=height*96/72)
    for ext in ['.pdf','.png']:
        outlined.with_suffix(ext).replace(target.with_suffix(ext))
    assert not ET.parse(target).getroot().findall(f'.//{{{NS}}}image')


def main():
    time_ms, activation, metadata = extract()
    apply_style()
    limit = float(MaxNLocator(nbins=1,steps=[1,2,2.5,5,10],min_n_ticks=1)
        .tick_values(0,float(np.max(np.abs(activation))))[-1])
    cmap = LinearSegmentedColormap.from_list('encoder_signed', ['#a26832','#f1f0ed','#474e80'])
    norm = Normalize(-limit,limit)
    edges = np.r_[time_ms-10,time_ms[-1]+10]
    feature_edges = np.arange(.5,257)

    def draw(ax):
        mesh = ax.pcolormesh(edges,feature_edges,activation.T,cmap=cmap,norm=norm,
            shading='flat',edgecolors='none',antialiased=False,rasterized=False)
        ax.set(xlim=(edges[0],edges[-1]),ylim=(256.5,.5))
        return mesh

    fig, ax = standard_axes('Feature index','Time (ms)')
    # A zero-reference line is useful for a signed scatter, not on a heatmap.
    for line in list(ax.lines):
        line.remove()
    draw(ax)
    ax.set(xticks=[time_ms[0],time_ms[len(time_ms)//2],0],yticks=[1,128,256])
    sns.despine(ax=ax,trim=True,offset=SPINE_OFFSET_PT)
    export(fig,STEM)

    fig = plt.figure(figsize=(INSET_WIDTH_IN,INSET_HEIGHT_IN))
    ax = fig.add_axes([.01,.04,.98,.92])
    draw(ax)
    ax.axis('off')
    export(fig,STEM+'-inset')

    # Separate editable key: can be placed once beside the architecture figure.
    fig = plt.figure(figsize=(1.05,.55))
    ax = fig.add_axes([.16,.68,.70,.13])
    key_edges = np.linspace(-limit,limit,257)
    ax.pcolormesh(key_edges,[0,1],((key_edges[:-1]+key_edges[1:])/2)[None],
        cmap=cmap,norm=norm,shading='flat',edgecolors='none',antialiased=False,rasterized=False)
    ax.set(yticks=[],xticks=[-limit,0,limit],xlabel='Activation value')
    ax.xaxis.labelpad = 1
    ax.tick_params(axis='x',length=2,pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    export(fig,STEM+'-key')

    for stem,n_cells in [(STEM,activation.size),(STEM+'-inset',activation.size),(STEM+'-key',256)]:
        affinity_export(stem,n_cells)
    np.savez_compressed(OUT/f'{STEM}-data.npz',time_ms=time_ms,
        feature_indices=np.arange(1,257),activation=activation)
    rows=np.column_stack([np.repeat(time_ms,256),np.tile(np.arange(1,257),len(time_ms)),activation.ravel()])
    np.savetxt(OUT/f'{STEM}.csv',rows,delimiter=',',header='time_ms,feature_index_1based,activation',
        comments='',fmt=['%.1f','%d','%.9g'])
    metadata.update(colour_limits=[-limit,limit],colour_mapping='One linear diverging scale for all raw activations; zero is white.',
        colour_clipping=False,labelled_plot_size_inches=[PLOT_WIDTH_IN,PLOT_HEIGHT_IN],
        inset_size_inches=[INSET_WIDTH_IN,INSET_HEIGHT_IN],trim=True,spine_offset_points=SPINE_OFFSET_PT,
        outputs={p.name:digest(p) for p in sorted(OUT.glob(STEM+'*')) if p.suffix in {'.svg','.pdf','.png','.csv','.npz'}})
    (OUT/f'{STEM}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (OUT/f'{STEM}.md').write_text(
        '# Temporal features after residual block 2\n\n'
        'A replacement for the intermediate dot plot in the encoder architecture diagram. '
        'Place after the second residual addition (+), before block 3; label “Temporal features (block 2/4)”. '
        'The original EMG and final-vector figures are unchanged.\n\n'
        'Each column contains all 256 activations at one output time. Columns run from −160 to 0 ms '
        'in 20 ms steps; rows retain the model’s original feature order. These are the nine block-2 '
        'outputs supported by a full 320 ms recorded input history within the same illustrated 480 ms '
        'EMG segment. The eight earlier outputs that depend on startup padding are excluded. '
        'The final column is exactly the previously plotted block-2 vector. This remains a recorded '
        'DPose participant-1/electrode-1 training example for illustration, not an evaluation result.\n\n'
        'Colours show signed raw activations on one shared linear scale, before the final LayerNorm. '
        'There is no per-feature normalization, sorting, interpolation, smoothing, or clipping. '
        'Use the separate colour key when the figure must be interpreted quantitatively. '
        'This is an activation map, not an attention map or an electrode-layout map.\n\n'
        'Files: `residual-block2-featuremap-affinity.svg` (labelled), '
        '`residual-block2-featuremap-inset-affinity.svg` (clean insert), '
        '`residual-block2-featuremap-key-affinity.svg` (colour key). '
        'All cells are editable vector paths; PDFs have outlined text.\n\n'
        'Reproduce: `uv run python -m paper.figures.plot_residual_featuremap`.\n')
    print(json.dumps({k:metadata[k] for k in ['activation_shape_time_by_feature',
        'activation_min','activation_max','colour_limits','final_feature_vector_max_abs_difference']},indent=2))


if __name__=='__main__':
    main()
