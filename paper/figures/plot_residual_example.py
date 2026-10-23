"""Plot a real halfway encoder activation, matched to the recorded figure example."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from paper.figures.plot_encoder_examples import (
    ROOT, OUT, PLOT_WIDTH_IN, PLOT_HEIGHT_IN, INSET_WIDTH_IN, INSET_HEIGHT_IN,
    SPINE_OFFSET_PT, apply_style, standard_axes, export, rel,
)
from paper.figures.export_encoder_examples_affinity import simple_svg, outline_text, write
from paper.figures.make_encoder_replacement_draft import render
from myoji.learned.stream import Tokenizer
from paper.common import paths

STEM = 'residual-block2-vector'
BLOCK_INDEX = 1  # Zero-based: the second of the four residual blocks.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def extract():
    torch.set_num_threads(2)
    provenance_path = OUT/'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    source = OUT/'example-data.npz'
    checkpoint = paths.LEARNED_TOKENIZER
    assert digest(source) == provenance['outputs'][source.name]
    assert digest(checkpoint) == provenance['sources'][rel(checkpoint)]
    data = np.load(source)
    tokenizer = Tokenizer(checkpoint, device='cpu')
    model = tokenizer.model
    assert not model.training
    assert len(model.temporal_encoder) == 4
    segment = torch.from_numpy(data['normalized_emg'].copy())[None, None, :]
    assert segment.shape == (1, 1, 960)
    captured = []

    def capture_output(module, inputs, output):
        captured.append(output.detach().clone())

    handle = model.temporal_encoder[BLOCK_INDEX].register_forward_hook(capture_output)
    channel = provenance['channel_zero_based']
    try:
        output = model(segment, sensor='hd',
            channel_descriptors=model.sensor_positions('hd')[channel:channel+1],
            include_spatial=False)
    finally:
        handle.remove()
    assert len(captured) == 1 and captured[0].shape == (1, 17, 256)
    intermediate = captured[0][0, -1].cpu().numpy().copy()
    final = output.continuous[0, 0, -1].cpu().numpy()
    np.testing.assert_allclose(final, data['prequantization_vector'], rtol=2e-5, atol=2e-5)
    token_id = int(output.tokens[0,0,-1])
    assert token_id == provenance['token_id']
    assert intermediate.shape == (256,) and np.isfinite(intermediate).all()
    # Two causal blocks add 2 * 4 preceding token steps to the 160 ms patch.
    lookback_samples = sum(block.left_padding for block in model.temporal_encoder[:2]) * tokenizer.config.hop_samples
    receptive_ms = 1000*(tokenizer.config.receptive_samples+lookback_samples)/tokenizer.config.sample_rate_hz
    metadata = dict(
        dataset=provenance['dataset'], participant=provenance['participant'],
        source_split=provenance['source_split'], electrode_one_based=channel+1,
        task=provenance['task'], token_id=token_id,
        recording_sample_endpoint_inclusive=provenance['recording_sample_endpoint_inclusive'],
        block_one_based=2, total_blocks=4,
        hook_module='temporal_encoder.1',
        stage='Output of residual block 2, after its residual addition and before block 3.',
        before_final_layernorm=True, before_quantization=True,
        effective_receptive_ms=receptive_ms,
        feature_dimension=256, feature_order='Original order; displayed indices 1–256 run downward.',
        intermediate_min=float(intermediate.min()), intermediate_max=float(intermediate.max()),
        matching_final_vector_max_abs_difference=float(np.max(np.abs(final-data['prequantization_vector']))),
        matching_token_agreement=True,
        smoothing=False, rescaling=False, new_training=False, synthetic_data=False,
        source_selection='Same fixed electrode and endpoint as the normalized EMG and prequantization plots.',
        sources={rel(p):digest(p) for p in [source, provenance_path, checkpoint,
            paths.REPO/'src/myoji/learned/encoder.py',
            ROOT/'figures/plot_encoder_examples.py',
            ROOT/'figures/export_encoder_examples_affinity.py', Path(__file__)]},
    )
    return intermediate, metadata


def main():
    features, metadata = extract()
    indices = np.arange(1,257)
    apply_style()
    extent = float(np.max(np.abs(features)))
    limit = float(MaxNLocator(nbins=1, steps=[1, 2, 5, 10], min_n_ticks=1)
                  .tick_values(0, extent)[-1])
    bounds = (-1.05*limit, 1.05*limit)
    fig, ax = standard_axes('Feature index', 'Activation value')
    ax.scatter(features, indices, s=4, color='#5c618e', linewidths=0, zorder=2)
    ax.set(xlim=bounds, ylim=(261,-4), xticks=[-limit,0,limit], yticks=[1,128,256])
    sns.despine(ax=ax, trim=True, offset=SPINE_OFFSET_PT)
    export(fig, STEM)

    fig = plt.figure(figsize=(INSET_WIDTH_IN, INSET_HEIGHT_IN))
    ax = fig.add_axes([.01,.04,.98,.92])
    ax.axvline(0, color='#939393', lw=.5, zorder=3)
    ax.scatter(features, indices, s=4, color='#5c618e', linewidths=0, zorder=2)
    ax.set(xlim=bounds, ylim=(261,-4))
    ax.axis('off')
    export(fig, STEM+'-inset')

    for stem in [STEM, STEM+'-inset']:
        tree, coordinates = simple_svg(OUT/f'{stem}.svg')
        target = OUT/f'{stem}-affinity.svg'
        write(tree, target)
        outlined = OUT/f'{stem}-affinity-outlined.svg'
        write(outline_text(tree), outlined)
        width, height = map(float, tree.get('viewBox').split()[2:])
        render(outlined, width=width*96/72, height=height*96/72)
        for ext in ['.pdf','.png']:
            outlined.with_suffix(ext).replace(target.with_suffix(ext))
        points = ET.parse(target).getroot().findall('.//{http://www.w3.org/2000/svg}circle')
        np.testing.assert_array_equal(coordinates, [[float(p.get('cx')),float(p.get('cy'))] for p in points])
        assert len(points)==256
        assert np.all(np.diff(coordinates[:,1])>0)

    np.savetxt(OUT/f'{STEM}.csv', np.column_stack([indices,features]),
        delimiter=',', header='feature_index_1based,activation', comments='', fmt=['%d','%.9g'])
    np.savez_compressed(OUT/f'{STEM}-data.npz', feature_indices=indices, activation=features)
    metadata.update(
        labelled_plot_size_inches=[PLOT_WIDTH_IN,PLOT_HEIGHT_IN],
        inset_size_inches=[INSET_WIDTH_IN,INSET_HEIGHT_IN],
        activation_axis_limits=list(bounds), trim=True, spine_offset_points=SPINE_OFFSET_PT,
        outputs={p.name:digest(p) for p in sorted(OUT.glob(STEM+'*')) if p.suffix in {'.svg','.pdf','.png','.csv','.npz'}},
    )
    (OUT/f'{STEM}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (OUT/f'{STEM}.md').write_text(
        '# Intermediate features after residual block 2\n\n'
        'Place this example **after the residual addition (+) of block 2 of 4**, before block 3. '
        'For a single diagram block marked “×4”, call the inset “Intermediate features (block 2/4)” '
        'so it is not mistaken for an operation inside every block.\n\n'
        'The dots are the actual 256 output activations of `temporal_encoder.1`, captured by a forward hook '
        'from the frozen released tokenizer. They use the same HD-EMG subject 1, electrode 1, '
        'and final time point as the normalized EMG and prequantization examples. '
        'These activations are **before the final LayerNorm and quantization**; their magnitudes '
        'are plotted directly, without rescaling or smoothing. The block-2 output has 320 ms '
        'of effective input context. The full encoder still reproduces the saved final vector and token 93.\n\n'
        'The vertical axis is feature index (top to bottom), not time or physical electrode position. '
        'Use `residual-block2-vector-affinity.svg` for the labelled plot, or '
        '`residual-block2-vector-inset-affinity.svg` for the clean insert. '
        'Their size, dot colour, zero line and spine settings match the existing examples. '
        'The activation axis has its own scale because this representation has not passed the final LayerNorm.\n\n'
        'Reproduce: `uv run python -m paper.figures.plot_residual_example`.\n'
    )
    print(json.dumps({k:metadata[k] for k in ['stage','feature_dimension','effective_receptive_ms',
        'intermediate_min','intermediate_max','matching_final_vector_max_abs_difference','token_id']},indent=2))


if __name__=='__main__':
    main()
