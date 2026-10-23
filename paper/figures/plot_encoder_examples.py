"""Recorded EMG and its actual pre-quantization feature vector for figure insets.

Select the first channel and first endpoint of the existing verified replay
fixture by index. No fitting, smoothing, rescaling of learned features, or
selection by model error. Outputs are separate editable SVG/PDF/PNG assets.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import NullLocator
import numpy as np
import seaborn as sns
import torch

from myoji.learned.stream import Tokenizer
from myoji.preprocessing import calibration_normalize
from paper.common import paths

ROOT = Path(__file__).resolve().parents[1]
OUT = paths.OUTPUT/'figures/encoder_examples'
FIXTURE = ROOT/'results/raw_tokenizers_cortex_m4_v1'
STYLE = ROOT/'common/simpetru.mplstyle'  # figstyle's `simpetru` sheet
CHANNEL, ENDPOINT_INDEX = 0, 0
PLOT_WIDTH_IN = 1.05
PLOT_HEIGHT_IN = 1.4
INSET_WIDTH_IN = .425
INSET_HEIGHT_IN = 1.4
SPINE_OFFSET_PT = 5


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rel(path: Path) -> str:
    """Source-hash key: relative to paper/, else to the repository (library), else to the asset bundle."""
    for base in (ROOT, paths.REPO, paths.ASSETS):
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    raise ValueError(f'{path} is outside paper/, the repository and the asset bundle')


@torch.inference_mode()
def extract():
    torch.set_num_threads(2)
    receipt = json.loads((FIXTURE/'fixture_receipt.json').read_text())
    checkpoint = paths.LEARNED_TOKENIZER
    raw_path = FIXTURE/'raw_320x2120.f32'
    reference_path = FIXTURE/'reference.npz'
    assert digest(raw_path) == receipt['raw_fixture_sha256']
    assert digest(reference_path) == receipt['reference_sha256']
    assert digest(checkpoint) == receipt['learned_checkpoint_sha256']
    source = receipt['source']
    raw = np.fromfile(raw_path, dtype='<f4').reshape(320, 2120)
    reference = np.load(reference_path)
    tokenizer = Tokenizer(checkpoint, device='cpu')
    config = tokenizer.config
    endpoint = int(receipt['endpoints'][ENDPOINT_INDEX])
    start = endpoint - config.effective_receptive_samples + 1
    assert start >= tokenizer.calibration_samples
    emg = torch.from_numpy(raw[CHANNEL:CHANNEL+1].copy())[None]
    normalized, center, scale = calibration_normalize(emg, tokenizer.calibration_samples)
    np.testing.assert_array_equal(normalized.numpy()[0, 0], reference['normalized'][CHANNEL])
    segment = normalized[..., start:endpoint+1]
    # Use the actual complete model forward. Geometry is unused for primitive
    # channel encoding/quantization; the single descriptor only satisfies shape.
    model = tokenizer.model
    output = model(segment, sensor='hd',
        channel_descriptors=model.sensor_positions('hd')[CHANNEL:CHANNEL+1],
        include_spatial=False)
    h = output.continuous[0, 0, -1].numpy().copy()
    token = int(output.tokens[0, 0, -1])
    saved_h = reference['learned_features'][ENDPOINT_INDEX, CHANNEL]
    difference = float(np.max(np.abs(h-saved_h)))
    np.testing.assert_allclose(h, saved_h, rtol=2e-5, atol=2e-5)
    assert token == int(reference['learned_ids'][ENDPOINT_INDEX, CHANNEL])
    book = model.quantizer.codebook.detach().numpy()
    # Independent float64 squared-distance calculation must recover the same ID.
    distances = ((book.astype(np.float64)-h.astype(np.float64))**2).sum(axis=1)
    assert int(distances.argmin()) == token
    waveform = segment[0, 0].numpy().copy()
    assert len(waveform) == 960 and h.shape == (256,)
    assert np.isfinite(waveform).all() and np.isfinite(h).all()
    # Time is elapsed since the first sample of the 480 ms support interval.
    # Sample centers run 0..479.5 ms; the interval's right boundary is 480 ms.
    time_ms = np.arange(len(waveform), dtype=float)*1000/config.sample_rate_hz
    metadata = dict(
        complete=True, dataset=source['dataset'], participant=source['participant'],
        source_split=source['source_split'], channel_zero_based=CHANNEL,
        channel_display=CHANNEL+1, task=source['task'], cached_window=source['cached_window'],
        fixture_endpoint_index=ENDPOINT_INDEX, fixture_sample_start=start,
        fixture_sample_endpoint_inclusive=endpoint,
        recording_sample_start=source['recording_start_sample']+start,
        recording_sample_endpoint_inclusive=source['recording_start_sample']+endpoint,
        selection='First channel and first saved replay endpoint by index; no outcome-based selection.',
        input_sample_interval_ms=1000/config.sample_rate_hz,
        plotted_sample_count=len(waveform), plotted_support_ms=len(waveform)*1000/config.sample_rate_hz,
        plotted_sample_times_ms=[float(time_ms[0]), float(time_ms[-1])],
        calibration_samples=tokenizer.calibration_samples, calibration_ms=500,
        normalization_center=float(center.item()), normalization_scale=float(scale.item()),
        normalization='Prefix lower median; lower MAD times 1.4826; scale floor 1e-4; clip at ±12.',
        plotted_clipped_samples=int(np.count_nonzero(np.abs(waveform)==12)),
        feature_stage='After final pre_quantizer_norm, before vector quantization.',
        feature_dimension=256, feature_index_order='Original model order; labels are 1-based.',
        token_id=token, saved_vector_max_absolute_difference=difference,
        saved_vector_agreement=True, saved_token_agreement=True,
        independent_float64_nearest_code_agreement=True,
        feature_min=float(h.min()), feature_max=float(h.max()),
        new_training=False, synthetic_data=False, smoothing=False,
        feature_rescaling=False, geometry_used_for_feature_vector=False,
        torch_version=str(torch.__version__), device='cpu', precision='float32',
        patch_encoder_source=dict(path=inspect.getfile(type(model.frontend.channel_encoder)),
            sha256=digest(Path(inspect.getfile(type(model.frontend.channel_encoder))))),
        sources={rel(p):digest(p) for p in [checkpoint, raw_path,
            reference_path, FIXTURE/'fixture_receipt.json',
            paths.REPO/'src/myoji/learned/encoder.py',
            paths.REPO/'src/myoji/learned/quantizer.py',
            paths.REPO/'src/myoji/preprocessing.py', Path(__file__)]},
    )
    return time_ms, waveform, h, metadata


def export(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.axison:
            continue
        for label in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            if label.get_visible() and label.get_text():
                box = label.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, label.get_text()
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(OUT/f'{stem}.{ext}', transparent=True, bbox_inches=None, dpi=300)
    plt.close(fig)


def standard_axes(ylabel, xlabel):
    fig = plt.figure(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))
    # Half the previous vertical height; keep room for readable bottom labels.
    ax = fig.add_axes([.43, .40, .49, .50])
    ax.set(xlabel=xlabel.replace(' ', '\n', 1), ylabel=ylabel)
    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 2
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(length=2, pad=1)
    ax.axvline(0, color='#d4d4d4', lw=.45, zorder=0)
    return fig, ax


def apply_style():
    plt.style.use(STYLE)
    # These are copyable inserts for the user's sans-serif architecture diagram.
    # Retain their figstyle colors, thin spines, vector output and editable text.
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':8, 'axes.labelsize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'text.usetex':False,
        'savefig.bbox':None, 'savefig.pad_inches':0})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    time_ms, waveform, features, metadata = extract()
    apply_style()

    limit = float(np.ceil(max(abs(waveform.min()), abs(waveform.max()))))
    fig, ax = standard_axes('Time (ms)', 'Normalized EMG')
    ax.plot(waveform, time_ms, color='#47494c', lw=.6)
    ax.set(xlim=(-limit, limit), ylim=(480, 0), xticks=[-limit, 0, limit], yticks=[0, 240, 480])
    sns.despine(ax=ax, trim=True, offset=SPINE_OFFSET_PT)
    export(fig, 'normalized-emg-channel')

    fig, ax = standard_axes('Feature index', 'Feature value')
    indices = np.arange(1, 257)
    ax.scatter(features, indices, s=4, color='#5c618e', linewidths=0, zorder=2)
    lower = float(5*np.floor(features.min()/5)); upper = float(5*np.ceil(features.max()/5))
    ax.set(xlim=(lower-1, upper+1), ylim=(261, -4), xticks=[lower, 0, upper], yticks=[1, 128, 256])
    sns.despine(ax=ax, trim=True, offset=SPINE_OFFSET_PT)
    export(fig, 'prequantization-vector')

    fig = plt.figure(figsize=(INSET_WIDTH_IN, INSET_HEIGHT_IN))
    ax = fig.add_axes([.01, .04, .98, .92])
    ax.axvline(0, color='#939393', lw=.5, zorder=3)
    ax.scatter(features, indices, s=4, color='#5c618e', linewidths=0, zorder=2)
    ax.set(xlim=(lower-1, upper+1), ylim=(261, -4))
    ax.axis('off')
    export(fig, 'prequantization-vector-inset')

    fig = plt.figure(figsize=(INSET_WIDTH_IN, INSET_HEIGHT_IN))
    ax = fig.add_axes([.01, .04, .98, .92])
    ax.plot(waveform, time_ms, color='#47494c', lw=.65)
    ax.set(xlim=(-limit, limit), ylim=(480, 0))
    ax.axis('off')
    export(fig, 'normalized-emg-inset')

    # Every one of the 256 values becomes one vector rectangle. This is a
    # 1-D feature strip, not an electrode map, time axis or schematic random data.
    color_limit = float(np.ceil(np.max(np.abs(features))))
    cmap = LinearSegmentedColormap.from_list('signed_feature', ['#2b2e4c', '#ffffff', '#8b0020'])
    norm = Normalize(-color_limit, color_limit)
    fig = plt.figure(figsize=(3.5, .25))
    ax = fig.add_axes([.01, .09, .98, .82])
    ax.pcolormesh(np.arange(257), [0, 1], features[None], cmap=cmap, norm=norm,
                  shading='flat', rasterized=False, edgecolors='none')
    ax.set(xlim=(0, 256), ylim=(0, 1))
    ax.axis('off')
    export(fig, 'prequantization-vector-strip')

    fig = plt.figure(figsize=(3.5, .7))
    ax = fig.add_axes([.05, .60, .90, .20])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.solids.set_rasterized(False)
    cb.set_ticks([-color_limit, 0, color_limit])
    cb.set_label('Feature value', labelpad=2)
    cb.outline.set_linewidth(.5)
    export(fig, 'prequantization-vector-key')

    np.savez_compressed(OUT/'example-data.npz', time_ms=time_ms, normalized_emg=waveform,
        feature_indices=indices, prequantization_vector=features,
        calibration_center=metadata['normalization_center'], calibration_scale=metadata['normalization_scale'])
    np.savetxt(OUT/'normalized-emg.csv', np.column_stack([time_ms, waveform]),
        delimiter=',', header='time_ms,normalized_emg', comments='')
    np.savetxt(OUT/'prequantization-vector.csv', np.column_stack([indices, features]),
        delimiter=',', header='feature_index_1based,value', comments='', fmt=['%d','%.9g'])
    metadata.update(feature_strip_color_limits=[-color_limit, color_limit], color_clipping=False,
        waveform_axis_limits=[-limit, limit], vector_axis_limits=[lower-1, upper+1],
        vector_plot='256 dots showing signed feature values, in original feature order.',
        labelled_plot_size_inches=[PLOT_WIDTH_IN, PLOT_HEIGHT_IN],
        inset_size_inches=[INSET_WIDTH_IN, INSET_HEIGHT_IN],
        orientation='Vertical: time and feature index increase from top to bottom; values run horizontally.',
        spine_style=dict(function='seaborn.despine', trim=True, offset_points=SPINE_OFFSET_PT),
        outputs={p.name:digest(p) for p in sorted(OUT.iterdir())
            if p.suffix in {'.svg','.pdf','.png','.csv','.npz'} and '-affinity' not in p.stem})
    (OUT/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    caption=(f"One recorded example from DPose participant {metadata['participant']}, electrode {CHANNEL+1} "
        f"(index {CHANNEL}), task {metadata['task']}, cached training window {metadata['cached_window']}. "
        "The waveform shows the complete 480 ms input support for one encoder output; the last 160 ms is its current patch. "
        "Time is relative to the first plotted sample (sample centers 0–479.5 ms). Normalization uses a preceding, separate "
        "500 ms prefix: subtract its lower median, divide by 1.4826 times its lower MAD with scale floor 1e-4, and clip to ±12. "
        f"No plotted samples are clipped. The 256 values are the actual output after the encoder's final LayerNorm, before "
        f"quantization, in original feature order. Each dot shows one signed feature value. "
        f"This example maps to token {metadata['token_id']}. "
        "In the waveform and scatter plots, time and feature index increase from top to bottom. "
        "The strip uses the same values with a linear, symmetric color scale; it is not a spatial electrode map. "
        "No smoothing, feature rescaling, fitting, or selection by error was performed. This is an illustration, not a held-out performance result.")
    (OUT/'README.md').write_text('# Recorded encoder example\n\n'+caption+'\n\n'
        '## Copy into the architecture diagram\n\n'
        '- `normalized-emg-inset.svg`: transparent waveform without axes.\n'
        '- `prequantization-vector-inset.svg`: 256 feature-value dots without axes.\n'
        '- `prequantization-vector-strip.svg`: all 256 values, left to right, as editable vector rectangles.\n'
        '- `prequantization-vector-key.svg`: a separate color key for the strip.\n\n'
        '## Separate labelled plots\n\n'
        '- `normalized-emg-channel.svg`: time in ms runs from top to bottom; normalized amplitude runs horizontally.\n'
        '- `prequantization-vector.svg`: feature index runs from top to bottom; actual feature values run horizontally.\n\n'
        'Both labelled plots are 1.05 × 1.4 inches, with the previous width retained and vertical height halved. '
        'Both use `sns.despine(trim=True, offset=5)` with three major ticks per axis. '
        'Their axis-free waveform/scatter inserts are 0.425 × 1.4 inches. '
        'All data values and their order are unchanged.\n\n'
        'Every asset also has PDF and PNG versions. SVG labels remain editable text. '
        'The CSV/NPZ files contain the plotted values; `provenance.json` records source hashes and verification.\n\n'
        'Reproduce from the repository root: `uv run python -m paper.figures.plot_encoder_examples` (requires seaborn).\n')
    print(json.dumps({k:metadata[k] for k in ['participant','channel_zero_based','token_id','plotted_support_ms',
        'feature_dimension','saved_vector_max_absolute_difference','feature_min','feature_max']}, indent=2))


if __name__ == '__main__':
    main()
