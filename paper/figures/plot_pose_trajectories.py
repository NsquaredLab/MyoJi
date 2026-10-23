"""Current DPose models on the original, target-only selected test windows."""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator
from paper.emg2pose import build_pose_paper_assets as data
from paper.common import switzer_font

from paper.common import figstyle as sf

SEED = 260911
MEASURED = '#BDBDBD'   # wide light underlay: the reference greys, separable from every model in greyscale
GRID, GRID_WIDTH = '#DEDEDE', .55
# Line form, not only hue, separates the models, so the figure survives greyscale print.
MODELS = [
    ('SPAR Tokens', 'dpose_spar_simple_full_v1', 'tokens', '#d80032', '-'),
    ('Learned Tokens', 'dpose_v20_v1', 'tokens', '#570014', (0, (4, 1.6))),
    ('Adapted CNN', 'dpose_raulnet_v1', 'raulnet', '#6d6f72', (0, (1, 1.3)))]
STRATA = ['Lower', 'Middle', 'Upper']   # training-defined speed bins


def windows(campaign, arm, subject):
    fit = data.npz(f'results/{campaign}/subject_{subject:02d}/fits/{arm}_{SEED}/test_predictions.npz')
    shape = (-1, 50, 20, 3)
    return fit['target'].astype(float).reshape(shape), fit['prediction'].astype(float).reshape(shape)


def select_examples():
    training, candidates = [], []
    for subject in range(1, 10):
        root = f'results/dpose_matched_v1/subject_{subject:02d}'
        train = data.npz(root+'/baseline_training_targets.npz')['target'].astype(float).reshape(-1, 50, 20, 3)
        training.extend(np.linalg.norm(np.diff(train, axis=1)*50, axis=-1).mean((1, 2)).tolist())
        target, _ = windows('dpose_matched_v1', 'tcn_tokens', subject)
        for w, speed in enumerate(np.linalg.norm(np.diff(target, axis=1)*50, axis=-1).mean((1, 2))):
            candidates.append((float(speed), subject, w))
    thresholds = np.quantile(training, [1/3, 2/3])
    selected = []
    for stratum in range(3):
        # Strata come from training windows; the median is over the test windows inside each stratum.
        pool = sorted(v for v in candidates if np.searchsorted(thresholds, v[0], side='right') == stratum)
        midpoint = np.median([v[0] for v in pool])
        selected.append(min(pool, key=lambda v: (abs(v[0]-midpoint), v[1], v[2])))
    # These examples were selected before the current models were plotted.
    assert [(u, w) for _, u, w in selected] == [(5, 8), (2, 13), (9, 6)]
    return thresholds, selected


def moving_landmark(target):
    return int(np.var(target, axis=0).sum(-1).argmax())


def context(selected):
    """Where each plotted error sits among all test windows of the same model and seed.

    Each window contributes its own greatest-variance landmark, the rule used for the
    plotted examples, so the comparison is like for like.
    """
    out = {}
    for name, campaign, arm, _, _ in MODELS:
        errors, keys = [], []
        for subject in range(1, 10):
            target, prediction = windows(campaign, arm, subject)
            for w in range(len(target)):
                lm = moving_landmark(target[w])
                errors.append(np.linalg.norm(prediction[w, :, lm]-target[w, :, lm], axis=-1).mean())
                keys.append((subject, w))
        errors = np.array(errors)
        picked = [errors[keys.index((u, w))] for _, u, w in selected]
        out[name] = {'windows': len(errors), 'mean_mm': float(errors.mean()),
                     'plotted_mean_mm': [float(v) for v in picked],
                     'plotted_percentile': [float((errors < v).mean()*100) for v in picked]}
    return out


def apply_style():
    plt.style.use(['science', 'ieee', 'simpetru-large'])
    plt.rcParams.update(switzer_font.rcparams())   # after style.use; see common/switzer_font.py
    plt.rcParams.update({'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.labelsize': 9.5, 'axes.labelpad': 5})


def draw():
    apply_style()
    thresholds, selected = select_examples()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.3), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.2, top=.86, wspace=.36, hspace=.16)
    time = np.arange(50)/50
    examples, spans = [], []
    for col, (speed, subject, w) in enumerate(selected):
        reference_target, _ = windows('dpose_matched_v1', 'tcn_tokens', subject)
        target = reference_target[w]
        landmark = moving_landmark(target)
        coordinate = int(np.ptp(target[:, landmark], axis=0).argmax())
        top, bottom = axes[:, col]
        top.plot(time, target[:, landmark, coordinate], color=MEASURED, lw=2.2, solid_capstyle='butt', zorder=1)
        shown = [target[:, landmark, coordinate]]
        model_details = []
        for name, campaign, arm, color, ls in MODELS:
            model_target, prediction = windows(campaign, arm, subject)
            np.testing.assert_array_equal(model_target, reference_target)
            prediction = prediction[w, :, landmark]
            error = np.linalg.norm(prediction-target[:, landmark], axis=-1)
            top.plot(time, prediction[:, coordinate], color=color, ls=ls, lw=1.2, zorder=2)
            bottom.plot(time, error, color=color, ls=ls, lw=1.2, zorder=2)
            shown.append(prediction[:, coordinate])
            model_details.append({'name': name, 'mean_error_mm': float(error.mean()),
                                  'coordinate_mm': prediction[:, coordinate].tolist(),
                                  'selected_landmark_error_mm': error.tolist()})
        spans.append((min(map(np.min, shown)), max(map(np.max, shown))))
        top.set_ylabel(f'Position {"xyz"[coordinate]} (mm)')
        fig.text(axes[0, col].get_position().x0 - .07, .955, 'ABC'[col], fontsize=11, weight='bold', va='top')
        fig.text(axes[0, col].get_position().x0 - .045, .955, 
                 f'{STRATA[col]} Speed Bin\nHand Mean Speed {speed:.1f} mm/s', linespacing=1.3, fontsize=8.5, va='top')
        examples.append({'speed_stratum': STRATA[col], 'participant': subject, 'window_index': w,
                         'target_speed_mm_s': speed,   # mean over all 20 landmarks (header value)
                         'landmark_speed_mm_s': float(np.linalg.norm(np.diff(target[:, landmark], axis=0)*50, axis=-1).mean()),
                         'landmark_number': landmark+1,   # 1-20; stored 0-based index is landmark, wrist point 0 excluded
                         'landmark_stored_index': landmark,
                         'coordinate': 'xyz'[coordinate], 'seed': SEED,
                         'target_coordinate_mm': target[:, landmark, coordinate].tolist(),
                         'models': model_details})
    # Matched range: every position panel spans the same millimetres, centred on
    # its own traces, so the eye compares tracking, not zoom. Ticks share one step.
    step = 25.0
    span = np.ceil(1.06*max(h-l for l, h in spans)/step)*step
    for ax, (l, h) in zip(axes[0], spans):
        lo = (l+h-span)/2
        ax.set_ylim(lo, lo+span); ax.set_yticks([t for t in np.arange(np.ceil(lo/step)*step, lo+span, step) if t-lo > .3*step])
    high = max(ax.get_ylim()[1] for ax in axes[1])
    ticks = MaxNLocator(3).tick_values(0, high)
    for ax in axes[1]:
        ax.set_ylim(0, ticks[-1]); ax.set_yticks(ticks)
    axes[1, 0].set_ylabel('3D Landmark Error (mm)')
    axes[1, 1].set_xlabel('Time Within Scored Second (s)')
    for ax in axes.flat:
        ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1])
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=GRID_WIDTH)
        ax.grid(axis='x', visible=False)
        sf.trim_spines(ax, which=('bottom',), offset=0)
        ax.tick_params(which='both', length=3, top=False, right=False, left=False)
    for ax in axes[1, 1:]:
        ax.tick_params(labelleft=False)
    for ax in axes[0]:   # time is read off the error row; the grid carries position
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
    fig.legend(handles=[Line2D([], [], color=MEASURED, lw=2.2, solid_capstyle='butt', label='Measured')] +
                       [Line2D([], [], color=c, ls=ls, lw=1.2, label=name) for name, _, _, c, ls in MODELS],
               loc='lower center', bbox_to_anchor=(.53, .0), ncol=4, frameon=False,
               columnspacing=1.8, handlelength=2.6)
    assert all(not ax.get_title() for ax in fig.axes) and fig._suptitle is None
    assert not sf.check_font_sizes(fig)
    receipt = {
        'selection': 'Training-window mean-speed tertiles; within each, the test window nearest the median speed of the test windows in that stratum; ties by participant/window. Landmark with greatest target temporal variance; coordinate with greatest target range. Target-only, fixed seed; no alignment, filtering, recentering or ensemble.',
        'thresholds_from_training_mm_s': thresholds.tolist(), 'examples': examples,
        'position_span_mm': float(span), 'position_tick_step_mm': float(step),
        'context_greatest_variance_landmark': context(selected),
        'anatomy': 'DPose millimetres; landmark numbers 1-20 relative to wrist point 0, no inferred finger names.',
        'error': 'Euclidean 3D error of the selected landmark at each scored time; not the aggregate over all 20 landmarks.',
        'new_inference': False}
    (data.OUT / 'pose_trajectories.json').write_text(json.dumps(receipt, indent=1) + '\n')
    data.save(fig, 'pose_trajectories', receipt)


if __name__ == '__main__':
    data.configure(); draw()
