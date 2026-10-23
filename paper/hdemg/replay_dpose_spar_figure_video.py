"""Benchmark and replay the two full-cohort SPAR checkpoints; never fit a model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import h5py
import joblib
import numpy as np
from scipy.signal import firwin, upfirdn
import torch

from paper.common import paths
from paper.hdemg.run_dpose_spar_simple import SparPose, normalize_raw, ENDS, sha, write
from myoji.spar import grid_layout, spar_simple_dataset_batched
from myoji.codebook import standardize, tokenize
from paper.hdemg.dpose_one_euro import apply
from paper.hdemg.benchmark_dpose_all_variants import measure, storage_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=paths.RESULTS/'dpose_spar_simple_full_v1/subject_01')
    p.add_argument('--previous-video', type=Path, default=paths.RESULTS/'dpose_movement_video')
    p.add_argument('--output', type=Path, required=True,
                   help="the campaign root results/dpose_spar_simple_full_v1 with --benchmark-only, "
                        "or results/dpose_movement_video_spar for the movement replay")
    p.add_argument('--benchmark-only', action='store_true')
    p.add_argument('--refresh-benchmark', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(260911)
    assert torch.cuda.device_count() == 1
    source = np.load(a.run/'figure-video-inputs.npz')
    original = json.loads((a.previous_video/'inference-receipt.json').read_text())
    reference = json.loads((a.previous_video/'model-reference.json').read_text())
    protocol = json.loads((a.run/'protocol.json').read_text())
    assert original['complete'] and original['seed'] == 260911
    assert sha(a.previous_video/'movement-predictions.npz') == original['prediction_sha256']
    book, mask = source['codebook'], source['mask']
    benchmark = dict(complete=False, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        precision='FP32; TF32 disabled', batch_size=1, input_indices=[0, 1, 2, 3],
        input_split='validation', cpu_threads=4, source_sha256=sha(a.run/'figure-video-inputs.npz'),
        scope='Decoder only, full context recomputation, final pose; excludes SPAR extraction, standardization, quantization, transfers and target denormalization.', models=[])
    models = []
    cost_path = a.output/'resource-benchmark.json'
    if a.refresh_benchmark and cost_path.exists():
        archived = a.output/'resource-benchmark-before-input-cast-alignment.json'
        if not archived.exists(): archived.write_bytes(cost_path.read_bytes())
    previous_cost = json.loads(cost_path.read_text()) if cost_path.exists() and not a.refresh_benchmark else None
    for arm in ['tokens', 'continuous']:
        path = a.run/'fits'/f'{arm}_260911/best.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = SparPose(torch.from_numpy(book), torch.from_numpy(mask))
        model.load_state_dict(checkpoint['model'], strict=True)
        np.testing.assert_array_equal(model.decoder.codebook.numpy(), book)
        np.testing.assert_array_equal(model.mask.numpy(), mask)
        model.eval().requires_grad_(False).to('cuda')
        x = torch.from_numpy(source['val_'+arm]).to('cuda')
        if previous_cost:
            assert previous_cost['complete'] and previous_cost['source_sha256']==benchmark['source_sha256']
            record, = [r for r in previous_cost['models'] if r['arm']==arm]
            assert record['checkpoint_sha256']==sha(path)
        else:
            # Match the earlier benchmark: stored uint16 receiver inputs are
            # converted to lookup indices inside the measured decoder call.
            timing = measure(lambda i: model(x[i:i+1].long() if arm=='tokens' else x[i:i+1].float(), arm)[:, -1])
            record = dict(arm=arm, checkpoint_sha256=sha(path), **storage_record(model, {}), **timing)
            record['latency_mean_ms'] = float(np.mean(timing['samples_ms']))
            record['latency_sd_ms'] = float(np.std(timing['samples_ms'], ddof=1))
        benchmark['models'].append(record)
        models.append((arm, model, checkpoint, record))
        print(json.dumps({k:record[k] for k in ['arm','parameter_count','latency_mean_ms','latency_sd_ms']}), flush=True)
    benchmark['complete'] = True
    benchmark['token_index_cast_included'] = True
    benchmark['benchmark_source_sha256'] = sha(__file__)
    write(a.output/'resource-benchmark.json', benchmark)
    if a.benchmark_only: return

    # Recreate only SPAR inputs for the exact previously selected decoder windows.
    # Their prefix normalization and target indices match the full-cohort run.
    t0 = time.monotonic()
    windows = original['windows']
    layout = grid_layout(protocol['geometry']['grids'])
    indices = ENDS[:, None] - np.arange(319, -1, -1)[None]
    emg = joblib.load(paths.HDEMG/'Sub1_emg.pkl')
    kin = joblib.load(paths.HDEMG/'Sub1_kinematics.pkl')
    for suffix in ['emg', 'kinematics']:
        name = f'Sub1_{suffix}.pkl'
        assert sha(paths.HDEMG/name) == protocol['data_sha256'][name]
    taps = firwin(10241, 800, fs=256000, window=('kaiser', 8.6)) * 125
    continuous = np.empty((len(windows), 320, 100, 18), np.float32)
    targets = np.empty((len(windows), 50, 60), np.float32)
    for task in sorted({w['task'] for w in windows}):
        raw = emg[str(task)]
        signal = upfirdn(taps, raw.astype(np.float32), up=125, down=128, axis=-1)[..., :raw.shape[-1]*125//128].astype(np.float32)
        for i, w in enumerate(windows):
            if w['task'] != task: continue
            start = w['start']
            norm = normalize_raw(signal[:, start:start+6000])
            value = spar_simple_dataset_batched(norm[:, indices].transpose(1, 2, 0).copy(),
                layout=layout, channelwise=True, batch_size=16, device='cuda').transpose(1, 0, 2)
            continuous[i] = standardize(value[None], source['feature_mean'], source['feature_scale'], mask)[0]
            points = kin[str(task)][..., ((start+ENDS)*128)//125].transpose(2, 0, 1).astype(np.float32)
            targets[i] = (points[:, 1:]-points[:, :1]).reshape(100, 60)[-50:]
        print(f'SPAR movement {task}: {time.monotonic()-t0:.1f}s', flush=True)
    tokens = tokenize(continuous, book, device='cuda')
    with h5py.File(a.previous_video/'movement-inputs.h5', 'r') as f:
        np.testing.assert_array_equal(targets, f['truth'][:])
    for i in range(2):
        assert windows[i]['task'] == int(source['test_tasks'][i])
        assert windows[i]['start'] == int(source['test_starts'][i])
    np.testing.assert_array_equal(targets[:2], source['test_targets'][:, -50:])
    inputs_audit = dict(token_id_differences=int(np.count_nonzero(tokens[:2] != source['test_tokens'])),
        feature_max_abs_difference=float(np.max(abs(continuous[:2]-source['test_continuous']))),
        target_identity_verified=True)
    # Cross-device float32 eigensolver differences are measured, never hidden.
    assert inputs_audit['token_id_differences'] / tokens[:2].size < .005
    assert inputs_audit['feature_max_abs_difference'] < .02
    outputs = dict(np.load(a.previous_video/'movement-predictions.npz'))
    receipt = copy.deepcopy(original)
    receipt.update(complete=False, original_receipt_sha256=sha(a.previous_video/'inference-receipt.json'),
        spar_input_audit=inputs_audit, spar_replay_code_sha256=sha(__file__), training_performed=False,
        spar_full_subject_protocol_sha256=sha(a.run/'protocol.json'))
    for arm, model, checkpoint, resource in models:
        mi = len(reference['models'])
        values = torch.from_numpy(tokens.astype(np.int64) if arm == 'tokens' else continuous).to('cuda')
        with torch.inference_mode():
            prediction = torch.cat([model(values[i:i+2], arm).cpu() for i in range(0, len(values), 2)])
        prediction = (prediction * checkpoint['scale'] + checkpoint['mean'])[:, -50:].numpy()
        expected = np.load(a.run/'fits'/f'{arm}_260911/test_predictions.npz')['prediction'][:2]
        delta = prediction[:2] - expected
        audit = dict(id=f'dpose_spar_simple_full_v1/{arm}', checkpoint_identity_verified=True,
            source_target_identity_verified=True, max_coordinate_difference_mm=float(abs(delta).max()),
            mean_3d_difference_mm=float(np.linalg.norm(delta.reshape(-1,20,3),axis=-1).mean()))
        audit['passed'] = audit['max_coordinate_difference_mm'] < 1 and audit['mean_3d_difference_mm'] < .1
        # Keep the original diagnostic, including a failed maximum-coordinate
        # check caused by cross-device descriptor/nearest-code differences.
        # Independently gate the decoder replay on the canonical cached inputs.
        canonical = torch.from_numpy(source['test_'+arm]).to('cuda')
        if arm=='tokens': canonical=canonical.long()
        with torch.inference_mode():
            canonical_prediction = (model(canonical,arm).cpu()*checkpoint['scale']+checkpoint['mean'])[:,-50:].numpy()
        same_input_delta = canonical_prediction-expected
        audit['canonical_input_max_coordinate_difference_mm'] = float(abs(same_input_delta).max())
        audit['canonical_input_mean_3d_difference_mm'] = float(np.linalg.norm(same_input_delta.reshape(-1,20,3),axis=-1).mean())
        audit['canonical_input_replay_passed'] = (audit['canonical_input_max_coordinate_difference_mm']<1
            and audit['canonical_input_mean_3d_difference_mm']<.1)
        assert audit['canonical_input_replay_passed'], audit
        for segment in receipt['selection']['segments']:
            raw = prediction[segment['window_indices']].reshape(-1, 60)
            filtered = apply(raw, reference['smoothing'])
            lo = segment['warmup_frames']; hi = lo+segment['display_frames']
            outputs[f'model_{mi}_task_{segment["task"]}_raw'] = raw[lo:hi]
            outputs[f'model_{mi}_task_{segment["task"]}_filtered'] = filtered[lo:hi].astype(np.float32)
        m = copy.deepcopy(reference['models'][9 if arm=='tokens' else 10])
        m.update(id=audit['id'], campaign='dpose_spar_simple_full_v1', arm=arm, fit_arm=arm, family='spar',
            name='Proposed spatial CNN + TCN · SPAR '+('tokens' if arm=='tokens' else 'features'),
            short_name='Spatial CNN + TCN', representation=arm,
            parameter_count=resource['parameter_count'],
            representation_label='SPAR '+('tokens' if arm=='tokens' else 'features'),
            checkpoint_sha256=resource['checkpoint_sha256'])
        reference['models'].append(m)
        receipt['models'].append(dict(m, loaded_state_unchanged=True))
        receipt['audit'].append(audit)
        np.savez_compressed(a.output/f'model-{mi:02d}-predictions.npz', prediction=prediction)
        print(json.dumps(audit), flush=True)
    reference['models'][9]['representation_label'] = 'Learned tokens'
    reference['models'][10]['representation_label'] = 'Learned features'
    np.savez_compressed(a.output/'movement-predictions.npz', **outputs)
    write(a.output/'model-reference.json', reference)
    receipt.update(complete=True, prediction_sha256=sha(a.output/'movement-predictions.npz'),
        model_reference_sha256=sha(a.output/'model-reference.json'), spar_replay_seconds=time.monotonic()-t0)
    write(a.output/'inference-receipt.json', receipt)
    print('Finished full-cohort SPAR benchmark and movement replay.', flush=True)


if __name__ == '__main__':
    main()
