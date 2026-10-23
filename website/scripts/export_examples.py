"""Export two existing manuscript examples; no fitting or new inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'website/static/data'


def example(vector, book, expected, **metadata):
    distances = ((book.astype('float64') - vector.astype('float64')) ** 2).sum(1)
    nearest = np.argsort(distances, kind='stable')[:8]
    assert int(nearest[0]) == expected
    ids = sorted(nearest.tolist())
    return dict(**metadata, token=expected, bits=format(expected, '010b'),
                dimensions=len(vector), vector=vector.tolist(),
                candidates=[dict(id=i, distance=float(distances[i]), vector=book[i].tolist()) for i in ids],
                ranking=nearest.tolist(), codebook_size=len(book),
                distance='squared Euclidean; all feature dimensions and all 1,024 codes')


paths = [ROOT/'manuscript/figures/encoder_examples/example-data.npz',
         ROOT/'checkpoints/clean_release.pt',
         ROOT/'results/hdemg_spar_simple_full_v1/subject_01/figure-video-inputs.npz']
with np.load(paths[0], allow_pickle=False) as learned:
    book = torch.load(paths[1], map_location='cpu', weights_only=True)['state_dict']['quantizer.codebook'].numpy()
    # Min/max pairs preserve peaks in each four-sample display bin.
    signal = learned['normalized_emg'].reshape(-1, 4)
    trace = np.column_stack((signal.min(1), signal.max(1))).reshape(-1).tolist()
    a = example(learned['prequantization_vector'], book, 93,
                name='Learned encoder', waveform=trace,
                provenance='HD-EMG participant 1, electrode 1, task 2, training window 2. The recorded example used in the manuscript encoder figure.',
                context_ms=480, patch_ms=160, update_ms=20)
with np.load(paths[2], allow_pickle=False) as spar:
    assert spar['mask'][27].all()
    b = example(spar['val_continuous'][0, 27, 49], spar['codebook'], 239,
                name='SPAR', waveform=None,
                provenance='HD-EMG participant 1, electrode 28 (zero-based index 27), validation window 0, frame 49. The recorded example used in the manuscript mechanism assets.',
                context_ms=160, patch_ms=160, update_ms=20)
payload = dict(learned=a, spar=b)
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'examples.json').write_text(json.dumps(payload, separators=(',', ':')) + '\n')
(OUT/'provenance.json').write_text(json.dumps(dict(
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    checked_codes_per_example=1024,
    synthetic_values=False,
    scene='The 2D flow illustrates processing order. The eight nearest candidates are displayed in distance order. SPAR input traces are schematic; numerical vectors, distances and IDs are recorded.',
    waveform_display='Min/max pairs per four input samples; 480 ms support; no smoothing.',
    feature_display='All dimensions in original order; grayscale symmetric around zero, independently scaled per example.'
), indent=2) + '\n')
print('Exported verified examples: learned → 93; SPAR → 239.')
