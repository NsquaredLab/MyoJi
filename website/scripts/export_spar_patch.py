"""Export SPAR patches and one recorded update for forearm grid 1; no fitting or new inference.

Patches come from the frozen layout code (grid_layout with participant 1's protocol geometry).
The features and tokens are the update the tokenizer figure already shows: participant 1,
validation window 0, frame 49 (electrode 28 gives token 239). Features are standardized;
couplings to a substituted neighbour are structural zeros.

    uv run python website/scripts/export_spar_patch.py
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/dpose_spar_simple_full_v1'
sys.path.insert(0, str(RUN / 'code/scripts'))
from dpose_spar_simple import grid_layout  # noqa: E402

OUT = ROOT / 'website/static/data/spar-patch.json'
grids = json.loads((RUN / 'subject_01/protocol.json').read_text())['geometry']['grids']
assert np.array_equal(grids[0], np.arange(64).reshape(8, 8)), 'forearm grid 1 is electrodes 1-64, row by row'
layout = grid_layout(grids)
with np.load(RUN / 'subject_01/figure-video-inputs.npz', allow_pickle=False) as z:
    mask, features, tokens = z['mask'], z['val_continuous'][0, :, 49], z['val_tokens'][0, :, 49]
assert int(tokens[27]) == 239, 'not the update shown in the tokenizer figure'
assert (features[~mask] == 0).all()

electrodes = []
for ch in range(64):
    members = layout.members[ch]                      # centre, auxiliary, horizontal, vertical
    assert (members < 64).all() and len(set(members.tolist())) == 4
    right, down = bool(layout.right[ch] >= 0), bool(layout.down[ch] >= 0)
    # Per band: centre diagonal, horizontal coupling, vertical coupling.
    assert mask[ch, 0::3].all() and (mask[ch, 1::3] == right).all() and (mask[ch, 2::3] == down).all()
    electrodes.append(dict(members=(members + 1).tolist(), right=right, down=down,
                           features=np.round(features[ch], 2).tolist(), token=int(tokens[ch])))
payload = dict(
    participant=1, rows=8, cols=8,
    bands=['20–40', '40–60', '60–100', '100–160', '160–250', '250–500'],
    electrodes=electrodes,
    provenance='Participant 1, forearm grid 1 (electrodes 1-64), validation window 0, update 49: the '
               'update shown in the tokenizer figure. Standardized SPAR features; tokens are the '
               'nearest of the participant\'s 1,024 codebook entries.')
OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
print(OUT, f'{OUT.stat().st_size / 1e3:.1f} kB', {e: electrodes[e - 1]['members'] for e in (8, 28, 60, 64)})
