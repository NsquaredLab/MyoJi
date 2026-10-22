"""Export 2 s of recorded SPAR tokens for all 320 electrodes for the hero grid; no fitting or new inference.

Participant 1, validation window 0: the window the tokenizer figure and the SPAR patch come from
(update 49 holds electrode 28's token 239). Activity is each electrode's standardized centre
(diagonal) feature averaged over the six bands, scaled to 0-9 between the 5th and 95th percentiles;
it only shades the cells.

    uv run python website/scripts/export_token_stream.py
"""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/dpose_spar_simple_full_v1'
OUT = ROOT / 'website/static/data/token-stream.json'
with np.load(RUN / 'subject_01/figure-video-inputs.npz', allow_pickle=False) as z:
    tokens, features, mask = z['val_tokens'][0], z['val_continuous'][0], z['mask']
assert tokens.shape == (320, 100) and int(tokens[27, 49]) == 239, 'not the window shown in the tokenizer figure'
patch = json.loads((ROOT / 'website/static/data/spar-patch.json').read_text())
assert tokens[:64, 49].tolist() == [e['token'] for e in patch['electrodes']]
assert mask[:, 0::3].all(), 'every electrode keeps its centre features'

centre = features[:, :, 0::3].mean(-1)                 # (electrode, update)
lo, hi = np.percentile(centre, [5, 95])
activity = np.clip(np.round((centre - lo) / (hi - lo) * 9), 0, 9).astype(int)
payload = dict(
    tokens=tokens.T.tolist(),                          # [update][electrode]
    activity=[''.join(map(str, row)) for row in activity.T],
    provenance='Participant 1, validation window 0: 100 recorded updates (2 s) of SPAR tokens for all '
               '320 electrodes. Activity is the standardized centre band power, 0-9, for shading only.')
OUT.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
print(OUT, f'{OUT.stat().st_size / 1e3:.1f} kB')
