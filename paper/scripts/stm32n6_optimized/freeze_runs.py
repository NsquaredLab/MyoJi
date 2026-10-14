"""Freeze matching hardware runs for plotting, retaining failed validations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build import OUT, ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--seed', type=int, default=260923)
    parser.add_argument('--elf-sha256', help='Defaults to the current build receipt')
    parser.add_argument('--output', type=Path, default=OUT / 'measurement_manifest.json')
    args = parser.parse_args()
    elf = args.elf_sha256 or json.loads((OUT / 'build.json').read_text())['elf_sha256']
    latest = {}
    attempts = []
    for folder in sorted((OUT / 'runs').iterdir()):
        config_path = folder / 'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get('plan_only') or any(config.get(key) != value for key, value in {
            'elf_sha256': elf, 'frames': args.frames,
            'warmup': args.warmup, 'seed': args.seed,
        }.items()):
            continue
        relative = str(folder.relative_to(ROOT))
        receipt_path = folder / 'receipt.json'
        complete = receipt_path.exists() and (folder / 'frames.csv').exists()
        verified = json.loads(receipt_path.read_text())['output_verified'] if complete else None
        attempts.append({'path': relative, 'complete': complete, 'output_verified': verified})
        # Selection is chronological, never based on successful validation.
        latest[(config['route'], config['channels'])] = (folder, complete)
    if not latest:
        raise RuntimeError('No runs match this firmware, frame count, warmup and seed')
    incomplete = [str(folder) for folder, complete in latest.values() if not complete]
    if incomplete:
        raise RuntimeError('Latest attempted configurations are incomplete: ' + ', '.join(incomplete))
    runs = [{'path': str(folder.relative_to(ROOT)),
             'receipt_sha256': sha(folder / 'receipt.json'),
             'frames_sha256': sha(folder / 'frames.csv')}
            for _, (folder, _) in sorted(latest.items())]
    manifest = {'schema': 1, 'frames_per_run': args.frames, 'warmup_frames': args.warmup,
                'seed': args.seed, 'board': 'NUCLEO-N657X0-Q', 'cpu_hz': 600000000,
                'elf_sha256': elf,
                'selection': 'Latest matching attempt per route/channel count, regardless of validation outcome.',
                'runs': runs, 'attempts': attempts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Frozen {len(runs)} configurations from {len(attempts)} attempts: {args.output}')


if __name__ == '__main__':
    main()
