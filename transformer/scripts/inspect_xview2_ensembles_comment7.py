#!/usr/bin/env python3
"""Collect an xView2 winner inference-code inventory WITHOUT opening checkpoint data.

Purpose: prepare a fair, complete-ensemble throughput/GPU-memory benchmark.
Runs on the LOGIN NODE (no GPU or ML libraries needed).
Does not execute arbitrary project code or load any model weights.

Usage:
  python3 transformer/scripts/inspect_xview2_ensembles_comment7.py
  python3 transformer/scripts/inspect_xview2_ensembles_comment7.py --root /path/to/DisasterAdaptiveNet

Produces:
  output/Reviewer2_Comment7_SourceInspection/ensemble_inventory.txt
  output/Reviewer2_Comment7_SourceInspection/ensemble_source_bundle.zip

Review the bundle before sharing. It contains selected .py/config/README/source files,
not model weights, datasets, prediction outputs, git metadata or logs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from pathlib import Path

WINNERS = {
    'first': 'xview2_first_place',
    'second': 'xview2_second_place',
    'third': 'xview2_third_place',
}
SRC_EXTS = {'.py', '.sh', '.sbatch', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.md', '.txt', '.json'}
SKIP_DIRS = {
    '.git', '.hg', '.svn', '__pycache__', '.venv', 'venv', 'env', 'logs',
    'oof_preds', 'predictions', 'preds', 'data', 'datasets', 'images',
    'output', 'outputs', 'checkpoints', 'weights', 'models_weights',
    'node_modules', '.mypy_cache', '.pytest_cache', 'wandb',
}
SKIP_FILENAMES = {
    '.env', 'id_rsa', 'id_ed25519', 'credentials.json',
    'token.json', 'secrets.json', 'settings.local.json',
}
RELEVANT = re.compile(r'xview2|xview|first.?place|second.?place|third.?place|ensemble|inference|predict', re.I)
SOURCE_FILE_LIMIT = 260
SOURCE_TOTAL_LIMIT = 16 * 1024 * 1024
SOURCE_SINGLE_LIMIT = 2 * 1024 * 1024
WEIGHT_SAMPLE_LIMIT = 90


def list_checkpoint_metadata(root: Path) -> tuple[list[dict], int]:
    """Read names + sizes only; never read checkpoint bytes, unzip or torch.load them."""
    records = []
    count = 0
    if not root.is_dir():
        return records, count
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.') and d != '__pycache__')
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            count += 1
            if len(records) < WEIGHT_SAMPLE_LIMIT:
                records.append({'path': str(path.relative_to(root)), 'size_mb': round(size / 1e6, 2)})
    return records, count


def candidate_sources(root: Path) -> list[dict]:
    baselines = root / 'baselines'
    winners = baselines / 'xview2_winners'
    results = []
    if not baselines.is_dir():
        return results
    for dirpath, dirnames, filenames in os.walk(baselines, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.'))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink() or name.startswith('.') or name.lower() in SKIP_FILENAMES:
                continue
            ext = p.suffix.lower()
            if ext not in SRC_EXTS and name.lower() not in {'readme', 'requirements', 'dockerfile'}:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if not 0 < size <= SOURCE_SINGLE_LIMIT:
                continue
            rel = p.relative_to(root)
            try:
                inside_winners = p.is_relative_to(winners)
            except AttributeError:
                inside_winners = str(p).startswith(str(winners) + os.sep)
            if not inside_winners:
                if RELEVANT.search(str(rel)) is None:
                    try:
                        with p.open('rb') as fh:
                            snippet = fh.read(24000).decode('utf-8', errors='replace')
                        if RELEVANT.search(snippet) is None:
                            continue
                    except (OSError, UnicodeError):
                        continue
            important = bool(re.search(r'infer|predict|ensemble|test|submit|model|load|config', name, re.I))
            results.append({'path': str(rel), 'bytes': size, 'likely_inference_related': important})
    # Pick most likely relevant sources first if cap applies.
    results.sort(key=lambda x: (not x['likely_inference_related'], x['path']))
    return results


def additional_project_entrypoints(root: Path) -> list[dict]:
    """Inspect only script filenames outside baseline directories (not file contents)."""
    scriptdir = root / 'transformer' / 'scripts'
    if not scriptdir.is_dir():
        return []
    out = []
    for p in sorted(scriptdir.glob('*.py')):
        if RELEVANT.search(p.name):
            out.append({'path': str(p.relative_to(root)), 'bytes': p.stat().st_size})
    return out[:100]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2],
                        help='Absolute DisasterAdaptiveNet project directory')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    winners = root / 'baselines' / 'xview2_winners'
    if not winners.is_dir():
        raise SystemExit(f'Cannot find: {winners}; check --root')
    destination = root / 'output' / 'Reviewer2_Comment7_SourceInspection'
    destination.mkdir(parents=True, exist_ok=True)
    inv_path = destination / 'ensemble_inventory.txt'
    zip_path = destination / 'ensemble_source_bundle.zip'

    lines = [
        'REVIEWER 2 COMMENT 7: XVIEW2 ENSEMBLE SOURCE INVENTORY',
        '=' * 66,
        f'Project root: {root}',
        'This report lists source code and checkpoint names/sizes only; no weights loaded.',
        '',
    ]
    weight_info = {}
    for label, subdir in WINNERS.items():
        base = winners / subdir
        lines.append(f'[{label.upper()} PLACE] {base}')
        if not base.is_dir():
            lines.extend(['  MISSING DIRECTORY', ''])
            weight_info[label] = {'present': False}
            continue
        lines.append('  Top-level entries: ' + ', '.join(sorted(p.name for p in base.iterdir())[:65]))
        weight_directories = {
            'first': [base / 'weights' / 'weights', base / 'weights'],
            'second': [base / 'weights'],
            'third': [base / 'weights' / 'extracted' / 'ensemble'],
        }[label]
        existing = next((p for p in weight_directories if p.is_dir()), None)
        if existing is None:
            lines.append('  Expected checkpoint folder not found; inspect manually.')
            weight_info[label] = {'present': True, 'checkpoint_dir_found': False}
        else:
            records, total = list_checkpoint_metadata(existing)
            lines.extend([
                f'  Checkpoint directory: {existing}',
                f'  Total files below directory (not necessarily all checkpoints): {total}',
                '  First filename/size entries (only):',
            ])
            for item in records:
                lines.append(f"    {item['size_mb']:>10.2f} MB  {item['path']}")
            if total > len(records):
                lines.append(f'    ... +{total - len(records)} additional files')
            weight_info[label] = {
                'present': True, 'checkpoint_dir': str(existing),
                'files_total': total, 'entries': records,
            }
        lines.append('')

    sources = candidate_sources(root)
    extra = additional_project_entrypoints(root)
    lines.extend([
        'SOURCE FILES (small text/code only; no checkpoints)',
        '-' * 66,
        f'Candidates identified: {len(sources)}',
    ])
    for entry in sources:
        lines.append(('  [POSSIBLE ENTRY] ' if entry['likely_inference_related'] else '  ') + entry['path'])
    lines.extend(['', 'OTHER POTENTIALLY RELEVANT PROJECT SCRIPTS', '-' * 66])
    for entry in extra:
        lines.append('  ' + entry['path'])
    lines.extend(['', 'NOTES', '-' * 66,
                  'The number of files is not necessarily the number of distinct networks.',
                  'Each ensemble must be benchmarked with its actual loader, models, TTA,',
                  'fusion, preprocessing, and postprocessing; do not substitute one model.',
                  'Benchmark cuDNN disabled across systems due to verified WSU GPU stall.',
                  'Source ZIP excludes checkpoint data; review ZIP for private configuration',
                  'or tokens before uploading it to ChatGPT.'])

    manifest = {
        'root': str(root), 'checkpoints': weight_info,
        'candidate_sources': sources, 'project_scripts': extra,
        'limits': {'max_files': SOURCE_FILE_LIMIT, 'max_total_bytes': SOURCE_TOTAL_LIMIT,
                   'max_single_source_bytes': SOURCE_SINGLE_LIMIT},
    }
    count = 0
    total_size = 0
    selected = []
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for entry in sources + extra:
            if count >= SOURCE_FILE_LIMIT or total_size + entry['bytes'] > SOURCE_TOTAL_LIMIT:
                continue
            rel = entry['path']
            p = root / rel
            try:
                if not p.is_file() or p.is_symlink():
                    continue
                # Recheck size to avoid including files modified during the scan.
                if p.stat().st_size > SOURCE_SINGLE_LIMIT:
                    continue
                zipf.write(p, arcname=rel)
            except OSError:
                continue
            selected.append(rel)
            total_size += entry['bytes']
            count += 1
        manifest['bundled_source_paths'] = selected
        manifest['bundled_source_count'] = count
        zipf.writestr('inventory.json', json.dumps(manifest, indent=2) + '\n')
        zipf.writestr('inventory.txt', '\n'.join(lines) + '\n')
    inv_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Inventory saved: {inv_path}')
    print(f'Source ZIP saved: {zip_path}')
    print(f'Source files bundled: {count}, source bytes: {total_size:,}')
    print('No checkpoint files were included. Review the ZIP for secrets before sharing.')
    print('\nQuick summary:')
    for label in WINNERS:
        w = weight_info.get(label, {})
        print(f"  {label}: files in weight directory = {w.get('files_total', 'not found')}")
    print(f'  potential source files: {len(sources)}')


if __name__ == '__main__':
    main()
