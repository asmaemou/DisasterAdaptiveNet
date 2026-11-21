#!/usr/bin/env python3
"""Evaluate the NEW source-trained event-disjoint RS-BDNet on five EBD tests.

Inference only. Uses the verified original evaluator's data preflight, exact
pixel-level metric conventions, class confusion matrix and reporting. Never
selects settings from EBD labels. Separate output from all earlier experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import transformer.scripts.evaluate_rsbdnet_xbd_zeroshot_ebd as baseline
import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal as arch

DG_OUTPUT = PROJECT_ROOT / 'output/RSBDNet_xBD_DomainGeneralization_v1'
DEFAULT_OUTPUT = PROJECT_ROOT / 'output/RSBDNet_xBD_DomainGeneralization_v1_ZeroShot_EBD'
OLD_OUTPUTS = [
    PROJECT_ROOT / 'output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3',
    PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_EBD',
    PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_OrdinalFusion',
    PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_Calibration',
    DG_OUTPUT,
]
ORIGINAL_SHA256 = 'c781c1d9539618e2d5c67bc6365daa24557e9c01e64f87fc7d92282fe90aea77'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=DG_OUTPUT / 'checkpoints/best.pt')
    p.add_argument('--training-manifest', type=Path, default=DG_OUTPUT / 'experiment_manifest.json')
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--xbd-root', type=Path, default=Path('/homes/j244s673/documents/wsu/phd/xview2'))
    p.add_argument('--ebd-base', type=Path, default=Path('/homes/j244s673/documents/wsu/phd'))
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--hash-input-files', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    if out_dir != DEFAULT_OUTPUT.resolve():
        raise ValueError(f'Only designated fresh output directory is permitted: {DEFAULT_OUTPUT}')
    for protected in OLD_OUTPUTS:
        pr = protected.resolve()
        if out_dir == pr or pr in out_dir.parents or out_dir in pr.parents:
            raise RuntimeError(f'Output overlaps protected directory: {pr}')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing results: {out_dir}')
    expected_checkpoint = (DG_OUTPUT / 'checkpoints/best.pt').resolve()
    if args.checkpoint.resolve() != expected_checkpoint:
        raise ValueError('Refusing checkpoint not produced by DG v1 training run')
    if args.training_manifest.resolve() != (DG_OUTPUT / 'experiment_manifest.json').resolve():
        raise ValueError('Refusing unrelated training manifest')
    if args.xbd_root.resolve() != Path('/homes/j244s673/documents/wsu/phd/xview2').resolve():
        raise ValueError('Unexpected xBD source root')
    if args.ebd_base.resolve() != Path('/homes/j244s673/documents/wsu/phd').resolve():
        raise ValueError('Unexpected EBD base root')
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid batch size or number of workers')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    if not args.checkpoint.is_file() or not args.training_manifest.is_file():
        raise FileNotFoundError('DG training checkpoint/manifest not found; run training first')
    training = json.loads(args.training_manifest.read_text(encoding='utf-8'))
    if training.get('status') != 'completed' or not training.get('no_ebd_used'):
        raise RuntimeError('DG training did not complete source-only selection')
    if training.get('original_checkpoint_sha256_post') != ORIGINAL_SHA256:
        raise RuntimeError('Training did not verify the original checkpoint SHA256 on exit')
    ck_hash = baseline.sha256(args.checkpoint)
    if ck_hash != training.get('created_checkpoint_sha256'):
        raise RuntimeError('New checkpoint SHA256 does not match the DG manifest')
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except TypeError as exc:
        raise RuntimeError('PyTorch with weights_only=True is required') from exc
    if ckpt.get('protocol') != 'source-only-xBD-event-disjoint-DG-v1':
        raise RuntimeError('Unexpected checkpoint protocol')
    if set(ckpt['train_events']) & set(ckpt['val_events']):
        raise RuntimeError('Checkpoint contains overlapping source train/validation events')
    if ckpt['best_epoch'] != training['best_epoch']:
        raise RuntimeError('Selected epoch does not agree with manifest')
    if abs(float(ckpt['best_threshold']) - float(training['selected_threshold_source_only'])) > 1e-8:
        raise RuntimeError('Selected threshold does not agree with manifest')
    if int(ckpt['args']['image_size']) != 896:
        raise RuntimeError('Unexpected image size')
    threshold = float(ckpt['best_threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Invalid selected localization threshold')
    # Only checkpoints and input datasets are read before creating any output.
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.set_seed(42)
    device = torch.device('cuda:0')
    model = arch.BuildingGuidedCrossAttentionOrdinalNet(image_size=896, width=96).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval().requires_grad_(False)
    print(f'[START] New DG checkpoint epoch={ckpt["best_epoch"]} threshold={threshold:.3f}', flush=True)
    baseline.dependencies_and_sources(out_dir)
    manifest = {
        'protocol': 'Frozen source-only xBD event-disjoint DG checkpoint -> five EBD tests; no EBD adaptation',
        'training_manifest': str(args.training_manifest),
        'source_checkpoint': str(args.checkpoint),
        'new_checkpoint_sha256': ck_hash,
        'checkpoint_best_epoch': int(ckpt['best_epoch']),
        'threshold_source_selected_on_xbd_events': threshold,
        'xbd_source_train_events': ckpt['train_events'],
        'xbd_source_validation_events': ckpt['val_events'],
        'seed': 42, 'image_size': 896, 'batch_size': args.batch_size,
        'no_ebd_data_used_for_model_selection': True,
        'exploratory_warning': 'EBD original results were inspected before this method was designed. '
                               'Do not describe any later gain as independently validated.',
        'dataset_preflights': {}, 'completed_utc': None,
    }
    baseline.write_json(out_dir / 'experiment_manifest.json', manifest)

    # Reproduce THIS model's xBD Test numbers, not the old checkpoint's 0.7756.
    test_ds = runner.XBDOriginalDataset(root=args.xbd_root, split='test',
                                         image_size=896, training=False, conditioning_id=0)
    test_result = baseline.evaluate(model, test_ds, args.batch_size, args.num_workers,
                                     threshold, device, out_dir / 'xbd_test_reproduction', 'xbd_test')
    saved_test = training.get('xbd_test_results')
    if saved_test:
        observed = test_result['damage_macro']['f1']
        expected = float(saved_test['damage_macro_f1'])
        if abs(observed - expected) > 1.5e-4:
            raise RuntimeError(f'New model xBD Test reproduction mismatch: {observed} vs {expected}')
    manifest['xbd_test_reproduction'] = {
        'damage_macro_f1': test_result['damage_macro']['f1'],
        'localization_f1': test_result['localization']['f1'],
        'status': 'PASS' if saved_test else 'MEASURED (training skipped xBD Test)',
    }
    baseline.write_json(out_dir / 'experiment_manifest.json', manifest)

    old_summary_path = PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_EBD/cross_dataset_summary.csv'
    original_macro = {}
    if old_summary_path.is_file():
        import csv
        with old_summary_path.open(newline='', encoding='utf-8') as f:
            original_macro = {r['dataset']: float(r['damage_macro_f1']) for r in csv.DictReader(f)}
    summary_rows = []
    compare_rows = []
    for name, (folder, expected_pairs) in baseline.EBD_DATASETS.items():
        print(f'[EBD] {name} ({expected_pairs} pairs)', flush=True)
        dataset_root = args.ebd_base / folder
        ds_output = out_dir / name
        ds_output.mkdir(parents=True, exist_ok=False)
        preflight = baseline.dataset_preflight(dataset_root, expected_pairs,
                                               ds_output / 'input_manifest.csv', args.hash_input_files)
        manifest['dataset_preflights'][name] = preflight
        ds = runner.XBDOriginalDataset(root=dataset_root, split='test',
                                        image_size=896, training=False, conditioning_id=0)
        if len(ds) != expected_pairs:
            raise RuntimeError(f'{name}: loader unexpectedly returned {len(ds)} pairs')
        result = baseline.evaluate(model, ds, args.batch_size, args.num_workers,
                                   threshold, device, ds_output, name)
        result['reproducibility'] = {
            'new_checkpoint_sha256': ck_hash, 'threshold_selected_using_xbd_only': threshold,
            'input_manifest_sha256': preflight['input_manifest_sha256'],
            'no_target_finetuning_or_selection': True,
        }
        baseline.write_json(ds_output / 'metrics.json', result)
        r = baseline.summary_row(name, result)
        summary_rows.append(r)
        old = original_macro.get(name)
        new_macro = float(result['damage_macro']['f1'])
        compare_rows.append([name, new_macro, old if old is not None else '',
                             new_macro - old if old is not None else ''])
        baseline.write_csv(out_dir / 'cross_dataset_summary.csv', list(r),
                           [[item.get(k, '') for k in r] for item in summary_rows])
        baseline.write_csv(out_dir / 'comparison_to_original_exploratory.csv',
                           ['dataset', 'dg_new_macro_f1', 'original_baseline_macro_f1', 'delta'],
                           compare_rows)
        baseline.write_json(out_dir / 'experiment_manifest.json', manifest)
        print(f'[RESULT] {name}: new DG macro-F1={new_macro:.6f}', flush=True)
    manifest['completed_utc'] = baseline.utc_now()
    manifest['original_checkpoint_sha256_post'] = baseline.sha256(
        PROJECT_ROOT / 'output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt')
    manifest['new_checkpoint_sha256_post'] = baseline.sha256(args.checkpoint)
    if manifest['original_checkpoint_sha256_post'] != ORIGINAL_SHA256 or manifest['new_checkpoint_sha256_post'] != ck_hash:
        raise RuntimeError('Protected or newly trained checkpoint hash changed!')
    baseline.write_json(out_dir / 'experiment_manifest.json', manifest)
    print(f'[DONE] {out_dir / "comparison_to_original_exploratory.csv"}', flush=True)


if __name__ == '__main__':
    main()
