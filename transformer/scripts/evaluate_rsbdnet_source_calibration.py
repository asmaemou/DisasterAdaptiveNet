#!/usr/bin/env python3
"""Frozen RS-BDNet source-only class-logit calibration.

STAGES (two distinct SLURM submissions):
  source: fit three relative damage-class biases on a SOURCE-only subset of
          xBD Hold; select calibrated vs categorical using DISJOINT held-out
          xBD Hold event groups; persist immutable choice and evidence.
  target: only when the source stage selected calibration, apply frozen biases
          to five EBD TEST sets without any target training/selection.

PROTECTION: original training, original zero-shot and ordinal-fusion directories
are NEVER write targets. This script permits only one exact new output folder,
rejects symlinks and preexisting source output, and strictly verifies a separate
checkpoint copy against the original published SHA256. No optimizer or train().

SCORING: identical pixel-level conventions as the user's original runner:
 sigmoid(localization)>0.50, damage argmax gated by localization, valid GT damage
 only where GT localization is 1 and damage target != 255; 4-class Macro-F1
 is the unweighted arithmetic mean of four class F1 scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import Subset
from transformer.scripts import evaluate_rsbdnet_xbd_zeroshot_ebd as base
from transformer.scripts import train_xbd_supervised_disasteradaptivenet as runner
from transformer.scripts import train_xbd_bitemporal_building_crossattention_ordinal as architecture

KNOWN_SHA = 'c781c1d9539618e2d5c67bc6365daa24557e9c01e64f87fc7d92282fe90aea77'
SOURCE_CHECKPOINT = PROJECT_ROOT / 'output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt'
PREVIOUS_ZERO_SHOT = PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_EBD'
PREVIOUS_ORDINAL = PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_OrdinalFusion'
NEW_OUTPUT = PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_Calibration'
XBD_ROOT = Path('/homes/j244s673/documents/wsu/phd/xview2')
EBD_BASE = Path('/homes/j244s673/documents/wsu/phd')
CLASS_NAMES = base.CLASS_NAMES


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def protected_path_check(out_dir: Path) -> None:
    # Require the single prescribed NEW output directory.  Parent symlinks are
    # detected by equality after resolution against the expected real path.
    if out_dir.absolute() != NEW_OUTPUT.absolute():
        raise RuntimeError('Refusing an output path other than the dedicated new calibration folder')
    if out_dir.is_symlink() or (out_dir / 'input_checkpoint').is_symlink():
        raise RuntimeError('Output/checkpoint-copy symlinks are not permitted')
    for item in (SOURCE_CHECKPOINT.parent.parent, PREVIOUS_ZERO_SHOT, PREVIOUS_ORDINAL):
        real = item.resolve(strict=True)
        proposed = out_dir.resolve(strict=False)
        if proposed == real or proposed in real.parents or real in proposed.parents:
            raise RuntimeError(f'Unsafe output path overlaps historical experiment: {item}')
    # Additionally require the project output tree not to be a symlink.
    if (PROJECT_ROOT / 'output').is_symlink():
        raise RuntimeError('Refusing to run through a symlinked output parent')


def safe_checkpoint_copy(out_dir: Path) -> Path:
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    if base.sha256(SOURCE_CHECKPOINT) != KNOWN_SHA:
        raise RuntimeError('Original checkpoint SHA256 is not the known published epoch-35 checkpoint')
    dest_dir = out_dir / 'input_checkpoint'
    dest_dir.mkdir(exist_ok=False)
    dest = dest_dir / 'best_xbd_epoch35.pt'
    fd = os.open(str(dest), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    try:
        with os.fdopen(fd, 'wb') as target, SOURCE_CHECKPOINT.open('rb') as source:
            shutil.copyfileobj(source, target, 4 * 1024 * 1024)
    except BaseException:
        dest.unlink(missing_ok=True)  # Only the new copy, never the original
        raise
    if base.sha256(dest) != KNOWN_SHA:
        raise RuntimeError('Copied checkpoint hash mismatch; refusing to continue')
    dest.chmod(0o444)  # only modifies the independent copy
    return dest


def load_frozen_model(checkpoint_copy: Path, device: torch.device):
    if base.sha256(checkpoint_copy) != KNOWN_SHA:
        raise RuntimeError('Independent checkpoint copy hash mismatch')
    ckpt = torch.load(checkpoint_copy, map_location='cpu', weights_only=True)
    if int(ckpt['epoch']) != 35 or abs(float(ckpt['best_threshold']) - 0.5) > 1e-8:
        raise RuntimeError('Checkpoint epoch or source-selected localization threshold changed')
    if int(ckpt.get('args', {}).get('img_size', 896)) != 896:
        raise RuntimeError('Checkpoint image size mismatch')
    model = architecture.make_model(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def event_group(stem: str) -> str:
    # Standard xBD stems have event name followed by underscore and tile ID.
    # If no underscore exists, each distinct stem is its own group.
    return stem.rsplit('_', 1)[0] if '_' in stem else stem


def event_group_split(dataset, seed: int, validation_fraction: float = .30):
    groups: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        groups[event_group(sample.stem)].append(i)
    if len(groups) < 4:
        raise RuntimeError(f'Need >=4 distinct source-event groups; found {len(groups)}')
    event_order = sorted(groups, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    validation_groups = []
    n_validation = 0
    desired = round(len(dataset) * validation_fraction)
    for g in event_order:
        if n_validation >= desired and len(validation_groups) >= 2:
            break
        # Leave at least two event groups for source fitting.
        if len(validation_groups) >= len(groups) - 2:
            break
        validation_groups.append(g)
        n_validation += len(groups[g])
    val_group_set = set(validation_groups)
    cal_groups = [g for g in groups if g not in val_group_set]
    cal_indices = sorted(i for g in cal_groups for i in groups[g])
    val_indices = sorted(i for g in validation_groups for i in groups[g])
    if len(cal_indices) < 50 or len(val_indices) < 50:
        raise RuntimeError('Source event-group split too small to support calibration/validation')
    assert set(cal_indices).isdisjoint(val_indices)
    return cal_indices, val_indices, sorted(cal_groups), sorted(validation_groups)


def predict_damage(logits: torch.Tensor, loc_pred: torch.Tensor,
                   relative_bias: np.ndarray) -> torch.Tensor:
    # Relative biases [b_no=0,b_minor,b_major,b_destroyed]. This is a categorical
    # logit adjustment, NOT an ordinal-channel or target-label transformation.
    adjustment = torch.as_tensor(relative_bias, device=logits.device, dtype=torch.float32)
    pred = (logits[:, 1:5].float() + adjustment.view(1, 4, 1, 1)).argmax(dim=1) + 1
    return pred * loc_pred


@torch.inference_mode()
def sample_source_logits(model, data, *, device: torch.device, batch_size: int,
                         num_workers: int, cap: int, seed: int):
    """Stratified sample of SOURCE building pixels, with pixel-count weights.

    Each image contributes up to cap pixels of each true class. The per-sample
    weight is (#pixels of this class in this image)/(#sampled), reproducing
    the true source class/image proportions in expectation. Missed localization
    pixels remain classed as 'missed' in the optimization objective.
    """
    loader = runner.make_eval_loader(data, batch_size, num_workers)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    sampled_logits, sampled_truth, sampled_loc, sampled_weights = [], [], [], []
    n_seen = 0
    for batch in loader:
        images = batch['img'].to(device, non_blocking=True)
        gt_loc = batch['loc'].to(device, non_blocking=True).long()
        gt_dmg = batch['dmg'].to(device, non_blocking=True).long()
        cond = batch['cond_id'].to(device, non_blocking=True)
        logits = model(images, cond)
        if logits.shape[1] != 8 or not bool(torch.isfinite(logits).all()):
            raise RuntimeError('Incorrect or non-finite RS-BDNet logits')
        pred_loc = (torch.sigmoid(logits[:, 0]) > 0.5).long()
        for j in range(images.size(0)):
            td = gt_dmg[j].flatten()
            tl = gt_loc[j].flatten()
            lp = pred_loc[j].flatten()
            dflat = logits[j, 1:5].float().reshape(4, -1)
            for cls in range(4):
                idx = torch.nonzero((tl == 1) & (td == cls), as_tuple=False).flatten()
                total = int(idx.numel())
                if not total:
                    continue
                n = min(cap, total)
                if total > n:
                    # CPU random choice for deterministic sampling across GPU variants.
                    chosen_cpu = torch.randperm(total, generator=generator)[:n]
                    pick = idx[chosen_cpu.to(device=idx.device)]
                else:
                    pick = idx
                sampled_logits.append(dflat[:, pick].T.cpu().numpy())
                sampled_truth.append(np.full(n, cls, dtype=np.int64))
                sampled_loc.append(lp[pick].cpu().numpy().astype(np.int64))
                sampled_weights.append(np.full(n, total / n, dtype=np.float64))
        n_seen += images.shape[0]
        if n_seen % 100 < batch_size or n_seen == len(data):
            print(f'  [SOURCE SAMPLE] {n_seen}/{len(data)} pairs', flush=True)
    if n_seen != len(data) or not sampled_logits:
        raise RuntimeError('Incomplete source calibration sample')
    return (np.concatenate(sampled_logits).astype(np.float32),
            np.concatenate(sampled_truth), np.concatenate(sampled_loc),
            np.concatenate(sampled_weights))


def sample_macro_f1(source_logits: np.ndarray, truth: np.ndarray,
                    loc: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> float:
    predicted = np.argmax(source_logits + biases[None, :], axis=1).astype(np.int64) + 1
    predicted = predicted * loc
    cm = np.bincount(truth * 5 + predicted, weights=weights,
                     minlength=20).reshape(4, 5)
    tp = np.diag(cm[:, 1:])
    fp = cm[:, 1:].sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    return float(np.divide(2 * tp, denominator,
                           out=np.zeros(4, dtype=np.float64), where=denominator > 0).mean())


def fit_relative_biases(logits: np.ndarray, truth: np.ndarray,
                        loc: np.ndarray, weights: np.ndarray, bound: float):
    """Deterministic source-sample coordinate search; no target data consulted."""
    biases = np.zeros(4, dtype=np.float32)
    best = sample_macro_f1(logits, truth, loc, weights, biases)
    trace = [{'round': -1, 'coordinate': 'baseline', 'biases': biases.tolist(),
              'weighted_sample_macro_f1': best}]
    for round_id, step in enumerate((0.5, 0.25, 0.125)):
        for sweep in range(3):
            improved = False
            for coordinate in (1, 2, 3):
                if round_id == 0:
                    candidates = np.arange(-bound, bound + 0.001, step)
                else:
                    candidates = np.arange(float(biases[coordinate]) - 3 * step,
                                           float(biases[coordinate]) + 3.1 * step, step)
                for value in candidates:
                    value = float(np.clip(value, -bound, bound))
                    candidate = biases.copy()
                    candidate[coordinate] = value
                    score = sample_macro_f1(logits, truth, loc, weights, candidate)
                    if score > best + 1e-9:  # strict improvement only, deterministic ties
                        best, biases = score, candidate
                        improved = True
                trace.append({'round': round_id, 'sweep': sweep,
                              'coordinate': CLASS_NAMES[coordinate],
                              'biases': biases.tolist(), 'weighted_sample_macro_f1': best})
            if not improved:
                break
    return biases, best, trace


@torch.inference_mode()
def exact_evaluation(model, dataset, relative_bias: np.ndarray, device: torch.device,
                     batch_size: int, num_workers: int, name: str,
                     out_folder: Path | None = None):
    loader = runner.make_eval_loader(dataset, batch_size, num_workers)
    loc_counts = np.zeros(4, dtype=np.int64)
    confusion = np.zeros((4, 5), dtype=np.int64)
    per_image_rows = []
    seen = 0
    for batch in loader:
        images = batch['img'].to(device, non_blocking=True)
        gt_loc = batch['loc'].to(device, non_blocking=True).long()
        gt_dmg = batch['dmg'].to(device, non_blocking=True).long()
        cond = batch['cond_id'].to(device, non_blocking=True)
        logits = model(images, cond)
        if logits.shape[1] != 8 or logits.shape[-2:] != images.shape[-2:]:
            raise RuntimeError('Unexpected RS-BDNet output shape')
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError('Non-finite RS-BDNet logits')
        pred_loc = (torch.sigmoid(logits[:, 0]) > .5).long()
        pred_dmg = predict_damage(logits, pred_loc, relative_bias)
        for j in range(images.size(0)):
            ltrue, lpred = gt_loc[j], pred_loc[j]
            loc_i = np.array([((lpred == 1) & (ltrue == 1)).sum().item(),
                              ((lpred == 1) & (ltrue == 0)).sum().item(),
                              ((lpred == 0) & (ltrue == 1)).sum().item(),
                              ((lpred == 0) & (ltrue == 0)).sum().item()], dtype=np.int64)
            mask = (ltrue == 1) & (gt_dmg[j] != 255)
            if bool(mask.any()):
                a, p = gt_dmg[j][mask], pred_dmg[j][mask]
                if not bool(((a >= 0) & (a <= 3)).all()):
                    raise RuntimeError(f'Invalid GT target encoding: {batch["stem"][j]}')
                cm_i = torch.bincount(a * 5 + p, minlength=20).reshape(4, 5).cpu().numpy().astype(np.int64)
            else:
                cm_i = np.zeros((4, 5), dtype=np.int64)
            loc_counts += loc_i
            confusion += cm_i
            if out_folder is not None:
                ind = base.metrics_from_counts(loc_i, cm_i, 1)
                per_image_rows.append([str(batch['stem'][j]), ind['localization']['f1'],
                                       ind['damage_macro']['f1'], int(cm_i.sum()),
                                       int(cm_i[:, 0].sum())])
        seen += images.size(0)
        if seen % 100 < batch_size or seen == len(dataset):
            print(f'  [{name}] {seen}/{len(dataset)} pairs', flush=True)
    if seen != len(dataset):
        raise RuntimeError(f'{name}: expected {len(dataset)} pairs, evaluated {seen}')
    metrics = base.metrics_from_counts(loc_counts, confusion, seen)
    if out_folder is not None:
        out_folder.mkdir(exist_ok=False)
        base.write_json(out_folder / 'metrics.json', metrics)
        base.write_csv(out_folder / 'damage_confusion_matrix.csv',
                       ['ground_truth\\prediction', *base.PREDICTION_NAMES],
                       [[cls, *confusion[i].tolist()] for i, cls in enumerate(CLASS_NAMES)])
        base.write_csv(out_folder / 'class_metrics.csv',
                       ['category', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'support_pixels'],
                       [[cls, *[metrics['damage_classes'][cls][key] for key in
                               ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'support_valid_gt_pixels')]]
                        for cls in CLASS_NAMES])
        base.write_csv(out_folder / 'per_image_metrics.csv',
                       ['stem', 'localization_f1', 'damage_macro_f1',
                        'valid_gt_damage_pixels', 'missed_gt_damage_pixels'], per_image_rows)
    return metrics


def read_original_baseline() -> dict[str, float]:
    path = PREVIOUS_ZERO_SHOT / 'cross_dataset_summary.csv'
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    baseline = {row['dataset']: float(row['damage_macro_f1']) for row in rows}
    if set(baseline) != set(base.EBD_DATASETS):
        raise RuntimeError('Baseline CSV has unexpected datasets')
    return baseline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['source', 'target'], required=True)
    parser.add_argument('--output-dir', type=Path, default=NEW_OUTPUT)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-per-image-class', type=int, default=96)
    parser.add_argument('--bias-bound', type=float, default=2.0)
    parser.add_argument('--min-validation-gain', type=float, default=.005)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.sample_per_image_class < 1:
        raise ValueError('Invalid batch/worker/sample settings')
    if not 0 < args.bias_bound <= 4 or not 0 <= args.min_validation_gain <= .1:
        raise ValueError('Invalid bias bound or validation margin')
    protected_path_check(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required to run the RS-BDNet model')
    runner.set_seed(args.seed)
    device = torch.device('cuda:0')
    output = args.output_dir
    script_hash = base.sha256(Path(__file__).resolve())
    if base.sha256(SOURCE_CHECKPOINT) != KNOWN_SHA:
        raise RuntimeError('Source checkpoint SHA256 mismatch; refusing to run')

    if args.stage == 'source':
        # Exclusive directory creation; NEVER overwrite or resume prior results.
        output.mkdir(parents=False, exist_ok=False)
        copied = safe_checkpoint_copy(output)
        model = load_frozen_model(copied, device)
        env = base.dependencies_and_sources(output)
        env['source_sha256'][str(Path(__file__).resolve())] = script_hash
        base.write_json(output / 'environment.json', env)
        dataset = runner.XBDOriginalDataset(root=XBD_ROOT, split='hold', image_size=896,
                                            training=False, conditioning_id=0)
        cal_idx, val_idx, cal_events, val_events = event_group_split(dataset, args.seed)
        print(f'[SOURCE] xBD Hold pairs={len(dataset)}; fit={len(cal_idx)} '
              f'({len(cal_events)} events), validation={len(val_idx)} '
              f'({len(val_events)} events)', flush=True)
        calibration_data, validation_data = Subset(dataset, cal_idx), Subset(dataset, val_idx)
        logits, truth, localization, weights = sample_source_logits(
            model, calibration_data, device=device, batch_size=args.batch_size,
            num_workers=args.num_workers, cap=args.sample_per_image_class, seed=args.seed)
        if set(np.unique(truth)) != {0, 1, 2, 3}:
            raise RuntimeError('Source-fitting subset does not contain all damage classes')
        biases, approx_f1, trace = fit_relative_biases(logits, truth, localization,
                                                      weights, args.bias_bound)
        print(f'[SOURCE] Fitted relative biases [no, minor, major, destroyed]: {biases.tolist()}', flush=True)
        print(f'[SOURCE] Weighted calibration sample Macro-F1={approx_f1:.6f}', flush=True)
        base.write_json(output / 'calibration_search_trace.json', trace)
        # Selection is on disjoint SOURCE events, using exact pixel-level scoring.
        baseline = exact_evaluation(model, validation_data, np.zeros(4), device,
                                    args.batch_size, args.num_workers,
                                    'xbd_hold_event_validation_categorical')
        calibrated = exact_evaluation(model, validation_data, biases, device,
                                      args.batch_size, args.num_workers,
                                      'xbd_hold_event_validation_calibrated')
        source_validation_support = {
            name: baseline['damage_classes'][name]['support_valid_gt_pixels']
            for name in CLASS_NAMES
        }
        if any(value == 0 for value in source_validation_support.values()):
            raise RuntimeError(
                f'Event-held-out source validation lacks a damage class: {source_validation_support}. '
                'Cannot select calibration on this split.'
            )
        before = baseline['damage_macro']['f1']
        after = calibrated['damage_macro']['f1']
        gain = after - before
        selected = 'calibrated' if gain >= args.min_validation_gain else 'categorical'
        base.write_json(output / 'xbd_hold_validation_categorical.json', baseline)
        base.write_json(output / 'xbd_hold_validation_calibrated.json', calibrated)
        base.write_csv(output / 'xbd_hold_calibration_comparison.csv',
                       ['method', 'n_validation_pairs', 'localization_f1',
                        'no_damage_f1', 'minor_damage_f1', 'major_damage_f1',
                        'destroyed_f1', 'damage_macro_f1'],
                       [[method, len(val_idx), result['localization']['f1'],
                         *[result['damage_classes'][c]['f1'] for c in CLASS_NAMES],
                         result['damage_macro']['f1']]
                        for method, result in [('categorical', baseline), ('calibrated', calibrated)]])
        manifest = {
            'status': 'source_ready', 'stage': 'source',
            'created_utc': timestamp(), 'source_only_no_ebd_opened': True,
            'protected_directories_read_only': [str(SOURCE_CHECKPOINT.parent.parent),
                                                 str(PREVIOUS_ZERO_SHOT), str(PREVIOUS_ORDINAL)],
            'independent_checkpoint_copy': str(copied),
            'original_checkpoint_sha256': KNOWN_SHA,
            'copied_checkpoint_sha256': base.sha256(copied),
            'script_sha256_source_stage': script_hash,
            'checkpoint_epoch': 35, 'fixed_localization_threshold': .5,
            'seed': args.seed, 'image_size': 896, 'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'source_dataset': str(XBD_ROOT / 'hold'),
            'source_hold_total_pairs': len(dataset),
            'source_calibration_fit_pairs': len(cal_idx),
            'source_validation_pairs': len(val_idx),
            'source_calibration_event_groups': cal_events,
            'source_validation_event_groups': val_events,
            'split_rule': 'Deterministic seed-42 SHA256 event-group split; no event shared by fit and validation',
            'sampling': f'Maximum {args.sample_per_image_class} GT pixels per image and damage class, weighted by original pixel counts',
            'bias_search': 'Source-weighted pixel Macro-F1 coordinate search, biases relative to no_damage=0',
            'bias_bounds': [-args.bias_bound, args.bias_bound],
            'fitted_biases_no_minor_major_destroyed': biases.tolist(),
            'source_fit_weighted_sample_macro_f1': approx_f1,
            'source_validation_baseline_macro_f1': before,
            'source_validation_calibrated_macro_f1': after,
            'source_validation_macro_f1_delta': gain,
            'min_source_validation_gain_to_select_calibration': args.min_validation_gain,
            'selected_method_frozen': selected,
            'frozen_biases': biases.tolist() if selected == 'calibrated' else [0., 0., 0., 0.],
            'selection_data': 'xBD Hold SOURCE event-disjoint validation ONLY',
            'xbd_hold_also_used_for_original_checkpoint_selection': True,
            'ebd_results_seen_prior_to_method_design': True,
            'caveat': 'Exploratory source-only calibration; EBD test scores never used to fit or select biases. '
                      'Original xBD checkpoint selected on xBD Hold, so source split is not a wholly independent checkpoint-validation set.',
            'source_code_sha256': env['source_sha256'],
            'completed_utc': timestamp(),
        }
        base.write_json(output / 'calibration_manifest.json', manifest)
        print(f'[VALIDATION] xBD Hold event-held-out Macro-F1 baseline={before:.6f}, '
              f'calibrated={after:.6f}, gain={gain:+.6f}', flush=True)
        print(f'[SOURCE SELECTED] {selected}', flush=True)
        print(f'[OUTPUT] {output / "xbd_hold_calibration_comparison.csv"}', flush=True)
    else:
        if not output.is_dir():
            raise FileNotFoundError('Missing source-stage calibration directory; run --stage source first')
        manifest_path = output / 'calibration_manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError('Missing completed source-stage calibration manifest')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'source_ready':
            raise RuntimeError('Source-stage manifest not in source_ready state; refusing target run')
        if manifest.get('script_sha256_source_stage') != script_hash:
            raise RuntimeError('Source-calibration script changed between stages; aborting')
        if manifest.get('original_checkpoint_sha256') != KNOWN_SHA:
            raise RuntimeError('Source calibration based on unexpected checkpoint')
        if args.seed != manifest.get('seed') or args.batch_size != manifest.get('batch_size') or args.num_workers != manifest.get('num_workers'):
            raise RuntimeError('Target-stage seed/batch/worker arguments must match the frozen source-stage settings')
        if manifest.get('selected_method_frozen') != 'calibrated':
            print('[NO SOURCE VALIDATION GAIN] Categorical baseline selected; '
                  'EBD target evaluation skipped to protect independence.', flush=True)
            return
        if any((output / name).exists() for name in
               ('cross_dataset_comparison.csv', 'ebd_experiment_manifest.json', *base.EBD_DATASETS)):
            raise FileExistsError('Target-stage output already exists; refusing to overwrite')
        frozen_biases = np.array(manifest['frozen_biases'], dtype=np.float32)
        if frozen_biases.shape != (4,) or frozen_biases[0] != 0 or not np.isfinite(frozen_biases).all():
            raise RuntimeError('Malformed frozen source-only bias vector')
        copied = output / 'input_checkpoint/best_xbd_epoch35.pt'
        model = load_frozen_model(copied, device)
        print(f'[TARGET] Applying frozen SOURCE biases {frozen_biases.tolist()}', flush=True)
        target_record = {
            'protocol': 'Frozen xBD checkpoint, source-only calibrated on disjoint xBD Hold groups, EBD test only',
            'started_utc': timestamp(), 'completed_utc': None,
            'source_manifest_sha256_at_start': base.sha256(manifest_path),
            'original_checkpoint_sha256': KNOWN_SHA,
            'frozen_biases': frozen_biases.tolist(),
            'selection_before_target_evaluation': True,
            'source_selected_method': 'calibrated',
            'ebd_results': {},
        }
        base.write_json(output / 'ebd_experiment_manifest.json', target_record)
        rows = []
        for name, (dir_name, expected_pairs) in base.EBD_DATASETS.items():
            root = EBD_BASE / dir_name
            dataset_folder = output / name
            # Only test split is opened; per-file manifest recorded for reproducibility.
            preflight = base.dataset_preflight(root, expected_pairs,
                                                output / f'{name}_input_manifest.csv', False)
            dataset = runner.XBDOriginalDataset(root=root, split='test', image_size=896,
                                                training=False, conditioning_id=0)
            if len(dataset) != expected_pairs:
                raise RuntimeError(f'{name}: expected {expected_pairs} pairs, got {len(dataset)}')
            result = exact_evaluation(model, dataset, frozen_biases, device,
                                      args.batch_size, args.num_workers, name, dataset_folder)
            result['reproducibility'] = {
                'checkpoint_sha256': KNOWN_SHA, 'frozen_biases': frozen_biases.tolist(),
                'localization_threshold': 0.5, 'source_only_selection': True,
                'no_target_train_or_val_used': True, 'preflight': preflight,
            }
            base.write_json(dataset_folder / 'metrics.json', result)
            row = base.summary_row(name, result)
            row['method'] = 'source_calibrated'
            rows.append(row)
            target_record['ebd_results'][name] = {
                'n_test_pairs': len(dataset), 'damage_macro_f1': result['damage_macro']['f1'],
                'localization_f1': result['localization']['f1'],
                'input_manifest_sha256': preflight['input_manifest_sha256'],
            }
            base.write_json(output / 'ebd_experiment_manifest.json', target_record)
            print(f'[EBD] {name}: damage Macro-F1={result["damage_macro"]["f1"]:.6f}', flush=True)
        # The old EBD scores are opened only AFTER frozen inference has finished.
        original = read_original_baseline()
        compare_rows = []
        for row in rows:
            name = row['dataset']
            old = original[name]
            compare_rows.append([name, row['n_test_pairs'], 'source_calibrated',
                                 row['localization_f1'],
                                 *[row[f'{c}_f1'] for c in CLASS_NAMES],
                                 row['damage_macro_f1'], old, row['damage_macro_f1'] - old])
        base.write_csv(output / 'cross_dataset_comparison.csv',
                       ['dataset', 'n_test_pairs', 'method', 'localization_f1',
                        'no_damage_f1', 'minor_damage_f1', 'major_damage_f1', 'destroyed_f1',
                        'damage_macro_f1', 'original_categorical_macro_f1_read_only',
                        'macro_f1_delta_exploratory'], compare_rows)
        target_record['completed_utc'] = timestamp()
        target_record['five_ebd_mean_calibrated_macro_f1'] = sum(
            row['damage_macro_f1'] for row in rows) / len(rows)
        target_record['five_ebd_mean_original_macro_f1'] = sum(original.values()) / len(original)
        target_record['five_ebd_mean_delta_exploratory'] = (
            target_record['five_ebd_mean_calibrated_macro_f1'] -
            target_record['five_ebd_mean_original_macro_f1'])
        base.write_json(output / 'ebd_experiment_manifest.json', target_record)
        print(f'[DONE] New calibrated EBD Macro-F1 mean: '
              f'{target_record["five_ebd_mean_calibrated_macro_f1"]:.6f}', flush=True)
    if base.sha256(SOURCE_CHECKPOINT) != KNOWN_SHA:
        raise RuntimeError('ORIGINAL CHECKPOINT HASH CHANGED (unexpected): investigate immediately')
    if base.sha256(output / 'input_checkpoint/best_xbd_epoch35.pt') != KNOWN_SHA:
        raise RuntimeError('Checkpoint copy hash changed (unexpected)')
    print('[PROTECTED] Original and copied checkpoint SHA256 verified unchanged', flush=True)


if __name__ == '__main__':
    main()
