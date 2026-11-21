#!/usr/bin/env python3
"""Source-only xBD event-disjoint domain-generalization training for RS-BDNet.

IMPORTANT: fresh full-model training, NOT warm-started from the old xBD checkpoint.
All train/validation pairs come from xBD train+tier3.  Entire disaster events
are disjoint across these subsets.  EBD is never accessed during training.
This is an exploratory DG recipe, not a guarantee of target-domain improvement.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable
import transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal as arch

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

SOURCE_ORIGINAL = PROJECT_ROOT / 'output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3'
OTHER_PROTECTED = [
    PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_EBD',
    PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_OrdinalFusion',
    PROJECT_ROOT / 'output/RSBDNet_xBD_ZeroShot_Calibration',
]
DEFAULT_OUT = PROJECT_ROOT / 'output/RSBDNet_xBD_DomainGeneralization_v1'
EXPECTED_OLD_SHA256 = 'c781c1d9539618e2d5c67bc6365daa24557e9c01e64f87fc7d92282fe90aea77'
CLASS_NAMES = ('no_damage', 'minor_damage', 'major_damage', 'destroyed')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def json_safe_args(args):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_save(state, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(state, tmp)
    os.replace(tmp, path)


def validate_output(path: Path) -> Path:
    """Allow writes ONLY to the dedicated new directory, never protected trees."""
    root = path.expanduser().resolve()
    if root != DEFAULT_OUT.resolve():
        raise ValueError(f'Refusing unexpected output path {root}; exact allowed path is {DEFAULT_OUT}')
    for protected in [SOURCE_ORIGINAL, *OTHER_PROTECTED]:
        p = protected.resolve()
        if root == p or p in root.parents or root in p.parents:
            raise RuntimeError(f'Output path overlaps a protected directory: {p}')
    return root


def event_id(stem: str) -> str:
    """xBD tile stems are disaster-name_NNNNNNNN. Never silently infer events."""
    match = re.fullmatch(r'(.+)_([0-9]+)', str(stem))
    if match is None:
        raise ValueError(f'Unexpected xBD tile stem; cannot guarantee event-disjoint split: {stem}')
    return match.group(1).lower()


def deterministic_group_split(samples, fraction: float, seed: int):
    event_counts = Counter(event_id(s.stem) for s in samples)
    hazard_to_events = defaultdict(list)
    for event in sorted(event_counts):
        hazard_to_events[stable.hazard_id(event)].append(event)
    rng = random.Random(seed)
    val_events = set()
    for hazard in sorted(hazard_to_events):
        events = sorted(hazard_to_events[hazard])
        if len(events) < 2:
            print(f'[EVENT SPLIT] Hazard {stable.HAZARDS[hazard]} has only one event; leave in training', flush=True)
            continue
        rng.shuffle(events)
        n = min(len(events) - 1, max(1, round(len(events) * fraction)))
        val_events.update(events[:n])
    train_events = set(event_counts) - val_events
    if not val_events or not train_events or train_events & val_events:
        raise RuntimeError('Event-disjoint split failed: missing groups or overlap')
    # A source tile's event is entirely assigned to one side, across train+tier3.
    train_samples = [s for s in samples if event_id(s.stem) in train_events]
    val_samples = [s for s in samples if event_id(s.stem) in val_events]
    if len(train_samples) < 10 or len(val_samples) < 10:
        raise RuntimeError('Too few images in event-disjoint source subsets')
    return train_events, val_events, event_counts


class DomainGeneralizationDataset(stable.MultiSplitHazardDataset):
    """Original localization/label mapping + source-only photometric augmentation."""
    def __init__(self, *args, augment_photometric=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_photometric = bool(augment_photometric)

    def _normalize_images(self, image_list):
        if not self.training or not self.augment_photometric:
            return super()._normalize_images(image_list)
        # Use shared acquisition perturbation on pre/post to preserve change cues.
        imgs = [x.astype(np.float32) / 255.0 for x in image_list]
        if np.random.rand() < 0.85:
            gamma = float(np.random.uniform(0.85, 1.18))
            gain = np.random.uniform(0.87, 1.14, size=(1, 1, 3)).astype(np.float32)
            bias = np.random.uniform(-0.045, 0.045, size=(1, 1, 3)).astype(np.float32)
            imgs = [np.clip(np.power(np.clip(x, 0, 1), gamma) * gain + bias, 0, 1) for x in imgs]
        if np.random.rand() < 0.25:
            sigma = float(np.random.uniform(0.35, 1.0))
            imgs = [cv2.GaussianBlur(x, (3, 3), sigmaX=sigma) for x in imgs]
        # Very small post-acquisition gain shift, not a synthetic damage mask.
        if np.random.rand() < 0.20:
            imgs[1] = np.clip(imgs[1] * np.random.uniform(0.98, 1.02), 0, 1)
        to_uint8 = [np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8) for x in imgs]
        return super()._normalize_images(to_uint8)


def scan_source_masks(dataset, csv_path: Path):
    """Training labels ONLY. Count valid pixels and per-tile class presence."""
    stats = []
    loc_pos = loc_neg = 0
    class_counts = np.zeros(4, dtype=np.int64)
    rows = []
    for i, sample in enumerate(dataset.samples):
        loc = runner.XBDOriginalDataset._read_mask(sample.pre_target_path) > 0
        damage = runner.XBDOriginalDataset._read_mask(sample.post_target_path)
        if loc.shape != damage.shape:
            raise RuntimeError(f'Misaligned ground-truth masks: {sample.stem}')
        npos = int(loc.sum())
        loc_pos += npos
        loc_neg += int(loc.size - npos)
        counts = np.array([int(((damage == c + 1) & loc).sum()) for c in range(4)], dtype=np.int64)
        class_counts += counts
        stats.append(counts)
        rows.append([sample.stem, event_id(sample.stem), *map(int, counts)])
        if (i + 1) % 1000 == 0:
            print(f'[SOURCE SCAN] {i+1}/{len(dataset)} source training masks', flush=True)
    write_csv(csv_path, ['stem', 'event', *CLASS_NAMES], rows)
    return np.stack(stats), loc_pos, loc_neg, class_counts


def build_sampler(dataset, stats: np.ndarray, event_alpha: float, boosts: tuple[float, float, float], seed: int):
    counts_by_event = Counter(event_id(s.stem) for s in dataset.samples)
    weights = []
    for sample, count_vec in zip(dataset.samples, stats):
        baseline = counts_by_event[event_id(sample.stem)] ** (-event_alpha)
        multiplier = 1.0 + sum(boost for boost, n in zip(boosts, count_vec[1:]) if n > 0)
        weights.append(baseline * multiplier)
    weights_tensor = torch.tensor(weights, dtype=torch.double)
    weights_tensor = weights_tensor / weights_tensor.mean()
    if not torch.isfinite(weights_tensor).all() or (weights_tensor <= 0).any():
        raise RuntimeError('Invalid sampler weights')
    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(dataset),
                                    replacement=True, generator=gen)
    return sampler, gen


def mean_macro_f1(results: dict) -> float:
    return float(np.mean([results[f'damage_f1_{name}'] for name in CLASS_NAMES]))


def metrics_summary(results: dict) -> dict:
    return {
        'localization_f1': float(results['localization_f1']),
        'damage_macro_f1': mean_macro_f1(results),
        'no_damage_f1': float(results['damage_f1_no_damage']),
        'minor_damage_f1': float(results['damage_f1_minor_damage']),
        'major_damage_f1': float(results['damage_f1_major_damage']),
        'destroyed_f1': float(results['damage_f1_destroyed']),
        'damage_harmonic_f1_original_runner': float(results['damage_f1']),
    }


def choose_device_and_seed(seed):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this training experiment')
    runner.set_seed(seed)
    return torch.device('cuda:0')


def load_weights_only(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError as exc:
        raise RuntimeError('PyTorch with torch.load(weights_only=True) is required') from exc


def make_checkpoint(model, optimizer, scheduler, epoch, best_epoch, best_score,
                    threshold, args, train_events, val_events, history):
    return {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': int(epoch),
        'best_epoch': int(best_epoch),
        'best_score': float(best_score),
        'best_threshold': float(threshold),
        'args': json_safe_args(args),
        'protocol': 'source-only-xBD-event-disjoint-DG-v1',
        'train_events': sorted(train_events),
        'val_events': sorted(val_events),
        'history': history,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--xbd-root', type=Path,
                   default=Path('/homes/j244s673/documents/wsu/phd/xview2'))
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--image-size', type=int, default=896)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--early-stop-patience', type=int, default=12)
    p.add_argument('--val-event-fraction', type=float, default=0.25)
    p.add_argument('--event-balance-alpha', type=float, default=0.75)
    p.add_argument('--minor-sampling-boost', type=float, default=0.50)
    p.add_argument('--major-sampling-boost', type=float, default=0.60)
    p.add_argument('--destroyed-sampling-boost', type=float, default=0.20)
    p.add_argument('--minor-loss-multiplier', type=float, default=1.25)
    p.add_argument('--major-loss-multiplier', type=float, default=1.25)
    p.add_argument('--aux-ce-weight', type=float, default=0.25)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume only from THIS experiment last.pt')
    p.add_argument('--skip-xbd-test', action='store_true')
    p.add_argument('--thresholds', type=float, nargs='+',
                   default=[0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = validate_output(args.output_dir)
    expected_source = Path('/homes/j244s673/documents/wsu/phd/xview2').resolve()
    if args.xbd_root.resolve() != expected_source:
        raise ValueError('Only the designated xBD source root is permitted; no EBD inputs')
    if args.image_size != 896 or args.batch_size < 1 or args.num_workers < 0 or args.epochs < 1:
        raise ValueError('Expected image_size=896, valid batch_size/num_workers/epochs')
    if not 0.0 < args.val_event_fraction < 0.5 or not 0 <= args.event_balance_alpha <= 1:
        raise ValueError('Invalid event split fraction or balancing exponent')
    if args.aux_ce_weight < 0 or args.minor_loss_multiplier < 1 or args.major_loss_multiplier < 1:
        raise ValueError('Invalid loss hyperparameters')
    if args.seed != 42:
        raise ValueError('This v1 experiment uses frozen seed 42')
    original_ckpt = SOURCE_ORIGINAL / 'checkpoints/best.pt'
    if not original_ckpt.is_file():
        raise FileNotFoundError(original_ckpt)
    orig_hash = sha256(original_ckpt)
    if orig_hash != EXPECTED_OLD_SHA256:
        raise RuntimeError(f'Original checkpoint hash unexpected: {orig_hash}; not running')
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f'Nonempty DG output directory: {out_dir}; use --resume only for a valid run')
    if args.resume and not (out_dir / 'checkpoints/last.pt').is_file():
        raise FileNotFoundError('Cannot resume: this experiment has no last.pt')
    if args.resume and (out_dir / 'experiment_manifest.json').is_file():
        past_status = json.loads((out_dir / 'experiment_manifest.json').read_text()).get('status')
        if past_status == 'completed':
            raise RuntimeError('DG v1 already completed; refusing to rerun and overwrite its results')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'checkpoints').mkdir(exist_ok=True)
    device = choose_device_and_seed(args.seed)
    print(f'[DG] Fresh model training, source-only. {utc_now()}', flush=True)
    print(f'[PROTECTED] original checkpoint SHA256={orig_hash}', flush=True)
    print(f'[OUTPUT] {out_dir}', flush=True)

    ds_train = DomainGeneralizationDataset(root=args.xbd_root, split='train+tier3',
                                            image_size=args.image_size, training=True)
    ds_val = DomainGeneralizationDataset(root=args.xbd_root, split='train+tier3',
                                          image_size=args.image_size, training=False)
    all_samples = list(ds_train.samples)
    train_events, val_events, original_event_counts = deterministic_group_split(
        all_samples, args.val_event_fraction, args.seed)
    ds_train.samples = [s for s in ds_train.samples if event_id(s.stem) in train_events]
    ds_val.samples = [s for s in ds_val.samples if event_id(s.stem) in val_events]
    if len(ds_train) + len(ds_val) != len(all_samples):
        raise RuntimeError('Event split lost source images')
    if not ds_train.samples or not ds_val.samples:
        raise RuntimeError('Empty event-disjoint source subset')
    train_stems = {s.stem for s in ds_train.samples}
    val_stems = {s.stem for s in ds_val.samples}
    if train_stems & val_stems or train_events & val_events:
        raise RuntimeError('Source train/validation leakage')
    split_manifest = {
        'source': 'xBD train+tier3 only',
        'split_rule': 'Seed-42 stratified selection of entire event groups by hazard',
        'seed': args.seed,
        'train_events': sorted(train_events),
        'val_events': sorted(val_events),
        'all_event_sample_counts': dict(sorted(original_event_counts.items())),
        'n_train_pairs': len(ds_train), 'n_val_pairs': len(ds_val),
        'n_all_pairs': len(all_samples),
        'train_stems_sha256': hashlib.sha256('\n'.join(sorted(train_stems)).encode()).hexdigest(),
        'val_stems_sha256': hashlib.sha256('\n'.join(sorted(val_stems)).encode()).hexdigest(),
        'overlapping_events': [],
    }
    if args.resume:
        stored = json.loads((out_dir / 'source_event_split.json').read_text())
        if stored != split_manifest:
            raise RuntimeError('Cannot resume: source event split changed')
    else:
        write_json(out_dir / 'source_event_split.json', split_manifest)
    print(f'[SPLIT] Train={len(ds_train)}, validation={len(ds_val)}; '
          f'events={len(train_events)} / {len(val_events)} disjoint', flush=True)
    print(f'[VAL EVENTS] {sorted(val_events)}', flush=True)

    stats, loc_pos, loc_neg, damage_counts = scan_source_masks(
        ds_train, out_dir / 'source_training_class_counts.csv')
    if min(damage_counts) <= 0 or loc_pos <= 0:
        raise RuntimeError(f'Missing necessary source train classes: {damage_counts}')
    # Exactly the existing source-only inverse-frequency recipe, with modest
    # fixed emphasis on intermediate severities; no EBD class counts used.
    raw_inv = damage_counts.sum() / damage_counts.astype(np.float64)
    class_weights = raw_inv / raw_inv.sum() * 4
    class_weights *= np.array([1.0, args.minor_loss_multiplier,
                               args.major_loss_multiplier, 1.0])
    class_weights *= 4.0 / class_weights.sum()
    pos_weight = min(10.0, max(1.0, loc_neg / loc_pos))
    val_class_counts = ds_val.get_damage_class_counts()
    print(f'[SOURCE VAL CLASS COUNTS] {dict(zip(CLASS_NAMES, map(int, val_class_counts)))}', flush=True)
    if any(int(c) == 0 for c in val_class_counts):
        print('[WARNING] A damage class is absent from source event-disjoint validation. '
              'Its F1 is zero by the original metric convention.', flush=True)
    sampler, sampler_generator = build_sampler(
        ds_train, stats, args.event_balance_alpha,
        (args.minor_sampling_boost, args.major_sampling_boost, args.destroyed_sampling_boost), args.seed)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = runner.make_eval_loader(ds_val, args.batch_size, args.num_workers)
    loc_criterion = runner.BCEDiceLoss(pos_weight=torch.tensor([pos_weight], device=device)).to(device)
    dmg_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
    dmg_criterion = torch.nn.CrossEntropyLoss(weight=dmg_weights, ignore_index=255).to(device)

    model = arch.BuildingGuidedCrossAttentionOrdinalNet(image_size=896, width=96).to(device)
    # We intentionally NEVER load original model weights: that model has seen
    # every event in xBD train+tier3, violating event-disjoint validation.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    milestones = sorted(set(max(1, int(args.epochs * x)) for x in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    start_epoch = 1
    best_macro, best_epoch = -1.0, 0
    history = []
    if args.resume:
        ckpt = load_weights_only(out_dir / 'checkpoints/last.pt')
        if ckpt.get('protocol') != 'source-only-xBD-event-disjoint-DG-v1':
            raise RuntimeError('Resume checkpoint protocol does not match')
        if ckpt.get('train_events') != sorted(train_events) or ckpt.get('val_events') != sorted(val_events):
            raise RuntimeError('Resume checkpoint event group mismatch')
        prior_args = dict(ckpt.get('args', {}))
        current_args = json_safe_args(args)
        prior_args.pop('resume', None)
        current_args.pop('resume', None)
        if prior_args != current_args:
            raise RuntimeError('Resume arguments differ from original training job')
        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = int(ckpt['epoch']) + 1
        best_epoch, best_macro = int(ckpt['best_epoch']), float(ckpt['best_score'])
        history = list(ckpt['history'])
        # Each epoch samples reproducibly from a newly seeded generator.
        print(f'[RESUME] Starting at epoch {start_epoch}, best={best_macro:.6f}', flush=True)

    manifest = {
        'status': 'training', 'created_or_resumed_utc': utc_now(),
        'experiment': 'RS-BDNet xBD-only domain-generalization v1',
        'protocol': 'Fresh ImageNet-encoder initialization; train on source xBD train+tier3 except '
                    'entire held-out events; checkpoint selection on disjoint source events; '
                    'never use EBD for training, checkpoint selection, loss weights or thresholds',
        'original_checkpoint_read_only': str(original_ckpt),
        'original_checkpoint_sha256': orig_hash,
        'original_checkpoint_loaded_as_model_weights': False,
        'source_event_split': str(out_dir / 'source_event_split.json'),
        'train_sample_count': len(ds_train), 'validation_sample_count': len(ds_val),
        'source_train_class_pixel_counts': dict(zip(CLASS_NAMES, map(int, damage_counts))),
        'source_validation_class_pixel_counts': dict(zip(CLASS_NAMES, map(int, val_class_counts))),
        'source_loss_class_weights': list(map(float, class_weights)),
        'source_localization_positive_weight': float(pos_weight),
        'checkpoint_selection_metric': 'arithmetic mean of four source event-disjoint val damage F1 scores',
        'validation_label_semantics': 'runner.evaluate_f1, unchanged, localization-gated damage predictions',
        'source_threshold_scan_only_after_best_epoch': args.thresholds,
        'parameters': json_safe_args(args), 'seed': args.seed,
        'torch_version': torch.__version__, 'python_version': sys.version,
        'gpu': torch.cuda.get_device_name(device),
        'hostname': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'source_files_sha256': {str(path): sha256(path) for path in [
            Path(__file__),
            PROJECT_ROOT / 'transformer/scripts/train_xbd_supervised_disasteradaptivenet.py',
            PROJECT_ROOT / 'transformer/scripts/train_xbd_resnet34_swin_film_gated.py',
            PROJECT_ROOT / 'transformer/scripts/train_xbd_bitemporal_building_crossattention_ordinal.py',
        ]},
        'created_checkpoint_sha256': None, 'completed_utc': None,
    }
    write_json(out_dir / 'experiment_manifest.json', manifest)
    best_file = out_dir / 'checkpoints/best.pt'
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # Reset all training RNGs per epoch for predictable resumed training.
        epoch_seed = args.seed * 100003 + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed_all(epoch_seed)
        sampler_generator.manual_seed(epoch_seed)
        running_loss, processed = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            imgs = batch['img'].to(device, non_blocking=True)
            loc = batch['loc'].to(device, non_blocking=True)
            dmg = batch['dmg'].to(device, non_blocking=True)
            cond = batch['cond_id'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            context = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                       if args.amp else contextlib.nullcontext())
            with context:
                logits = model(imgs, cond)
                total, loc_bce, loc_dice, dmg_loss = arch.compute_losses(
                    logits, loc, dmg, loc_criterion, dmg_criterion, device,
                    argparse.Namespace(loc_bce_weight=1.0, loc_dice_weight=1.0, dmg_ce_weight=1.0))
                if bool((dmg != 255).any()):
                    ce = F.cross_entropy(logits[:, 1:5].float(), dmg, weight=dmg_weights,
                                         ignore_index=255)
                    total = total + args.aux_ce_weight * ce
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(f'Nonfinite training loss at epoch={epoch}, step={step}')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(total.detach().item()) * imgs.size(0)
            processed += imgs.size(0)
            if step % 50 == 0 or step == len(train_loader):
                print(f'[EPOCH {epoch}/{args.epochs}] step={step}/{len(train_loader)} '
                      f'loss={running_loss/max(1, processed):.5f}', flush=True)
        scheduler.step()
        results = runner.evaluate_f1(model, val_loader, device, 0.50)
        val = metrics_summary(results)
        improved = val['damage_macro_f1'] > best_macro + 1e-9
        if improved:
            best_macro, best_epoch = val['damage_macro_f1'], epoch
        record = {'epoch': epoch, 'train_loss': running_loss/max(1, processed),
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'best_macro_so_far': best_macro, 'best_epoch_so_far': best_epoch, **val}
        history.append(record)
        write_json(out_dir / 'history.json', history)
        state = make_checkpoint(model, optimizer, scheduler, epoch, best_epoch, best_macro,
                                0.50, args, train_events, val_events, history)
        if improved:
            atomic_save(state, best_file)
            print(f'[BEST] epoch={epoch} val_macro_f1={best_macro:.6f}', flush=True)
        atomic_save(state, out_dir / 'checkpoints/last.pt')
        print(f'[VALID] epoch={epoch} macro={val["damage_macro_f1"]:.6f} '
              f'loc={val["localization_f1"]:.6f} best_epoch={best_epoch}', flush=True)
        if epoch - best_epoch >= args.early_stop_patience:
            print(f'[EARLY STOP] {args.early_stop_patience} epochs without improvement', flush=True)
            break

    if not best_file.is_file():
        raise RuntimeError('No best checkpoint was written')
    best_ckpt = load_weights_only(best_file)
    model.load_state_dict(best_ckpt['model'], strict=True)
    model.eval()
    selected_threshold, best_threshold_results = 0.5, None
    threshold_rows = []
    for thr in args.thresholds:
        r = runner.evaluate_f1(model, val_loader, device, float(thr))
        macro = mean_macro_f1(r)
        threshold_rows.append([thr, macro, r['localization_f1'], r['damage_f1']])
        if best_threshold_results is None or macro > mean_macro_f1(best_threshold_results) + 1e-10:
            selected_threshold, best_threshold_results = float(thr), r
    write_csv(out_dir / 'source_validation_threshold_scan.csv',
              ['threshold', 'damage_macro_f1', 'localization_f1', 'original_harmonic_damage_f1'],
              threshold_rows)
    assert best_threshold_results is not None
    # Final best checkpoint may update its source-selected threshold and metadata,
    # but is written ONLY into this experiment's new directory.
    best_ckpt['best_threshold'] = selected_threshold
    best_ckpt['best_score'] = mean_macro_f1(best_threshold_results)
    best_ckpt['model_selection_score_at_fixed_threshold_0_5'] = best_macro
    best_ckpt['checkpoint_selection_source'] = 'xBD train+tier3 event-held-out validation'
    # Keep optimizer state for audit, though inference only loads model weights.
    atomic_save(best_ckpt, best_file)
    val_summary = metrics_summary(best_threshold_results)
    write_json(out_dir / 'source_validation_metrics.json', val_summary)
    print(f'[SOURCE SELECTION] best_epoch={best_epoch} threshold={selected_threshold} '
          f'heldout_macro={val_summary["damage_macro_f1"]:.6f}', flush=True)

    if not args.skip_xbd_test:
        ds_test = runner.XBDOriginalDataset(root=args.xbd_root, split='test',
                                             image_size=896, training=False, conditioning_id=0)
        test_loader = runner.make_eval_loader(ds_test, args.batch_size, args.num_workers)
        test_results = runner.evaluate_f1(model, test_loader, device, selected_threshold)
        write_json(out_dir / 'xbd_test_metrics.json', metrics_summary(test_results))
        manifest['xbd_test_results'] = metrics_summary(test_results)
        print(f'[xBD TEST] macro={mean_macro_f1(test_results):.6f}', flush=True)

    original_sha_post = sha256(original_ckpt)
    if original_sha_post != orig_hash:
        raise RuntimeError('Protected original checkpoint hash changed during training!')
    manifest.update({
        'status': 'completed', 'completed_utc': utc_now(),
        'best_epoch': best_epoch, 'best_val_macro_at_fixed_threshold_0_5': best_macro,
        'selected_threshold_source_only': selected_threshold,
        'selected_source_val_macro_f1': val_summary['damage_macro_f1'],
        'new_checkpoint': str(best_file),
        'created_checkpoint_sha256': sha256(best_file),
        'original_checkpoint_sha256_post': original_sha_post,
        'no_ebd_used': True,
    })
    write_json(out_dir / 'experiment_manifest.json', manifest)
    print(f'[COMPLETE] new checkpoint={best_file}', flush=True)
    print(f'[COMPLETE] source-only manifest={out_dir / "experiment_manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
