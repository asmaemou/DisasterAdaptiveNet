#!/usr/bin/env python3
"""
HRTBDA v5 for IDA-BD with automatic 80/10/10 split.

Pipeline:
  1) Preprocess IDA-BD into deterministic train/val/test splits.
  2) Train Phase I localization from scratch on IDA-BD.
  3) Train Phase II multi-label rare-crop damage model initialized from Phase I.
  4) Run validation ablation over Phase-I thresholds and dilation modes.
  5) Test once on the IDA-BD test split using the best validation setting.

Expected IDA-BD structure:
  ROOT/
    images/
      <tile>_pre_disaster.png|jpg|tif|...
      <tile>_post_disaster.png|jpg|tif|...
    masks/
      <tile>_post_disaster.png|jpg|tif|...
      optionally <tile>_pre_disaster.png|...

Mask labels:
  0 background
  1 no damage
  2 minor damage
  3 major damage
  4 destroyed
  255 ignore

This script reuses the HRTBDA v5 architecture/loss code from:
  train_xbd_hrtbda_v5_multilabel_crop_cascade.py
Place both scripts in the same transformer/scripts directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import the v5 architecture and utility functions from the xBD script.
# This keeps IDA-BD training using exactly the same HRTBDA v5 model components.
from train_xbd_hrtbda_v5_multilabel_crop_cascade import (  # noqa: E402
    HRTBDAPhase1,
    HRTBDAPhase2,
    BinaryFocalDiceLoss,
    MultilabelDamageFocalDiceLoss,
    AverageMeter,
    F1Recorder,
    apply_shared_augmentations,
    apply_extra_photometric_augmentations,
    rare_damage_candidate_crop,
    target5_to_multilabel_damage4,
    get_damage_logits,
    get_aux_loc_logits,
    evaluate_phase1,
    scan_phase1_thresholds,
    evaluate_phase2_cascade,
    make_loc_pos_weight,
    make_damage4_class_weights,
    load_phase1_backbone_into_phase2,
    save_checkpoint,
    load_model_weights,
    make_scaler,
    USE_TORCH_AMP,
    autocast,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# IDA-BD sample discovery and split preprocessing
# -----------------------------
@dataclass(frozen=True)
class IDABDSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    post_mask_path: Path
    pre_mask_path: Optional[Path] = None


def tile_id_from_name(path_or_name: str | Path) -> str:
    base = Path(path_or_name).stem
    return base.replace("_pre_disaster", "").replace("_post_disaster", "")


def list_images_by_split(images_dir: Path, split: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in IMG_EXTS:
        for p in images_dir.glob(f"*_{split}_disaster{ext}"):
            out[tile_id_from_name(p)] = p
    return dict(sorted(out.items()))


def find_mask(mask_dir: Path, stem: str, split: str = "post") -> Optional[Path]:
    for ext in IMG_EXTS:
        cand = mask_dir / f"{stem}_{split}_disaster{ext}"
        if cand.exists():
            return cand
    return None


def collect_idabd_samples(root: str | Path) -> List[IDABDSample]:
    root = Path(root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists():
        raise FileNotFoundError(f"Expected images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Expected masks directory not found: {masks_dir}")

    pre_map = list_images_by_split(images_dir, "pre")
    post_map = list_images_by_split(images_dir, "post")
    stems = sorted(set(pre_map.keys()) & set(post_map.keys()))

    samples: List[IDABDSample] = []
    for stem in stems:
        post_mask = find_mask(masks_dir, stem, split="post")
        if post_mask is None:
            continue
        pre_mask = find_mask(masks_dir, stem, split="pre")
        samples.append(
            IDABDSample(
                stem=stem,
                pre_image_path=pre_map[stem],
                post_image_path=post_map[stem],
                post_mask_path=post_mask,
                pre_mask_path=pre_mask,
            )
        )
    if not samples:
        raise RuntimeError(f"No valid IDA-BD triplets found under {root}. Check images/ and masks/ filenames.")
    return samples


def make_idabd_splits(
    samples: List[IDABDSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    stems = [s.stem for s in samples]
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Keep all leftovers in test so every sample is used exactly once.
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(1, n)
    n_val = min(max(n_val, 1), n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train)

    train = sorted(stems[:n_train])
    val = sorted(stems[n_train:n_train + n_val])
    test = sorted(stems[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


def prepare_or_load_splits(args: argparse.Namespace, samples: List[IDABDSample]) -> Dict[str, List[str]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = Path(args.split_file) if args.split_file else output_dir / f"idabd_splits_seed{args.seed}_80_10_10.json"

    if split_file.exists() and not args.force_resplit:
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        print(f"Loaded existing split file: {split_file}", flush=True)
    else:
        splits = make_idabd_splits(
            samples=samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"Wrote split file: {split_file}", flush=True)

    all_stems = {s.stem for s in samples}
    for key in ["train", "val", "test"]:
        splits[key] = [x for x in splits.get(key, []) if x in all_stems]
        if not splits[key]:
            raise RuntimeError(f"Split '{key}' is empty after filtering. Check split file and dataset.")

    print("===== IDA-BD SPLIT SUMMARY =====", flush=True)
    print(f"Train: {len(splits['train'])}", flush=True)
    print(f"Val:   {len(splits['val'])}", flush=True)
    print(f"Test:  {len(splits['test'])}", flush=True)
    print("=================================", flush=True)
    return splits


# -----------------------------
# Dataset
# -----------------------------
def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        # If RGB mask is present, use first channel/grayscale-like behavior.
        m = m[..., 0]
    m = m.astype(np.int64)
    # Keep legal labels, convert unknown labels to ignore.
    legal = (m == 0) | (m == 1) | (m == 2) | (m == 3) | (m == 4) | (m == 255)
    m = np.where(legal, m, 255).astype(np.uint8)
    return m


def resize_img_mask_pair(
    image_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    image_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    out_i: List[np.ndarray] = []
    out_m: List[np.ndarray] = []
    for img in image_list:
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out_i.append(img)
    for m in mask_list:
        if m.shape[:2] != (image_size, image_size):
            m = cv2.resize(m, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        out_m.append(m)
    return out_i, out_m


class IDABDHRTBDADataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        samples_by_stem: Dict[str, IDABDSample],
        stems: List[str],
        image_size: int,
        training: bool,
        crop_size: int = 0,
        crop_candidate_count: int = 1,
        crop_class_weights: Tuple[float, float, float, float] = (1.0, 12.0, 12.0, 4.0),
        extra_photometric: bool = False,
    ):
        self.root = Path(root)
        self.samples_by_stem = samples_by_stem
        self.stems = list(stems)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.crop_candidate_count = int(crop_candidate_count)
        self.crop_class_weights = tuple(float(x) for x in crop_class_weights)
        self.extra_photometric = bool(extra_photometric)

    def __len__(self) -> int:
        return len(self.stems)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        stem = self.stems[index]
        s = self.samples_by_stem[stem]

        pre = read_rgb(s.pre_image_path)
        post = read_rgb(s.post_image_path)
        target5 = read_mask(s.post_mask_path)

        # Use the post disaster damage mask as the building mask for Phase I target.
        # Any valid damage label 1..4 is a building pixel.
        loc_raw = np.isin(target5, [1, 2, 3, 4]).astype(np.uint8)

        [pre, post], [loc_raw, target5] = resize_img_mask_pair(
            image_list=[pre, post],
            mask_list=[loc_raw, target5],
            image_size=self.image_size,
        )

        [pre, post], [loc_raw, target5] = apply_shared_augmentations(
            image_list=[pre, post],
            mask_list=[loc_raw, target5],
            training=self.training,
            image_size=self.image_size,
        )

        if self.training and self.extra_photometric:
            [pre, post] = apply_extra_photometric_augmentations([pre, post], training=True)

        if self.training and self.crop_size > 0:
            [pre, post], [loc_raw, target5] = rare_damage_candidate_crop(
                image_list=[pre, post],
                mask_list=[loc_raw, target5],
                crop_size=self.crop_size,
                candidate_count=self.crop_candidate_count,
                class_weights=self.crop_class_weights,
            )

        loc = (loc_raw > 0).astype(np.float32)

        return {
            "pre": torch.from_numpy(self._normalize(pre)).float(),
            "post": torch.from_numpy(self._normalize(post)).float(),
            "loc": torch.from_numpy(loc).float(),
            "target5": torch.from_numpy(target5.astype(np.int64)).long(),
            "stem": stem,
            "split": "idabd",
        }

    def localization_counts(self) -> Tuple[int, int]:
        pos = 0
        neg = 0
        for stem in self.stems:
            m = read_mask(self.samples_by_stem[stem].post_mask_path)
            loc = np.isin(m, [1, 2, 3, 4])
            pos += int(loc.sum())
            neg += int((~loc).sum())
        return pos, neg

    def class5_counts(self) -> np.ndarray:
        counts = np.zeros(5, dtype=np.int64)
        for stem in self.stems:
            m = read_mask(self.samples_by_stem[stem].post_mask_path)
            valid = m != 255
            vals, freqs = np.unique(m[valid], return_counts=True)
            for v, f in zip(vals.tolist(), freqs.tolist()):
                if 0 <= int(v) <= 4:
                    counts[int(v)] += int(f)
        counts[counts == 0] = 1
        return counts


# -----------------------------
# Loaders/schedulers
# -----------------------------
def make_loaders(args: argparse.Namespace, phase2_training: bool = False):
    samples = collect_idabd_samples(args.idabd_root)
    sample_map = {s.stem: s for s in samples}
    splits = prepare_or_load_splits(args, samples)

    crop_weights = (
        args.crop_weight_no_damage,
        args.crop_weight_minor,
        args.crop_weight_major,
        args.crop_weight_destroyed,
    )

    train_ds = IDABDHRTBDADataset(
        root=args.idabd_root,
        samples_by_stem=sample_map,
        stems=splits["train"],
        image_size=args.img_size,
        training=True,
        crop_size=args.phase2_crop_size if phase2_training else 0,
        crop_candidate_count=args.crop_candidate_count if phase2_training else 1,
        crop_class_weights=crop_weights,
        extra_photometric=args.extra_photometric_aug if phase2_training else False,
    )
    val_ds = IDABDHRTBDADataset(args.idabd_root, sample_map, splits["val"], args.img_size, training=False)
    test_ds = IDABDHRTBDADataset(args.idabd_root, sample_map, splits["test"], args.img_size, training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.phase2_batch_size if phase2_training else args.phase1_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, train_ds


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def backward_step(loss, model, optimizer, scaler, args):
    scaler.scale(loss).backward()
    if args.max_grad_norm is not None and args.max_grad_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()


# -----------------------------
# Phase I training
# -----------------------------
def train_phase1(args: argparse.Namespace, device: torch.device) -> Path:
    print("\n================ IDA-BD PHASE I: BUILDING LOCALIZATION ================", flush=True)
    train_loader, val_loader, _, train_ds = make_loaders(args, phase2_training=False)
    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)

    model = HRTBDAPhase1(args.base_channels, args.decoder_channels, args.window_size).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    loc_pos_weight = make_loc_pos_weight(train_ds).to(device)
    print(f"Phase I loc pos_weight: {loc_pos_weight.detach().cpu().numpy().tolist()}", flush=True)
    criterion = BinaryFocalDiceLoss(pos_weight=loc_pos_weight, gamma=args.focal_gamma).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = make_scheduler(optimizer, args.phase1_epochs, args.warmup_epochs)
    scaler = make_scaler(args, device)

    out = Path(args.output_dir)
    ckpt_dir = out / "checkpoints"
    scores_dir = out / "scores"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    history = []

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        total_meter, focal_meter, dice_meter = AverageMeter(), AverageMeter(), AverageMeter()
        print(f"\nPhase I epoch {epoch}/{args.phase1_epochs} | LR={optimizer.param_groups[0]['lr']:.8f}", flush=True)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            pre = batch["pre"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    logits = model(pre)
                    loss, focal, dice = criterion(logits, loc)
                    loss = args.loc_loss_weight * loss
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(pre)
                    loss, focal, dice = criterion(logits, loc)
                    loss = args.loc_loss_weight * loss

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite Phase I loss at epoch={epoch} step={step}; skipping step.", flush=True)
                continue

            backward_step(loss, model, optimizer, scaler, args)
            bs = pre.size(0)
            total_meter.update(loss.item(), bs)
            focal_meter.update(focal.item(), bs)
            dice_meter.update(dice.item(), bs)

            if step % args.log_every == 0 or step == len(train_loader):
                print(
                    f"Phase I Epoch {epoch}/{args.phase1_epochs} | Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | focal={focal_meter.avg:.4f} | dice={dice_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()
        scan_csv = scores_dir / f"phase1_epoch_{epoch:03d}_threshold_scan.csv"
        th, val_res = scan_phase1_thresholds(model, val_loader, device, args.thresholds, scan_csv)
        val_f1 = float(val_res["localization_f1"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_focal": focal_meter.avg,
            "train_dice": dice_meter.avg,
            "val_best_threshold": th,
            "val_localization_f1": val_f1,
        }
        history.append(row)
        with open(out / "history_phase1.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"Phase I Epoch {epoch:03d} | val_loc_f1={val_f1:.6f} | threshold={th:.2f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_threshold = th
            save_checkpoint(
                ckpt_dir / "phase1_best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_f1,
                args,
                extra={"best_threshold": best_threshold, "dataset": "IDABD"},
            )
            print(f"Saved Phase I best | epoch={epoch} | loc_f1={best_f1:.6f} | threshold={best_threshold:.2f}", flush=True)

        save_checkpoint(
            ckpt_dir / "phase1_last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_f1,
            args,
            extra={"best_threshold": best_threshold, "dataset": "IDABD"},
        )
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(
                ckpt_dir / f"phase1_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_f1,
                args,
                extra={"best_threshold": best_threshold, "dataset": "IDABD"},
            )

    print(f"Phase I done. Best epoch={best_epoch}, best val loc F1={best_f1:.6f}, threshold={best_threshold:.2f}", flush=True)
    return ckpt_dir / "phase1_best.pt"


def load_phase1_model(args: argparse.Namespace, device: torch.device, phase1_ckpt: Path) -> Tuple[nn.Module, float, Dict[str, object]]:
    model = HRTBDAPhase1(args.base_channels, args.decoder_channels, args.window_size).to(device)
    ckpt = load_model_weights(model, phase1_ckpt, device)
    threshold = float(ckpt.get("best_threshold", 0.5))
    meta = {
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": float(ckpt.get("best_metric", -1.0)),
        "best_threshold": threshold,
    }
    model.eval()
    print("Loaded Phase I model for IDA-BD cascade.", flush=True)
    print(f"Phase I checkpoint: {phase1_ckpt}", flush=True)
    print(f"Phase I epoch: {meta['epoch']} | best_metric={meta['best_metric']:.6f} | threshold={threshold:.2f}", flush=True)
    return model, threshold, meta


# -----------------------------
# Phase II training + validation ablation + test
# -----------------------------
def train_phase2(args: argparse.Namespace, device: torch.device, phase1_ckpt: Path) -> Path:
    print("\n================ IDA-BD PHASE II: MULTI-LABEL RARE-CROP DAMAGE ================", flush=True)
    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")

    train_loader, val_loader, _, train_ds = make_loaders(args, phase2_training=True)
    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)
    print(f"Phase II crop size: {args.phase2_crop_size}", flush=True)
    print(f"Crop candidates: {args.crop_candidate_count}", flush=True)

    phase1_eval_model, phase1_threshold, phase1_meta = load_phase1_model(args, device, phase1_ckpt)

    model = HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size, num_classes=4).to(device)
    load_phase1_backbone_into_phase2(model, phase1_ckpt, device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    class_weights = make_damage4_class_weights(train_ds, args=args).to(device)
    criterion = MultilabelDamageFocalDiceLoss(class_weights=class_weights, gamma=args.focal_gamma).to(device)

    loc_pos_weight = make_loc_pos_weight(train_ds).to(device)
    aux_loc_criterion = BinaryFocalDiceLoss(pos_weight=loc_pos_weight, gamma=args.focal_gamma).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = make_scheduler(optimizer, args.phase2_epochs, args.warmup_epochs)
    scaler = make_scaler(args, device)

    out = Path(args.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    history = []
    accum = max(1, int(args.grad_accum_steps))

    def run_training_epoch(epoch: int, total_epochs: int, phase_name: str):
        model.train()
        total_meter, focal_meter, dice_meter, aux_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        print(f"\n{phase_name} epoch {epoch}/{total_epochs} | LR={optimizer.param_groups[0]['lr']:.8f}", flush=True)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            target5 = batch["target5"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            damage_target, valid_mask = target5_to_multilabel_damage4(target5)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    out_model = model(pre, post)
                    damage_logits = get_damage_logits(out_model)
                    aux_loc = get_aux_loc_logits(out_model)
                    loss_damage, focal, dice = criterion(damage_logits, damage_target, valid_mask)
                    if aux_loc is not None and args.aux_loc_weight > 0:
                        loss_aux, _, _ = aux_loc_criterion(aux_loc, loc)
                    else:
                        loss_aux = damage_logits.sum() * 0.0
                    loss = args.cls_loss_weight * loss_damage + args.aux_loc_weight * loss_aux
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    out_model = model(pre, post)
                    damage_logits = get_damage_logits(out_model)
                    aux_loc = get_aux_loc_logits(out_model)
                    loss_damage, focal, dice = criterion(damage_logits, damage_target, valid_mask)
                    if aux_loc is not None and args.aux_loc_weight > 0:
                        loss_aux, _, _ = aux_loc_criterion(aux_loc, loc)
                    else:
                        loss_aux = damage_logits.sum() * 0.0
                    loss = args.cls_loss_weight * loss_damage + args.aux_loc_weight * loss_aux

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite Phase II loss at epoch={epoch} step={step}; skipping step.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(train_loader):
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = pre.size(0)
            total_meter.update(loss.item(), bs)
            focal_meter.update(focal.item(), bs)
            dice_meter.update(dice.item(), bs)
            aux_meter.update(loss_aux.item(), bs)

            if step % args.log_every == 0 or step == len(train_loader):
                print(
                    f"{phase_name} Epoch {epoch}/{total_epochs} | Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | focal={focal_meter.avg:.4f} | dice={dice_meter.avg:.4f} | aux={aux_meter.avg:.4f}",
                    flush=True,
                )
        return total_meter, focal_meter, dice_meter, aux_meter

    def validate_and_save(epoch_label: int, meters, phase_label: str):
        nonlocal best_score, best_epoch, no_improve
        total_meter, focal_meter, dice_meter, aux_meter = meters
        val_results = evaluate_phase2_cascade(
            phase1_model=phase1_eval_model,
            phase2_model=model,
            loader=val_loader,
            device=device,
            phase1_threshold=phase1_threshold,
            postprocess_dilation=args.postprocess_dilation,
            dilation_kernel=args.dilation_kernel,
        )
        val_score = float(val_results["score"])
        row = {
            "epoch": epoch_label,
            "phase_label": phase_label,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_focal": focal_meter.avg,
            "train_dice": dice_meter.avg,
            "train_aux_loc": aux_meter.avg,
            "val_score_cascade": val_score,
            "val_localization_f1_from_phase1_mask": float(val_results["localization_f1"]),
            "val_damage_f1": float(val_results["damage_f1"]),
            "val_no_damage_f1": float(val_results["damage_f1_no_damage"]),
            "val_minor_damage_f1": float(val_results["damage_f1_minor_damage"]),
            "val_major_damage_f1": float(val_results["damage_f1_major_damage"]),
            "val_destroyed_f1": float(val_results["damage_f1_destroyed"]),
            "phase1_threshold": phase1_threshold,
            "postprocess_dilation": args.postprocess_dilation,
        }
        history.append(row)
        with open(Path(args.output_dir) / "history_phase2.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"{phase_label} Epoch {epoch_label:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_score={row['val_score_cascade']:.6f} | val_loc={row['val_localization_f1_from_phase1_mask']:.6f} | "
            f"val_damage={row['val_damage_f1']:.6f} | no={row['val_no_damage_f1']:.6f} | "
            f"minor={row['val_minor_damage_f1']:.6f} | major={row['val_major_damage_f1']:.6f} | destroyed={row['val_destroyed_f1']:.6f}",
            flush=True,
        )

        extra = {
            "phase1_checkpoint": str(phase1_ckpt),
            "phase1_threshold": phase1_threshold,
            "phase1_best_metric_val": phase1_meta["best_metric"],
            "cascade_validation": True,
            "multilabel_damage_heads": True,
            "rare_crop_training": True,
            "dataset": "IDABD",
        }
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch_label
            no_improve = 0
            save_checkpoint(ckpt_dir / "phase2_best.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)
            print(f"Saved Phase II best | epoch={epoch_label} | val cascade score={best_score:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"Phase II no improvement for {no_improve} epoch(s). Best epoch={best_epoch}", flush=True)

        save_checkpoint(ckpt_dir / "phase2_last.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)
        if epoch_label % max(1, args.save_every) == 0:
            save_checkpoint(ckpt_dir / f"phase2_epoch_{epoch_label:03d}.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)

    for epoch in range(1, args.phase2_epochs + 1):
        meters = run_training_epoch(epoch, args.phase2_epochs, phase_name="Phase II")
        scheduler.step()
        validate_and_save(epoch, meters, phase_label="main")
        if no_improve >= args.early_stopping_patience:
            print(f"Phase II early stopping at epoch {epoch}.", flush=True)
            break

    if args.finetune_epochs > 0:
        print("\n================ IDA-BD PHASE II SHORT FINE-TUNING ================", flush=True)
        for ft_epoch in range(1, args.finetune_epochs + 1):
            for g in optimizer.param_groups:
                g["lr"] = float(args.finetune_lr) * (0.5 ** (ft_epoch - 1))
            epoch_label = args.phase2_epochs + ft_epoch
            meters = run_training_epoch(ft_epoch, args.finetune_epochs, phase_name="Fine-tune")
            validate_and_save(epoch_label, meters, phase_label="finetune")

    print(f"Phase II done. Best epoch={best_epoch}, best val cascade score={best_score:.6f}", flush=True)
    return ckpt_dir / "phase2_best.pt"


def evaluate_val_ablation_and_test(args: argparse.Namespace, device: torch.device, phase1_ckpt: Path, phase2_ckpt: Path) -> None:
    print("\n================ IDA-BD VALIDATION ABLATION + FINAL TEST ================", flush=True)
    _, val_loader, test_loader, _ = make_loaders(args, phase2_training=False)

    phase1_model, stored_threshold, phase1_meta = load_phase1_model(args, device, phase1_ckpt)
    phase2_model = HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size, num_classes=4).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        phase2_model = nn.DataParallel(phase2_model)
    ckpt2 = load_model_weights(phase2_model, phase2_ckpt, device)
    phase2_epoch = int(ckpt2.get("epoch", -1))

    thresholds = args.ablation_thresholds
    dilations = args.ablation_dilations
    out = Path(args.output_dir)
    scores_dir = out / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best = None
    for th in thresholds:
        for dil in dilations:
            res = evaluate_phase2_cascade(
                phase1_model=phase1_model,
                phase2_model=phase2_model,
                loader=val_loader,
                device=device,
                phase1_threshold=float(th),
                postprocess_dilation=dil,
                dilation_kernel=args.dilation_kernel,
            )
            row = {
                "threshold": float(th),
                "dilation": dil,
                "localization_f1": float(res["localization_f1"]),
                "no_damage_f1": float(res["damage_f1_no_damage"]),
                "minor_damage_f1": float(res["damage_f1_minor_damage"]),
                "major_damage_f1": float(res["damage_f1_major_damage"]),
                "destroyed_f1": float(res["damage_f1_destroyed"]),
                "damage_f1": float(res["damage_f1"]),
                "overall_score": float(res["score"]),
            }
            rows.append(row)
            if best is None or row["overall_score"] > best["overall_score"]:
                best = row
            print(
                f"VAL ABLATION | th={th:.2f} dil={dil} | loc={row['localization_f1']:.6f} | "
                f"no={row['no_damage_f1']:.6f} | minor={row['minor_damage_f1']:.6f} | "
                f"major={row['major_damage_f1']:.6f} | destroyed={row['destroyed_f1']:.6f} | "
                f"damage={row['damage_f1']:.6f} | overall={row['overall_score']:.6f}",
                flush=True,
            )

    assert best is not None
    with open(scores_dir / "idabd_val_threshold_dilation_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(scores_dir / "idabd_val_threshold_dilation_ablation.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "best": best}, f, indent=2)

    print("\n===== BEST VALIDATION SETTING =====", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    print("===================================", flush=True)

    # Final test only once using validation-selected setting.
    test_res = evaluate_phase2_cascade(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=test_loader,
        device=device,
        phase1_threshold=float(best["threshold"]),
        postprocess_dilation=str(best["dilation"]),
        dilation_kernel=args.dilation_kernel,
    )

    final = {
        "experiment": "IDABD HRTBDA v5 multi-label rare-crop cascade",
        "phase1_checkpoint": str(phase1_ckpt),
        "phase2_checkpoint": str(phase2_ckpt),
        "phase1_epoch": phase1_meta["epoch"],
        "phase1_stored_threshold": stored_threshold,
        "phase2_best_epoch_selected_on_val": phase2_epoch,
        "selected_threshold_from_val": best["threshold"],
        "selected_dilation_from_val": best["dilation"],
        "val_best": best,
        "test": test_res,
    }
    with open(scores_dir / "idabd_final_test_selected_setting.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    lines = [
        "Experiment: IDA-BD HRTBDA v5 multi-label rare-crop cascade 80/10/10 split",
        f"IDA-BD root: {args.idabd_root}",
        f"Phase I checkpoint: {phase1_ckpt}",
        f"Phase I stored epoch: {phase1_meta['epoch']}",
        f"Phase I stored val Localization F1: {phase1_meta['best_metric']:.6f}",
        f"Phase II checkpoint: {phase2_ckpt}",
        f"Best Phase II epoch selected on val: {phase2_epoch}",
        f"Selected Phase-I threshold from val ablation: {best['threshold']:.2f}",
        f"Selected dilation from val ablation: {best['dilation']}",
        "",
        "Validation-selected setting:",
        f"Val Localization F1: {best['localization_f1']:.6f}",
        f"Val No Damage F1:    {best['no_damage_f1']:.6f}",
        f"Val Minor Damage F1: {best['minor_damage_f1']:.6f}",
        f"Val Major Damage F1: {best['major_damage_f1']:.6f}",
        f"Val Destroyed F1:    {best['destroyed_f1']:.6f}",
        f"Val Damage F1:       {best['damage_f1']:.6f}",
        f"Val Overall Score:   {best['overall_score']:.6f}",
        "",
        "Final test result:",
        f"Test Localization F1 from Phase I mask: {test_res['localization_f1']:.6f}",
        f"No Damage F1:    {test_res['damage_f1_no_damage']:.6f}",
        f"Minor Damage F1: {test_res['damage_f1_minor_damage']:.6f}",
        f"Major Damage F1: {test_res['damage_f1_major_damage']:.6f}",
        f"Destroyed F1:    {test_res['damage_f1_destroyed']:.6f}",
        f"Damage F1:       {test_res['damage_f1']:.6f}",
        f"Overall Score:   {test_res['score']:.6f}",
    ]
    summary_path = scores_dir / "summary_idabd_hrtbda_v5_selected_test.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"Wrote: {summary_path}", flush=True)


# -----------------------------
# Args/main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train HRTBDA v5 on IDA-BD with 80/10/10 split")
    p.add_argument("--phase", default="both", choices=["prepare", "phase1", "phase2", "test", "both"])
    p.add_argument("--idabd-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-file", default="")
    p.add_argument("--force-resplit", action="store_true")
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)

    p.add_argument("--phase1-checkpoint", default="")
    p.add_argument("--phase2-checkpoint", default="")
    p.add_argument("--phase1-epochs", type=int, default=150)
    p.add_argument("--phase2-epochs", type=int, default=60)
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--finetune-lr", type=float, default=5e-5)

    p.add_argument("--phase1-batch-size", type=int, default=1)
    p.add_argument("--phase2-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--phase2-crop-size", type=int, default=608)
    p.add_argument("--crop-candidate-count", type=int, default=8)
    p.add_argument("--crop-weight-no-damage", type=float, default=1.0)
    p.add_argument("--crop-weight-minor", type=float, default=12.0)
    p.add_argument("--crop-weight-major", type=float, default=12.0)
    p.add_argument("--crop-weight-destroyed", type=float, default=4.0)
    p.add_argument("--extra-photometric-aug", action="store_true")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--early-stopping-patience", type=int, default=999)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--loc-loss-weight", type=float, default=1.0)
    p.add_argument("--cls-loss-weight", type=float, default=1.0)
    p.add_argument("--aux-loc-weight", type=float, default=0.2)
    p.add_argument("--minor-damage-boost", type=float, default=1.5)
    p.add_argument("--major-damage-boost", type=float, default=1.5)
    p.add_argument("--max-damage-class-weight", type=float, default=10.0)

    p.add_argument("--postprocess-dilation", default="none", choices=["none", "minor", "minor_major"])
    p.add_argument("--dilation-kernel", type=int, default=3)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90])
    p.add_argument("--ablation-thresholds", type=float, nargs="+", default=[0.40,0.45,0.50,0.55])
    p.add_argument("--ablation-dilations", nargs="+", default=["none", "minor"], choices=["none", "minor", "minor_major"])
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "scores").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== IDA-BD HRTBDA V5 MULTI-LABEL RARE-CROP CASCADED TRAINING =====", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"IDA-BD root: {args.idabd_root}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}", flush=True)
    print(f"Phase I epochs: {args.phase1_epochs}", flush=True)
    print(f"Phase II epochs: {args.phase2_epochs}", flush=True)
    print(f"Phase I batch size: {args.phase1_batch_size}", flush=True)
    print(f"Phase II batch size: {args.phase2_batch_size}", flush=True)
    print(f"Effective Phase II batch size: {args.phase2_batch_size * args.grad_accum_steps}", flush=True)
    print(f"Phase II crop size: {args.phase2_crop_size}", flush=True)
    print(f"Ablation thresholds: {args.ablation_thresholds}", flush=True)
    print(f"Ablation dilations: {args.ablation_dilations}", flush=True)
    print("Architecture: HRTBDA v5 4-branch HRNet-style + DCSwin + CSF fusion", flush=True)
    print("Inference: Phase I mask gates Phase II multi-label damage prediction.", flush=True)
    print("======================================================================", flush=True)

    # Always collect and prepare/load split first.
    samples = collect_idabd_samples(args.idabd_root)
    prepare_or_load_splits(args, samples)

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    phase1_ckpt = Path(args.phase1_checkpoint) if args.phase1_checkpoint else ckpt_dir / "phase1_best.pt"
    phase2_ckpt = Path(args.phase2_checkpoint) if args.phase2_checkpoint else ckpt_dir / "phase2_best.pt"

    if args.phase == "prepare":
        return
    if args.phase == "phase1":
        train_phase1(args, device)
    elif args.phase == "phase2":
        train_phase2(args, device, phase1_ckpt)
    elif args.phase == "test":
        evaluate_val_ablation_and_test(args, device, phase1_ckpt, phase2_ckpt)
    elif args.phase == "both":
        phase1_ckpt = train_phase1(args, device)
        phase2_ckpt = train_phase2(args, device, phase1_ckpt)
        evaluate_val_ablation_and_test(args, device, phase1_ckpt, phase2_ckpt)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
