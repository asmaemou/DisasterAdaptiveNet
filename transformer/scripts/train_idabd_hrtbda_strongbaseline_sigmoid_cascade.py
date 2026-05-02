#!/usr/bin/env python3
"""
IDA-BD HRTBDA strong-baseline-inspired cascaded training.

Keeps your HRTBDA v2 two-stage cascade:
  Phase I: pre-disaster image -> binary building localization mask
  Phase II: pre/post images -> damage severity inside Phase-I mask
  Final inference: outside Phase-I mask = background; inside mask = Phase-II damage class

Borrows the strongest useful ideas from the strong-baseline paper:
  - Phase-II predicts independent sigmoid channels instead of a softmax
  - Phase-II has 5 channels: aux localization + 4 damage channels
  - binary focal + soft Dice loss per channel
  - inverse-frequency class weighting with extra rare-class boosts
  - rare-damage crop selection for Phase-II training, resized to 608
  - optional conservative minor/destroyed dilation at inference
  - validation threshold/dilation ablation, then real test using the best validation setting

This script imports the HRTBDA v2 model blocks from:
  train_xbd_hrtbda_v2_cascaded_phase1mask.py
Keep that file in transformer/scripts/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse your HRTBDA v2 architecture blocks and Phase-I training.
import train_xbd_hrtbda_v2_cascaded_phase1mask as v2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

try:
    GradScaler = torch.amp.GradScaler
    autocast = torch.amp.autocast
    USE_TORCH_AMP = True
except AttributeError:
    from torch.cuda.amp import GradScaler, autocast
    USE_TORCH_AMP = False

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


# -----------------------------
# IDA-BD discovery and split
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
    for suffix in [
        "_pre_disaster_target",
        "_post_disaster_target",
        "_pre_disaster_mask",
        "_post_disaster_mask",
        "_pre_disaster",
        "_post_disaster",
        "_target",
        "_mask",
    ]:
        base = base.replace(suffix, "")
    return base


def list_images_by_split(images_dir: Path, split: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in IMG_EXTS:
        for p in images_dir.glob(f"*_{split}_disaster{ext}"):
            out[tile_id_from_name(p)] = p
    return dict(sorted(out.items()))


def find_mask(masks_dir: Path, stem: str, split: str = "post") -> Optional[Path]:
    candidate_bases = [
        f"{stem}_{split}_disaster_target",
        f"{stem}_{split}_disaster_mask",
        f"{stem}_{split}_disaster",
    ]
    if split == "post":
        candidate_bases += [f"{stem}_target", f"{stem}_mask", stem]

    for base in candidate_bases:
        for ext in IMG_EXTS:
            p = masks_dir / f"{base}{ext}"
            if p.exists():
                return p
    return None


def collect_idabd_samples(root: str | Path) -> List[IDABDSample]:
    root = Path(root)
    images_dir = root / "images"
    masks_dir = root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Expected IDA-BD images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Expected IDA-BD masks directory not found: {masks_dir}")

    pre_map = list_images_by_split(images_dir, "pre")
    post_map = list_images_by_split(images_dir, "post")
    stems = sorted(set(pre_map.keys()) & set(post_map.keys()))

    samples: List[IDABDSample] = []
    missing_masks: List[str] = []
    for stem in stems:
        post_mask = find_mask(masks_dir, stem, split="post")
        if post_mask is None:
            missing_masks.append(stem)
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
        raise RuntimeError(
            f"No valid IDA-BD paired samples found under {root}. Expected images/*_pre_disaster.*, "
            "images/*_post_disaster.*, and matching masks."
        )
    if missing_masks:
        print(f"WARNING: skipped {len(missing_masks)} sample(s) with missing post masks.", flush=True)
    return samples


def make_idabd_splits(
    samples: List[IDABDSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    stems = [s.stem for s in samples]
    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    if n < 3:
        raise RuntimeError(f"Need at least 3 samples for train/val/test split. Found {n}.")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    return {
        "train": sorted(stems[:n_train]),
        "val": sorted(stems[n_train:n_train + n_val]),
        "test": sorted(stems[n_train + n_val:]),
    }


def prepare_or_load_splits(args: argparse.Namespace, samples: List[IDABDSample]) -> Dict[str, List[str]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file = Path(args.split_file) if args.split_file else output_dir / f"idabd_splits_seed{args.seed}_80_10_10.json"
    if split_file.exists() and not args.force_resplit:
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        print(f"Loaded existing IDA-BD split file: {split_file}", flush=True)
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
        print(f"Wrote IDA-BD split file: {split_file}", flush=True)

    all_stems = {s.stem for s in samples}
    for key in ["train", "val", "test"]:
        splits[key] = [s for s in splits.get(key, []) if s in all_stems]
        if not splits[key]:
            raise RuntimeError(f"Split '{key}' is empty. Check split file or dataset discovery.")

    print("===== IDA-BD SPLIT SUMMARY =====", flush=True)
    print(f"Train: {len(splits['train'])}", flush=True)
    print(f"Val:   {len(splits['val'])}", flush=True)
    print(f"Test:  {len(splits['test'])}", flush=True)
    print("=================================", flush=True)
    return splits


# -----------------------------
# IO / preprocessing
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
        m = m[..., 0]
    m = m.astype(np.int64)
    legal = (m == 0) | (m == 1) | (m == 2) | (m == 3) | (m == 4) | (m == 255)
    m = np.where(legal, m, 255).astype(np.uint8)
    return m


def apply_extra_photometric_aug(image_list: List[np.ndarray]) -> List[np.ndarray]:
    """Extra photometric augmentation borrowed from the strong-baseline/winning-solution family."""
    out = image_list

    # RGB channel shift.
    if np.random.rand() < 0.35:
        shifts = np.random.uniform(-12, 12, size=(1, 1, 3)).astype(np.float32)
        out = [np.clip(img.astype(np.float32) + shifts, 0, 255).astype(np.uint8) for img in out]

    # HSV saturation/value jitter.
    if np.random.rand() < 0.35:
        jittered = []
        for img in out:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= np.random.uniform(0.80, 1.25)
            hsv[..., 2] *= np.random.uniform(0.80, 1.25)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            jittered.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        out = jittered

    # CLAHE on luminance.
    if np.random.rand() < 0.20:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahed = []
        for img in out:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            clahed.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        out = clahed

    # Contrast jitter.
    if np.random.rand() < 0.35:
        alpha = np.random.uniform(0.80, 1.25)
        beta = np.random.uniform(-8, 8)
        out = [np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8) for img in out]

    return out


class IDABDHRTBDADataset(Dataset):
    """IDA-BD dataset exposing the same keys as your xBD datasets."""

    def __init__(
        self,
        root: str | Path,
        samples_by_stem: Dict[str, IDABDSample],
        stems: List[str],
        image_size: int,
        training: bool,
        phase2_crop_size: int = 0,
        crop_candidates: int = 1,
        crop_class_weights: Tuple[float, float, float, float] = (1.0, 8.0, 8.0, 25.0),
        random_crop_min: int = 529,
        random_crop_max: int = 715,
        extra_photometric_aug: bool = False,
    ):
        self.root = Path(root)
        self.samples_by_stem = samples_by_stem
        self.stems = list(stems)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.phase2_crop_size = int(phase2_crop_size)
        self.crop_candidates = max(1, int(crop_candidates))
        self.crop_class_weights = np.array(crop_class_weights, dtype=np.float32)
        self.random_crop_min = int(random_crop_min)
        self.random_crop_max = int(random_crop_max)
        self.extra_photometric_aug = bool(extra_photometric_aug)

    def __len__(self) -> int:
        return len(self.stems)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def _loc_from_sample(self, s: IDABDSample) -> np.ndarray:
        if s.pre_mask_path is not None:
            pre_mask = read_mask(s.pre_mask_path)
            return (pre_mask > 0).astype(np.uint8)
        post_mask = read_mask(s.post_mask_path)
        return np.isin(post_mask, [1, 2, 3, 4]).astype(np.uint8)

    def _crop_score(self, target5: np.ndarray, top: int, left: int, side: int) -> float:
        crop = target5[top:top + side, left:left + side]
        score = 0.0
        for i, cls_id in enumerate([1, 2, 3, 4]):
            score += float(self.crop_class_weights[i]) * float((crop == cls_id).sum())
        # Small building bonus so completely empty crops are avoided.
        score += 0.05 * float(np.isin(crop, [1, 2, 3, 4]).sum())
        return score

    def _rare_damage_crop(
        self,
        image_list: List[np.ndarray],
        mask_list: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        crop_out = self.phase2_crop_size
        if not self.training or crop_out <= 0:
            return image_list, mask_list

        h, w = image_list[0].shape[:2]
        min_side = min(max(16, self.random_crop_min), h, w)
        max_side = min(max(min_side, self.random_crop_max), h, w)
        target5 = mask_list[1]

        best = None
        best_score = -1.0
        for _ in range(self.crop_candidates):
            side = int(np.random.randint(min_side, max_side + 1))
            top = int(np.random.randint(0, h - side + 1))
            left = int(np.random.randint(0, w - side + 1))
            score = self._crop_score(target5, top, left, side)
            # tiny random tie-breaker
            score += float(np.random.rand()) * 1e-3
            if score > best_score:
                best_score = score
                best = (top, left, side)

        assert best is not None
        top, left, side = best
        cropped_imgs = [img[top:top + side, left:left + side] for img in image_list]
        cropped_masks = [m[top:top + side, left:left + side] for m in mask_list]

        cropped_imgs = [cv2.resize(img, (crop_out, crop_out), interpolation=cv2.INTER_LINEAR) for img in cropped_imgs]
        cropped_masks = [cv2.resize(m, (crop_out, crop_out), interpolation=cv2.INTER_NEAREST) for m in cropped_masks]
        return cropped_imgs, cropped_masks

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        stem = self.stems[index]
        s = self.samples_by_stem[stem]

        pre = read_rgb(s.pre_image_path)
        post = read_rgb(s.post_image_path)
        loc_raw = self._loc_from_sample(s)
        target5 = read_mask(s.post_mask_path)

        [pre, post], [loc_raw, target5] = v2.resize_rgb_and_masks(
            image_list=[pre, post],
            mask_list=[loc_raw, target5],
            image_size=self.image_size,
        )

        [pre, post], [loc_raw, target5] = v2.apply_shared_augmentations(
            image_list=[pre, post],
            mask_list=[loc_raw, target5],
            training=self.training,
            image_size=self.image_size,
        )

        if self.training and self.extra_photometric_aug:
            [pre, post] = apply_extra_photometric_aug([pre, post])

        [pre, post], [loc_raw, target5] = self._rare_damage_crop(
            image_list=[pre, post],
            mask_list=[loc_raw, target5],
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
            loc = self._loc_from_sample(self.samples_by_stem[stem]) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())
        return pos, neg

    def class5_counts(self) -> np.ndarray:
        counts = np.zeros(5, dtype=np.int64)
        for stem in self.stems:
            m = read_mask(self.samples_by_stem[stem].post_mask_path)
            valid = m != 255
            vals, freqs = np.unique(m[valid], return_counts=True)
            for val, freq in zip(vals.tolist(), freqs.tolist()):
                if 0 <= int(val) <= 4:
                    counts[int(val)] += int(freq)
        counts[counts == 0] = 1
        return counts

    def foreground_damage_counts(self) -> np.ndarray:
        counts = np.zeros(4, dtype=np.int64)
        for stem in self.stems:
            m = read_mask(self.samples_by_stem[stem].post_mask_path)
            for i, cls_id in enumerate([1, 2, 3, 4]):
                counts[i] += int((m == cls_id).sum())
        counts[counts == 0] = 1
        return counts


# -----------------------------
# Loaders
# -----------------------------
def build_idabd_datasets(args: argparse.Namespace, phase2_train: bool = False):
    samples = collect_idabd_samples(args.idabd_root)
    sample_map = {s.stem: s for s in samples}
    splits = prepare_or_load_splits(args, samples)

    crop_size = int(args.phase2_crop_size) if phase2_train else 0
    crop_weights = tuple(float(x) for x in args.crop_class_weights)

    train_ds = IDABDHRTBDADataset(
        args.idabd_root,
        sample_map,
        splits["train"],
        args.img_size,
        training=True,
        phase2_crop_size=crop_size,
        crop_candidates=args.crop_candidates,
        crop_class_weights=crop_weights,
        random_crop_min=args.random_crop_min,
        random_crop_max=args.random_crop_max,
        extra_photometric_aug=args.extra_photometric_aug,
    )
    val_ds = IDABDHRTBDADataset(args.idabd_root, sample_map, splits["val"], args.img_size, training=False)
    test_ds = IDABDHRTBDADataset(args.idabd_root, sample_map, splits["test"], args.img_size, training=False)
    return train_ds, val_ds, test_ds


def make_loaders_for_phase1(args: argparse.Namespace):
    train_ds, val_ds, test_ds = build_idabd_datasets(args, phase2_train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds


def make_loaders_for_phase2(args: argparse.Namespace):
    train_ds, val_ds, test_ds = build_idabd_datasets(args, phase2_train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds


# -----------------------------
# Strong-baseline-style loss
# -----------------------------
def make_binary_targets(target5: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """Return Bx5xHxW targets: loc, no, minor, major, destroyed."""
    b, h, w = target5.shape
    y = torch.zeros((b, 5, h, w), device=target5.device, dtype=torch.float32)
    y[:, 0] = loc.float()
    for ch, cls_id in enumerate([1, 2, 3, 4], start=1):
        y[:, ch] = (target5 == cls_id).float()
    return y


class WeightedBinaryComboLoss(nn.Module):
    """Per-channel binary focal + soft Dice loss with channel weights."""

    def __init__(self, channel_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("channel_weights", channel_weights.float())
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # logits/target: B,C,H,W; valid_mask: B,1,H,W
        valid = valid_mask.float()
        weights = self.channel_weights.to(logits.device)
        weights = weights / weights.sum().clamp_min(1e-7)

        total = logits.sum() * 0.0
        focal_total = logits.sum() * 0.0
        dice_total = logits.sum() * 0.0

        for c in range(logits.shape[1]):
            logit_c = logits[:, c:c + 1]
            target_c = target[:, c:c + 1]

            bce = F.binary_cross_entropy_with_logits(logit_c, target_c, reduction="none")
            pt = torch.exp(-bce)
            focal = ((1.0 - pt) ** self.gamma * bce * valid).sum() / valid.sum().clamp_min(1.0)

            prob = torch.sigmoid(logit_c) * valid
            tgt = target_c * valid
            inter = (prob * tgt).sum(dim=(0, 2, 3))
            denom = prob.sum(dim=(0, 2, 3)) + tgt.sum(dim=(0, 2, 3))
            dice = 1.0 - ((2.0 * inter + 1e-7) / (denom + 1e-7)).mean()

            combo = focal + dice
            total = total + weights[c] * combo
            focal_total = focal_total + weights[c] * focal
            dice_total = dice_total + weights[c] * dice

        return total, focal_total, dice_total


def make_strong_channel_weights(args: argparse.Namespace, train_ds: IDABDHRTBDADataset) -> torch.Tensor:
    # loc channel uses localization balance; damage channels use foreground damage class frequencies.
    loc_pos, loc_neg = train_ds.localization_counts()
    loc_freq = max(loc_pos, 1) / max(loc_pos + loc_neg, 1)
    loc_w = min(1.0 / max(loc_freq, 1e-6), args.max_class_weight)

    dmg_counts = train_ds.foreground_damage_counts().astype(np.float64)
    dmg_freq = dmg_counts / max(float(dmg_counts.sum()), 1.0)
    dmg_w = 1.0 / np.sqrt(dmg_freq + 1e-12)
    dmg_w = dmg_w / dmg_w.mean()
    dmg_w[1] *= args.minor_boost
    dmg_w[2] *= args.major_boost
    dmg_w[3] *= args.destroyed_boost
    dmg_w = np.clip(dmg_w, 0.05, args.max_class_weight)

    weights = np.concatenate([[loc_w * args.aux_loc_weight], dmg_w]).astype(np.float32)
    print(f"damage counts [no,minor,major,destroyed]: {dmg_counts.astype(int).tolist()}", flush=True)
    print(f"strong channel weights [loc,no,minor,major,destroyed]: {weights.tolist()}", flush=True)
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": float(best_metric),
        "args": vars(args),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_model_weights(model: nn.Module, path: Path, device: torch.device) -> Dict[str, object]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"]
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return ckpt


def make_scaler(args: argparse.Namespace, device: torch.device):
    enabled = bool(args.amp and device.type == "cuda")
    if USE_TORCH_AMP:
        return GradScaler(device.type, enabled=enabled)
    return GradScaler(enabled=enabled)


def amp_context(args: argparse.Namespace, device: torch.device):
    if USE_TORCH_AMP:
        return autocast(device_type=device.type, enabled=args.amp and device.type == "cuda")
    return autocast(enabled=args.amp and device.type == "cuda")


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# -----------------------------
# Evaluation
# -----------------------------
def apply_damage_dilation(final_np: np.ndarray, loc_np: np.ndarray, mode: str, kernel_size: int) -> np.ndarray:
    if mode == "none" or kernel_size <= 1:
        return final_np

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = final_np.copy()
    loc_bool = loc_np.astype(bool)

    if mode in {"minor", "minor_destroyed", "all"}:
        minor = (out == 2).astype(np.uint8)
        dil = cv2.dilate(minor, kernel, iterations=1).astype(bool)
        # Conservative: minor only overwrites no-damage inside Phase-I mask.
        out[dil & loc_bool & (out == 1)] = 2

    if mode in {"destroyed", "minor_destroyed", "all"}:
        destroyed = (out == 4).astype(np.uint8)
        dil = cv2.dilate(destroyed, kernel, iterations=1).astype(bool)
        # Conservative: destroyed overwrites no/minor inside Phase-I mask, not major.
        out[dil & loc_bool & ((out == 1) | (out == 2))] = 4

    return out


@torch.no_grad()
def evaluate_strong_cascade(
    args: argparse.Namespace,
    phase1_model: nn.Module,
    phase2_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    phase1_threshold: float,
    dilation_mode: str = "none",
    dilation_kernel: int = 3,
) -> Dict[str, object]:
    phase1_model.eval()
    phase2_model.eval()

    loc_tp = loc_fp = loc_fn = 0
    cls_counts = {1: {"tp": 0, "fp": 0, "fn": 0}, 2: {"tp": 0, "fp": 0, "fn": 0}, 3: {"tp": 0, "fp": 0, "fn": 0}, 4: {"tp": 0, "fp": 0, "fn": 0}}

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        target = batch["target5"].to(device, non_blocking=True).long()

        loc_logits = phase1_model(pre)
        loc_pred = (torch.sigmoid(loc_logits) > phase1_threshold).long()

        logits = phase2_model(pre, post)
        probs = torch.sigmoid(logits)
        damage_pred = torch.argmax(probs[:, 1:5], dim=1) + 1  # 1..4
        final = torch.where(loc_pred == 1, damage_pred, torch.zeros_like(damage_pred))

        if dilation_mode != "none":
            final_cpu = final.detach().cpu().numpy().astype(np.uint8)
            loc_cpu = loc_pred.detach().cpu().numpy().astype(np.uint8)
            for i in range(final_cpu.shape[0]):
                final_cpu[i] = apply_damage_dilation(final_cpu[i], loc_cpu[i], dilation_mode, dilation_kernel)
            final = torch.from_numpy(final_cpu).to(device=device, dtype=torch.long)

        loc_final = (final > 0).long()
        loc_tp += int(((loc_final == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_final == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_final == 0) & (loc_true == 1)).sum().item())

        valid_building = (target >= 1) & (target <= 4)
        pred_valid = final[valid_building]
        true_valid = target[valid_building]
        for cls in [1, 2, 3, 4]:
            cls_counts[cls]["tp"] += int(((pred_valid == cls) & (true_valid == cls)).sum().item())
            cls_counts[cls]["fp"] += int(((pred_valid == cls) & (true_valid != cls)).sum().item())
            cls_counts[cls]["fn"] += int(((pred_valid != cls) & (true_valid == cls)).sum().item())

    loc_rec = v2.F1Recorder(loc_tp, loc_fp, loc_fn)
    no = v2.F1Recorder(cls_counts[1]["tp"], cls_counts[1]["fp"], cls_counts[1]["fn"])
    minor = v2.F1Recorder(cls_counts[2]["tp"], cls_counts[2]["fp"], cls_counts[2]["fn"])
    major = v2.F1Recorder(cls_counts[3]["tp"], cls_counts[3]["fp"], cls_counts[3]["fn"])
    destroyed = v2.F1Recorder(cls_counts[4]["tp"], cls_counts[4]["fp"], cls_counts[4]["fn"])

    damage_f1 = v2.harmonic_mean([no.f1, minor.f1, major.f1, destroyed.f1])
    score = 0.3 * loc_rec.f1 + 0.7 * damage_f1

    return {
        "score": score,
        "localization_f1": loc_rec.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no.f1,
        "damage_f1_minor_damage": minor.f1,
        "damage_f1_major_damage": major.f1,
        "damage_f1_destroyed": destroyed.f1,
        "phase1_threshold": phase1_threshold,
        "postprocess_dilation": dilation_mode,
        "dilation_kernel": dilation_kernel,
        "details": {
            "localization": loc_rec.as_dict(),
            "no_damage": no.as_dict(),
            "minor_damage": minor.as_dict(),
            "major_damage": major.as_dict(),
            "destroyed": destroyed.as_dict(),
        },
    }


def load_phase1_model(args: argparse.Namespace, device: torch.device, phase1_ckpt: Path):
    return v2.load_phase1_model_for_cascade(args=args, device=device, phase1_ckpt=phase1_ckpt)


def make_phase2_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = v2.HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=5,
    ).to(device)
    return model


# -----------------------------
# Training Phase II strong
# -----------------------------
def train_phase2_strong(args: argparse.Namespace, device: torch.device, phase1_ckpt: Path) -> Path:
    print("\n================ PHASE II: STRONG-BASELINE SIGMOID DAMAGE HEAD ================", flush=True)
    train_loader, val_loader, _, train_ds = make_loaders_for_phase2(args)
    print(f"Train samples: {len(train_loader.dataset)} | steps/epoch={len(train_loader)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)
    print(f"Phase-II crop size: {args.phase2_crop_size}", flush=True)
    print(f"Crop candidates: {args.crop_candidates}", flush=True)
    print(f"Crop class weights [no,minor,major,destroyed]: {args.crop_class_weights}", flush=True)

    model = make_phase2_model(args, device)
    if phase1_ckpt.exists():
        v2.load_phase1_backbone_into_phase2(model, phase1_ckpt, device)
    else:
        print(f"WARNING: Phase I checkpoint not found, Phase II backbone starts random: {phase1_ckpt}", flush=True)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    channel_weights = make_strong_channel_weights(args, train_ds).to(device)
    criterion = WeightedBinaryComboLoss(channel_weights=channel_weights, gamma=args.focal_gamma).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = make_scheduler(optimizer, args.phase2_epochs, warmup_epochs=args.warmup_epochs)
    scaler = make_scaler(args, device)

    phase1_model, phase1_threshold, phase1_meta = load_phase1_model(args, device, phase1_ckpt)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    scores_dir = output_dir / "scores"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    history: List[Dict[str, float | int]] = []

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        total_meter = v2.AverageMeter()
        focal_meter = v2.AverageMeter()
        dice_meter = v2.AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        print(f"\nPhase II epoch {epoch}/{args.phase2_epochs} | LR={optimizer.param_groups[0]['lr']:.8f}", flush=True)

        for step, batch in enumerate(train_loader, start=1):
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            target5 = batch["target5"].to(device, non_blocking=True)
            valid = (target5 != 255).unsqueeze(1).float()
            target_bin = make_binary_targets(target5, loc)

            with amp_context(args, device):
                logits = model(pre, post)
                loss, focal, dice = criterion(logits, target_bin, valid)
                loss = loss / max(1, args.grad_accum_steps)

            if not torch.isfinite(loss):
                print(f"ERROR: non-finite loss at epoch={epoch} step={step}. Stop and use the best checkpoint so far.", flush=True)
                return checkpoints_dir / "phase2_best.pt"

            scaler.scale(loss).backward()

            do_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
            if do_step:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = pre.size(0)
            total_meter.update(loss.item() * max(1, args.grad_accum_steps), bs)
            focal_meter.update(focal.item(), bs)
            dice_meter.update(dice.item(), bs)

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Phase II Epoch {epoch}/{args.phase2_epochs} | Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | focal={focal_meter.avg:.4f} | dice={dice_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()

        val_results = evaluate_strong_cascade(
            args=args,
            phase1_model=phase1_model,
            phase2_model=model,
            loader=val_loader,
            device=device,
            phase1_threshold=phase1_threshold,
            dilation_mode="none",
            dilation_kernel=args.dilation_kernel,
        )
        val_score = float(val_results["score"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_focal": focal_meter.avg,
            "train_dice": dice_meter.avg,
            "hold_score_cascade": val_score,
            "hold_loc_f1_from_phase1": float(val_results["localization_f1"]),
            "hold_damage_f1": float(val_results["damage_f1"]),
            "hold_no_damage_f1": float(val_results["damage_f1_no_damage"]),
            "hold_minor_damage_f1": float(val_results["damage_f1_minor_damage"]),
            "hold_major_damage_f1": float(val_results["damage_f1_major_damage"]),
            "hold_destroyed_f1": float(val_results["damage_f1_destroyed"]),
        }
        history.append(row)

        print(
            f"strong Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"hold_score_cascade={row['hold_score_cascade']:.6f} | "
            f"hold_loc_f1_from_phase1={row['hold_loc_f1_from_phase1']:.6f} | "
            f"hold_damage_f1={row['hold_damage_f1']:.6f} | "
            f"no={row['hold_no_damage_f1']:.6f} | minor={row['hold_minor_damage_f1']:.6f} | "
            f"major={row['hold_major_damage_f1']:.6f} | destroyed={row['hold_destroyed_f1']:.6f}",
            flush=True,
        )

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                path=checkpoints_dir / "phase2_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_score,
                args=args,
                extra={
                    "phase1_checkpoint": str(phase1_ckpt),
                    "phase1_threshold": float(phase1_threshold),
                    "strong_baseline_sigmoid_head": True,
                    "channel_weights": channel_weights.detach().cpu().numpy().tolist(),
                },
            )
            print(f"Saved Phase II best checkpoint | epoch={epoch} | cascade_score={best_score:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"Phase II no improvement for {no_improve} epoch(s). Best epoch={best_epoch}", flush=True)

        save_checkpoint(
            path=checkpoints_dir / "phase2_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_score,
            args=args,
        )
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(
                path=checkpoints_dir / f"phase2_epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_score,
                args=args,
            )
        with open(output_dir / "history_phase2_strong.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.early_stopping_patience:
            print(f"Phase II early stopping at epoch {epoch}.", flush=True)
            break

    print(f"Phase II done. Best epoch={best_epoch}, best hold score={best_score:.6f}", flush=True)
    return checkpoints_dir / "phase2_best.pt"


# -----------------------------
# Test and ablation
# -----------------------------
def test_strong(
    args: argparse.Namespace,
    device: torch.device,
    phase1_ckpt: Path,
    phase2_ckpt: Path,
    threshold: Optional[float] = None,
    dilation_mode: Optional[str] = None,
    output_suffix: str = "test",
) -> Dict[str, object]:
    print("\n================ CASCADED TESTING: STRONG SIGMOID HEAD ================", flush=True)
    _, _, test_loader, _ = make_loaders_for_phase2(args)
    phase1_model, stored_threshold, phase1_meta = load_phase1_model(args, device, phase1_ckpt)
    th = float(stored_threshold if threshold is None else threshold)
    dil = args.postprocess_dilation if dilation_mode is None else dilation_mode

    phase2_model = make_phase2_model(args, device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        phase2_model = nn.DataParallel(phase2_model)
    ckpt = load_model_weights(phase2_model, phase2_ckpt, device)
    best_epoch = int(ckpt.get("epoch", -1))

    results = evaluate_strong_cascade(
        args=args,
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=test_loader,
        device=device,
        phase1_threshold=th,
        dilation_mode=dil,
        dilation_kernel=args.dilation_kernel,
    )
    results.update({
        "phase1_checkpoint": str(phase1_ckpt),
        "phase2_checkpoint": str(phase2_ckpt),
        "phase1_epoch": int(phase1_meta.get("epoch", -1)),
        "phase1_best_metric_hold": float(phase1_meta.get("best_metric", -1.0)),
        "phase2_best_epoch_selected_on_hold": best_epoch,
    })

    scores_dir = Path(args.output_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    json_path = scores_dir / f"scores_idabd_strong_sigmoid_{output_suffix}.json"
    txt_path = scores_dir / f"summary_idabd_strong_sigmoid_{output_suffix}.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "Experiment: IDA-BD HRTBDA strong-baseline sigmoid damage head cascade train80 -> val10 -> test10",
        f"Phase I checkpoint: {phase1_ckpt}",
        f"Phase I stored best epoch: {phase1_meta.get('epoch', -1)}",
        f"Phase I stored hold Localization F1: {float(phase1_meta.get('best_metric', -1.0)):.6f}",
        f"Phase I threshold used for mask: {th:.2f}",
        f"Postprocess dilation: {dil}",
        f"Best Phase II epoch selected on hold cascade score: {best_epoch}",
        f"Test Localization F1 from Phase I mask: {results['localization_f1']:.6f}",
        f"No Damage F1:    {results['damage_f1_no_damage']:.6f}",
        f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}",
        f"Major Damage F1: {results['damage_f1_major_damage']:.6f}",
        f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}",
        f"Damage F1:       {results['damage_f1']:.6f}",
        f"Overall Score:   {results['score']:.6f}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)
    return results


def run_validation_ablation_then_test(args: argparse.Namespace, device: torch.device, phase1_ckpt: Path, phase2_ckpt: Path) -> None:
    print("\n================ VALIDATION THRESHOLD/DILATION ABLATION ================", flush=True)
    _, val_loader, test_loader, _ = make_loaders_for_phase2(args)
    phase1_model, stored_threshold, phase1_meta = load_phase1_model(args, device, phase1_ckpt)

    phase2_model = make_phase2_model(args, device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        phase2_model = nn.DataParallel(phase2_model)
    phase2_meta = load_model_weights(phase2_model, phase2_ckpt, device)

    thresholds = [float(x) for x in args.ablation_thresholds]
    dilations = [str(x) for x in args.ablation_dilations]
    rows: List[Dict[str, object]] = []

    for th in thresholds:
        for dil in dilations:
            res = evaluate_strong_cascade(
                args=args,
                phase1_model=phase1_model,
                phase2_model=phase2_model,
                loader=val_loader,
                device=device,
                phase1_threshold=th,
                dilation_mode=dil,
                dilation_kernel=args.dilation_kernel,
            )
            row = {
                "threshold": th,
                "dilation": dil,
                "localization_f1": float(res["localization_f1"]),
                "no_damage_f1": float(res["damage_f1_no_damage"]),
                "minor_damage_f1": float(res["damage_f1_minor_damage"]),
                "major_damage_f1": float(res["damage_f1_major_damage"]),
                "destroyed_f1": float(res["damage_f1_destroyed"]),
                "damage_f1": float(res["damage_f1"]),
                "overall": float(res["score"]),
            }
            rows.append(row)
            print(
                f"VAL th={th:.2f} dil={dil:15s} | loc={row['localization_f1']:.6f} | "
                f"no={row['no_damage_f1']:.6f} minor={row['minor_damage_f1']:.6f} "
                f"major={row['major_damage_f1']:.6f} destroyed={row['destroyed_f1']:.6f} | "
                f"damage={row['damage_f1']:.6f} overall={row['overall']:.6f}",
                flush=True,
            )

    rows = sorted(rows, key=lambda r: float(r["overall"]), reverse=True)
    best = rows[0]

    scores_dir = Path(args.output_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scores_dir / "validation_threshold_dilation_ablation.csv"
    json_path = scores_dir / "validation_threshold_dilation_ablation.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\n================ VALIDATION ABLATION RANKING ================", flush=True)
    print("Rank | Threshold | Dilation | Loc F1 | No F1 | Minor F1 | Major F1 | Destroyed F1 | Damage F1 | Overall", flush=True)
    for i, r in enumerate(rows, start=1):
        print(
            f"{i:4d} | {float(r['threshold']):.2f} | {str(r['dilation']):15s} | "
            f"{float(r['localization_f1']):.6f} | {float(r['no_damage_f1']):.6f} | "
            f"{float(r['minor_damage_f1']):.6f} | {float(r['major_damage_f1']):.6f} | "
            f"{float(r['destroyed_f1']):.6f} | {float(r['damage_f1']):.6f} | {float(r['overall']):.6f}",
            flush=True,
        )
    print(f"Wrote: {csv_path}", flush=True)
    print(f"Wrote: {json_path}", flush=True)

    if args.run_final_test:
        best_th = float(best["threshold"])
        best_dil = str(best["dilation"])
        print(f"\nSelected on validation: threshold={best_th:.2f}, dilation={best_dil}. Running real test.", flush=True)
        test_res = evaluate_strong_cascade(
            args=args,
            phase1_model=phase1_model,
            phase2_model=phase2_model,
            loader=test_loader,
            device=device,
            phase1_threshold=best_th,
            dilation_mode=best_dil,
            dilation_kernel=args.dilation_kernel,
        )
        test_res.update({
            "selected_on_validation_threshold": best_th,
            "selected_on_validation_dilation": best_dil,
            "phase1_checkpoint": str(phase1_ckpt),
            "phase2_checkpoint": str(phase2_ckpt),
            "phase1_epoch": int(phase1_meta.get("epoch", -1)),
            "phase1_best_metric_hold": float(phase1_meta.get("best_metric", -1.0)),
            "phase2_best_epoch_selected_on_hold": int(phase2_meta.get("epoch", -1)),
        })
        test_json = scores_dir / "final_test_selected_by_validation.json"
        test_txt = scores_dir / "summary_final_test_selected_by_validation.txt"
        with open(test_json, "w", encoding="utf-8") as f:
            json.dump(test_res, f, indent=2)

        lines = [
            "Experiment: IDA-BD HRTBDA strong-baseline sigmoid damage head cascade",
            f"Selected validation threshold: {best_th:.2f}",
            f"Selected validation dilation: {best_dil}",
            f"Phase I checkpoint: {phase1_ckpt}",
            f"Phase II checkpoint: {phase2_ckpt}",
            f"Test Localization F1 from Phase I mask: {test_res['localization_f1']:.6f}",
            f"No Damage F1:    {test_res['damage_f1_no_damage']:.6f}",
            f"Minor Damage F1: {test_res['damage_f1_minor_damage']:.6f}",
            f"Major Damage F1: {test_res['damage_f1_major_damage']:.6f}",
            f"Destroyed F1:    {test_res['damage_f1_destroyed']:.6f}",
            f"Damage F1:       {test_res['damage_f1']:.6f}",
            f"Overall Score:   {test_res['score']:.6f}",
        ]
        with open(test_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        print(f"Wrote: {test_json}", flush=True)
        print(f"Wrote: {test_txt}", flush=True)


# -----------------------------
# Args / main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("IDA-BD HRTBDA strong-baseline sigmoid cascade")

    parser.add_argument("--phase", type=str, default="phase2_test", choices=["both", "phase1", "phase2", "phase2_test", "test", "ablate", "inspect_phase1"])
    parser.add_argument("--idabd-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-file", type=str, default="")
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)

    # Kept for compatibility with v2 print strings.
    parser.add_argument("--train-split", type=str, nargs="+", default=["train80"])
    parser.add_argument("--val-split", type=str, default="val10")
    parser.add_argument("--test-split", type=str, default="test10")

    parser.add_argument("--phase1-checkpoint", type=str, default="")
    parser.add_argument("--phase2-checkpoint", type=str, default="")

    parser.add_argument("--phase1-epochs", type=int, default=150)
    parser.add_argument("--phase2-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--loc-loss-weight", type=float, default=1.0)  # used by imported Phase I
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)  # compatibility

    parser.add_argument("--phase2-crop-size", type=int, default=608)
    parser.add_argument("--crop-candidates", type=int, default=16)
    parser.add_argument("--random-crop-min", type=int, default=529)
    parser.add_argument("--random-crop-max", type=int, default=715)
    parser.add_argument("--crop-class-weights", type=float, nargs=4, default=[1.0, 10.0, 10.0, 30.0], help="Rare-crop selection weights [no, minor, major, destroyed]")
    parser.add_argument("--extra-photometric-aug", action="store_true")

    parser.add_argument("--aux-loc-weight", type=float, default=0.25)
    parser.add_argument("--minor-boost", type=float, default=2.0)
    parser.add_argument("--major-boost", type=float, default=2.0)
    parser.add_argument("--destroyed-boost", type=float, default=8.0)
    parser.add_argument("--max-class-weight", type=float, default=25.0)

    parser.add_argument("--postprocess-dilation", type=str, default="none", choices=["none", "minor", "destroyed", "minor_destroyed", "all"])
    parser.add_argument("--dilation-kernel", type=int, default=3)
    parser.add_argument("--ablation-thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--ablation-dilations", type=str, nargs="+", default=["none", "minor", "destroyed", "minor_destroyed"])
    parser.add_argument("--run-final-test", action="store_true")

    # v2 Phase-I threshold scanning.
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.xbd_root = args.idabd_root  # compatibility for imported v2 code
    v2.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "scores").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints_dir = output_dir / "checkpoints"
    phase1_ckpt = Path(args.phase1_checkpoint) if args.phase1_checkpoint else checkpoints_dir / "phase1_best.pt"
    phase2_ckpt = Path(args.phase2_checkpoint) if args.phase2_checkpoint else checkpoints_dir / "phase2_best.pt"

    print("===== IDA-BD HRTBDA STRONG-BASELINE SIGMOID CASCADE =====", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print(f"IDA-BD root: {args.idabd_root}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Split file: {args.split_file or '(create in output dir)'}", flush=True)
    print(f"Device: {device}", flush=True)
    print("Architecture: HRTBDA v2 4-branch transformer + CSF, Phase-II 5 sigmoid channels", flush=True)
    print("Phase-II loss: binary focal + dice per channel, inverse-frequency rare-class weighting", flush=True)
    print("Training: rare-damage crop selection + optional photometric augmentation", flush=True)
    print("Final inference: Phase-I mask gates background; Phase-II predicts damage inside mask", flush=True)
    print("========================================================", flush=True)

    # Monkey-patch only for imported Phase-I training.
    v2.make_loaders = make_loaders_for_phase1

    if args.phase == "inspect_phase1":
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
        v2.print_phase1_checkpoint_summary(phase1_ckpt, device)
        return

    if args.phase == "phase1":
        v2.train_phase1(args, device)

    elif args.phase == "phase2":
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
        train_phase2_strong(args, device, phase1_ckpt)

    elif args.phase == "phase2_test":
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
        phase2_ckpt = train_phase2_strong(args, device, phase1_ckpt)
        run_validation_ablation_then_test(args, device, phase1_ckpt, phase2_ckpt)

    elif args.phase == "test":
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
        if not phase2_ckpt.exists():
            raise FileNotFoundError(f"Phase II checkpoint not found: {phase2_ckpt}")
        # If no explicit ablation is requested, use checkpoint threshold and args.postprocess_dilation.
        test_strong(args, device, phase1_ckpt, phase2_ckpt, output_suffix="manual")

    elif args.phase == "ablate":
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
        if not phase2_ckpt.exists():
            raise FileNotFoundError(f"Phase II checkpoint not found: {phase2_ckpt}")
        run_validation_ablation_then_test(args, device, phase1_ckpt, phase2_ckpt)

    elif args.phase == "both":
        phase1_ckpt = v2.train_phase1(args, device)
        phase2_ckpt = train_phase2_strong(args, device, phase1_ckpt)
        run_validation_ablation_then_test(args, device, phase1_ckpt, phase2_ckpt)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
