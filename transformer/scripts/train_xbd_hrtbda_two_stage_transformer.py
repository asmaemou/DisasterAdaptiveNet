#!/usr/bin/env python3
"""
HRTBDA v2 four-branch warmup AMP experiment.

This is based on the first HRTBDA-inspired script, with targeted upgrades:
  - train on xBD train + tier3, validate on hold, test on test
  - 1024x1024 input support
  - 4-resolution HRNet-style backbone branch instead of 3 branches
  - warmup + cosine LR schedule instead of linear decay from epoch 1
  - capped localization pos_weight to avoid overly aggressive BCE+focal weighting
  - richer augmentation: flips, rotations, scale crop, brightness jitter, blur, noise
  - AMP enabled from the sbatch for speed/memory
  - Phase I localization checkpoint initializes Phase II Siamese damage model
  - Cascaded two-phase inference: Phase I mask is the final building/background mask
  - Phase II is trained as foreground-only 4-class damage severity classification
  - Rare-damage weighted sampler for Phase II, inspired by the older crop-focused pipeline
  - Optional vector-scaling calibration on hold logits with class-balanced damage pixels
  - Final Phase-I mask threshold sweep on hold for the end-to-end cascaded score

This is still an HRTBDA-inspired experimental implementation, not official author code.
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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    GradScaler = torch.amp.GradScaler
    autocast = torch.amp.autocast
    USE_TORCH_AMP = True
except AttributeError:
    from torch.cuda.amp import GradScaler, autocast
    USE_TORCH_AMP = False


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Better speed for fixed-size 1024 images.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Dataset
# -----------------------------
@dataclass(frozen=True)
class XBDSample:
    stem: str
    split: str
    pre_image_path: Path
    post_image_path: Path
    pre_target_path: Path
    post_target_path: Path


def resize_rgb_and_masks(
    image_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    image_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    out_imgs: List[np.ndarray] = []
    out_masks: List[np.ndarray] = []

    for img in image_list:
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(img)

    for mask in mask_list:
        if mask.shape[:2] != (image_size, image_size):
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        out_masks.append(mask)

    return out_imgs, out_masks


def apply_shared_augmentations(
    image_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    training: bool,
    image_size: int = 1024,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Apply shared geometric augmentation to images/masks and photometric augmentation to images."""
    if not training:
        return image_list, mask_list

    if np.random.rand() < 0.5:
        image_list = [np.flip(x, axis=1).copy() for x in image_list]
        mask_list = [np.flip(x, axis=1).copy() for x in mask_list]

    if np.random.rand() < 0.5:
        image_list = [np.flip(x, axis=0).copy() for x in image_list]
        mask_list = [np.flip(x, axis=0).copy() for x in mask_list]

    k = np.random.randint(0, 4)
    if k:
        image_list = [np.rot90(x, k=k).copy() for x in image_list]
        mask_list = [np.rot90(x, k=k).copy() for x in mask_list]

    # Scale jitter: crop 75-100% of the image and resize back.
    if np.random.rand() < 0.5:
        h, w = image_list[0].shape[:2]
        scale = np.random.uniform(0.75, 1.0)
        crop_h, crop_w = max(1, int(h * scale)), max(1, int(w * scale))
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        image_list = [
            cv2.resize(x[top:top + crop_h, left:left + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
            for x in image_list
        ]
        mask_list = [
            cv2.resize(x[top:top + crop_h, left:left + crop_w], (w, h), interpolation=cv2.INTER_NEAREST)
            for x in mask_list
        ]

    # Brightness jitter.
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.75, 1.25)
        image_list = [np.clip(x.astype(np.float32) * factor, 0, 255).astype(np.uint8) for x in image_list]

    # Blur.
    if np.random.rand() < 0.3:
        ksize = int(np.random.choice([3, 5]))
        image_list = [cv2.GaussianBlur(x, (ksize, ksize), 0) for x in image_list]

    # Mild Gaussian noise.
    if np.random.rand() < 0.25:
        noise_std = np.random.uniform(2.0, 8.0)
        noisy = []
        for img in image_list:
            n = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
            noisy.append(np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8))
        image_list = noisy

    return image_list, mask_list

class XBDHRTBDADataset(Dataset):
    """
    Expected structure:

    /homes/j244s673/documents/wsu/phd/xview2/
      train/
        images/
        targets/
      tier3/
        images/
        targets/
      hold/
        images/
        targets/
      test/
        images/
        targets/

    Supports one or more training splits, for example:
      --train-split train tier3

    Uses:
      *_pre_disaster.png
      *_post_disaster.png
      *_pre_disaster_target.png
      *_post_disaster_target.png

    Phase I target:
      loc: 0 background, 1 building

    Phase II target:
      0   background
      1   no damage
      2   minor damage
      3   major damage
      4   destroyed
      255 ignored / other labels
    """

    def __init__(self, root: str | Path, split: str | List[str] | Tuple[str, ...], image_size: int, training: bool):
        self.root = Path(root)

        if isinstance(split, (list, tuple)):
            self.splits = [str(s) for s in split]
        else:
            self.splits = [str(split)]

        self.split = "+".join(self.splits)
        self.image_size = int(image_size)
        self.training = bool(training)

        for split_name in self.splits:
            split_root = self.root / split_name
            images_dir = split_root / "images"
            targets_dir = split_root / "targets"

            if not images_dir.exists():
                raise FileNotFoundError(f"Expected images dir not found: {images_dir}")
            if not targets_dir.exists():
                raise FileNotFoundError(f"Expected targets dir not found: {targets_dir}")

        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.root} for splits {self.splits}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    @staticmethod
    def _target5_from_masks(loc: np.ndarray, dmg: np.ndarray) -> np.ndarray:
        loc_bin = loc > 0
        target = np.zeros(loc.shape, dtype=np.uint8)

        target[(dmg == 1) & loc_bin] = 1
        target[(dmg == 2) & loc_bin] = 2
        target[(dmg == 3) & loc_bin] = 3
        target[(dmg == 4) & loc_bin] = 4

        other_building = loc_bin & ~np.isin(dmg, [1, 2, 3, 4])
        target[other_building] = 255

        return target

    def _collect_samples(self) -> List[XBDSample]:
        samples: List[XBDSample] = []
        seen: set[str] = set()

        for split_name in self.splits:
            split_root = self.root / split_name
            images_dir = split_root / "images"
            targets_dir = split_root / "targets"

            post_images: List[Path] = []

            for pattern in [
                "*_post_disaster.png",
                "*_post_disaster.jpg",
                "*_post_disaster.jpeg",
                "*_post_disaster.tif",
                "*_post_disaster.tiff",
                "*_post_disaster.bmp",
            ]:
                post_images.extend(images_dir.glob(pattern))

            post_images = sorted(post_images)

            for post_path in post_images:
                prefix = post_path.stem.replace("_post_disaster", "")
                ext = post_path.suffix

                pre_path = images_dir / f"{prefix}_pre_disaster{ext}"
                pre_tgt = targets_dir / f"{prefix}_pre_disaster_target.png"
                post_tgt = targets_dir / f"{prefix}_post_disaster_target.png"

                # Avoid accidental duplicates if the same sample exists in both train and tier3.
                key = prefix

                if key in seen:
                    continue

                if pre_path.exists() and pre_tgt.exists() and post_tgt.exists():
                    seen.add(key)
                    samples.append(
                        XBDSample(
                            stem=prefix,
                            split=split_name,
                            pre_image_path=pre_path,
                            post_image_path=post_path,
                            pre_target_path=pre_tgt,
                            post_target_path=post_tgt,
                        )
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        x = (x - self.mean) / self.std
        return x

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        s = self.samples[index]

        pre = self._read_rgb(s.pre_image_path)
        post = self._read_rgb(s.post_image_path)
        loc_raw = self._read_mask(s.pre_target_path)
        dmg_raw = self._read_mask(s.post_target_path)

        target5 = self._target5_from_masks(loc_raw, dmg_raw)

        [pre, post], [loc_raw, target5] = resize_rgb_and_masks(
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

        loc = (loc_raw > 0).astype(np.float32)

        return {
            "pre": torch.from_numpy(self._normalize(pre)).float(),
            "post": torch.from_numpy(self._normalize(post)).float(),
            "loc": torch.from_numpy(loc).float(),
            "target5": torch.from_numpy(target5).long(),
            "stem": s.stem,
            "split": s.split,
        }

    def localization_counts(self) -> Tuple[int, int]:
        pos = 0
        neg = 0

        for s in self.samples:
            loc = self._read_mask(s.pre_target_path) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())

        return pos, neg

    def class5_counts(self) -> np.ndarray:
        counts = np.zeros(5, dtype=np.int64)

        for s in self.samples:
            loc = self._read_mask(s.pre_target_path)
            dmg = self._read_mask(s.post_target_path)
            tgt = self._target5_from_masks(loc, dmg)
            valid = tgt != 255

            vals, freqs = np.unique(tgt[valid], return_counts=True)
            for value, freq in zip(vals.tolist(), freqs.tolist()):
                counts[int(value)] += int(freq)

        counts[counts == 0] = 1
        return counts


# -----------------------------
# Utility metrics
# -----------------------------
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += int(n)


class F1Recorder:
    def __init__(self, tp: int, fp: int, fn: int):
        self.tp = int(tp)
        self.fp = int(fp)
        self.fn = int(fn)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 0.0 if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 0.0 if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 0.0 if p == 0.0 or r == 0.0 else 2.0 * p * r / (p + r)

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def harmonic_mean(values: List[float]) -> float:
    return len(values) / sum((float(x) + 1e-6) ** -1 for x in values)


# -----------------------------
# Model blocks
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, c)


class WindowSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        hp, wp = x.shape[-2:]
        x_hw = x.permute(0, 2, 3, 1).contiguous()
        windows = window_partition(x_hw, ws)

        out, _ = self.attn(windows, windows, windows, need_weights=False)

        x_hw = window_reverse(out, ws, hp, wp, b)
        x = x_hw.permute(0, 3, 1, 2).contiguous()

        if pad_h or pad_w:
            x = x[:, :, :h, :w]

        return x


class DCMLP(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCSwinBlock(nn.Module):
    """
    Simplified Depthwise-Convolutional Swin block:
      LN -> window MSA / shifted-window MSA -> residual
      LN -> DCMLP -> residual
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int = 8,
        shift: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.shift = bool(shift)
        self.shift_size = self.window_size // 2 if self.shift else 0

        self.norm1 = LayerNorm2d(channels)
        self.attn = WindowSelfAttention(channels, num_heads=num_heads, window_size=window_size)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = DCMLP(channels, mlp_ratio=4.0, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        y = self.norm1(x)

        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        y = self.attn(y)

        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = shortcut + y
        x = x + self.mlp(self.norm2(x))
        return x


class HighResolutionTransformerBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 48, window_size: int = 8):
        super().__init__()

        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4
        c3 = base_channels * 8  # NEW 4th branch

        self.channels = [c0, c1, c2, c3]  # now 4 channels

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c0 // 2, kernel_size=3, stride=2),
            ConvBNAct(c0 // 2, c0, kernel_size=3, stride=2),
        )

        # --- Stage 1 (same) ---
        self.stage1 = nn.Sequential(
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=True),
        )
        self.down01 = ConvBNAct(c0, c1, kernel_size=3, stride=2)

        # --- Stage 2 (same) ---
        self.stage2_b0 = nn.Sequential(
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=True),
        )
        self.stage2_b1 = nn.Sequential(
            DCSwinBlock(c1, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c1, num_heads=4, window_size=window_size, shift=True),
        )
        self.fuse10_to_0 = nn.Conv2d(c1, c0, kernel_size=1)
        self.fuse0_to_1 = ConvBNAct(c0, c1, kernel_size=3, stride=2)
        self.down12 = ConvBNAct(c1, c2, kernel_size=3, stride=2)

        # --- Stage 3 (same) ---
        self.stage3_b0 = nn.Sequential(
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=True),
        )
        self.stage3_b1 = nn.Sequential(
            DCSwinBlock(c1, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c1, num_heads=4, window_size=window_size, shift=True),
        )
        self.stage3_b2 = nn.Sequential(
            DCSwinBlock(c2, num_heads=8, window_size=window_size, shift=False),
            DCSwinBlock(c2, num_heads=8, window_size=window_size, shift=True),
        )
        self.fuse1_to_0_s3 = nn.Conv2d(c1, c0, kernel_size=1)
        self.fuse2_to_0_s3 = nn.Conv2d(c2, c0, kernel_size=1)
        self.fuse0_to_1_s3 = ConvBNAct(c0, c1, kernel_size=3, stride=2)
        self.fuse2_to_1_s3 = nn.Conv2d(c2, c1, kernel_size=1)
        self.fuse0_to_2_s3 = nn.Sequential(
            ConvBNAct(c0, c1, kernel_size=3, stride=2),
            ConvBNAct(c1, c2, kernel_size=3, stride=2),
        )
        self.fuse1_to_2_s3 = ConvBNAct(c1, c2, kernel_size=3, stride=2)
        self.down23 = ConvBNAct(c2, c3, kernel_size=3, stride=2)

        # --- Stage 4: 4 branches ---
        self.stage4_b0 = nn.Sequential(
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=True),
        )
        self.stage4_b1 = nn.Sequential(
            DCSwinBlock(c1, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c1, num_heads=4, window_size=window_size, shift=True),
        )
        self.stage4_b2 = nn.Sequential(
            DCSwinBlock(c2, num_heads=8, window_size=window_size, shift=False),
            DCSwinBlock(c2, num_heads=8, window_size=window_size, shift=True),
        )
        self.stage4_b3 = nn.Sequential(
            DCSwinBlock(c3, num_heads=8, window_size=window_size, shift=False),
            DCSwinBlock(c3, num_heads=8, window_size=window_size, shift=True),
        )

        # Stage 4 fusion (all 4 branches fuse into all 4)
        self.s4_fuse1_to_0 = nn.Conv2d(c1, c0, kernel_size=1)
        self.s4_fuse2_to_0 = nn.Conv2d(c2, c0, kernel_size=1)
        self.s4_fuse3_to_0 = nn.Conv2d(c3, c0, kernel_size=1)

        self.s4_fuse0_to_1 = ConvBNAct(c0, c1, kernel_size=3, stride=2)
        self.s4_fuse2_to_1 = nn.Conv2d(c2, c1, kernel_size=1)
        self.s4_fuse3_to_1 = nn.Conv2d(c3, c1, kernel_size=1)

        self.s4_fuse0_to_2 = nn.Sequential(
            ConvBNAct(c0, c1, kernel_size=3, stride=2),
            ConvBNAct(c1, c2, kernel_size=3, stride=2),
        )
        self.s4_fuse1_to_2 = ConvBNAct(c1, c2, kernel_size=3, stride=2)
        self.s4_fuse3_to_2 = nn.Conv2d(c3, c2, kernel_size=1)

        self.s4_fuse0_to_3 = nn.Sequential(
            ConvBNAct(c0, c1, kernel_size=3, stride=2),
            ConvBNAct(c1, c2, kernel_size=3, stride=2),
            ConvBNAct(c2, c3, kernel_size=3, stride=2),
        )
        self.s4_fuse1_to_3 = nn.Sequential(
            ConvBNAct(c1, c2, kernel_size=3, stride=2),
            ConvBNAct(c2, c3, kernel_size=3, stride=2),
        )
        self.s4_fuse2_to_3 = ConvBNAct(c2, c3, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Stem
        x0 = self.stem(x)
        x0 = self.stage1(x0)

        # Stage 2
        x1 = self.down01(x0)
        a0 = self.stage2_b0(x0)
        a1 = self.stage2_b1(x1)
        x0 = a0 + F.interpolate(self.fuse10_to_0(a1), size=a0.shape[-2:], mode="bilinear", align_corners=False)
        x1 = a1 + self.fuse0_to_1(a0)

        # Stage 3
        x2 = self.down12(x1)
        b0 = self.stage3_b0(x0)
        b1 = self.stage3_b1(x1)
        b2 = self.stage3_b2(x2)
        x0 = (b0
              + F.interpolate(self.fuse1_to_0_s3(b1), size=b0.shape[-2:], mode="bilinear", align_corners=False)
              + F.interpolate(self.fuse2_to_0_s3(b2), size=b0.shape[-2:], mode="bilinear", align_corners=False))
        x1 = (b1
              + self.fuse0_to_1_s3(b0)
              + F.interpolate(self.fuse2_to_1_s3(b2), size=b1.shape[-2:], mode="bilinear", align_corners=False))
        x2 = b2 + self.fuse0_to_2_s3(b0) + self.fuse1_to_2_s3(b1)

        # Stage 4
        x3 = self.down23(x2)
        c0 = self.stage4_b0(x0)
        c1 = self.stage4_b1(x1)
        c2 = self.stage4_b2(x2)
        c3 = self.stage4_b3(x3)

        y0 = (c0
              + F.interpolate(self.s4_fuse1_to_0(c1), size=c0.shape[-2:], mode="bilinear", align_corners=False)
              + F.interpolate(self.s4_fuse2_to_0(c2), size=c0.shape[-2:], mode="bilinear", align_corners=False)
              + F.interpolate(self.s4_fuse3_to_0(c3), size=c0.shape[-2:], mode="bilinear", align_corners=False))
        y1 = (c1
              + self.s4_fuse0_to_1(c0)
              + F.interpolate(self.s4_fuse2_to_1(c2), size=c1.shape[-2:], mode="bilinear", align_corners=False)
              + F.interpolate(self.s4_fuse3_to_1(c3), size=c1.shape[-2:], mode="bilinear", align_corners=False))
        y2 = (c2
              + self.s4_fuse0_to_2(c0)
              + self.s4_fuse1_to_2(c1)
              + F.interpolate(self.s4_fuse3_to_2(c3), size=c2.shape[-2:], mode="bilinear", align_corners=False))
        y3 = c3 + self.s4_fuse0_to_3(c0) + self.s4_fuse1_to_3(c1) + self.s4_fuse2_to_3(c2)

        return [y0, y1, y2, y3]

class MultiScaleDecoder(nn.Module):
    def __init__(self, in_channels: List[int], decoder_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, decoder_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.GELU(),
                )
                for c in in_channels
            ]
        )

        self.fuse = nn.Sequential(
            ConvBNAct(decoder_channels * len(in_channels), decoder_channels, kernel_size=3, stride=1),
            ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, stride=1),
        )

        self.out = nn.Conv2d(decoder_channels, out_channels, kernel_size=1)

    def forward(self, features: List[torch.Tensor], output_size: Tuple[int, int]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        xs = []

        for feat, proj in zip(features, self.proj):
            y = proj(feat)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
            xs.append(y)

        x = torch.cat(xs, dim=1)
        x = self.fuse(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.out(x)


class CSFModule(nn.Module):
    """
    CSF-inspired cross-spatial fusion module.
    Recalibrates pre/post features using channel and spatial attention,
    then fuses them through 1x1/3x3 convolutions.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        ch = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))
        x = x * ch

        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp = self.spatial_conv(torch.cat([avg_sp, max_sp], dim=1))
        return x * sp

    def forward(self, fpre: torch.Tensor, fpost: torch.Tensor) -> torch.Tensor:
        apre = self._attention(fpre)
        apost = self._attention(fpost)

        diff = torch.abs(apost - apre)
        pre_refined = apre + diff
        post_refined = apost + diff

        return self.fuse(torch.cat([pre_refined, post_refined], dim=1))


class HRTBDAPhase1(nn.Module):
    def __init__(self, base_channels: int, decoder_channels: int, window_size: int):
        super().__init__()
        self.backbone = HighResolutionTransformerBackbone(
            in_channels=3,
            base_channels=base_channels,
            window_size=window_size,
        )

        self.decoder = MultiScaleDecoder(
            in_channels=self.backbone.channels,
            decoder_channels=decoder_channels,
            out_channels=1,
        )

    def forward(self, pre: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pre)
        return self.decoder(features, output_size=pre.shape[-2:]).squeeze(1)


class HRTBDAPhase2(nn.Module):
    def __init__(self, base_channels: int, decoder_channels: int, window_size: int, num_classes: int = 4):
        super().__init__()
        self.backbone = HighResolutionTransformerBackbone(
            in_channels=3,
            base_channels=base_channels,
            window_size=window_size,
        )

        self.csf = nn.ModuleList([CSFModule(c) for c in self.backbone.channels])

        self.decoder = MultiScaleDecoder(
            in_channels=self.backbone.channels,
            decoder_channels=decoder_channels,
            out_channels=num_classes,
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        fpre = self.backbone(pre)
        fpost = self.backbone(post)

        fused = [module(a, b) for module, a, b in zip(self.csf, fpre, fpost)]
        return self.decoder(fused, output_size=pre.shape[-2:])


# -----------------------------
# Losses
# -----------------------------
class BinaryFocalDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(1))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma * bce).mean()

        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * inter + 1e-7) / (denom + 1e-7)).mean()

        return focal + dice, focal, dice


class MulticlassFocalDiceLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)

        if class_weights is None:
            class_weights = torch.ones(4, dtype=torch.float32)

        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multiclass Focal + Dice loss.

        Important fix:
        - pt is computed from unweighted CE.
        - class weights are applied after pt is computed.
        This avoids making rare-class examples artificially look harder only because their CE was weighted.
        """
        ce_plain = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = target != self.ignore_index

        if valid.any():
            ce_valid = ce_plain[valid]
            pt = torch.exp(-ce_valid)

            target_valid = target[valid]
            weight_valid = self.class_weights[target_valid]

            focal = (((1.0 - pt) ** self.gamma) * ce_valid * weight_valid).mean()
        else:
            focal = logits.sum() * 0.0

        probs = torch.softmax(logits, dim=1)
        target_safe = target.clone()
        target_safe[target_safe == self.ignore_index] = 0

        one_hot = F.one_hot(target_safe, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)
        denom = (probs * probs).sum(dim=dims) + (one_hot * one_hot).sum(dim=dims)

        dice_per_class = 1.0 - (2.0 * inter + 1e-7) / (denom + 1e-7)

        w = self.class_weights / self.class_weights.sum().clamp_min(1e-7)
        dice = (dice_per_class * w).sum()

        return focal + dice, focal, dice


def make_loc_pos_weight(dataset: XBDHRTBDADataset) -> torch.Tensor:
    pos, neg = dataset.localization_counts()
    raw = max(1.0, neg / max(pos, 1))
    capped = min(raw, 10.0)
    return torch.tensor([capped], dtype=torch.float32)


def make_class_weights(dataset: XBDHRTBDADataset) -> torch.Tensor:
    """Legacy 5-class weights: [background, no, minor, major, destroyed]."""
    counts = dataset.class5_counts().astype(np.float64)

    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def make_damage4_class_weights(
    dataset: XBDHRTBDADataset,
    minor_boost: float = 2.0,
    major_boost: float = 1.5,
    destroyed_boost: float = 1.0,
) -> torch.Tensor:
    """
    Foreground-only damage class weights:
      index 0 = no damage
      index 1 = minor damage
      index 2 = major damage
      index 3 = destroyed

    Background is not part of this loss. This mirrors the older two-stage
    pipeline: Stage/Phase II specializes in severity only after localization.
    """
    counts5 = dataset.class5_counts().astype(np.float64)
    counts4 = counts5[1:5].copy()
    counts4[counts4 == 0] = 1.0

    freq = counts4 / counts4.sum()
    weights = 1.0 / np.sqrt(freq + 1e-12)

    # Manual emphasis for weak rare classes. Defaults are intentionally moderate
    # because Focal+Dice and rare-damage sampling already add pressure.
    weights[1] *= float(minor_boost)
    weights[2] *= float(major_boost)
    weights[3] *= float(destroyed_boost)

    weights = weights / weights.mean()

    print(
        "Phase II damage counts [no,minor,major,destroyed]: "
        f"{counts4.astype(int).tolist()}",
        flush=True,
    )
    print(
        "Phase II damage class weights [no,minor,major,destroyed]: "
        f"{weights.tolist()}",
        flush=True,
    )

    return torch.tensor(weights, dtype=torch.float32)


def make_rare_damage_sample_weights(
    dataset: XBDHRTBDADataset,
    minor_weight: float = 4.0,
    major_weight: float = 2.5,
    destroyed_weight: float = 1.5,
) -> List[float]:
    """
    Image-level rare-damage sampler inspired by the old pipeline's
    rare/foreground-aware cropping.

    A tile gets the maximum matching weight:
      base = 1.0
      contains destroyed -> at least destroyed_weight
      contains major     -> at least major_weight
      contains minor     -> at least minor_weight

    Minor is usually the weakest xBD class, so it defaults to the largest boost.
    """
    weights: List[float] = []

    for s in dataset.samples:
        loc = dataset._read_mask(s.pre_target_path)
        dmg = dataset._read_mask(s.post_target_path)
        tgt = dataset._target5_from_masks(loc, dmg)

        w = 1.0
        if np.any(tgt == 4):
            w = max(w, float(destroyed_weight))
        if np.any(tgt == 3):
            w = max(w, float(major_weight))
        if np.any(tgt == 2):
            w = max(w, float(minor_weight))

        weights.append(w)

    arr = np.array(weights, dtype=np.float64)
    print(
        "Rare-damage sampler tile weights: "
        f"min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}; "
        f"boosted={(arr > 1.0).sum()}/{len(arr)}",
        flush=True,
    )

    return weights


def target5_to_damage4(target5: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Convert 5-class map:
      0 background, 1 no, 2 minor, 3 major, 4 destroyed, 255 ignored
    to foreground-only 4-class map:
      0 no, 1 minor, 2 major, 3 destroyed, 255 ignored/background

    This makes Phase II learn severity only on building pixels.
    """
    out = torch.full_like(target5, fill_value=ignore_index)
    valid = (target5 >= 1) & (target5 <= 4)
    out[valid] = target5[valid] - 1
    return out


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_phase1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, object]:
    model.eval()

    tp = fp = fn = 0

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()

        logits = model(pre)
        pred = (torch.sigmoid(logits) > threshold).long()

        tp += int(((pred == 1) & (loc_true == 1)).sum().item())
        fp += int(((pred == 1) & (loc_true == 0)).sum().item())
        fn += int(((pred == 0) & (loc_true == 1)).sum().item())

    rec = F1Recorder(tp, fp, fn)
    return {
        "threshold": threshold,
        "localization_f1": rec.f1,
        "details": rec.as_dict(),
    }


@torch.no_grad()
def scan_phase1_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: List[float],
    csv_path: Path,
) -> Tuple[float, Dict[str, object]]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    best_threshold = thresholds[0]
    best_results: Dict[str, object] = {}
    best_f1 = -1.0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "localization_f1", "precision", "recall", "tp", "fp", "fn"])

        for th in thresholds:
            res = evaluate_phase1(model, loader, device, th)
            details = res["details"]

            writer.writerow([
                th,
                res["localization_f1"],
                details["precision"],
                details["recall"],
                details["tp"],
                details["fp"],
                details["fn"],
            ])

            if float(res["localization_f1"]) > best_f1:
                best_f1 = float(res["localization_f1"])
                best_threshold = th
                best_results = res

    return best_threshold, best_results


@torch.no_grad()
def evaluate_phase2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()

    loc_tp = loc_fp = loc_fn = 0

    cls_counts = {
        1: {"tp": 0, "fp": 0, "fn": 0},
        2: {"tp": 0, "fp": 0, "fn": 0},
        3: {"tp": 0, "fp": 0, "fn": 0},
        4: {"tp": 0, "fp": 0, "fn": 0},
    }

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        target = batch["target5"].to(device, non_blocking=True).long()

        logits = model(pre, post)
        pred = torch.argmax(logits, dim=1)

        loc_pred = (pred > 0).long()

        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())

        valid_building = (target >= 1) & (target <= 4)

        pred_valid = pred[valid_building]
        true_valid = target[valid_building]

        for cls in [1, 2, 3, 4]:
            tp = ((pred_valid == cls) & (true_valid == cls)).sum()
            fp = ((pred_valid == cls) & (true_valid != cls)).sum()
            fn = ((pred_valid != cls) & (true_valid == cls)).sum()

            cls_counts[cls]["tp"] += int(tp.item())
            cls_counts[cls]["fp"] += int(fp.item())
            cls_counts[cls]["fn"] += int(fn.item())

    loc = F1Recorder(loc_tp, loc_fp, loc_fn)
    no_damage = F1Recorder(cls_counts[1]["tp"], cls_counts[1]["fp"], cls_counts[1]["fn"])
    minor = F1Recorder(cls_counts[2]["tp"], cls_counts[2]["fp"], cls_counts[2]["fn"])
    major = F1Recorder(cls_counts[3]["tp"], cls_counts[3]["fp"], cls_counts[3]["fn"])
    destroyed = F1Recorder(cls_counts[4]["tp"], cls_counts[4]["fp"], cls_counts[4]["fn"])

    damage_f1 = harmonic_mean([no_damage.f1, minor.f1, major.f1, destroyed.f1])
    score = 0.3 * loc.f1 + 0.7 * damage_f1

    return {
        "score": score,
        "localization_f1": loc.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no_damage.f1,
        "damage_f1_minor_damage": minor.f1,
        "damage_f1_major_damage": major.f1,
        "damage_f1_destroyed": destroyed.f1,
        "details": {
            "localization": loc.as_dict(),
            "no_damage": no_damage.as_dict(),
            "minor_damage": minor.as_dict(),
            "major_damage": major.as_dict(),
            "destroyed": destroyed.as_dict(),
        },
    }



@torch.no_grad()
def evaluate_phase2_cascade(
    phase1_model: nn.Module,
    phase2_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    phase1_threshold: float,
) -> Dict[str, object]:
    """
    Strict two-phase/cascaded evaluation:

      Phase I: pre image -> building mask
      Phase II: pre/post images -> 4 damage classes
      Final map:
        outside Phase I mask = background
        inside Phase I mask  = Phase II severity class

    This makes the reported localization come from Phase I, not from Phase II argmax.
    """
    phase1_model.eval()
    phase2_model.eval()

    loc_tp = loc_fp = loc_fn = 0

    cls_counts = {
        1: {"tp": 0, "fp": 0, "fn": 0},
        2: {"tp": 0, "fp": 0, "fn": 0},
        3: {"tp": 0, "fp": 0, "fn": 0},
        4: {"tp": 0, "fp": 0, "fn": 0},
    }

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        target = batch["target5"].to(device, non_blocking=True).long()

        phase1_logits = phase1_model(pre)
        loc_pred = (torch.sigmoid(phase1_logits) > phase1_threshold).long()

        damage_logits = phase2_model(pre, post)  # [B, 4, H, W]
        damage_pred = torch.argmax(damage_logits, dim=1).long() + 1  # -> 1..4

        final_pred = torch.zeros_like(damage_pred)
        final_pred[loc_pred.bool()] = damage_pred[loc_pred.bool()]

        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())

        valid_building = (target >= 1) & (target <= 4)

        pred_valid = final_pred[valid_building]
        true_valid = target[valid_building]

        for cls in [1, 2, 3, 4]:
            tp = ((pred_valid == cls) & (true_valid == cls)).sum()
            fp = ((pred_valid == cls) & (true_valid != cls)).sum()
            fn = ((pred_valid != cls) & (true_valid == cls)).sum()

            cls_counts[cls]["tp"] += int(tp.item())
            cls_counts[cls]["fp"] += int(fp.item())
            cls_counts[cls]["fn"] += int(fn.item())

    loc = F1Recorder(loc_tp, loc_fp, loc_fn)
    no_damage = F1Recorder(cls_counts[1]["tp"], cls_counts[1]["fp"], cls_counts[1]["fn"])
    minor = F1Recorder(cls_counts[2]["tp"], cls_counts[2]["fp"], cls_counts[2]["fn"])
    major = F1Recorder(cls_counts[3]["tp"], cls_counts[3]["fp"], cls_counts[3]["fn"])
    destroyed = F1Recorder(cls_counts[4]["tp"], cls_counts[4]["fp"], cls_counts[4]["fn"])

    damage_f1 = harmonic_mean([no_damage.f1, minor.f1, major.f1, destroyed.f1])
    score = 0.3 * loc.f1 + 0.7 * damage_f1

    return {
        "score": score,
        "localization_f1": loc.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no_damage.f1,
        "damage_f1_minor_damage": minor.f1,
        "damage_f1_major_damage": major.f1,
        "damage_f1_destroyed": destroyed.f1,
        "phase1_threshold": float(phase1_threshold),
        "details": {
            "localization": loc.as_dict(),
            "no_damage": no_damage.as_dict(),
            "minor_damage": minor.as_dict(),
            "major_damage": major.as_dict(),
            "destroyed": destroyed.as_dict(),
        },
    }




def apply_damage_calibration(
    logits: torch.Tensor,
    calibration: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Apply 4-class vector scaling to Phase II damage logits.

    logits: [B, 4, H, W]
    W: [4, 4], b: [4]
    """
    if calibration is None:
        return logits

    W = calibration["W"].to(device=logits.device, dtype=torch.float32)
    b = calibration["b"].to(device=logits.device, dtype=torch.float32)

    z = logits.float().permute(0, 2, 3, 1).contiguous()
    z = torch.matmul(z, W.t()) + b
    return z.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def collect_damage_logits_balanced(
    phase2_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    per_class_pixels: int = 50000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect class-balanced validation pixels for damage calibration.

    This mirrors the old pipeline idea: calibration should not be dominated by
    no-damage pixels. We collect logits only on true building pixels/classes.
    """
    phase2_model.eval()
    rng = np.random.RandomState(seed)

    xs: List[List[np.ndarray]] = [[] for _ in range(4)]
    ys: List[List[np.ndarray]] = [[] for _ in range(4)]
    remaining = [int(per_class_pixels)] * 4
    collected = [0, 0, 0, 0]

    for batch in loader:
        if all(r <= 0 for r in remaining):
            break

        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        target5 = batch["target5"].to(device, non_blocking=True).long()
        target4 = target5_to_damage4(target5, ignore_index=255)

        logits = phase2_model(pre, post).float()
        feat = logits.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        target_flat = target4.view(-1)

        valid = target_flat != 255
        if valid.sum().item() == 0:
            continue

        feat_valid = feat[valid]
        target_valid = target_flat[valid]

        for c in range(4):
            need = remaining[c]
            if need <= 0:
                continue

            idx = torch.where(target_valid == c)[0]
            if idx.numel() == 0:
                continue

            take = min(need, int(idx.numel()))
            if int(idx.numel()) > take:
                pick_np = rng.choice(idx.detach().cpu().numpy(), size=take, replace=False)
                pick = torch.from_numpy(pick_np).to(device)
            else:
                pick = idx

            xs[c].append(feat_valid[pick].detach().cpu().numpy().astype(np.float32))
            ys[c].append(np.full((take,), c, dtype=np.int64))
            remaining[c] -= take
            collected[c] += take

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for c in range(4):
        if xs[c]:
            x_parts.append(np.concatenate(xs[c], axis=0))
            y_parts.append(np.concatenate(ys[c], axis=0))

    if not x_parts:
        raise RuntimeError("Could not collect any validation damage pixels for calibration.")

    X = np.concatenate(x_parts, axis=0)
    Y = np.concatenate(y_parts, axis=0)
    order = rng.permutation(X.shape[0])
    X = X[order]
    Y = Y[order]

    return X, Y, np.array(collected, dtype=np.int64)


def fit_vector_scaling(
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device,
    lr: float = 5e-2,
    epochs: int = 10,
    batch_size: int = 8192,
    weight_decay: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Fit 4-class vector scaling: z_cal = z @ W.T + b.
    """
    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    Y_t = torch.from_numpy(Y).to(device=device, dtype=torch.long)

    W = torch.eye(4, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(4, device=device, dtype=torch.float32, requires_grad=True)

    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    n = X_t.shape[0]
    for ep in range(1, int(epochs) + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        steps = 0

        for start in range(0, n, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            logits = X_t[idx] @ W.t() + b
            loss = ce(logits, Y_t[idx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1

        print(f"[CALIB] vector-scaling epoch {ep:02d}/{epochs} loss={total / max(1, steps):.6f}", flush=True)

    return {"W": W.detach().cpu(), "b": b.detach().cpu()}


def save_damage_calibration(
    calibration: Dict[str, torch.Tensor],
    path: Path,
    meta: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "W": calibration["W"],
        "b": calibration["b"],
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_damage_calibration(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    return {
        "W": payload["W"].to(device),
        "b": payload["b"].to(device),
    }


def maybe_fit_damage_calibration(
    args: argparse.Namespace,
    phase2_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    scores_dir: Path,
) -> Optional[Dict[str, torch.Tensor]]:
    if not getattr(args, "calibrate_damage", False):
        print("[CALIB] Damage calibration disabled.", flush=True)
        return None

    calib_path = scores_dir / "damage_vector_scaling_calibration.pt"

    print("[CALIB] Collecting class-balanced hold logits for 4 damage classes...", flush=True)
    X, Y, got = collect_damage_logits_balanced(
        phase2_model=phase2_model,
        loader=val_loader,
        device=device,
        per_class_pixels=args.calib_pixels_per_class,
        seed=args.seed,
    )
    print(
        "[CALIB] Collected pixels [no,minor,major,destroyed] = "
        f"{got.tolist()} | total={int(X.shape[0])}",
        flush=True,
    )

    calibration = fit_vector_scaling(
        X=X,
        Y=Y,
        device=device,
        lr=args.calib_lr,
        epochs=args.calib_epochs,
        batch_size=args.calib_batch,
        weight_decay=args.calib_weight_decay,
    )

    save_damage_calibration(
        calibration,
        calib_path,
        meta={
            "calib_pixels_per_class": args.calib_pixels_per_class,
            "collected_per_class": got.tolist(),
            "classes": ["no_damage", "minor_damage", "major_damage", "destroyed"],
        },
    )
    print(f"[CALIB] Saved calibration: {calib_path}", flush=True)

    return {k: v.to(device) for k, v in calibration.items()}


@torch.no_grad()
def evaluate_phase2_cascade_calibrated(
    phase1_model: nn.Module,
    phase2_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    phase1_threshold: float,
    calibration: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    """
    Same strict cascade as evaluate_phase2_cascade, with optional calibrated
    damage logits.
    """
    phase1_model.eval()
    phase2_model.eval()

    loc_tp = loc_fp = loc_fn = 0
    cls_counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in [1, 2, 3, 4]}

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        target = batch["target5"].to(device, non_blocking=True).long()

        phase1_logits = phase1_model(pre)
        loc_pred = (torch.sigmoid(phase1_logits) > phase1_threshold).long()

        damage_logits = phase2_model(pre, post)
        damage_logits = apply_damage_calibration(damage_logits, calibration)
        damage_pred = torch.argmax(damage_logits, dim=1).long() + 1

        final_pred = torch.zeros_like(damage_pred)
        final_pred[loc_pred.bool()] = damage_pred[loc_pred.bool()]

        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())

        valid_building = (target >= 1) & (target <= 4)
        pred_valid = final_pred[valid_building]
        true_valid = target[valid_building]

        for cls in [1, 2, 3, 4]:
            cls_counts[cls]["tp"] += int(((pred_valid == cls) & (true_valid == cls)).sum().item())
            cls_counts[cls]["fp"] += int(((pred_valid == cls) & (true_valid != cls)).sum().item())
            cls_counts[cls]["fn"] += int(((pred_valid != cls) & (true_valid == cls)).sum().item())

    loc = F1Recorder(loc_tp, loc_fp, loc_fn)
    no_damage = F1Recorder(cls_counts[1]["tp"], cls_counts[1]["fp"], cls_counts[1]["fn"])
    minor = F1Recorder(cls_counts[2]["tp"], cls_counts[2]["fp"], cls_counts[2]["fn"])
    major = F1Recorder(cls_counts[3]["tp"], cls_counts[3]["fp"], cls_counts[3]["fn"])
    destroyed = F1Recorder(cls_counts[4]["tp"], cls_counts[4]["fp"], cls_counts[4]["fn"])

    damage_f1 = harmonic_mean([no_damage.f1, minor.f1, major.f1, destroyed.f1])
    score = 0.3 * loc.f1 + 0.7 * damage_f1

    return {
        "score": score,
        "localization_f1": loc.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no_damage.f1,
        "damage_f1_minor_damage": minor.f1,
        "damage_f1_major_damage": major.f1,
        "damage_f1_destroyed": destroyed.f1,
        "phase1_threshold": float(phase1_threshold),
        "damage_calibrated": calibration is not None,
        "details": {
            "localization": loc.as_dict(),
            "no_damage": no_damage.as_dict(),
            "minor_damage": minor.as_dict(),
            "major_damage": major.as_dict(),
            "destroyed": destroyed.as_dict(),
        },
    }


@torch.no_grad()
def sweep_cascade_thresholds(
    phase1_model: nn.Module,
    phase2_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: List[float],
    csv_path: Path,
    calibration: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[float, Dict[str, object]]:
    """
    Sweep Phase-I mask thresholds on hold and select by full end-to-end score,
    not only pure localization F1.

    This copies the old pipeline philosophy: the final threshold should be good
    for the pipeline output, not just the standalone localization mask.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    best_threshold = float(thresholds[0])
    best_results: Dict[str, object] = {}
    best_score = -1.0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold",
            "score",
            "localization_f1",
            "damage_f1",
            "no_damage_f1",
            "minor_damage_f1",
            "major_damage_f1",
            "destroyed_f1",
            "damage_calibrated",
        ])

        for th in thresholds:
            res = evaluate_phase2_cascade_calibrated(
                phase1_model=phase1_model,
                phase2_model=phase2_model,
                loader=loader,
                device=device,
                phase1_threshold=float(th),
                calibration=calibration,
            )

            writer.writerow([
                th,
                res["score"],
                res["localization_f1"],
                res["damage_f1"],
                res["damage_f1_no_damage"],
                res["damage_f1_minor_damage"],
                res["damage_f1_major_damage"],
                res["damage_f1_destroyed"],
                res["damage_calibrated"],
            ])

            if float(res["score"]) > best_score:
                best_score = float(res["score"])
                best_threshold = float(th)
                best_results = res

    return best_threshold, best_results


def print_phase1_checkpoint_summary(phase1_ckpt: Path, device: torch.device) -> Dict[str, object]:
    ckpt = torch.load(phase1_ckpt, map_location=device)

    summary = {
        "path": str(phase1_ckpt),
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": float(ckpt.get("best_metric", -1.0)),
        "best_threshold": float(ckpt.get("best_threshold", 0.5)),
        "has_optimizer": ckpt.get("optimizer") is not None,
        "has_scheduler": ckpt.get("scheduler") is not None,
        "has_scaler": ckpt.get("scaler") is not None,
    }

    print("===== PHASE I CHECKPOINT SUMMARY =====", flush=True)
    print(f"path:           {summary['path']}", flush=True)
    print(f"epoch:          {summary['epoch']}", flush=True)
    print(f"best_metric:    {summary['best_metric']:.6f}", flush=True)
    print(f"best_threshold: {summary['best_threshold']:.2f}", flush=True)
    print(f"has_optimizer:  {summary['has_optimizer']}", flush=True)
    print(f"has_scheduler:  {summary['has_scheduler']}", flush=True)
    print(f"has_scaler:     {summary['has_scaler']}", flush=True)
    print("======================================", flush=True)

    return summary


def load_phase1_model_for_cascade(
    args: argparse.Namespace,
    device: torch.device,
    phase1_ckpt: Path,
) -> Tuple[nn.Module, float, Dict[str, object]]:
    model = HRTBDAPhase1(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
    ).to(device)

    ckpt = load_model_weights(model, phase1_ckpt, device)
    threshold = float(ckpt.get("best_threshold", args.phase1_threshold))

    meta = {
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": float(ckpt.get("best_metric", -1.0)),
        "best_threshold": threshold,
    }

    model.eval()

    print("Loaded Phase I localization model for cascaded mask.", flush=True)
    print(f"Phase I checkpoint: {phase1_ckpt}", flush=True)
    print(f"Phase I stored epoch: {meta['epoch']}", flush=True)
    print(f"Phase I stored best_metric: {meta['best_metric']:.6f}", flush=True)
    print(f"Phase I threshold used for mask: {threshold:.2f}", flush=True)

    return model, threshold, meta



# -----------------------------
# Checkpointing
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
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "args": vars(args),
    }

    if extra is not None:
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


def load_phase1_backbone_into_phase2(phase2: HRTBDAPhase2, phase1_ckpt: Path, device: torch.device) -> None:
    ckpt = torch.load(phase1_ckpt, map_location=device)
    state = ckpt["model"]

    backbone_state = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            backbone_state[k.replace("backbone.", "", 1)] = v

    missing, unexpected = phase2.backbone.load_state_dict(backbone_state, strict=False)
    print(f"Loaded Phase I backbone into Phase II from: {phase1_ckpt}", flush=True)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}", flush=True)


# -----------------------------
# Training
# -----------------------------
def make_loaders(
    args: argparse.Namespace,
    use_rare_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, XBDHRTBDADataset]:
    train_ds = XBDHRTBDADataset(args.xbd_root, args.train_split, args.img_size, training=True)
    val_ds = XBDHRTBDADataset(args.xbd_root, args.val_split, args.img_size, training=False)
    test_ds = XBDHRTBDADataset(args.xbd_root, args.test_split, args.img_size, training=False)

    sampler = None
    shuffle = True

    if use_rare_sampler:
        weights = make_rare_damage_sample_weights(
            train_ds,
            minor_weight=args.rare_minor_weight,
            major_weight=args.rare_major_weight,
            destroyed_weight=args.rare_destroyed_weight,
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
        print("Using rare-damage WeightedRandomSampler for Phase II training.", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_ds


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 10):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_scaler(args: argparse.Namespace, device: torch.device):
    enabled = bool(args.amp and device.type == "cuda")

    if USE_TORCH_AMP:
        return GradScaler(device.type, enabled=enabled)

    return GradScaler(enabled=enabled)


def backward_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    args: argparse.Namespace,
) -> None:
    scaler.scale(loss).backward()

    if args.max_grad_norm is not None and args.max_grad_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    scaler.step(optimizer)
    scaler.update()


def train_phase1(args: argparse.Namespace, device: torch.device) -> Path:
    print("\n================ PHASE I: BUILDING LOCALIZATION ================", flush=True)

    train_loader, val_loader, _, train_ds = make_loaders(args)

    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)

    model = HRTBDAPhase1(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    loc_pos_weight = make_loc_pos_weight(train_ds).to(device)
    print(f"Phase I localization pos_weight: {loc_pos_weight.detach().cpu().numpy().tolist()}", flush=True)

    criterion = BinaryFocalDiceLoss(
        pos_weight=loc_pos_weight,
        gamma=args.focal_gamma,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = make_scheduler(optimizer, args.phase1_epochs, warmup_epochs=args.warmup_epochs)
    scaler = make_scaler(args, device)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    scores_dir = output_dir / "scores"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    no_improve = 0
    history = []
    start_epoch = 1

    # Load existing Phase I history only when explicitly resuming.
    # This avoids accidentally reusing stale Phase-I best metrics/checkpoints
    # when running a new full two-stage experiment in an existing folder.
    history_path = output_dir / "history_phase1.json"
    resume_path = Path(args.resume_phase1_from) if getattr(args, "resume_phase1_from", "") else None

    if resume_path is None and history_path.exists():
        print(
            f"Ignoring existing Phase I history because --resume-phase1-from was not set: {history_path}",
            flush=True,
        )

    if resume_path is not None and history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)

            if history:
                best_row = max(history, key=lambda r: float(r.get("val_localization_f1", -1.0)))
                best_f1 = float(best_row.get("val_localization_f1", -1.0))
                best_epoch = int(best_row.get("epoch", 0))
                best_threshold = float(best_row.get("val_best_threshold", 0.5))

            print(
                f"Loaded existing Phase I history from {history_path} "
                f"with {len(history)} row(s).",
                flush=True,
            )
        except Exception as e:
            print(f"WARNING: Could not load existing Phase I history: {e}", flush=True)
            history = []

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-phase1-from was provided but does not exist: {resume_path}")

        print(f"Resuming Phase I from checkpoint: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device)

        state = ckpt["model"]
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)

        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        if ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"WARNING: Could not load scaler state: {e}", flush=True)

        resumed_epoch = int(ckpt.get("epoch", 0))
        start_epoch = resumed_epoch + 1

        best_f1 = float(ckpt.get("best_metric", best_f1))
        best_threshold = float(ckpt.get("best_threshold", best_threshold))

        # Keep only history rows before the next epoch to avoid duplicate rows
        # if the previous job died after writing partial output.
        history = [r for r in history if int(r.get("epoch", 0)) < start_epoch]

        if history:
            matching = [
                r for r in history
                if abs(float(r.get("val_localization_f1", -1.0)) - best_f1) < 1e-12
            ]
            if matching:
                best_epoch = int(matching[-1].get("epoch", best_epoch))
            else:
                best_row = max(history, key=lambda r: float(r.get("val_localization_f1", -1.0)))
                best_epoch = int(best_row.get("epoch", best_epoch))

        no_improve = max(0, start_epoch - best_epoch - 1)

        print(
            f"Resume summary: checkpoint_epoch={resumed_epoch}, "
            f"start_epoch={start_epoch}, best_epoch={best_epoch}, "
            f"best_f1={best_f1:.6f}, best_threshold={best_threshold:.2f}, "
            f"no_improve={no_improve}",
            flush=True,
        )

    if start_epoch > args.phase1_epochs:
        print(
            f"Phase I already reached epoch {start_epoch - 1}, "
            f"which is >= requested phase1_epochs={args.phase1_epochs}.",
            flush=True,
        )
        return checkpoints_dir / "phase1_best.pt"

    for epoch in range(start_epoch, args.phase1_epochs + 1):
        model.train()

        total_meter = AverageMeter()
        focal_meter = AverageMeter()
        dice_meter = AverageMeter()

        print(f"\nPhase I epoch {epoch}/{args.phase1_epochs} | LR={optimizer.param_groups[0]['lr']:.8f}", flush=True)

        iterator = tqdm(train_loader, desc=f"phase1 {epoch}") if (tqdm is not None and sys.stderr.isatty()) else train_loader

        for step, batch in enumerate(iterator, start=1):
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

            backward_step(loss, model, optimizer, scaler, args)

            bs = pre.size(0)
            total_meter.update(loss.item(), bs)
            focal_meter.update(focal.item(), bs)
            dice_meter.update(dice.item(), bs)

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Phase I Epoch {epoch}/{args.phase1_epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | "
                    f"focal={focal_meter.avg:.4f} | "
                    f"dice={dice_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()

        scan_csv = scores_dir / f"phase1_epoch_{epoch:03d}_threshold_scan.csv"
        threshold, val_results = scan_phase1_thresholds(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=args.thresholds,
            csv_path=scan_csv,
        )

        val_f1 = float(val_results["localization_f1"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_focal": focal_meter.avg,
            "train_dice": dice_meter.avg,
            "val_best_threshold": threshold,
            "val_localization_f1": val_f1,
        }
        history.append(row)

        print(
            f"Phase I Epoch {epoch:03d} | "
            f"train_loss={total_meter.avg:.4f} | "
            f"val_loc_f1={val_f1:.6f} | "
            f"threshold={threshold:.2f}",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_threshold = threshold
            no_improve = 0

            save_checkpoint(
                path=checkpoints_dir / "phase1_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_f1,
                args=args,
                extra={"best_threshold": best_threshold},
            )

            print(f"Saved Phase I best checkpoint | epoch={epoch} | loc_f1={best_f1:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"Phase I no improvement for {no_improve} epoch(s). Best epoch={best_epoch}", flush=True)

        save_checkpoint(
            path=checkpoints_dir / "phase1_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_f1,
            args=args,
            extra={"best_threshold": best_threshold},
        )

        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(
                path=checkpoints_dir / f"phase1_epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_f1,
                args=args,
                extra={"best_threshold": best_threshold},
            )

        with open(output_dir / "history_phase1.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.early_stopping_patience:
            print(f"Phase I early stopping at epoch {epoch}.", flush=True)
            break

    print(
        f"Phase I done. Best epoch={best_epoch}, best loc F1={best_f1:.6f}, threshold={best_threshold:.2f}",
        flush=True,
    )

    return checkpoints_dir / "phase1_best.pt"


def train_phase2(args: argparse.Namespace, device: torch.device, phase1_ckpt: Optional[Path]) -> Path:
    print("\n================ PHASE II: FOREGROUND-ONLY DAMAGE CLASSIFICATION ================", flush=True)

    if phase1_ckpt is None or not phase1_ckpt.exists():
        raise FileNotFoundError(
            "A valid Phase I checkpoint is required for cascaded Phase II training. "
            f"Got: {phase1_ckpt}"
        )

    train_loader, val_loader, _, train_ds = make_loaders(
        args,
        use_rare_sampler=getattr(args, "rare_sampler", False),
    )

    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)

    # Load the actual Phase I localization model. This model provides the final
    # building/background mask during validation and testing.
    phase1_eval_model, phase1_threshold, phase1_meta = load_phase1_model_for_cascade(
        args=args,
        device=device,
        phase1_ckpt=phase1_ckpt,
    )

    model = HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=4,  # no damage, minor, major, destroyed
    ).to(device)

    # Phase I still initializes the Phase II backbone, preserving building-extraction features.
    load_phase1_backbone_into_phase2(model, phase1_ckpt, device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    class_weights = make_damage4_class_weights(
        train_ds,
        minor_boost=args.minor_damage_boost,
        major_boost=args.major_damage_boost,
        destroyed_boost=args.destroyed_damage_boost,
    ).to(device)
    print(
        f"Phase II foreground damage class weights [no,minor,major,destroyed]: "
        f"{class_weights.detach().cpu().numpy().tolist()}",
        flush=True,
    )

    criterion = MulticlassFocalDiceLoss(
        class_weights=class_weights,
        gamma=args.focal_gamma,
        ignore_index=255,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = make_scheduler(optimizer, args.phase2_epochs, warmup_epochs=args.warmup_epochs)
    scaler = make_scaler(args, device)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    scores_dir = output_dir / "scores"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    with open(scores_dir / "phase1_checkpoint_used_for_cascade.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase1_checkpoint": str(phase1_ckpt),
                "phase1_epoch": phase1_meta["epoch"],
                "phase1_best_metric_hold": phase1_meta["best_metric"],
                "phase1_threshold": phase1_threshold,
            },
            f,
            indent=2,
        )

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()

        total_meter = AverageMeter()
        focal_meter = AverageMeter()
        dice_meter = AverageMeter()

        print(f"\nPhase II epoch {epoch}/{args.phase2_epochs} | LR={optimizer.param_groups[0]['lr']:.8f}", flush=True)

        iterator = tqdm(train_loader, desc=f"phase2 {epoch}") if (tqdm is not None and sys.stderr.isatty()) else train_loader

        for step, batch in enumerate(iterator, start=1):
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            target5 = batch["target5"].to(device, non_blocking=True)
            damage_target = target5_to_damage4(target5, ignore_index=255)

            optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    logits = model(pre, post)
                    loss, focal, dice = criterion(logits, damage_target)
                    loss = args.cls_loss_weight * loss
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(pre, post)
                    loss, focal, dice = criterion(logits, damage_target)
                    loss = args.cls_loss_weight * loss

            backward_step(loss, model, optimizer, scaler, args)

            bs = pre.size(0)
            total_meter.update(loss.item(), bs)
            focal_meter.update(focal.item(), bs)
            dice_meter.update(dice.item(), bs)

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Phase II Epoch {epoch}/{args.phase2_epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | "
                    f"focal={focal_meter.avg:.4f} | "
                    f"dice={dice_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()

        val_results = evaluate_phase2_cascade(
            phase1_model=phase1_eval_model,
            phase2_model=model,
            loader=val_loader,
            device=device,
            phase1_threshold=phase1_threshold,
        )
        val_score = float(val_results["score"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_focal": focal_meter.avg,
            "train_dice": dice_meter.avg,
            "hold_score_cascade": val_score,
            "hold_localization_f1_from_phase1_mask": float(val_results["localization_f1"]),
            "hold_damage_f1": float(val_results["damage_f1"]),
            "hold_no_damage_f1": float(val_results["damage_f1_no_damage"]),
            "hold_minor_damage_f1": float(val_results["damage_f1_minor_damage"]),
            "hold_major_damage_f1": float(val_results["damage_f1_major_damage"]),
            "hold_destroyed_f1": float(val_results["damage_f1_destroyed"]),
            "phase1_threshold": phase1_threshold,
            "phase1_best_metric_hold": phase1_meta["best_metric"],
        }

        history.append(row)

        print(
            f"Phase II Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"hold_score_cascade={row['hold_score_cascade']:.6f} | "
            f"hold_loc_f1_from_phase1={row['hold_localization_f1_from_phase1_mask']:.6f} | "
            f"hold_damage_f1={row['hold_damage_f1']:.6f} | "
            f"no={row['hold_no_damage_f1']:.6f} | "
            f"minor={row['hold_minor_damage_f1']:.6f} | "
            f"major={row['hold_major_damage_f1']:.6f} | "
            f"destroyed={row['hold_destroyed_f1']:.6f}",
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
                    "phase1_threshold": phase1_threshold,
                    "phase1_best_metric_hold": phase1_meta["best_metric"],
                    "cascade_validation": True,
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
            extra={
                "phase1_checkpoint": str(phase1_ckpt),
                "phase1_threshold": phase1_threshold,
                "phase1_best_metric_hold": phase1_meta["best_metric"],
                "cascade_validation": True,
            },
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
                extra={
                    "phase1_checkpoint": str(phase1_ckpt),
                    "phase1_threshold": phase1_threshold,
                    "phase1_best_metric_hold": phase1_meta["best_metric"],
                    "cascade_validation": True,
                },
            )

        with open(output_dir / "history_phase2.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.early_stopping_patience:
            print(f"Phase II early stopping at epoch {epoch}.", flush=True)
            break

    print(f"Phase II done. Best epoch={best_epoch}, best hold cascade score={best_score:.6f}", flush=True)

    return checkpoints_dir / "phase2_best.pt"


def test_phase2(args: argparse.Namespace, device: torch.device, checkpoint_path: Path, phase1_ckpt: Path) -> None:
    print("\n================ CASCADED TESTING: PHASE I MASK + PHASE II DAMAGE ================", flush=True)

    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Phase II checkpoint not found: {checkpoint_path}")

    _, val_loader, test_loader, _ = make_loaders(args)

    phase1_model, stored_phase1_threshold, phase1_meta = load_phase1_model_for_cascade(
        args=args,
        device=device,
        phase1_ckpt=phase1_ckpt,
    )

    phase2_model = HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=4,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        phase2_model = nn.DataParallel(phase2_model)

    ckpt = load_model_weights(phase2_model, checkpoint_path, device)
    best_epoch = int(ckpt.get("epoch", -1))

    output_dir = Path(args.output_dir)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Optional damage calibration using hold/validation logits only.
    calibration = maybe_fit_damage_calibration(
        args=args,
        phase2_model=phase2_model,
        val_loader=val_loader,
        device=device,
        scores_dir=scores_dir,
    )

    # Sweep Phase-I mask threshold on hold for the end-to-end cascade score.
    # This is separate from the stored Phase-I best_threshold because the best
    # pure localization threshold is not always the best full pipeline threshold.
    sweep_thresholds = args.mask_thresholds if args.mask_thresholds else [stored_phase1_threshold]

    uncal_threshold, uncal_hold = sweep_cascade_thresholds(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=val_loader,
        device=device,
        thresholds=sweep_thresholds,
        csv_path=scores_dir / "hold_threshold_sweep_uncalibrated.csv",
        calibration=None,
    )

    selected_threshold = uncal_threshold
    selected_calibrated = False
    selected_hold = uncal_hold
    selected_calibration = None

    if calibration is not None:
        cal_threshold, cal_hold = sweep_cascade_thresholds(
            phase1_model=phase1_model,
            phase2_model=phase2_model,
            loader=val_loader,
            device=device,
            thresholds=sweep_thresholds,
            csv_path=scores_dir / "hold_threshold_sweep_calibrated.csv",
            calibration=calibration,
        )

        if float(cal_hold["score"]) >= float(uncal_hold["score"]):
            selected_threshold = cal_threshold
            selected_calibrated = True
            selected_hold = cal_hold
            selected_calibration = calibration

    selection = {
        "stored_phase1_threshold": float(stored_phase1_threshold),
        "selected_phase1_threshold": float(selected_threshold),
        "selected_damage_calibrated": bool(selected_calibrated),
        "hold_selected_score": float(selected_hold["score"]),
        "hold_selected_localization_f1": float(selected_hold["localization_f1"]),
        "hold_selected_damage_f1": float(selected_hold["damage_f1"]),
        "hold_selected_no_damage_f1": float(selected_hold["damage_f1_no_damage"]),
        "hold_selected_minor_damage_f1": float(selected_hold["damage_f1_minor_damage"]),
        "hold_selected_major_damage_f1": float(selected_hold["damage_f1_major_damage"]),
        "hold_selected_destroyed_f1": float(selected_hold["damage_f1_destroyed"]),
    }

    with open(scores_dir / "final_pipeline_selection_on_hold.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    print("===== FINAL PIPELINE SELECTION ON HOLD =====", flush=True)
    print(f"stored Phase-I threshold:   {stored_phase1_threshold:.2f}", flush=True)
    print(f"selected Phase-I threshold: {selected_threshold:.2f}", flush=True)
    print(f"damage calibrated:          {selected_calibrated}", flush=True)
    print(f"hold score:                 {selected_hold['score']:.6f}", flush=True)
    print(f"hold loc F1:                {selected_hold['localization_f1']:.6f}", flush=True)
    print(f"hold damage F1:             {selected_hold['damage_f1']:.6f}", flush=True)
    print("===========================================", flush=True)

    # Test both uncalibrated/stored and final-selected settings for transparency.
    baseline_results = evaluate_phase2_cascade_calibrated(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=test_loader,
        device=device,
        phase1_threshold=stored_phase1_threshold,
        calibration=None,
    )

    selected_results = evaluate_phase2_cascade_calibrated(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=test_loader,
        device=device,
        phase1_threshold=selected_threshold,
        calibration=selected_calibration,
    )

    selected_results["phase1_checkpoint"] = str(phase1_ckpt)
    selected_results["phase2_checkpoint"] = str(checkpoint_path)
    selected_results["phase1_epoch"] = phase1_meta["epoch"]
    selected_results["phase1_best_metric_hold"] = phase1_meta["best_metric"]
    selected_results["phase1_threshold_stored"] = stored_phase1_threshold
    selected_results["phase1_threshold_selected_on_hold"] = selected_threshold
    selected_results["damage_calibrated_selected_on_hold"] = selected_calibrated
    selected_results["phase2_best_epoch_selected_on_hold"] = best_epoch
    selected_results["hold_selection"] = selection
    selected_results["baseline_stored_threshold_uncalibrated"] = baseline_results

    json_path = scores_dir / "scores_xbd_test_hrtbda_v2_pipeline_calibrated_thresholded.json"
    txt_path = scores_dir / "scores_xbd_test_hrtbda_v2_pipeline_calibrated_thresholded.txt"
    summary_path = scores_dir / "summary_hrtbda_v2_pipeline_calibrated_thresholded.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selected_results, f, indent=2)

    train_splits_text = "+".join(args.train_split) if isinstance(args.train_split, list) else str(args.train_split)

    lines = [
        f"Experiment: HRTBDA v2 two-stage transformer pipeline + calibrated damage + threshold sweep {train_splits_text} -> {args.val_split} -> {args.test_split}",
        f"Phase I checkpoint: {phase1_ckpt}",
        f"Phase I stored best epoch: {phase1_meta['epoch']}",
        f"Phase I stored hold Localization F1: {phase1_meta['best_metric']:.6f}",
        f"Phase I stored threshold: {stored_phase1_threshold:.2f}",
        f"Phase I threshold selected on hold pipeline score: {selected_threshold:.2f}",
        f"Damage calibration selected: {selected_calibrated}",
        f"Best Phase II epoch selected on hold cascade score: {best_epoch}",
        "",
        "Baseline test with stored Phase-I threshold and uncalibrated damage:",
        f"  Localization F1: {baseline_results['localization_f1']:.6f}",
        f"  No Damage F1:    {baseline_results['damage_f1_no_damage']:.6f}",
        f"  Minor Damage F1: {baseline_results['damage_f1_minor_damage']:.6f}",
        f"  Major Damage F1: {baseline_results['damage_f1_major_damage']:.6f}",
        f"  Destroyed F1:    {baseline_results['damage_f1_destroyed']:.6f}",
        f"  Damage F1:       {baseline_results['damage_f1']:.6f}",
        f"  Overall Score:   {baseline_results['score']:.6f}",
        "",
        "Selected test pipeline:",
        f"  Localization F1 from Phase I mask: {selected_results['localization_f1']:.6f}",
        f"  No Damage F1:    {selected_results['damage_f1_no_damage']:.6f}",
        f"  Minor Damage F1: {selected_results['damage_f1_minor_damage']:.6f}",
        f"  Major Damage F1: {selected_results['damage_f1_major_damage']:.6f}",
        f"  Destroyed F1:    {selected_results['damage_f1_destroyed']:.6f}",
        f"  Damage F1:       {selected_results['damage_f1']:.6f}",
        f"  Overall Score:   {selected_results['score']:.6f}",
    ]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines), flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)
    print(f"Wrote: {summary_path}", flush=True)


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HRTBDA-inspired xBD building damage assessment")

    parser.add_argument("--phase", type=str, default="both", choices=["both", "phase1", "phase2", "phase2_test", "test", "inspect_phase1"])

    parser.add_argument(
        "--resume-phase1-from",
        type=str,
        default="",
        help="Optional Phase I checkpoint path to resume from, e.g. checkpoints/phase1_last.pt",
    )

    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default="",
        help="Path to an existing Phase I best checkpoint. Use this when Phase I is already trained.",
    )

    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        default="",
        help="Optional Phase II checkpoint path for --phase test. Defaults to output-dir/checkpoints/phase2_best.pt.",
    )

    parser.add_argument(
        "--phase1-threshold",
        type=float,
        default=0.5,
        help="Fallback threshold if the Phase I checkpoint does not contain best_threshold.",
    )

    parser.add_argument("--xbd-root", type=str, required=True)

    parser.add_argument(
        "--train-split",
        type=str,
        nargs="+",
        default=["tier3"],
        help="One or more training splits, e.g. --train-split train tier3",
    )

    parser.add_argument("--val-split", type=str, default="hold")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--phase1-epochs", type=int, default=150)
    parser.add_argument("--phase2-epochs", type=int, default=30)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)

    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--loc-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)

    # Ideas borrowed from the older two-stage pipeline.
    parser.add_argument("--rare-sampler", action="store_true", help="Use rare-damage weighted sampling for Phase II training.")
    parser.add_argument("--rare-minor-weight", type=float, default=4.0)
    parser.add_argument("--rare-major-weight", type=float, default=2.5)
    parser.add_argument("--rare-destroyed-weight", type=float, default=1.5)

    parser.add_argument("--minor-damage-boost", type=float, default=2.0, help="Extra class-weight multiplier for minor damage in foreground-only Phase II loss.")
    parser.add_argument("--major-damage-boost", type=float, default=1.5, help="Extra class-weight multiplier for major damage in foreground-only Phase II loss.")
    parser.add_argument("--destroyed-damage-boost", type=float, default=1.0, help="Extra class-weight multiplier for destroyed damage in foreground-only Phase II loss.")

    parser.add_argument("--calibrate-damage", action="store_true", help="Fit vector-scaling calibration on hold damage logits after Phase II training.")
    parser.add_argument("--calib-pixels-per-class", type=int, default=50000)
    parser.add_argument("--calib-lr", type=float, default=5e-2)
    parser.add_argument("--calib-epochs", type=int, default=10)
    parser.add_argument("--calib-batch", type=int, default=8192)
    parser.add_argument("--calib-weight-decay", type=float, default=0.0)

    parser.add_argument(
        "--mask-thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        help="Phase-I mask thresholds to sweep on hold for final pipeline score.",
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "scores").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoints_dir = output_dir / "checkpoints"

    phase1_ckpt = Path(args.phase1_checkpoint) if args.phase1_checkpoint else checkpoints_dir / "phase1_best.pt"
    phase2_ckpt = Path(args.phase2_checkpoint) if args.phase2_checkpoint else checkpoints_dir / "phase2_best.pt"

    print("===== HRTBDA V2 CASCADED PHASE I MASK + FOREGROUND DAMAGE TRAINING =====", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print(f"Resume Phase I from: {args.resume_phase1_from if args.resume_phase1_from else 'none'}", flush=True)
    print(f"Existing Phase I checkpoint: {phase1_ckpt}", flush=True)
    print(f"Phase II checkpoint: {phase2_ckpt}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"xBD root: {args.xbd_root}", flush=True)
    print(f"Train split(s): {args.train_split}", flush=True)
    print(f"Val split: {args.val_split}", flush=True)
    print(f"Test split: {args.test_split}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Phase I epochs: {args.phase1_epochs}", flush=True)
    print(f"Phase II epochs: {args.phase2_epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Image size: {args.img_size}", flush=True)
    print(f"LR: {args.lr}", flush=True)
    print(f"Weight decay: {args.weight_decay}", flush=True)
    print(f"Base channels: {args.base_channels}", flush=True)
    print(f"Decoder channels: {args.decoder_channels}", flush=True)
    print(f"Window size: {args.window_size}", flush=True)
    print(f"Focal gamma: {args.focal_gamma}", flush=True)
    print(f"Max grad norm: {args.max_grad_norm}", flush=True)
    print(f"Warmup epochs: {args.warmup_epochs}", flush=True)
    print("Architecture: HRTBDA v2 4-branch HRNet-style + DCSwin + CSF fusion", flush=True)
    print("Final inference: Phase I mask gives localization; Phase II predicts 4 foreground damage classes.", flush=True)
    print(f"Rare sampler: {args.rare_sampler} | minor/major/destroyed tile weights: {args.rare_minor_weight}/{args.rare_major_weight}/{args.rare_destroyed_weight}", flush=True)
    print(f"Damage class boosts [minor,major,destroyed]: {args.minor_damage_boost}/{args.major_damage_boost}/{args.destroyed_damage_boost}", flush=True)
    print(f"Damage calibration: {args.calibrate_damage} | mask-threshold sweep: {args.mask_thresholds}", flush=True)
    print("Domain adaptation: none", flush=True)
    print("=======================================", flush=True)

    if args.phase == "inspect_phase1":
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
        print_phase1_checkpoint_summary(phase1_ckpt, device)
        return

    if args.phase == "phase1":
        train_phase1(args, device)

    elif args.phase == "phase2":
        train_phase2(args, device, phase1_ckpt)

    elif args.phase == "phase2_test":
        phase2_ckpt = train_phase2(args, device, phase1_ckpt)
        test_phase2(args, device, phase2_ckpt, phase1_ckpt)

    elif args.phase == "test":
        test_phase2(args, device, phase2_ckpt, phase1_ckpt)

    elif args.phase == "both":
        # Full run: train Phase I first, then train cascaded foreground-only Phase II.
        phase1_ckpt = train_phase1(args, device)
        phase2_ckpt = train_phase2(args, device, phase1_ckpt)
        test_phase2(args, device, phase2_ckpt, phase1_ckpt)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
