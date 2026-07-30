#!/usr/bin/env python3
"""
HRTBDA v5 multi-label rare-crop cascaded transformer experiment -- Swin
ImageNet-pretrained backbone variant.

This is a standalone copy of train_xbd_hrtbda_v5_multilabel_crop_cascade.py
(that original file is untouched). Everything is identical to the original
EXCEPT the backbone:

  - HighResolutionTransformerBackbone (a custom 4-branch HRNet-style network,
    where all resolution branches exist and exchange information
    simultaneously throughout the whole network, trained from scratch with
    no pretraining) is replaced by SwinPretrainedBackbone -- a standard
    ImageNet-pretrained Swin Transformer (via timm), used purely as a
    feature-pyramid encoder.

  - Structural trade-off: Swin is a single-path hierarchical/pyramid
    transformer (one branch, downsampled stage by stage: stride 4 -> 8 ->
    16 -> 32, channels doubling at each merge). It does NOT keep multiple
    resolution branches alive simultaneously the way the original backbone
    does -- that "always high-resolution" property is given up in exchange
    for real ImageNet pretraining. Everything downstream only depends on
    getting a 4-level [B,C,H,W] feature list back, so CSFModule,
    MultiScaleDecoder, the two-phase cascade, the multilabel damage head,
    rare-crop sampling, and the full training/eval pipeline are unchanged.

  - Input normalization is unchanged and already correct for this swap:
    XBDHRTBDADataset._normalize() already uses ImageNet mean/std
    ([0.485,0.456,0.406] / [0.229,0.224,0.225]), which is exactly what an
    ImageNet-pretrained backbone expects.

  - Resolution requirement: Swin's window attention needs every stage's
    feature map to divide evenly by the window size. For the default
    swin_*_patch4_window7_224 family (patch_size=4, window_size=7, 4
    stages) that means --img-size and --phase2-crop-size must be multiples
    of 224 (e.g. 896, 672, 448) -- SwinPretrainedBackbone raises a clear
    error at construction time if they aren't, instead of failing with an
    opaque shape mismatch deep inside attention.

Everything below this docstring that isn't the backbone class, the two
HRTBDAPhase1/HRTBDAPhase2 constructors, the four places that build those
models, or the --swin-* / --img-size / --phase2-crop-size CLI plumbing is
byte-for-byte identical to the original script.

This is still an HRTBDA-inspired experimental implementation, not official
author code.
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
from torch.utils.data import DataLoader, Dataset

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



def apply_extra_photometric_augmentations(
    image_list: List[np.ndarray],
    training: bool,
) -> List[np.ndarray]:
    """Additional xView2-style photometric augmentation for Phase II training."""
    if not training:
        return image_list

    out = image_list

    # RGB channel-wise additive shift, same random values for pre/post to avoid fake change.
    if np.random.rand() < 0.35:
        shift = np.random.uniform(-12.0, 12.0, size=(1, 1, 3)).astype(np.float32)
        out = [np.clip(x.astype(np.float32) + shift, 0, 255).astype(np.uint8) for x in out]

    # HSV saturation/value jitter, same factors for pre/post.
    if np.random.rand() < 0.35:
        sat_factor = float(np.random.uniform(0.85, 1.15))
        val_factor = float(np.random.uniform(0.85, 1.15))
        aug = []
        for rgb in out:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * sat_factor, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * val_factor, 0, 255)
            aug.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
        out = aug

    # CLAHE on luminance, per image.
    if np.random.rand() < 0.25:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        aug = []
        for rgb in out:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            aug.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        out = aug

    return out


def rare_damage_candidate_crop(
    image_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    crop_size: int,
    candidate_count: int,
    class_weights: Tuple[float, float, float, float],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Choose the best crop among random candidates using weighted damage pixels.

    class_weights is ordered as: no_damage, minor, major, destroyed.
    This borrows the xView2 strong-baseline idea of selecting crops that contain
    rare/important damage classes instead of training only on huge no-damage-heavy images.
    """
    if crop_size <= 0:
        return image_list, mask_list

    h, w = mask_list[0].shape[:2]
    size = int(crop_size)

    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        image_list = [cv2.copyMakeBorder(x, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101) for x in image_list]
        mask_list = [np.pad(x, ((0, pad_h), (0, pad_w)), mode="edge") for x in mask_list]
        h, w = mask_list[0].shape[:2]

    target5 = mask_list[1]
    n = max(1, int(candidate_count))
    weights = np.asarray(class_weights, dtype=np.float64)

    best_score = -1.0
    best_xy = (0, 0)

    for _ in range(n):
        y0 = np.random.randint(0, h - size + 1)
        x0 = np.random.randint(0, w - size + 1)
        crop = target5[y0:y0 + size, x0:x0 + size]
        counts = np.array([(crop == c).sum() for c in [1, 2, 3, 4]], dtype=np.float64)
        # Make empty/no-building crops very unattractive.
        score = float((counts * weights).sum())
        if score > best_score:
            best_score = score
            best_xy = (y0, x0)

    y0, x0 = best_xy
    image_list = [x[y0:y0 + size, x0:x0 + size].copy() for x in image_list]
    mask_list = [x[y0:y0 + size, x0:x0 + size].copy() for x in mask_list]
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

    def __init__(
        self,
        root: str | Path,
        split: str | List[str] | Tuple[str, ...],
        image_size: int,
        training: bool,
        crop_size: int = 0,
        crop_candidate_count: int = 1,
        crop_class_weights: Tuple[float, float, float, float] = (1.0, 10.0, 10.0, 4.0),
        extra_photometric: bool = False,
    ):
        self.root = Path(root)

        if isinstance(split, (list, tuple)):
            self.splits = [str(s) for s in split]
        else:
            self.splits = [str(split)]

        self.split = "+".join(self.splits)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.crop_candidate_count = int(crop_candidate_count)
        self.crop_class_weights = tuple(float(x) for x in crop_class_weights)
        self.extra_photometric = bool(extra_photometric)

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

            if self.crop_size != self.image_size:
                # SwinPretrainedBackbone is one instance shared across this
                # phase's train/val/test calls; timm's Swin (PatchEmbed's
                # strict_img_size can be patched around, but the shifted-window
                # attention mask is precomputed for a single fixed resolution
                # and does not recompute per forward call) only works
                # correctly at the resolution it was constructed with. Resize
                # the selected crop back up to image_size so every tensor this
                # backbone ever sees is the same size -- this keeps the
                # rare-class crop *selection* benefit (which pixels end up in
                # the frame) while giving up the smaller-tensor compute/memory
                # savings crop_size < image_size would otherwise provide.
                [pre, post], [loc_raw, target5] = resize_rgb_and_masks(
                    image_list=[pre, post],
                    mask_list=[loc_raw, target5],
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


class SwinPretrainedBackbone(nn.Module):
    """
    ImageNet-pretrained Swin Transformer backbone (via timm), used as a
    drop-in replacement for HighResolutionTransformerBackbone.

    Structural difference from the original backbone: this is a standard
    hierarchical/pyramid transformer -- one path, progressively downsampled
    by patch merging between 4 stages (stride 4 -> 8 -> 16 -> 32, channels
    doubling at each merge) -- not a network that keeps every resolution
    branch alive simultaneously. It still returns a 4-level [B,C,H,W]
    feature list, which is the only contract CSFModule / MultiScaleDecoder
    actually rely on (via self.channels), so nothing downstream changes.

    Weight sharing: HRTBDAPhase2 calls this module twice, once on the pre
    image and once on the post image, exactly like the original backbone --
    that Siamese-style weight sharing is unaffected by this swap.

    Input normalization: unchanged, no dataset changes needed.
    XBDHRTBDADataset._normalize() already uses ImageNet mean/std
    ([0.485,0.456,0.406] / [0.229,0.224,0.225]).

    Resolution requirement: Swin's relative position bias is defined per
    window (not per absolute image size, since ape=False in the standard
    configs), so it generalizes across input resolutions -- but H and W
    must both be evenly divisible by patch_size * window_size * 2^(num_stages-1)
    so every stage's feature map divides cleanly into windows. For the
    default swin_*_patch4_window7_224 family that's patch_size=4,
    window_size=7, 4 stages -> divisor 224. Raises ValueError at
    construction time if img_size doesn't satisfy this, instead of a
    cryptic shape mismatch deep inside attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        variant: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        img_size: int = 896,
        patch_size: int = 4,
        window_size: int = 7,
        num_stages: int = 4,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                f"SwinPretrainedBackbone expects 3-channel RGB input to match its "
                f"ImageNet pretraining; got in_channels={in_channels}."
            )

        divisor = patch_size * window_size * (2 ** (num_stages - 1))
        if img_size % divisor != 0:
            raise ValueError(
                f"img_size={img_size} is not divisible by patch_size*window_size*2^(num_stages-1)"
                f"={divisor} (patch_size={patch_size}, window_size={window_size}, "
                f"num_stages={num_stages}). Every Swin stage's feature map must divide "
                f"evenly into {window_size}x{window_size} windows. Use an --img-size / "
                f"--phase2-crop-size that is a multiple of {divisor}, e.g. "
                f"{divisor}, {2 * divisor}, {3 * divisor}, {4 * divisor}."
            )

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "SwinPretrainedBackbone requires the 'timm' package "
                "(pip install timm) for pretrained Swin Transformer weights."
            ) from exc

        # This backbone must handle two different resolutions with the SAME
        # instance: HRTBDAPhase2 trains on --phase2-crop-size crops but
        # validates/tests at --img-size (see make_loaders(): val/test datasets
        # always use args.img_size, only the train split uses phase2_crop_size).
        #
        # timm's PatchEmbed (timm/layers/patch_embed.py) gates its strict
        # input-resolution assertion on a per-instance self.strict_img_size
        # attribute:
        #     if self.img_size is not None:
        #         if self.strict_img_size:
        #             _assert(H == self.img_size[0], ...)
        #             _assert(W == self.img_size[1], ...)
        #         elif not self.dynamic_img_pad:
        #             _assert(H % patch_size == 0, ...)
        # Passing dynamic_img_size=True to timm.create_model() is NOT reliably
        # wired through to strict_img_size=False for every model builder
        # (confirmed empirically for the classic Swin v1 family via
        # smoke_test_swin_backbone_resolutions.py: the kwarg is accepted
        # without error but the assertion still fires). So instead of relying
        # on that kwarg, force strict_img_size=False directly on every
        # submodule that has the attribute, after construction. This targets
        # the exact flag forward() checks, not a guess at which constructor
        # kwarg maps onto it. dynamic_img_pad=True is also set as a no-cost
        # safety net for the patch-size divisibility branch (irrelevant for
        # any --img-size / --phase2-crop-size that already passes the
        # divisibility check above, since both are multiples of patch_size).
        if variant.startswith("twins_"):
            # timm 1.x supports multiscale Twins features, but the Twins
            # constructor does not accept the generic output_fmt keyword.
            # Its features_only wrapper already returns spatial feature maps.
            self.model = timm.create_model(
                variant,
                img_size=img_size,
                pretrained=pretrained,
                features_only=True,
            )
        elif variant.startswith("pvt_v2_"):
            # PVTv2 is resolution-flexible and does not accept img_size or
            # output_fmt in the timm constructor used on the HPC environment.
            self.model = timm.create_model(
                variant,
                pretrained=pretrained,
                features_only=True,
            )
        else:
            create_kwargs = dict(pretrained=pretrained, features_only=True, output_fmt="NCHW")
            try:
                self.model = timm.create_model(variant, img_size=img_size, **create_kwargs)
            except TypeError:
                # Some timm versions/variants don't accept img_size/output_fmt
                # for features_only models.
                self.model = timm.create_model(
                    variant,
                    **dict(pretrained=pretrained, features_only=True),
                )

        patched_modules = 0
        for module in self.model.modules():
            if hasattr(module, "strict_img_size"):
                module.strict_img_size = False
                patched_modules += 1
            if hasattr(module, "dynamic_img_pad"):
                module.dynamic_img_pad = True

        print(
            f"SwinPretrainedBackbone: constructed '{variant}', patched "
            f"strict_img_size=False on {patched_modules} submodule(s) so a single "
            f"instance can handle both --img-size and --phase2-crop-size.",
            flush=True,
        )
        if patched_modules == 0:
            print(
                "WARNING: found no submodule with a strict_img_size attribute to patch. "
                "This backbone may only work at a single fixed resolution -- if --img-size "
                "and --phase2-crop-size differ, expect a PatchEmbed input-size assertion "
                "failure the first time the other resolution is used.",
                flush=True,
            )

        feature_info = self.model.feature_info.get_dicts()
        self.channels = [f["num_chs"] for f in feature_info]
        self.strides = [f["reduction"] for f in feature_info]

        if len(self.channels) != num_stages:
            raise RuntimeError(
                f"Expected {num_stages} feature stages from timm variant '{variant}', "
                f"got {len(self.channels)} with channels={self.channels}. "
                f"CSFModule/MultiScaleDecoder assume exactly {num_stages} scales; "
                f"pick a different --swin-variant."
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = list(self.model(x))
        out = []
        for i, feat in enumerate(feats):
            expected_c = self.channels[i]
            if feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
                # timm returned this stage as NHWC (channels-last) instead of NCHW,
                # despite requesting output_fmt="NCHW" at construction -- observed on
                # at least one timm version for the Swin family. CSFModule /
                # MultiScaleDecoder are nn.Conv2d-based and require NCHW, so
                # self-correct here based on the actual tensor shape rather than
                # trusting the creation-time flag.
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return out

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
    def __init__(
        self,
        decoder_channels: int,
        swin_variant: str = "swin_tiny_patch4_window7_224",
        swin_pretrained: bool = True,
        img_size: int = 896,
        swin_patch_size: int = 4,
        swin_window_size: int = 7,
    ):
        super().__init__()
        self.backbone = SwinPretrainedBackbone(
            in_channels=3,
            variant=swin_variant,
            pretrained=swin_pretrained,
            img_size=img_size,
            patch_size=swin_patch_size,
            window_size=swin_window_size,
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
    """Phase II model.

    Outputs:
      damage_logits: [B, 4, H, W] independent sigmoid damage channels
      aux_loc:       [B, H, W] optional auxiliary building localization logit

    Final localization still comes from the saved Phase-I model during validation/test.
    """
    def __init__(
        self,
        decoder_channels: int,
        swin_variant: str = "swin_tiny_patch4_window7_224",
        swin_pretrained: bool = True,
        img_size: int = 672,
        swin_patch_size: int = 4,
        swin_window_size: int = 7,
        num_classes: int = 4,
    ):
        super().__init__()
        self.backbone = SwinPretrainedBackbone(
            in_channels=3,
            variant=swin_variant,
            pretrained=swin_pretrained,
            img_size=img_size,
            patch_size=swin_patch_size,
            window_size=swin_window_size,
        )

        self.csf = nn.ModuleList([CSFModule(c) for c in self.backbone.channels])

        self.decoder = MultiScaleDecoder(
            in_channels=self.backbone.channels,
            decoder_channels=decoder_channels,
            out_channels=num_classes,
        )

        self.aux_loc_head = nn.Sequential(
            ConvBNAct(self.backbone.channels[0], 64, kernel_size=3, stride=1),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fpre = self.backbone(pre)
        fpost = self.backbone(post)

        fused = [module(a, b) for module, a, b in zip(self.csf, fpre, fpost)]
        damage_logits = self.decoder(fused, output_size=pre.shape[-2:])
        aux_loc = F.interpolate(
            self.aux_loc_head(fpre[0]),
            size=pre.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return damage_logits, aux_loc


# -----------------------------
# Losses
# -----------------------------
class BinaryFocalDiceLoss(nn.Module):
    """Binary Focal + Dice with focal pt computed from unweighted BCE."""
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(1))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_plain = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce_plain)
        focal_weight = (1.0 - pt) ** self.gamma
        pw_map = torch.where(target > 0.5, self.pos_weight, torch.ones_like(target))
        focal = (focal_weight * bce_plain * pw_map).mean()

        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * inter + 1e-7) / (denom + 1e-7)).mean()

        return focal + dice, focal, dice


class MultilabelDamageFocalDiceLoss(nn.Module):
    """Per-channel sigmoid Focal + Dice for foreground-only damage classification.

    Target shape is [B, 4, H, W], valid_mask shape is [B, 1, H, W].
    Background pixels are ignored for damage loss; among building pixels each
    class is trained as an independent binary channel.
    """
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("class_weights", class_weights.float().view(1, -1, 1, 1))

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = valid_mask.float()
        denom_pixels = valid.sum().clamp_min(1.0)

        bce_plain = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce_plain)
        focal = ((1.0 - pt) ** self.gamma) * bce_plain * self.class_weights * valid
        focal_loss = focal.sum() / (denom_pixels * logits.shape[1])

        prob = torch.sigmoid(logits) * valid
        tgt = target * valid
        dims = (0, 2, 3)
        inter = (prob * tgt).sum(dim=dims)
        denom = prob.sum(dim=dims) + tgt.sum(dim=dims)
        dice_per_class = 1.0 - (2.0 * inter + 1e-7) / (denom + 1e-7)
        w = self.class_weights.view(-1)
        w = w / w.sum().clamp_min(1e-7)
        dice_loss = (dice_per_class * w).sum()

        return focal_loss + dice_loss, focal_loss, dice_loss


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


def make_damage4_class_weights(dataset: XBDHRTBDADataset, args: Optional[argparse.Namespace] = None) -> torch.Tensor:
    """Foreground-only damage weights [no, minor, major, destroyed].

    Stronger than inverse-sqrt: use inverse frequency, clipped, with optional
    extra boosts for minor and major damage.
    """
    counts5 = dataset.class5_counts().astype(np.float64)
    counts4 = counts5[1:5].copy()
    counts4[counts4 == 0] = 1.0

    freq = counts4 / counts4.sum()
    weights = 1.0 / (freq + 1e-12)
    weights = weights / weights.mean()

    if args is not None:
        weights[1] *= float(args.minor_damage_boost)
        weights[2] *= float(args.major_damage_boost)
        max_w = float(args.max_damage_class_weight)
        if max_w > 0:
            weights = np.minimum(weights, max_w)
        weights = weights / weights.mean()

    print(f"  damage counts [no,minor,major,destroyed]: {counts4.astype(int).tolist()}", flush=True)
    print(f"  damage weights [no,minor,major,destroyed]: {weights.tolist()}", flush=True)
    return torch.tensor(weights, dtype=torch.float32)


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



def target5_to_multilabel_damage4(target5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert target5 into multilabel damage target and foreground valid mask.

    Returns:
      y4:   [B, 4, H, W], one-hot damage class among building pixels
      mask: [B, 1, H, W], 1 only for valid building pixels with class 1..4
    """
    valid = ((target5 >= 1) & (target5 <= 4)).unsqueeze(1).float()
    y = torch.zeros((target5.shape[0], 4, target5.shape[1], target5.shape[2]), device=target5.device, dtype=torch.float32)
    for raw_cls in [1, 2, 3, 4]:
        y[:, raw_cls - 1] = (target5 == raw_cls).float()
    y = y * valid
    return y, valid


def get_damage_logits(model_output):
    return model_output[0] if isinstance(model_output, (tuple, list)) else model_output


def get_aux_loc_logits(model_output):
    if isinstance(model_output, (tuple, list)) and len(model_output) > 1:
        return model_output[1]
    return None


def damage_logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    """Independent sigmoid channels during training; argmax for final single label."""
    probs = torch.sigmoid(logits)
    return torch.argmax(probs, dim=1).long() + 1


def apply_damage_dilation(damage_pred: torch.Tensor, loc_pred: torch.Tensor, mode: str = "none", kernel_size: int = 3) -> torch.Tensor:
    """Optional test-time dilation for minority classes, only inside Phase-I mask.

    Conservative behavior: dilated minor/major only overwrite no-damage pixels.
    """
    if mode == "none" or kernel_size <= 1:
        return damage_pred
    out = damage_pred.clone()
    pad = kernel_size // 2
    classes = []
    if mode in {"minor", "minor_major"}:
        classes.append(2)
    if mode == "minor_major":
        classes.append(3)
    for cls in classes:
        m = (damage_pred == cls).float().unsqueeze(1)
        dil = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad).squeeze(1).bool()
        overwrite = dil & loc_pred.bool() & (out == 1)
        out[overwrite] = cls
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
    postprocess_dilation: str = "none",
    dilation_kernel: int = 3,
) -> Dict[str, object]:
    """Strict two-phase/cascaded evaluation with multi-label damage heads."""
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

        out = phase2_model(pre, post)
        damage_logits = get_damage_logits(out)
        damage_pred = damage_logits_to_pred(damage_logits)
        damage_pred = apply_damage_dilation(damage_pred, loc_pred, mode=postprocess_dilation, kernel_size=dilation_kernel)

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
        "postprocess_dilation": postprocess_dilation,
        "details": {
            "localization": loc.as_dict(),
            "no_damage": no_damage.as_dict(),
            "minor_damage": minor.as_dict(),
            "major_damage": major.as_dict(),
            "destroyed": destroyed.as_dict(),
        },
    }


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
        decoder_channels=args.decoder_channels,
        swin_variant=args.swin_variant,
        swin_pretrained=args.swin_pretrained,
        img_size=args.img_size,
        swin_patch_size=args.swin_patch_size,
        swin_window_size=args.swin_window_size,
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
def make_loaders(args: argparse.Namespace, phase2_training: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader, XBDHRTBDADataset]:
    crop_weights = (
        float(args.crop_weight_no_damage),
        float(args.crop_weight_minor),
        float(args.crop_weight_major),
        float(args.crop_weight_destroyed),
    )

    train_ds = XBDHRTBDADataset(
        args.xbd_root,
        args.train_split,
        args.img_size,
        training=True,
        crop_size=args.phase2_crop_size if phase2_training else 0,
        crop_candidate_count=args.crop_candidate_count if phase2_training else 1,
        crop_class_weights=crop_weights,
        extra_photometric=args.extra_photometric_aug if phase2_training else False,
    )
    val_ds = XBDHRTBDADataset(args.xbd_root, args.val_split, args.img_size, training=False)
    test_ds = XBDHRTBDADataset(args.xbd_root, args.test_split, args.img_size, training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
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
        decoder_channels=args.decoder_channels,
        swin_variant=args.swin_variant,
        swin_pretrained=args.swin_pretrained,
        img_size=args.img_size,
        swin_patch_size=args.swin_patch_size,
        swin_window_size=args.swin_window_size,
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

    # Load existing Phase I history so a resumed run keeps the same record
    # and does not duplicate already-finished epochs.
    history_path = output_dir / "history_phase1.json"
    if history_path.exists():
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

    resume_path = Path(args.resume_phase1_from) if getattr(args, "resume_phase1_from", "") else None

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
    print("\n================ PHASE II: MULTI-LABEL RARE-CROP DAMAGE CLASSIFICATION ================", flush=True)

    if phase1_ckpt is None or not phase1_ckpt.exists():
        raise FileNotFoundError(
            "A valid Phase I checkpoint is required for cascaded Phase II training. "
            f"Got: {phase1_ckpt}"
        )

    train_loader, val_loader, _, train_ds = make_loaders(args, phase2_training=True)

    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)
    print(f"Phase II training crop size: {args.phase2_crop_size}", flush=True)
    print(f"Crop candidates per sample: {args.crop_candidate_count}", flush=True)
    print(
        "Crop weights [no,minor,major,destroyed]: "
        f"[{args.crop_weight_no_damage}, {args.crop_weight_minor}, {args.crop_weight_major}, {args.crop_weight_destroyed}]",
        flush=True,
    )

    phase1_eval_model, phase1_threshold, phase1_meta = load_phase1_model_for_cascade(
        args=args,
        device=device,
        phase1_ckpt=phase1_ckpt,
    )

    model = HRTBDAPhase2(
        decoder_channels=args.decoder_channels,
        swin_variant=args.swin_variant,
        swin_pretrained=args.swin_pretrained,
        img_size=args.img_size,
        swin_patch_size=args.swin_patch_size,
        swin_window_size=args.swin_window_size,
        num_classes=4,
    ).to(device)

    load_phase1_backbone_into_phase2(model, phase1_ckpt, device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    class_weights = make_damage4_class_weights(train_ds, args=args).to(device)
    criterion = MultilabelDamageFocalDiceLoss(
        class_weights=class_weights,
        gamma=args.focal_gamma,
    ).to(device)

    loc_pos_weight = make_loc_pos_weight(train_ds).to(device)
    aux_loc_criterion = BinaryFocalDiceLoss(pos_weight=loc_pos_weight, gamma=args.focal_gamma).to(device)

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
    accumulation_steps = max(1, int(args.grad_accum_steps))

    def run_training_epoch(epoch: int, total_epochs: int, finetune: bool = False):
        model.train()
        total_meter = AverageMeter()
        focal_meter = AverageMeter()
        dice_meter = AverageMeter()
        aux_meter = AverageMeter()

        phase_name = "Fine-tune" if finetune else "Phase II"
        print(f"\n{phase_name} epoch {epoch}/{total_epochs} | LR={optimizer.param_groups[0]['lr']:.8f}", flush=True)
        iterator = tqdm(train_loader, desc=f"p2-{epoch}") if (tqdm is not None and sys.stderr.isatty()) else train_loader
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(iterator, start=1):
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            target5 = batch["target5"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            damage_target, valid_mask = target5_to_multilabel_damage4(target5)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    out = model(pre, post)
                    damage_logits = get_damage_logits(out)
                    aux_loc = get_aux_loc_logits(out)
                    loss_damage, focal, dice = criterion(damage_logits, damage_target, valid_mask)
                    if aux_loc is not None and args.aux_loc_weight > 0:
                        loss_aux, _, _ = aux_loc_criterion(aux_loc, loc)
                    else:
                        loss_aux = damage_logits.sum() * 0.0
                    loss = args.cls_loss_weight * loss_damage + args.aux_loc_weight * loss_aux
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    out = model(pre, post)
                    damage_logits = get_damage_logits(out)
                    aux_loc = get_aux_loc_logits(out)
                    loss_damage, focal, dice = criterion(damage_logits, damage_target, valid_mask)
                    if aux_loc is not None and args.aux_loc_weight > 0:
                        loss_aux, _, _ = aux_loc_criterion(aux_loc, loc)
                    else:
                        loss_aux = damage_logits.sum() * 0.0
                    loss = args.cls_loss_weight * loss_damage + args.aux_loc_weight * loss_aux

            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % accumulation_steps == 0 or step == len(train_loader):
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

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"{phase_name} Epoch {epoch}/{total_epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | focal={focal_meter.avg:.4f} | "
                    f"dice={dice_meter.avg:.4f} | aux={aux_meter.avg:.4f}",
                    flush=True,
                )

        return total_meter, focal_meter, dice_meter, aux_meter

    def validate_and_save(epoch_label: int, train_meters, phase_label: str):
        nonlocal best_score, best_epoch, no_improve
        total_meter, focal_meter, dice_meter, aux_meter = train_meters
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
            "hold_score_cascade": val_score,
            "hold_localization_f1_from_phase1_mask": float(val_results["localization_f1"]),
            "hold_damage_f1": float(val_results["damage_f1"]),
            "hold_no_damage_f1": float(val_results["damage_f1_no_damage"]),
            "hold_minor_damage_f1": float(val_results["damage_f1_minor_damage"]),
            "hold_major_damage_f1": float(val_results["damage_f1_major_damage"]),
            "hold_destroyed_f1": float(val_results["damage_f1_destroyed"]),
            "phase1_threshold": phase1_threshold,
            "phase1_best_metric_hold": phase1_meta["best_metric"],
            "postprocess_dilation": args.postprocess_dilation,
        }
        history.append(row)

        print(
            f"{phase_label} Epoch {epoch_label:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"hold_score_cascade={row['hold_score_cascade']:.6f} | "
            f"hold_loc_f1_from_phase1={row['hold_localization_f1_from_phase1_mask']:.6f} | "
            f"hold_damage_f1={row['hold_damage_f1']:.6f} | "
            f"no={row['hold_no_damage_f1']:.6f} | minor={row['hold_minor_damage_f1']:.6f} | "
            f"major={row['hold_major_damage_f1']:.6f} | destroyed={row['hold_destroyed_f1']:.6f}",
            flush=True,
        )

        extra = {
            "phase1_checkpoint": str(phase1_ckpt),
            "phase1_threshold": phase1_threshold,
            "phase1_best_metric_hold": phase1_meta["best_metric"],
            "cascade_validation": True,
            "multilabel_damage_heads": True,
            "rare_crop_training": True,
            "postprocess_dilation": args.postprocess_dilation,
        }

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch_label
            no_improve = 0
            save_checkpoint(checkpoints_dir / "phase2_best.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)
            print(f"Saved Phase II best checkpoint | epoch={epoch_label} | cascade_score={best_score:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"Phase II no improvement for {no_improve} epoch(s). Best epoch={best_epoch}", flush=True)

        save_checkpoint(checkpoints_dir / "phase2_last.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)
        if epoch_label % max(1, args.save_every) == 0:
            save_checkpoint(checkpoints_dir / f"phase2_epoch_{epoch_label:03d}.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)
        with open(output_dir / "history_phase2.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    for epoch in range(1, args.phase2_epochs + 1):
        meters = run_training_epoch(epoch, args.phase2_epochs, finetune=False)
        scheduler.step()
        validate_and_save(epoch, meters, phase_label="main")
        if no_improve >= args.early_stopping_patience:
            print(f"Phase II early stopping at epoch {epoch}.", flush=True)
            break

    # Optional short tuning stage inspired by the xView2 winner's warm-restart fine-tuning.
    if args.finetune_epochs > 0:
        print("\n================ PHASE II SHORT FINE-TUNING STAGE ================", flush=True)
        for g in optimizer.param_groups:
            g["lr"] = float(args.finetune_lr)
        for ft_epoch in range(1, args.finetune_epochs + 1):
            for g in optimizer.param_groups:
                g["lr"] = float(args.finetune_lr) * (0.5 ** (ft_epoch - 1))
            epoch_label = args.phase2_epochs + ft_epoch
            meters = run_training_epoch(ft_epoch, args.finetune_epochs, finetune=True)
            validate_and_save(epoch_label, meters, phase_label="finetune")

    print(f"Phase II done. Best epoch={best_epoch}, best hold cascade score={best_score:.6f}", flush=True)
    return checkpoints_dir / "phase2_best.pt"


def test_phase2(args: argparse.Namespace, device: torch.device, checkpoint_path: Path, phase1_ckpt: Path) -> None:
    print("\n================ CASCADED TESTING: PHASE I MASK + PHASE II DAMAGE ================", flush=True)

    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Phase II checkpoint not found: {checkpoint_path}")

    _, _, test_loader, _ = make_loaders(args)

    phase1_model, phase1_threshold, phase1_meta = load_phase1_model_for_cascade(
        args=args,
        device=device,
        phase1_ckpt=phase1_ckpt,
    )

    phase2_model = HRTBDAPhase2(
        decoder_channels=args.decoder_channels,
        swin_variant=args.swin_variant,
        swin_pretrained=args.swin_pretrained,
        img_size=args.img_size,
        swin_patch_size=args.swin_patch_size,
        swin_window_size=args.swin_window_size,
        num_classes=4,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        phase2_model = nn.DataParallel(phase2_model)

    ckpt = load_model_weights(phase2_model, checkpoint_path, device)
    best_epoch = int(ckpt.get("epoch", -1))

    results = evaluate_phase2_cascade(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=test_loader,
        device=device,
        phase1_threshold=phase1_threshold,
        postprocess_dilation=args.postprocess_dilation,
        dilation_kernel=args.dilation_kernel,
    )

    output_dir = Path(args.output_dir)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    results["phase1_checkpoint"] = str(phase1_ckpt)
    results["phase2_checkpoint"] = str(checkpoint_path)
    results["phase1_epoch"] = phase1_meta["epoch"]
    results["phase1_best_metric_hold"] = phase1_meta["best_metric"]
    results["phase1_threshold"] = phase1_threshold
    results["phase2_best_epoch_selected_on_hold"] = best_epoch

    json_path = scores_dir / "scores_xbd_test_hrtbda_v5_multilabel_crop_cascade.json"
    txt_path = scores_dir / "scores_xbd_test_hrtbda_v5_multilabel_crop_cascade.txt"
    summary_path = scores_dir / "summary_hrtbda_v5_multilabel_crop_cascade.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    train_splits_text = "+".join(args.train_split) if isinstance(args.train_split, list) else str(args.train_split)

    lines = [
        f"Experiment: HRTBDA v5 multi-label rare-crop cascade {train_splits_text} -> {args.val_split} -> {args.test_split}",
        f"Phase I checkpoint: {phase1_ckpt}",
        f"Phase I stored best epoch: {phase1_meta['epoch']}",
        f"Phase I stored hold Localization F1: {phase1_meta['best_metric']:.6f}",
        f"Phase I threshold used for mask: {phase1_threshold:.2f}",
        f"Damage post-processing dilation: {args.postprocess_dilation}",
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

    parser.add_argument("--phase", type=str, default="phase2_test", choices=["both", "phase1", "phase2", "phase2_test", "test", "inspect_phase1"])

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

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--img-size",
        type=int,
        default=896,
        help="Must be a multiple of swin-patch-size * swin-window-size * 8 (224 for the "
        "default variant). 896 = 224*4, closest clean multiple to the original "
        "backbone's 1024.",
    )
    parser.add_argument(
        "--phase2-crop-size",
        type=int,
        default=672,
        help="Must be a multiple of swin-patch-size * swin-window-size * 8 (224 for the "
        "default variant). 672 = 224*3, closest clean multiple to the original "
        "backbone's 608.",
    )
    parser.add_argument("--crop-candidate-count", type=int, default=8)
    parser.add_argument("--crop-weight-no-damage", type=float, default=1.0)
    parser.add_argument("--crop-weight-minor", type=float, default=12.0)
    parser.add_argument("--crop-weight-major", type=float, default=12.0)
    parser.add_argument("--crop-weight-destroyed", type=float, default=4.0)
    parser.add_argument("--extra-photometric-aug", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--swin-variant",
        type=str,
        default="swin_tiny_patch4_window7_224",
        help="timm model name for the ImageNet-pretrained Swin backbone, e.g. "
        "swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, "
        "swin_base_patch4_window7_224.",
    )
    parser.add_argument(
        "--no-imagenet-pretrained",
        dest="swin_pretrained",
        action="store_false",
        help="Disable ImageNet pretrained weights (random init instead). Useful as an "
        "ablation to isolate the effect of pretraining vs. the Swin architecture itself.",
    )
    parser.set_defaults(swin_pretrained=True)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument(
        "--swin-patch-size",
        type=int,
        default=4,
        help="Must match --swin-variant's patch size (4 for the standard swin_*_patch4_* family).",
    )
    parser.add_argument(
        "--swin-window-size",
        type=int,
        default=7,
        help="Must match --swin-variant's window size (7 for the standard swin_*_window7_* family). "
        "--img-size and --phase2-crop-size must each be a multiple of "
        "swin-patch-size * swin-window-size * 8.",
    )

    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--loc-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--aux-loc-weight", type=float, default=0.2)
    parser.add_argument("--minor-damage-boost", type=float, default=1.5)
    parser.add_argument("--major-damage-boost", type=float, default=1.5)
    parser.add_argument("--max-damage-class-weight", type=float, default=10.0)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--finetune-lr", type=float, default=5e-5)
    parser.add_argument("--postprocess-dilation", type=str, default="none", choices=["none", "minor", "minor_major"])
    parser.add_argument("--dilation-kernel", type=int, default=3)

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

    swin_divisor = args.swin_patch_size * args.swin_window_size * 8
    for size_name, size_value in [("--img-size", args.img_size), ("--phase2-crop-size", args.phase2_crop_size)]:
        if size_value % swin_divisor != 0:
            raise ValueError(
                f"{size_name}={size_value} is not divisible by "
                f"swin-patch-size*swin-window-size*8={swin_divisor} "
                f"(swin-patch-size={args.swin_patch_size}, swin-window-size={args.swin_window_size}). "
                f"Both --img-size and --phase2-crop-size are used with the same Swin backbone "
                f"instance (train crops use --phase2-crop-size, validation/test always use "
                f"--img-size), so both must independently be a multiple of {swin_divisor}, "
                f"e.g. {swin_divisor}, {2 * swin_divisor}, {3 * swin_divisor}, {4 * swin_divisor}."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "scores").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoints_dir = output_dir / "checkpoints"

    phase1_ckpt = Path(args.phase1_checkpoint) if args.phase1_checkpoint else checkpoints_dir / "phase1_best.pt"
    phase2_ckpt = Path(args.phase2_checkpoint) if args.phase2_checkpoint else checkpoints_dir / "phase2_best.pt"

    print("===== HRTBDA V5 MULTI-LABEL RARE-CROP CASCADED TRAINING =====", flush=True)
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
    print(f"Eval batch size: {args.eval_batch_size}", flush=True)
    print(f"Gradient accumulation steps: {args.grad_accum_steps}", flush=True)
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}", flush=True)
    print(f"Image size: {args.img_size}", flush=True)
    print(f"Phase II crop size: {args.phase2_crop_size}", flush=True)
    print(f"Crop candidate count: {args.crop_candidate_count}", flush=True)
    print(f"LR: {args.lr}", flush=True)
    print(f"Weight decay: {args.weight_decay}", flush=True)
    print(f"Swin variant: {args.swin_variant}", flush=True)
    print(f"Swin ImageNet pretrained: {args.swin_pretrained}", flush=True)
    print(f"Swin patch size: {args.swin_patch_size}", flush=True)
    print(f"Swin window size: {args.swin_window_size}", flush=True)
    print(f"Decoder channels: {args.decoder_channels}", flush=True)
    print(f"Focal gamma: {args.focal_gamma}", flush=True)
    print(f"Max grad norm: {args.max_grad_norm}", flush=True)
    print(f"Warmup epochs: {args.warmup_epochs}", flush=True)
    print("Architecture: HRTBDA v5 pretrained-Swin-Transformer backbone + DCSwin-free CSF fusion cascade", flush=True)
    print("Damage head: 4 independent sigmoid channels + auxiliary localization head", flush=True)
    print("Final inference: Phase I mask gives localization; Phase II predicts 4 foreground damage classes.", flush=True)
    print(f"Postprocess dilation: {args.postprocess_dilation}", flush=True)
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
