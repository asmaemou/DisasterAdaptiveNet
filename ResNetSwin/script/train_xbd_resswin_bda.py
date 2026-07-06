#!/usr/bin/env python3
"""
ResSwin-BDA xBD experiment.

Train on existing xBD splits:
  train + tier3  -> training
  hold           -> validation / checkpoint selection
  test           -> unseen final test

Architecture:
  shared Siamese ResNet34 encoder
  feature fusion: [pre, post, |post-pre|] + 1x1 conv
  window/shifted-window Swin-style context on deep fused features
  FPN decoder
  two heads: building localization + 4-class damage classification

Expected data structure:
  xview2/{train,tier3,hold,test}/images/*_pre_disaster.png
  xview2/{train,tier3,hold,test}/images/*_post_disaster.png
  xview2/{train,tier3,hold,test}/targets/*_pre_disaster_target.png
  xview2/{train,tier3,hold,test}/targets/*_post_disaster_target.png

Final classes:
  0 background
  1 no_damage
  2 minor_damage
  3 major_damage
  4 destroyed
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

try:
    from torchvision import models
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for ResNet34. Please install/import torchvision in your conda env.") from exc

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    GradScaler = torch.amp.GradScaler
    autocast = torch.amp.autocast
    USE_TORCH_AMP = True
except AttributeError:  # pragma: no cover
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
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    return image_list, mask_list


def apply_photometric_augmentations(image_list: List[np.ndarray], training: bool) -> List[np.ndarray]:
    if not training:
        return image_list

    out = image_list

    # Same brightness factor for pre/post so we do not create fake change.
    if np.random.rand() < 0.5:
        factor = float(np.random.uniform(0.75, 1.25))
        out = [np.clip(x.astype(np.float32) * factor, 0, 255).astype(np.uint8) for x in out]

    # Same RGB shift for pre/post.
    if np.random.rand() < 0.35:
        shift = np.random.uniform(-12.0, 12.0, size=(1, 1, 3)).astype(np.float32)
        out = [np.clip(x.astype(np.float32) + shift, 0, 255).astype(np.uint8) for x in out]

    # Mild blur.
    if np.random.rand() < 0.25:
        ksize = int(np.random.choice([3, 5]))
        out = [cv2.GaussianBlur(x, (ksize, ksize), 0) for x in out]

    # Mild noise.
    if np.random.rand() < 0.25:
        std = float(np.random.uniform(2.0, 8.0))
        noisy = []
        for img in out:
            n = np.random.normal(0.0, std, img.shape).astype(np.float32)
            noisy.append(np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8))
        out = noisy

    return out


def rare_damage_candidate_crop(
    image_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    crop_size: int,
    candidate_count: int,
    class_weights: Tuple[float, float, float, float],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        score = float((counts * weights).sum())
        if score > best_score:
            best_score = score
            best_xy = (y0, x0)

    y0, x0 = best_xy
    image_list = [x[y0:y0 + size, x0:x0 + size].copy() for x in image_list]
    mask_list = [x[y0:y0 + size, x0:x0 + size].copy() for x in mask_list]
    return image_list, mask_list


class XBDResSwinDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str | List[str] | Tuple[str, ...],
        image_size: int,
        training: bool,
        crop_size: int = 0,
        crop_candidate_count: int = 1,
        crop_class_weights: Tuple[float, float, float, float] = (1.0, 12.0, 12.0, 4.0),
        photometric_aug: bool = False,
    ):
        self.root = Path(root)
        self.splits = [str(s) for s in split] if isinstance(split, (list, tuple)) else [str(split)]
        self.split = "+".join(self.splits)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.crop_size = int(crop_size)
        self.crop_candidate_count = int(crop_candidate_count)
        self.crop_class_weights = tuple(float(x) for x in crop_class_weights)
        self.photometric_aug = bool(photometric_aug)

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
        patterns = [
            "*_post_disaster.png", "*_post_disaster.jpg", "*_post_disaster.jpeg",
            "*_post_disaster.tif", "*_post_disaster.tiff", "*_post_disaster.bmp",
        ]

        for split_name in self.splits:
            images_dir = self.root / split_name / "images"
            targets_dir = self.root / split_name / "targets"
            post_images: List[Path] = []
            for pat in patterns:
                post_images.extend(images_dir.glob(pat))
            for post_path in sorted(post_images):
                prefix = post_path.stem.replace("_post_disaster", "")
                ext = post_path.suffix
                pre_path = images_dir / f"{prefix}_pre_disaster{ext}"
                pre_tgt = targets_dir / f"{prefix}_pre_disaster_target.png"
                post_tgt = targets_dir / f"{prefix}_post_disaster_target.png"
                key = prefix
                if key in seen:
                    continue
                if pre_path.exists() and pre_tgt.exists() and post_tgt.exists():
                    seen.add(key)
                    samples.append(XBDSample(prefix, split_name, pre_path, post_path, pre_tgt, post_tgt))
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
        )

        if self.training and self.photometric_aug:
            [pre, post] = apply_photometric_augmentations([pre, post], training=True)

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
            "target5": torch.from_numpy(target5).long(),
            "stem": s.stem,
            "split": s.split,
        }

    def loc_counts(self) -> Tuple[int, int]:
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
# Metrics
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
        self.sum += float(value) * int(n)
        self.count += int(n)


class F1Recorder:
    def __init__(self, tp: int = 0, fp: int = 0, fn: int = 0):
        self.tp = int(tp)
        self.fp = int(fp)
        self.fn = int(fn)

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        self.tp += int(((pred == 1) & (true == 1)).sum().item())
        self.fp += int(((pred == 1) & (true == 0)).sum().item())
        self.fn += int(((pred == 0) & (true == 1)).sum().item())

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
        p, r = self.precision, self.recall
        return 0.0 if p == 0.0 or r == 0.0 else 2.0 * p * r / (p + r)

    def as_dict(self) -> Dict[str, float | int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": self.precision, "recall": self.recall, "f1": self.f1}


def harmonic_mean(values: List[float]) -> float:
    vals = [max(float(v), 1e-6) for v in values]
    return len(vals) / sum(1.0 / v for v in vals)


# -----------------------------
# Model
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


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            try:
                weights = models.ResNet34_Weights.DEFAULT
            except AttributeError:  # older torchvision
                weights = "DEFAULT"
            net = models.resnet34(weights=weights)
        else:
            net = models.resnet34(weights=None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # 64, H/4
        self.layer2 = net.layer2  # 128, H/8
        self.layer3 = net.layer3  # 256, H/16
        self.layer4 = net.layer4  # 512, H/32
        self.channels = [64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class FusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            ConvBNAct(channels, channels, kernel_size=3, stride=1),
        )

    def forward(self, fpre: torch.Tensor, fpost: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(fpost - fpre)
        return self.fuse(torch.cat([fpre, fpost, diff], dim=1))


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
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, c)


class WindowSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.window_size = int(window_size)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

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


class SwinStyleBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int = 8, shift: bool = False, dropout: float = 0.0):
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = self.window_size // 2 if shift else 0
        self.norm1 = LayerNorm2d(channels)
        self.attn = WindowSelfAttention(channels, num_heads=num_heads, window_size=window_size)
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * 4
        self.mlp = nn.Sequential(
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
        y = self.norm1(x)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        y = self.attn(y)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class SwinContext(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int, depth: int = 2):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(SwinStyleBlock(channels, num_heads=num_heads, window_size=window_size, shift=bool(i % 2)))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class FPNDecoder(nn.Module):
    def __init__(self, in_channels: List[int], decoder_channels: int):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            )
            for c in in_channels
        ])
        self.smooth4 = ConvBNAct(decoder_channels, decoder_channels)
        self.smooth3 = ConvBNAct(decoder_channels, decoder_channels)
        self.smooth2 = ConvBNAct(decoder_channels, decoder_channels)
        self.smooth1 = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels),
            ConvBNAct(decoder_channels, decoder_channels),
        )

    def forward(self, features: List[torch.Tensor], output_size: Tuple[int, int]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        p4 = self.smooth4(self.lateral[3](f4))
        p3 = self.smooth3(self.lateral[2](f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(self.lateral[1](f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.smooth1(self.lateral[0](f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False))
        return F.interpolate(p1, size=output_size, mode="bilinear", align_corners=False)


class ResSwinBDA(nn.Module):
    def __init__(self, decoder_channels: int = 128, window_size: int = 8, swin_depth: int = 2, pretrained_resnet: bool = False):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=pretrained_resnet)  # shared Siamese encoder
        chs = self.encoder.channels
        self.fusion = nn.ModuleList([FusionBlock(c) for c in chs])

        # Apply Swin-style context only to deeper feature maps to keep memory manageable.
        self.ctx1 = nn.Identity()
        self.ctx2 = nn.Identity()
        self.ctx3 = SwinContext(chs[2], num_heads=8, window_size=window_size, depth=swin_depth)
        self.ctx4 = SwinContext(chs[3], num_heads=8, window_size=window_size, depth=swin_depth)

        self.decoder = FPNDecoder(chs, decoder_channels=decoder_channels)
        self.loc_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.damage_head = nn.Conv2d(decoder_channels, 4, kernel_size=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fpre = self.encoder(pre)
        fpost = self.encoder(post)
        fused = [m(a, b) for m, a, b in zip(self.fusion, fpre, fpost)]
        fused = [self.ctx1(fused[0]), self.ctx2(fused[1]), self.ctx3(fused[2]), self.ctx4(fused[3])]
        dec = self.decoder(fused, output_size=pre.shape[-2:])
        loc_logits = self.loc_head(dec).squeeze(1)
        damage_logits = self.damage_head(dec)
        return loc_logits, damage_logits


# -----------------------------
# Losses
# -----------------------------
class BinaryFocalDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(1))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_plain = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce_plain)
        focal_weight = (1.0 - pt) ** self.gamma
        pos_weight_map = torch.where(target > 0.5, self.pos_weight, torch.ones_like(target))
        focal = (focal_weight * bce_plain * pos_weight_map).mean()

        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * inter + 1e-7) / (denom + 1e-7)).mean()
        return focal + dice, focal, dice


class DamageFocalDiceLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # target: 0..3 for valid building pixels, 255 for background/ignore.
        ce_plain = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        valid = target != self.ignore_index
        if valid.any():
            ce_valid = ce_plain[valid]
            pt = torch.exp(-ce_valid)
            target_valid = target[valid]
            w = self.class_weights[target_valid]
            focal = (((1.0 - pt) ** self.gamma) * ce_valid * w).mean()
        else:
            focal = logits.sum() * 0.0

        probs = torch.softmax(logits, dim=1)
        safe_target = target.clone()
        safe_target[safe_target == self.ignore_index] = 0
        one_hot = F.one_hot(safe_target, num_classes=4).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice_per_class = 1.0 - (2.0 * inter + 1e-7) / (denom + 1e-7)
        w = self.class_weights / self.class_weights.sum().clamp_min(1e-7)
        dice = (dice_per_class * w).sum()
        return focal + dice, focal, dice


def target5_to_damage4(target5: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    out = torch.full_like(target5, fill_value=ignore_index)
    valid = (target5 >= 1) & (target5 <= 4)
    out[valid] = target5[valid] - 1
    return out


def make_loc_pos_weight(dataset: XBDResSwinDataset) -> torch.Tensor:
    pos, neg = dataset.loc_counts()
    raw = max(1.0, neg / max(pos, 1))
    return torch.tensor([min(raw, 10.0)], dtype=torch.float32)


def make_damage_class_weights(dataset: XBDResSwinDataset, minor_boost: float, major_boost: float, max_weight: float) -> torch.Tensor:
    counts5 = dataset.class5_counts().astype(np.float64)
    counts4 = counts5[1:5].copy()
    counts4[counts4 == 0] = 1.0
    freq = counts4 / counts4.sum()
    weights = 1.0 / (freq + 1e-12)
    weights = weights / weights.mean()
    weights[1] *= float(minor_boost)
    weights[2] *= float(major_boost)
    if max_weight > 0:
        weights = np.minimum(weights, float(max_weight))
    weights = weights / weights.mean()
    print(f"Damage counts [no, minor, major, destroyed]: {counts4.astype(int).tolist()}", flush=True)
    print(f"Damage weights [no, minor, major, destroyed]: {weights.tolist()}", flush=True)
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loc_threshold: float,
    save_pred_dir: Optional[Path] = None,
) -> Dict[str, object]:
    model.eval()
    if save_pred_dir is not None:
        save_pred_dir.mkdir(parents=True, exist_ok=True)

    loc_rec = F1Recorder()
    cls_counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in [1, 2, 3, 4]}

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        target5 = batch["target5"].to(device, non_blocking=True).long()

        loc_logits, damage_logits = model(pre, post)
        loc_pred = (torch.sigmoid(loc_logits) > loc_threshold).long()
        damage_pred = torch.argmax(damage_logits, dim=1).long() + 1
        final_pred = torch.zeros_like(damage_pred)
        final_pred[loc_pred.bool()] = damage_pred[loc_pred.bool()]

        loc_rec.update(loc_pred, loc_true)

        valid_building = (target5 >= 1) & (target5 <= 4)
        pred_valid = final_pred[valid_building]
        true_valid = target5[valid_building]
        for cls in [1, 2, 3, 4]:
            cls_counts[cls]["tp"] += int(((pred_valid == cls) & (true_valid == cls)).sum().item())
            cls_counts[cls]["fp"] += int(((pred_valid == cls) & (true_valid != cls)).sum().item())
            cls_counts[cls]["fn"] += int(((pred_valid != cls) & (true_valid == cls)).sum().item())

        if save_pred_dir is not None:
            stems = batch["stem"]
            pred_np = final_pred.detach().cpu().numpy().astype(np.uint8)
            loc_np = loc_pred.detach().cpu().numpy().astype(np.uint8)
            for i, stem in enumerate(stems):
                cv2.imwrite(str(save_pred_dir / f"{stem}_resSwinBDA_pred.png"), pred_np[i])
                cv2.imwrite(str(save_pred_dir / f"{stem}_resSwinBDA_loc.png"), loc_np[i])

    recs = {cls: F1Recorder(v["tp"], v["fp"], v["fn"]) for cls, v in cls_counts.items()}
    damage_f1 = harmonic_mean([recs[1].f1, recs[2].f1, recs[3].f1, recs[4].f1])
    score = 0.3 * loc_rec.f1 + 0.7 * damage_f1
    return {
        "score": score,
        "localization_f1": loc_rec.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": recs[1].f1,
        "damage_f1_minor_damage": recs[2].f1,
        "damage_f1_major_damage": recs[3].f1,
        "damage_f1_destroyed": recs[4].f1,
        "loc_threshold": float(loc_threshold),
        "details": {
            "localization": loc_rec.as_dict(),
            "no_damage": recs[1].as_dict(),
            "minor_damage": recs[2].as_dict(),
            "major_damage": recs[3].as_dict(),
            "destroyed": recs[4].as_dict(),
        },
    }


# -----------------------------
# Train / test
# -----------------------------
def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, XBDResSwinDataset]:
    crop_weights = (
        float(args.crop_weight_no_damage),
        float(args.crop_weight_minor),
        float(args.crop_weight_major),
        float(args.crop_weight_destroyed),
    )
    train_ds = XBDResSwinDataset(
        args.xbd_root,
        args.train_split,
        image_size=args.img_size,
        training=True,
        crop_size=args.crop_size,
        crop_candidate_count=args.crop_candidate_count,
        crop_class_weights=crop_weights,
        photometric_aug=args.photometric_aug,
    )
    val_ds = XBDResSwinDataset(args.xbd_root, args.val_split, image_size=args.img_size, training=False)
    test_ds = XBDResSwinDataset(args.xbd_root, args.test_split, image_size=args.img_size, training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader, train_ds


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": float(best_score),
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


def train(args: argparse.Namespace, device: torch.device) -> Path:
    print("\n================ TRAIN ResSwin-BDA ================", flush=True)
    train_loader, val_loader, _, train_ds = make_loaders(args)
    print(f"Train splits: {args.train_split}", flush=True)
    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val split: {args.val_split} | Val samples: {len(val_loader.dataset)}", flush=True)

    model = ResSwinBDA(
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        swin_depth=args.swin_depth,
        pretrained_resnet=args.pretrained_resnet,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    loc_pos_weight = make_loc_pos_weight(train_ds).to(device)
    damage_weights = make_damage_class_weights(
        train_ds,
        minor_boost=args.minor_damage_boost,
        major_boost=args.major_damage_boost,
        max_weight=args.max_damage_class_weight,
    ).to(device)
    loc_criterion = BinaryFocalDiceLoss(pos_weight=loc_pos_weight, gamma=args.focal_gamma).to(device)
    dmg_criterion = DamageFocalDiceLoss(class_weights=damage_weights, gamma=args.focal_gamma).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = make_scaler(args, device)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    scores_dir = output_dir / "scores"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    history: List[Dict[str, object]] = []
    accum = max(1, int(args.grad_accum_steps))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_meter = AverageMeter()
        loc_meter = AverageMeter()
        dmg_meter = AverageMeter()
        focal_meter = AverageMeter()
        dice_meter = AverageMeter()

        print(f"\nEpoch {epoch}/{args.epochs} | LR={optimizer.param_groups[0]['lr']:.8f}", flush=True)
        iterator = tqdm(train_loader, desc=f"epoch {epoch}") if (tqdm is not None and sys.stderr.isatty()) else train_loader
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(iterator, start=1):
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            target5 = batch["target5"].to(device, non_blocking=True)
            dmg_target = target5_to_damage4(target5)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    loc_logits, dmg_logits = model(pre, post)
                    loss_loc, loc_focal, loc_dice = loc_criterion(loc_logits, loc)
                    loss_dmg, dmg_focal, dmg_dice = dmg_criterion(dmg_logits, dmg_target)
                    loss = args.loc_loss_weight * loss_loc + args.damage_loss_weight * loss_dmg
            else:  # pragma: no cover
                with autocast(enabled=args.amp and device.type == "cuda"):
                    loc_logits, dmg_logits = model(pre, post)
                    loss_loc, loc_focal, loc_dice = loc_criterion(loc_logits, loc)
                    loss_dmg, dmg_focal, dmg_dice = dmg_criterion(dmg_logits, dmg_target)
                    loss = args.loc_loss_weight * loss_loc + args.damage_loss_weight * loss_dmg

            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = pre.size(0)
            total_meter.update(loss.item(), bs)
            loc_meter.update(loss_loc.item(), bs)
            dmg_meter.update(loss_dmg.item(), bs)
            focal_meter.update((loc_focal.item() + dmg_focal.item()) / 2.0, bs)
            dice_meter.update((loc_dice.item() + dmg_dice.item()) / 2.0, bs)

            if step % args.print_every == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | loc={loc_meter.avg:.4f} | dmg={dmg_meter.avg:.4f} | "
                    f"focal={focal_meter.avg:.4f} | dice={dice_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()

        val_results = evaluate(model, val_loader, device, loc_threshold=args.loc_threshold)
        val_score = float(val_results["score"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_loc_loss": loc_meter.avg,
            "train_damage_loss": dmg_meter.avg,
            "hold_score": val_score,
            "hold_localization_f1": float(val_results["localization_f1"]),
            "hold_damage_f1": float(val_results["damage_f1"]),
            "hold_no_damage_f1": float(val_results["damage_f1_no_damage"]),
            "hold_minor_damage_f1": float(val_results["damage_f1_minor_damage"]),
            "hold_major_damage_f1": float(val_results["damage_f1_major_damage"]),
            "hold_destroyed_f1": float(val_results["damage_f1_destroyed"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"hold_score={row['hold_score']:.6f} | hold_loc_f1={row['hold_localization_f1']:.6f} | "
            f"hold_damage_f1={row['hold_damage_f1']:.6f} | no={row['hold_no_damage_f1']:.6f} | "
            f"minor={row['hold_minor_damage_f1']:.6f} | major={row['hold_major_damage_f1']:.6f} | "
            f"destroyed={row['hold_destroyed_f1']:.6f}",
            flush=True,
        )

        extra = {"loc_threshold": float(args.loc_threshold), "hold_results": val_results}
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_score, args, extra=extra)
            print(f"Saved best checkpoint: epoch={epoch}, hold_score={best_score:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s). Best epoch={best_epoch}", flush=True)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_score, args, extra=extra)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, best_score, args, extra=extra)

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    print(f"Training done. Best epoch={best_epoch}, best hold score={best_score:.6f}", flush=True)
    return ckpt_dir / "best.pt"


def test(args: argparse.Namespace, device: torch.device, checkpoint_path: Path) -> None:
    print("\n================ TEST ResSwin-BDA ON UNSEEN TEST SET ================", flush=True)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, _, test_loader, _ = make_loaders(args)
    print(f"Test split: {args.test_split} | Test samples: {len(test_loader.dataset)}", flush=True)

    model = ResSwinBDA(
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        swin_depth=args.swin_depth,
        pretrained_resnet=False,
    ).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    ckpt = load_model_weights(model, checkpoint_path, device)
    best_epoch = int(ckpt.get("epoch", -1))
    loc_threshold = float(ckpt.get("loc_threshold", args.loc_threshold))

    pred_dir = Path(args.output_dir) / "predictions" / args.test_split if args.save_test_preds else None
    results = evaluate(model, test_loader, device, loc_threshold=loc_threshold, save_pred_dir=pred_dir)
    results["checkpoint"] = str(checkpoint_path)
    results["best_epoch_selected_on_hold"] = best_epoch
    results["train_splits"] = args.train_split
    results["val_split"] = args.val_split
    results["test_split"] = args.test_split

    scores_dir = Path(args.output_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    json_path = scores_dir / "scores_xbd_test_resswin_bda.json"
    txt_path = scores_dir / "scores_xbd_test_resswin_bda.txt"
    csv_path = scores_dir / "scores_xbd_test_resswin_bda.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        f"Experiment: ResSwin-BDA {'+'.join(args.train_split)} -> {args.val_split} -> {args.test_split}",
        f"Checkpoint: {checkpoint_path}",
        f"Best epoch selected on hold: {best_epoch}",
        f"Localization threshold: {loc_threshold:.2f}",
        f"Test Localization F1: {results['localization_f1']:.6f}",
        f"No Damage F1:    {results['damage_f1_no_damage']:.6f}",
        f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}",
        f"Major Damage F1: {results['damage_f1_major_damage']:.6f}",
        f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}",
        f"Damage F1:       {results['damage_f1']:.6f}",
        f"Overall Score:   {results['score']:.6f}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in [
            "score", "localization_f1", "damage_f1", "damage_f1_no_damage",
            "damage_f1_minor_damage", "damage_f1_major_damage", "damage_f1_destroyed",
        ]:
            writer.writerow([key, results[key]])

    print("\n".join(lines), flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)
    print(f"Wrote: {csv_path}", flush=True)
    if pred_dir is not None:
        print(f"Saved prediction masks under: {pred_dir}", flush=True)


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train/test ResSwin-BDA on existing xBD splits")
    p.add_argument("--phase", type=str, default="train_test", choices=["train", "test", "train_test"])
    p.add_argument("--xbd-root", type=str, required=True)
    p.add_argument("--train-split", type=str, nargs="+", default=["train", "tier3"])
    p.add_argument("--val-split", type=str, default="hold")
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="", help="Checkpoint for --phase test. Defaults to output-dir/checkpoints/best.pt")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--crop-size", type=int, default=608, help="Training crop size. Use 0 for full-image training.")
    p.add_argument("--crop-candidate-count", type=int, default=8)
    p.add_argument("--crop-weight-no-damage", type=float, default=1.0)
    p.add_argument("--crop-weight-minor", type=float, default=12.0)
    p.add_argument("--crop-weight-major", type=float, default=12.0)
    p.add_argument("--crop-weight-destroyed", type=float, default=4.0)
    p.add_argument("--photometric-aug", action="store_true")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--swin-depth", type=int, default=2)
    p.add_argument("--pretrained-resnet", action="store_true", help="Use ImageNet pretrained ResNet34 if weights are available in your environment/cache.")

    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--loc-loss-weight", type=float, default=1.0)
    p.add_argument("--damage-loss-weight", type=float, default=1.0)
    p.add_argument("--minor-damage-boost", type=float, default=1.5)
    p.add_argument("--major-damage-boost", type=float, default=1.5)
    p.add_argument("--max-damage-class-weight", type=float, default=10.0)
    p.add_argument("--loc-threshold", type=float, default=0.50)

    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--early-stopping-patience", type=int, default=999)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--save-test-preds", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "scores").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(args.output_dir) / "checkpoints" / "best.pt"

    print("===== ResSwin-BDA xBD TRAIN/TEST =====", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"xBD root: {args.xbd_root}", flush=True)
    print(f"Train split(s): {args.train_split}", flush=True)
    print(f"Val split: {args.val_split}", flush=True)
    print(f"Test split: {args.test_split}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Grad accumulation: {args.grad_accum_steps}", flush=True)
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}", flush=True)
    print(f"Image size: {args.img_size}", flush=True)
    print(f"Training crop size: {args.crop_size}", flush=True)
    print(f"Architecture: shared Siamese ResNet34 + abs-diff fusion + Swin-style context + FPN decoder + loc/damage heads", flush=True)
    print("======================================", flush=True)

    if args.phase in {"train", "train_test"}:
        checkpoint_path = train(args, device)

    if args.phase in {"test", "train_test"}:
        test(args, device, checkpoint_path)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
