#!/usr/bin/env python3
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Dataset
# -----------------------------
@dataclass(frozen=True)
class XBDSample:
    stem: str
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


class XBDHRTBDADataset(Dataset):
    """
    Expected structure:

    /homes/j244s673/documents/wsu/phd/xview2/
      tier3/
        images/
        targets/
      hold/
        images/
        targets/
      test/
        images/
        targets/

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

    def __init__(self, root: str | Path, split: str, image_size: int, training: bool):
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.training = bool(training)

        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        self.targets_dir = self.split_root / "targets"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images dir not found: {self.images_dir}")
        if not self.targets_dir.exists():
            raise FileNotFoundError(f"Expected targets dir not found: {self.targets_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.split_root}")

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
        post_images: List[Path] = []

        for pattern in [
            "*_post_disaster.png",
            "*_post_disaster.jpg",
            "*_post_disaster.jpeg",
            "*_post_disaster.tif",
            "*_post_disaster.tiff",
            "*_post_disaster.bmp",
        ]:
            post_images.extend(self.images_dir.glob(pattern))

        post_images = sorted(post_images)
        samples: List[XBDSample] = []

        for post_path in post_images:
            prefix = post_path.stem.replace("_post_disaster", "")
            ext = post_path.suffix

            pre_path = self.images_dir / f"{prefix}_pre_disaster{ext}"
            pre_tgt = self.targets_dir / f"{prefix}_pre_disaster_target.png"
            post_tgt = self.targets_dir / f"{prefix}_post_disaster_target.png"

            if pre_path.exists() and pre_tgt.exists() and post_tgt.exists():
                samples.append(
                    XBDSample(
                        stem=prefix,
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
        )

        loc = (loc_raw > 0).astype(np.float32)

        return {
            "pre": torch.from_numpy(self._normalize(pre)).float(),
            "post": torch.from_numpy(self._normalize(post)).float(),
            "loc": torch.from_numpy(loc).float(),
            "target5": torch.from_numpy(target5).long(),
            "stem": s.stem,
            "split": self.split,
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
    # x: B, H, W, C
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    # windows: B*nW, ws*ws, C
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
        # x: B,C,H,W
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
    """
    HRTBDA-inspired high-resolution backbone:
      - HRNet-style multi-resolution features
      - DCSwin blocks at each scale
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 48, window_size: int = 8):
        super().__init__()

        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4

        self.channels = [c0, c1, c2]

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c0 // 2, kernel_size=3, stride=2),
            ConvBNAct(c0 // 2, c0, kernel_size=3, stride=2),
        )

        self.stage1 = nn.Sequential(
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=False),
            DCSwinBlock(c0, num_heads=4, window_size=window_size, shift=True),
        )

        self.down01 = ConvBNAct(c0, c1, kernel_size=3, stride=2)

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

        self.fuse1_to_0 = nn.Conv2d(c1, c0, kernel_size=1)
        self.fuse2_to_0 = nn.Conv2d(c2, c0, kernel_size=1)
        self.fuse0_to_1_b = ConvBNAct(c0, c1, kernel_size=3, stride=2)
        self.fuse2_to_1 = nn.Conv2d(c2, c1, kernel_size=1)
        self.fuse0_to_2 = nn.Sequential(
            ConvBNAct(c0, c1, kernel_size=3, stride=2),
            ConvBNAct(c1, c2, kernel_size=3, stride=2),
        )
        self.fuse1_to_2 = ConvBNAct(c1, c2, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)
        x0 = self.stage1(x0)

        x1 = self.down01(x0)

        a0 = self.stage2_b0(x0)
        a1 = self.stage2_b1(x1)

        x0 = a0 + F.interpolate(self.fuse10_to_0(a1), size=a0.shape[-2:], mode="bilinear", align_corners=False)
        x1 = a1 + self.fuse0_to_1(a0)

        x2 = self.down12(x1)

        b0 = self.stage3_b0(x0)
        b1 = self.stage3_b1(x1)
        b2 = self.stage3_b2(x2)

        y0 = (
            b0
            + F.interpolate(self.fuse1_to_0(b1), size=b0.shape[-2:], mode="bilinear", align_corners=False)
            + F.interpolate(self.fuse2_to_0(b2), size=b0.shape[-2:], mode="bilinear", align_corners=False)
        )

        y1 = (
            b1
            + self.fuse0_to_1_b(b0)
            + F.interpolate(self.fuse2_to_1(b2), size=b1.shape[-2:], mode="bilinear", align_corners=False)
        )

        y2 = b2 + self.fuse0_to_2(b0) + self.fuse1_to_2(b1)

        return [y0, y1, y2]


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

        # Cross focus: emphasize differences and shared damage-relevant information.
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
    def __init__(self, base_channels: int, decoder_channels: int, window_size: int, num_classes: int = 5):
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
            class_weights = torch.ones(5, dtype=torch.float32)

        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = target != self.ignore_index
        if valid.any():
            ce_valid = ce[valid]
            pt = torch.exp(-ce_valid)
            focal = ((1.0 - pt) ** self.gamma * ce_valid).mean()
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

        # Weighted dice, normalized.
        w = self.class_weights / self.class_weights.sum().clamp_min(1e-7)
        dice = (dice_per_class * w).sum()

        return focal + dice, focal, dice


def make_loc_pos_weight(dataset: XBDHRTBDADataset) -> torch.Tensor:
    pos, neg = dataset.localization_counts()
    return torch.tensor([max(1.0, neg / max(pos, 1))], dtype=torch.float32)


def make_class_weights(dataset: XBDHRTBDADataset) -> torch.Tensor:
    counts = dataset.class5_counts().astype(np.float64)

    # sqrt inverse frequency avoids absurdly huge weights for rare classes.
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


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
def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, XBDHRTBDADataset]:
    train_ds = XBDHRTBDADataset(args.xbd_root, args.train_split, args.img_size, training=True)
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


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.0, 1.0 - (epoch / max(1, epochs))),
    )


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

    scheduler = make_scheduler(optimizer, args.phase1_epochs)
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

    for epoch in range(1, args.phase1_epochs + 1):
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
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(pre)
                    loss, focal, dice = criterion(logits, loc)

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
    print("\n================ PHASE II: DAMAGE CLASSIFICATION ================", flush=True)

    train_loader, val_loader, _, train_ds = make_loaders(args)

    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples:   {len(val_loader.dataset)}", flush=True)

    model = HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=5,
    ).to(device)

    if phase1_ckpt is not None and phase1_ckpt.exists():
        load_phase1_backbone_into_phase2(model, phase1_ckpt, device)
    else:
        print("WARNING: Phase I checkpoint not found. Phase II will train from scratch.", flush=True)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    class_weights = make_class_weights(train_ds).to(device)
    print(f"Phase II class weights [bg,no,minor,major,destroyed]: {class_weights.detach().cpu().numpy().tolist()}", flush=True)

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

    scheduler = make_scheduler(optimizer, args.phase2_epochs)
    scaler = make_scaler(args, device)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    scores_dir = output_dir / "scores"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

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
            target = batch["target5"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    logits = model(pre, post)
                    loss, focal, dice = criterion(logits, target)
                    loss = args.cls_loss_weight * loss
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(pre, post)
                    loss, focal, dice = criterion(logits, target)
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

        val_results = evaluate_phase2(model, val_loader, device)
        val_score = float(val_results["score"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_focal": focal_meter.avg,
            "train_dice": dice_meter.avg,
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
            f"Phase II Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"hold_score={row['hold_score']:.6f} | "
            f"hold_loc_f1={row['hold_localization_f1']:.6f} | "
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
            )

            print(f"Saved Phase II best checkpoint | epoch={epoch} | score={best_score:.6f}", flush=True)
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

        with open(output_dir / "history_phase2.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.early_stopping_patience:
            print(f"Phase II early stopping at epoch {epoch}.", flush=True)
            break

    print(f"Phase II done. Best epoch={best_epoch}, best hold score={best_score:.6f}", flush=True)

    return checkpoints_dir / "phase2_best.pt"


def test_phase2(args: argparse.Namespace, device: torch.device, checkpoint_path: Path) -> None:
    print("\n================ TESTING PHASE II BEST MODEL ================", flush=True)

    _, _, test_loader, _ = make_loaders(args)

    model = HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=5,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    ckpt = load_model_weights(model, checkpoint_path, device)
    best_epoch = int(ckpt.get("epoch", -1))

    results = evaluate_phase2(model, test_loader, device)

    output_dir = Path(args.output_dir)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    json_path = scores_dir / "scores_xbd_test_hrtbda.json"
    txt_path = scores_dir / "scores_xbd_test_hrtbda.txt"
    summary_path = scores_dir / "summary_hrtbda.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "Experiment: HRTBDA-inspired xBD tier3 -> hold -> test",
        f"Best Phase II epoch selected on hold: {best_epoch}",
        f"Localization F1: {results['localization_f1']:.6f}",
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

    parser.add_argument("--phase", type=str, default="both", choices=["both", "phase1", "phase2", "test"])

    parser.add_argument("--xbd-root", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="tier3")
    parser.add_argument("--val-split", type=str, default="hold")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--phase1-epochs", type=int, default=80)
    parser.add_argument("--phase2-epochs", type=int, default=40)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)

    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--loc-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)

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

    print("===== HRTBDA-INSPIRED XBD TRAINING =====", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"xBD root: {args.xbd_root}", flush=True)
    print(f"Train split: {args.train_split}", flush=True)
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
    print("Architecture: HRNet-style high-resolution branches + DCSwin blocks + CSF fusion", flush=True)
    print("Domain adaptation: none", flush=True)
    print("=======================================", flush=True)

    checkpoints_dir = output_dir / "checkpoints"

    if args.phase == "phase1":
        train_phase1(args, device)

    elif args.phase == "phase2":
        phase1_ckpt = checkpoints_dir / "phase1_best.pt"
        train_phase2(args, device, phase1_ckpt)

    elif args.phase == "test":
        phase2_ckpt = checkpoints_dir / "phase2_best.pt"
        test_phase2(args, device, phase2_ckpt)

    elif args.phase == "both":
        phase1_ckpt = train_phase1(args, device)
        phase2_ckpt = train_phase2(args, device, phase1_ckpt)
        test_phase2(args, device, phase2_ckpt)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()