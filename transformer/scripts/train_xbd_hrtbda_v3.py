#!/usr/bin/env python3
"""
HRTBDA v2 — key improvements over v2:

  1. BinaryFocalDiceLoss: pt now computed from UNWEIGHTED bce so focal
     modulation is correct (the v1 bug caused pt~0 for all positives,
     negating focal down-weighting of easy negatives).

  2. make_loc_pos_weight: capped at 8.0  (was ~31.9, which stacked badly
     with focal and stalled dice convergence).

  3. make_class_weights: inverse-frequency instead of inverse-sqrt-frequency,
     plus a 3x manual boost for minor damage — the single most impactful fix
     for minor-damage F1 (0.245 → expected ~0.50+).

  4. LR scheduler: warmup (default 10 ep) + cosine decay for BOTH phases.
     Phase II best-at-epoch-9 was a symptom of LR decaying to ~70% already
     by epoch 9 with the old linear schedule.

  5. Phase II epochs: default raised to 80 (was 30 — too short given best at 9).

  6. Richer augmentation: scale-jitter crop + brightness + Gaussian blur
     applied consistently to both pre and post images.

  7. CutMix for Phase II: patches from rare-damage tiles are mixed in,
     matching the paper's explicit mention of CutMix boosting minor damage.

  8. Auxiliary localization head in Phase II: an extra 1-ch output predicts
     building vs background. This gives the shared backbone a direct
     localization gradient during Phase II, closing the loc-F1 gap.

  9. MulticlassFocalDiceLoss: pt is now computed from UNWEIGHTED CE, then
     class weights are applied to the loss value. This avoids the same focal
     modulation bug that was already fixed for binary localization.

 10. CutMix now patches the auxiliary localization mask too, and the double
     probability gate was removed. With cutmix_prob=0.3 and batch_size=1,
     the old code applied CutMix only about 9% of the time; this version
     applies it at the requested per-sample probability.
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


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
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
    out_imgs, out_masks = [], []
    for img in image_list:
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(img)
    for mask in mask_list:
        if mask.shape[:2] != (image_size, image_size):
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        out_masks.append(mask)
    return out_imgs, out_masks


# FIX 6 — richer augmentations (scale-jitter, brightness, blur)
def apply_shared_augmentations(
    image_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    training: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not training:
        return image_list, mask_list

    # Horizontal flip
    if np.random.rand() < 0.5:
        image_list = [np.flip(x, axis=1).copy() for x in image_list]
        mask_list  = [np.flip(x, axis=1).copy() for x in mask_list]

    # Vertical flip
    if np.random.rand() < 0.5:
        image_list = [np.flip(x, axis=0).copy() for x in image_list]
        mask_list  = [np.flip(x, axis=0).copy() for x in mask_list]

    # 90° rotation
    k = np.random.randint(0, 4)
    if k:
        image_list = [np.rot90(x, k=k).copy() for x in image_list]
        mask_list  = [np.rot90(x, k=k).copy() for x in mask_list]

    # Scale-jitter crop: crop 75–100% then resize back to original size
    if np.random.rand() < 0.5:
        h, w = image_list[0].shape[:2]
        scale   = np.random.uniform(0.75, 1.0)
        ch, cw  = int(h * scale), int(w * scale)
        top     = np.random.randint(0, h - ch + 1)
        left    = np.random.randint(0, w - cw + 1)
        image_list = [
            cv2.resize(x[top:top+ch, left:left+cw], (w, h), interpolation=cv2.INTER_LINEAR)
            for x in image_list
        ]
        mask_list = [
            cv2.resize(x[top:top+ch, left:left+cw], (w, h), interpolation=cv2.INTER_NEAREST)
            for x in mask_list
        ]

    # Brightness jitter — same factor for both pre/post so relative change is preserved
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.75, 1.25)
        image_list = [
            np.clip(x.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            for x in image_list
        ]

    # Gaussian blur (mild)
    if np.random.rand() < 0.3:
        ksize = int(np.random.choice([3, 5]))
        image_list = [cv2.GaussianBlur(x, (ksize, ksize), 0) for x in image_list]

    return image_list, mask_list


class XBDHRTBDADataset(Dataset):
    """
    xBD dataset loader.

    Phase I  target: 0 background, 1 building
    Phase II target: 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed, 255 ignore
    """

    def __init__(
        self,
        root: str | Path,
        split: str | List[str] | Tuple[str, ...],
        image_size: int,
        training: bool,
    ):
        self.root       = Path(root)
        self.splits     = [str(s) for s in split] if isinstance(split, (list, tuple)) else [str(split)]
        self.split      = "+".join(self.splits)
        self.image_size = int(image_size)
        self.training   = bool(training)

        for sn in self.splits:
            sr = self.root / sn
            if not (sr / "images").exists():
                raise FileNotFoundError(f"images dir not found: {sr / 'images'}")
            if not (sr / "targets").exists():
                raise FileNotFoundError(f"targets dir not found: {sr / 'targets'}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.root} for splits {self.splits}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

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
        target  = np.zeros(loc.shape, dtype=np.uint8)
        for cls in [1, 2, 3, 4]:
            target[(dmg == cls) & loc_bin] = cls
        other = loc_bin & ~np.isin(dmg, [1, 2, 3, 4])
        target[other] = 255
        return target

    def _collect_samples(self) -> List[XBDSample]:
        samples, seen = [], set()
        for sn in self.splits:
            images_dir  = self.root / sn / "images"
            targets_dir = self.root / sn / "targets"
            post_images: List[Path] = []
            for pat in ["*_post_disaster.png", "*_post_disaster.jpg", "*_post_disaster.jpeg",
                        "*_post_disaster.tif", "*_post_disaster.tiff", "*_post_disaster.bmp"]:
                post_images.extend(images_dir.glob(pat))
            for pp in sorted(post_images):
                prefix  = pp.stem.replace("_post_disaster", "")
                pre_p   = images_dir  / f"{prefix}_pre_disaster{pp.suffix}"
                pre_tgt = targets_dir / f"{prefix}_pre_disaster_target.png"
                pst_tgt = targets_dir / f"{prefix}_post_disaster_target.png"
                if prefix in seen:
                    continue
                if pre_p.exists() and pre_tgt.exists() and pst_tgt.exists():
                    seen.add(prefix)
                    samples.append(XBDSample(
                        stem=prefix, split=sn,
                        pre_image_path=pre_p, post_image_path=pp,
                        pre_target_path=pre_tgt, post_target_path=pst_tgt,
                    ))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return (x - self.mean) / self.std

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        s = self.samples[index]

        pre     = self._read_rgb(s.pre_image_path)
        post    = self._read_rgb(s.post_image_path)
        loc_raw = self._read_mask(s.pre_target_path)
        dmg_raw = self._read_mask(s.post_target_path)
        target5 = self._target5_from_masks(loc_raw, dmg_raw)

        [pre, post], [loc_raw, target5] = resize_rgb_and_masks(
            [pre, post], [loc_raw, target5], self.image_size)
        [pre, post], [loc_raw, target5] = apply_shared_augmentations(
            [pre, post], [loc_raw, target5], self.training)

        loc = (loc_raw > 0).astype(np.float32)
        return {
            "pre":     torch.from_numpy(self._normalize(pre)).float(),
            "post":    torch.from_numpy(self._normalize(post)).float(),
            "loc":     torch.from_numpy(loc).float(),
            "target5": torch.from_numpy(target5).long(),
            "stem":    s.stem,
            "split":   s.split,
        }

    def localization_counts(self) -> Tuple[int, int]:
        pos = neg = 0
        for s in self.samples:
            loc  = self._read_mask(s.pre_target_path) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())
        return pos, neg

    def class5_counts(self) -> np.ndarray:
        counts = np.zeros(5, dtype=np.int64)
        for s in self.samples:
            tgt   = self._target5_from_masks(
                self._read_mask(s.pre_target_path),
                self._read_mask(s.post_target_path))
            valid = tgt != 255
            vals, freqs = np.unique(tgt[valid], return_counts=True)
            for v, f in zip(vals.tolist(), freqs.tolist()):
                counts[int(v)] += int(f)
        counts[counts == 0] = 1
        return counts

    # ── helpers used by CutMix ─────────────────────────────────────────────
    def get_rare_damage_indices(self) -> List[int]:
        """Return indices of samples that contain minor or major damage pixels."""
        rare = []
        for i, s in enumerate(self.samples):
            dmg = self._read_mask(s.post_target_path)
            loc = self._read_mask(s.pre_target_path)
            tgt = self._target5_from_masks(loc, dmg)
            if np.any((tgt == 2) | (tgt == 3)):      # minor or major
                rare.append(i)
        return rare


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.count = 0
    @property
    def avg(self): return self.sum / max(1, self.count)
    def update(self, v, n=1): self.sum += float(v)*n; self.count += int(n)


class F1Recorder:
    def __init__(self, tp, fp, fn): self.tp=int(tp); self.fp=int(fp); self.fn=int(fn)
    @property
    def precision(self):
        d = self.tp + self.fp; return 0.0 if d==0 else self.tp/d
    @property
    def recall(self):
        d = self.tp + self.fn; return 0.0 if d==0 else self.tp/d
    @property
    def f1(self):
        p,r = self.precision, self.recall
        return 0.0 if p==0 or r==0 else 2*p*r/(p+r)
    def as_dict(self):
        return {"tp":self.tp,"fp":self.fp,"fn":self.fn,
                "precision":self.precision,"recall":self.recall,"f1":self.f1}


def harmonic_mean(values: List[float]) -> float:
    return len(values) / sum((float(x) + 1e-6)**-1 for x in values)


# ─────────────────────────────────────────────
# Model blocks
# ─────────────────────────────────────────────
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x): return self.block(x)


class LayerNorm2d(nn.Module):
    def __init__(self, ch): super().__init__(); self.norm = nn.LayerNorm(ch)
    def forward(self, x):
        x = x.permute(0,2,3,1); x = self.norm(x)
        return x.permute(0,3,1,2).contiguous()


def window_partition(x, ws):
    b,h,w,c = x.shape
    x = x.view(b, h//ws, ws, w//ws, ws, c)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, ws*ws, c)


def window_reverse(windows, ws, h, w, b):
    c = windows.shape[-1]
    x = windows.view(b, h//ws, w//ws, ws, ws, c)
    return x.permute(0,1,3,2,4,5).contiguous().view(b,h,w,c)


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        b,c,h,w = x.shape; ws = self.window_size
        ph = (ws - h%ws)%ws; pw = (ws - w%ws)%ws
        if ph or pw: x = F.pad(x, (0,pw,0,ph))
        hp,wp = x.shape[-2:]
        x_hw = x.permute(0,2,3,1).contiguous()
        wins = window_partition(x_hw, ws)
        out,_ = self.attn(wins, wins, wins, need_weights=False)
        x_hw = window_reverse(out, ws, hp, wp, b)
        x = x_hw.permute(0,3,1,2).contiguous()
        if ph or pw: x = x[:,:,:h,:w]
        return x


class DCMLP(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hid = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hid, channels, 1, bias=False), nn.BatchNorm2d(channels),
            nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class DCSwinBlock(nn.Module):
    def __init__(self, channels, num_heads, window_size=8, shift=False, dropout=0.0):
        super().__init__()
        self.shift_size = window_size//2 if shift else 0
        self.norm1 = LayerNorm2d(channels)
        self.attn  = WindowSelfAttention(channels, num_heads=num_heads, window_size=window_size)
        self.norm2 = LayerNorm2d(channels)
        self.mlp   = DCMLP(channels, mlp_ratio=4.0, dropout=dropout)

    def forward(self, x):
        sc = x; y = self.norm1(x)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size,-self.shift_size), dims=(2,3))
        y = self.attn(y)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(2,3))
        x = sc + y
        return x + self.mlp(self.norm2(x))


class HighResolutionTransformerBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=48, window_size=8):
        super().__init__()
        c0, c1, c2, c3 = (base_channels * m for m in [1, 2, 4, 8])
        self.channels = [c0, c1, c2, c3]

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c0//2, 3, stride=2),
            ConvBNAct(c0//2, c0, 3, stride=2))

        # Stage 1
        self.stage1 = nn.Sequential(
            DCSwinBlock(c0, 4, window_size, shift=False),
            DCSwinBlock(c0, 4, window_size, shift=True))
        self.down01 = ConvBNAct(c0, c1, 3, stride=2)

        # Stage 2
        self.s2_b0 = nn.Sequential(DCSwinBlock(c0,4,window_size,False), DCSwinBlock(c0,4,window_size,True))
        self.s2_b1 = nn.Sequential(DCSwinBlock(c1,4,window_size,False), DCSwinBlock(c1,4,window_size,True))
        self.s2_f10_0 = nn.Conv2d(c1, c0, 1)
        self.s2_f0_1  = ConvBNAct(c0, c1, 3, stride=2)
        self.down12   = ConvBNAct(c1, c2, 3, stride=2)

        # Stage 3
        self.s3_b0 = nn.Sequential(DCSwinBlock(c0,4,window_size,False), DCSwinBlock(c0,4,window_size,True))
        self.s3_b1 = nn.Sequential(DCSwinBlock(c1,4,window_size,False), DCSwinBlock(c1,4,window_size,True))
        self.s3_b2 = nn.Sequential(DCSwinBlock(c2,8,window_size,False), DCSwinBlock(c2,8,window_size,True))
        self.s3_f1_0 = nn.Conv2d(c1, c0, 1)
        self.s3_f2_0 = nn.Conv2d(c2, c0, 1)
        self.s3_f0_1 = ConvBNAct(c0, c1, 3, stride=2)
        self.s3_f2_1 = nn.Conv2d(c2, c1, 1)
        self.s3_f0_2 = nn.Sequential(ConvBNAct(c0,c1,3,stride=2), ConvBNAct(c1,c2,3,stride=2))
        self.s3_f1_2 = ConvBNAct(c1, c2, 3, stride=2)
        self.down23  = ConvBNAct(c2, c3, 3, stride=2)

        # Stage 4  (4 branches)
        self.s4_b0 = nn.Sequential(DCSwinBlock(c0,4,window_size,False), DCSwinBlock(c0,4,window_size,True))
        self.s4_b1 = nn.Sequential(DCSwinBlock(c1,4,window_size,False), DCSwinBlock(c1,4,window_size,True))
        self.s4_b2 = nn.Sequential(DCSwinBlock(c2,8,window_size,False), DCSwinBlock(c2,8,window_size,True))
        self.s4_b3 = nn.Sequential(DCSwinBlock(c3,8,window_size,False), DCSwinBlock(c3,8,window_size,True))

        # Stage 4 fusions  (all-to-all across 4 branches)
        self.s4_f1_0 = nn.Conv2d(c1, c0, 1)
        self.s4_f2_0 = nn.Conv2d(c2, c0, 1)
        self.s4_f3_0 = nn.Conv2d(c3, c0, 1)

        self.s4_f0_1 = ConvBNAct(c0, c1, 3, stride=2)
        self.s4_f2_1 = nn.Conv2d(c2, c1, 1)
        self.s4_f3_1 = nn.Conv2d(c3, c1, 1)

        self.s4_f0_2 = nn.Sequential(ConvBNAct(c0,c1,3,stride=2), ConvBNAct(c1,c2,3,stride=2))
        self.s4_f1_2 = ConvBNAct(c1, c2, 3, stride=2)
        self.s4_f3_2 = nn.Conv2d(c3, c2, 1)

        self.s4_f0_3 = nn.Sequential(
            ConvBNAct(c0,c1,3,stride=2), ConvBNAct(c1,c2,3,stride=2), ConvBNAct(c2,c3,3,stride=2))
        self.s4_f1_3 = nn.Sequential(ConvBNAct(c1,c2,3,stride=2), ConvBNAct(c2,c3,3,stride=2))
        self.s4_f2_3 = ConvBNAct(c2, c3, 3, stride=2)

    def _up(self, x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stage1(self.stem(x))

        # Stage 2
        x1  = self.down01(x0)
        a0  = self.s2_b0(x0); a1 = self.s2_b1(x1)
        x0  = a0 + self._up(self.s2_f10_0(a1), a0.shape[-2:])
        x1  = a1 + self.s2_f0_1(a0)

        # Stage 3
        x2  = self.down12(x1)
        b0, b1, b2 = self.s3_b0(x0), self.s3_b1(x1), self.s3_b2(x2)
        x0  = b0 + self._up(self.s3_f1_0(b1), b0.shape[-2:]) + self._up(self.s3_f2_0(b2), b0.shape[-2:])
        x1  = b1 + self.s3_f0_1(b0) + self._up(self.s3_f2_1(b2), b1.shape[-2:])
        x2  = b2 + self.s3_f0_2(b0) + self.s3_f1_2(b1)

        # Stage 4
        x3  = self.down23(x2)
        c0, c1, c2, c3 = self.s4_b0(x0), self.s4_b1(x1), self.s4_b2(x2), self.s4_b3(x3)
        y0  = c0 + self._up(self.s4_f1_0(c1), c0.shape[-2:]) + self._up(self.s4_f2_0(c2), c0.shape[-2:]) + self._up(self.s4_f3_0(c3), c0.shape[-2:])
        y1  = c1 + self.s4_f0_1(c0) + self._up(self.s4_f2_1(c2), c1.shape[-2:]) + self._up(self.s4_f3_1(c3), c1.shape[-2:])
        y2  = c2 + self.s4_f0_2(c0) + self.s4_f1_2(c1) + self._up(self.s4_f3_2(c3), c2.shape[-2:])
        y3  = c3 + self.s4_f0_3(c0) + self.s4_f1_3(c1) + self.s4_f2_3(c2)
        return [y0, y1, y2, y3]


class MultiScaleDecoder(nn.Module):
    def __init__(self, in_channels: List[int], decoder_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, decoder_channels, 1, bias=False),
                          nn.BatchNorm2d(decoder_channels), nn.GELU())
            for c in in_channels])
        self.fuse = nn.Sequential(
            ConvBNAct(decoder_channels * len(in_channels), decoder_channels, 3),
            ConvBNAct(decoder_channels, decoder_channels, 3))
        self.out = nn.Conv2d(decoder_channels, out_channels, 1)

    def forward(self, features, output_size):
        tgt = features[0].shape[-2:]
        xs  = []
        for feat, proj in zip(features, self.proj):
            y = proj(feat)
            if y.shape[-2:] != tgt:
                y = F.interpolate(y, size=tgt, mode="bilinear", align_corners=False)
            xs.append(y)
        x = self.fuse(torch.cat(xs, dim=1))
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.out(x)


class CSFModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hid = max(8, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, 1, bias=False))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.GELU())

    def _attention(self, x):
        avg = F.adaptive_avg_pool2d(x, 1); mx = F.adaptive_max_pool2d(x, 1)
        ch  = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))
        x   = x * ch
        sp  = self.spatial_conv(torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], dim=1))
        return x * sp

    def forward(self, fpre, fpost):
        apre, apost = self._attention(fpre), self._attention(fpost)
        diff        = torch.abs(apost - apre)
        return self.fuse(torch.cat([apre + diff, apost + diff], dim=1))


class HRTBDAPhase1(nn.Module):
    def __init__(self, base_channels, decoder_channels, window_size):
        super().__init__()
        self.backbone = HighResolutionTransformerBackbone(3, base_channels, window_size)
        self.decoder  = MultiScaleDecoder(self.backbone.channels, decoder_channels, 1)

    def forward(self, pre):
        return self.decoder(self.backbone(pre), output_size=pre.shape[-2:]).squeeze(1)


class HRTBDAPhase2(nn.Module):
    # FIX 8 — auxiliary localization head keeps backbone sharp during Phase II
    def __init__(self, base_channels, decoder_channels, window_size, num_classes=5):
        super().__init__()
        self.backbone  = HighResolutionTransformerBackbone(3, base_channels, window_size)
        self.csf       = nn.ModuleList([CSFModule(c) for c in self.backbone.channels])
        self.decoder   = MultiScaleDecoder(self.backbone.channels, decoder_channels, num_classes)
        # Auxiliary head: predict building vs background from the shared backbone (pre branch)
        self.aux_head  = nn.Sequential(
            ConvBNAct(self.backbone.channels[0], 64, 3),
            nn.Conv2d(64, 1, 1))

    def forward(self, pre, post):
        fpre  = self.backbone(pre)
        fpost = self.backbone(post)
        fused = [m(a, b) for m, a, b in zip(self.csf, fpre, fpost)]
        logits_cls = self.decoder(fused, output_size=pre.shape[-2:])
        # Aux loc: upsample highest-res pre feature
        aux_loc = F.interpolate(self.aux_head(fpre[0]), size=pre.shape[-2:],
                                mode="bilinear", align_corners=False).squeeze(1)
        return logits_cls, aux_loc


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────
class BinaryFocalDiceLoss(nn.Module):
    """
    FIX 1 — pt is now computed from UNWEIGHTED BCE.

    v1 bug: passing pos_weight into BCE then computing pt=exp(-bce_weighted)
    made pt≈0 for ALL positive pixels (because bce was inflated 32×), so the
    focal factor (1-pt)^gamma ≈ 1 everywhere — focal modulation was disabled.
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(1))
        self.gamma = float(gamma)

    def forward(self, logits, target):
        # Unweighted BCE for focal modulation
        bce_plain = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt            = torch.exp(-bce_plain)
        focal_weight  = (1.0 - pt) ** self.gamma

        # Class weighting applied to the loss value, not to pt computation
        pw_map = torch.where(target > 0.5, self.pos_weight, torch.ones_like(target))
        focal  = (focal_weight * bce_plain * pw_map).mean()

        prob  = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice  = 1.0 - ((2.0 * inter + 1e-7) / (denom + 1e-7)).mean()

        return focal + dice, focal, dice


class MulticlassFocalDiceLoss(nn.Module):
    """
    Multiclass focal + Dice loss.

    Important: focal pt is computed from UNWEIGHTED CE. Class weights are
    applied after the focal factor is computed. This prevents boosted rare
    classes, especially minor damage, from making pt artificially tiny and
    disabling focal down-weighting.
    """
    def __init__(self, class_weights=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma        = float(gamma)
        self.ignore_index = int(ignore_index)
        if class_weights is None:
            class_weights = torch.ones(5, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits, target):
        valid = target != self.ignore_index

        # Unweighted CE for pt computation.
        ce_plain = F.cross_entropy(
            logits, target,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        if valid.any():
            ce_v = ce_plain[valid]
            pt   = torch.exp(-ce_v)

            target_v = target[valid]
            class_w  = self.class_weights[target_v]

            focal = (((1.0 - pt) ** self.gamma) * ce_v * class_w).mean()
        else:
            focal = logits.sum() * 0.0

        probs      = torch.softmax(logits, dim=1)
        tgt_safe   = target.clone(); tgt_safe[tgt_safe == self.ignore_index] = 0
        one_hot    = F.one_hot(tgt_safe, num_classes=logits.shape[1]).permute(0,3,1,2).float()
        vm         = valid.unsqueeze(1).float()
        probs      = probs * vm; one_hot = one_hot * vm
        dims       = (0, 2, 3)
        inter      = (probs * one_hot).sum(dim=dims)
        denom      = (probs**2).sum(dim=dims) + (one_hot**2).sum(dim=dims)
        dice_cls   = 1.0 - (2.0 * inter + 1e-7) / (denom + 1e-7)
        w          = self.class_weights / self.class_weights.sum().clamp_min(1e-7)
        dice       = (dice_cls * w).sum()
        return focal + dice, focal, dice


# FIX 2 — cap pos_weight at 8  (was ~31.9; stacked badly with focal)
def make_loc_pos_weight(dataset: XBDHRTBDADataset) -> torch.Tensor:
    pos, neg = dataset.localization_counts()
    raw      = max(1.0, neg / max(pos, 1))
    capped   = min(raw, 8.0)
    print(f"  loc pos_weight raw={raw:.2f}  capped={capped:.2f}", flush=True)
    return torch.tensor([capped], dtype=torch.float32)


# FIX 3 — inverse-frequency weights + 3× manual boost for minor damage
def make_class_weights(dataset: XBDHRTBDADataset, minor_boost: float = 3.0) -> torch.Tensor:
    counts = dataset.class5_counts().astype(np.float64)
    freq   = counts / counts.sum()
    # Inverse frequency (much stronger signal than inverse-sqrt)
    weights = 1.0 / (freq + 1e-12)
    # Extra boost for minor damage (class 2) — it is ~1.7% of pixels
    weights[2] *= minor_boost
    weights  = weights / weights.mean()
    print(f"  class weights [bg,no,minor,major,dest]: {weights.tolist()}", flush=True)
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# CutMix  (FIX 7)
# ─────────────────────────────────────────────
def cutmix_batch(
    pre: torch.Tensor,
    post: torch.Tensor,
    target5: torch.Tensor,
    loc: torch.Tensor,
    rare_indices: List[int],
    dataset: XBDHRTBDADataset,
    p: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-sample CutMix for Phase II.

    With probability p for each item in the batch, paste a random crop from a
    rare-damage tile into pre, post, target5, and loc. Patching loc matters
    because Phase II also trains the auxiliary localization head.
    """
    if not rare_indices or p <= 0.0:
        return pre, post, target5, loc

    B, C, H, W = pre.shape
    pre, post, target5, loc = pre.clone(), post.clone(), target5.clone(), loc.clone()

    for i in range(B):
        if np.random.rand() > p:
            continue

        j   = random.choice(rare_indices)
        src = dataset[j]

        s_pre  = src["pre"].unsqueeze(0).to(pre.device)
        s_post = src["post"].unsqueeze(0).to(post.device)
        s_tgt  = src["target5"].unsqueeze(0).to(target5.device)
        s_loc  = src["loc"].unsqueeze(0).to(loc.device)

        # Random crop size: 20–50% of image.
        lam    = np.random.uniform(0.2, 0.5)
        ch, cw = int(H * lam), int(W * lam)

        # Source crop position.
        sy     = np.random.randint(0, H - ch + 1)
        sx     = np.random.randint(0, W - cw + 1)

        # Destination paste position.
        dy     = np.random.randint(0, H - ch + 1)
        dx     = np.random.randint(0, W - cw + 1)

        pre[i, :, dy:dy+ch, dx:dx+cw]  = s_pre[0, :, sy:sy+ch, sx:sx+cw]
        post[i, :, dy:dy+ch, dx:dx+cw] = s_post[0, :, sy:sy+ch, sx:sx+cw]
        target5[i, dy:dy+ch, dx:dx+cw] = s_tgt[0, sy:sy+ch, sx:sx+cw]
        loc[i, dy:dy+ch, dx:dx+cw]     = s_loc[0, sy:sy+ch, sx:sx+cw]

    return pre, post, target5, loc


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_phase1(model, loader, device, threshold) -> Dict:
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        pre      = batch["pre"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        pred     = (torch.sigmoid(model(pre)) > threshold).long()
        tp += int(((pred==1)&(loc_true==1)).sum()); fp += int(((pred==1)&(loc_true==0)).sum()); fn += int(((pred==0)&(loc_true==1)).sum())
    rec = F1Recorder(tp, fp, fn)
    return {"threshold": threshold, "localization_f1": rec.f1, "details": rec.as_dict()}


@torch.no_grad()
def scan_phase1_thresholds(model, loader, device, thresholds, csv_path) -> Tuple[float, Dict]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    best_th, best_res, best_f1 = thresholds[0], {}, -1.0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["threshold","localization_f1","precision","recall","tp","fp","fn"])
        for th in thresholds:
            res = evaluate_phase1(model, loader, device, th)
            d   = res["details"]
            wr.writerow([th, res["localization_f1"], d["precision"], d["recall"],
                         d["tp"], d["fp"], d["fn"]])
            if float(res["localization_f1"]) > best_f1:
                best_f1 = float(res["localization_f1"]); best_th = th; best_res = res
    return best_th, best_res


@torch.no_grad()
def evaluate_phase2(model, loader, device) -> Dict:
    model.eval()
    loc_tp = loc_fp = loc_fn = 0
    cls_counts = {c: {"tp":0,"fp":0,"fn":0} for c in [1,2,3,4]}
    for batch in loader:
        pre      = batch["pre"].to(device, non_blocking=True)
        post     = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        target   = batch["target5"].to(device, non_blocking=True).long()
        # HRTBDAPhase2 now returns (logits_cls, aux_loc)
        out      = model(pre, post)
        logits   = out[0] if isinstance(out, (tuple,list)) else out
        pred     = torch.argmax(logits, dim=1)
        lp       = (pred > 0).long()
        loc_tp  += int(((lp==1)&(loc_true==1)).sum()); loc_fp += int(((lp==1)&(loc_true==0)).sum()); loc_fn += int(((lp==0)&(loc_true==1)).sum())
        vb       = (target>=1)&(target<=4)
        pv, tv   = pred[vb], target[vb]
        for cls in [1,2,3,4]:
            cls_counts[cls]["tp"] += int(((pv==cls)&(tv==cls)).sum())
            cls_counts[cls]["fp"] += int(((pv==cls)&(tv!=cls)).sum())
            cls_counts[cls]["fn"] += int(((pv!=cls)&(tv==cls)).sum())
    loc       = F1Recorder(loc_tp, loc_fp, loc_fn)
    nd,mi,ma,de = [F1Recorder(cls_counts[c]["tp"],cls_counts[c]["fp"],cls_counts[c]["fn"]) for c in [1,2,3,4]]
    dmg_f1    = harmonic_mean([nd.f1, mi.f1, ma.f1, de.f1])
    score     = 0.3*loc.f1 + 0.7*dmg_f1
    return {"score":score,"localization_f1":loc.f1,"damage_f1":dmg_f1,
            "damage_f1_no_damage":nd.f1,"damage_f1_minor_damage":mi.f1,
            "damage_f1_major_damage":ma.f1,"damage_f1_destroyed":de.f1,
            "details":{"localization":loc.as_dict(),"no_damage":nd.as_dict(),
                       "minor_damage":mi.as_dict(),"major_damage":ma.as_dict(),
                       "destroyed":de.as_dict()}}


# ─────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric, args, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"epoch":epoch,
             "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler else None,
             "scaler":    scaler.state_dict()    if scaler    else None,
             "best_metric": best_metric, "args": vars(args)}
    if extra: state.update(extra)
    torch.save(state, path)


def load_model_weights(model, path, device):
    ckpt  = torch.load(path, map_location=device)
    state = ckpt["model"]
    (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state, strict=True)
    return ckpt


def load_phase1_backbone_into_phase2(phase2, phase1_ckpt, device):
    ckpt  = torch.load(phase1_ckpt, map_location=device)
    state = {k.replace("backbone.","",1): v for k, v in ckpt["model"].items() if k.startswith("backbone.")}
    miss, unex = phase2.backbone.load_state_dict(state, strict=False)
    print(f"Loaded Phase I backbone → Phase II | missing={len(miss)} unexpected={len(unex)}", flush=True)


# ─────────────────────────────────────────────
# LR Scheduler  (FIX 4 — warmup + cosine)
# ─────────────────────────────────────────────
def make_scheduler(optimizer, epochs: int, warmup_epochs: int = 10):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)      # linear warmup
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))   # cosine decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_loaders(args):
    train_ds = XBDHRTBDADataset(args.xbd_root, args.train_split, args.img_size, training=True)
    val_ds   = XBDHRTBDADataset(args.xbd_root, args.val_split,   args.img_size, training=False)
    test_ds  = XBDHRTBDADataset(args.xbd_root, args.test_split,  args.img_size, training=False)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **kw)
    return train_loader, val_loader, test_loader, train_ds


def make_scaler(args, device):
    enabled = bool(args.amp and device.type == "cuda")
    return GradScaler(device.type, enabled=enabled) if USE_TORCH_AMP else GradScaler(enabled=enabled)


def backward_step(loss, model, optimizer, scaler, args):
    scaler.scale(loss).backward()
    if args.max_grad_norm and args.max_grad_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scaler.step(optimizer); scaler.update()


# ─────────────────────────────────────────────
# Phase I training
# ─────────────────────────────────────────────
def train_phase1(args, device) -> Path:
    print("\n================ PHASE I: BUILDING LOCALIZATION ================", flush=True)
    train_loader, val_loader, _, train_ds = make_loaders(args)
    print(f"Train={len(train_loader.dataset)}  Val={len(val_loader.dataset)}", flush=True)

    model = HRTBDAPhase1(args.base_channels, args.decoder_channels, args.window_size).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    loc_pw   = make_loc_pos_weight(train_ds).to(device)
    criterion = BinaryFocalDiceLoss(pos_weight=loc_pw, gamma=args.focal_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = make_scheduler(optimizer, args.phase1_epochs, warmup_epochs=args.warmup_epochs)
    scaler    = make_scaler(args, device)

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    scr_dir  = out_dir / "scores";      scr_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = best_threshold = -1.0; best_epoch = 0; no_improve = 0; history = []; start_epoch = 1
    hist_path = out_dir / "history_phase1.json"

    # Important: do NOT load old history/checkpoint state unless the user explicitly
    # resumes. Reusing the same output directory after changing the code can otherwise
    # leave stale best metrics/checkpoints from an older run and cause Phase II to load
    # the wrong phase1_best.pt.
    resume_path = Path(args.resume_phase1_from) if getattr(args,"resume_phase1_from","") else None
    if not resume_path and hist_path.exists():
        print(f"Ignoring existing Phase I history because --resume-phase1-from was not set: {hist_path}", flush=True)

    if resume_path and resume_path.exists():
        if hist_path.exists():
            try:
                history = json.load(open(hist_path, encoding="utf-8"))
                if history:
                    br       = max(history, key=lambda r: float(r.get("val_localization_f1", -1)))
                    best_f1  = float(br.get("val_localization_f1", -1))
                    best_epoch = int(br.get("epoch", 0))
                    best_threshold = float(br.get("val_best_threshold", 0.5))
                    print(f"Loaded Phase I history: {len(history)} rows | best_epoch={best_epoch} | best_f1={best_f1:.6f}", flush=True)
            except Exception as e:
                print(f"WARNING: Could not load Phase I history: {e}", flush=True); history = []
        print(f"Resuming Phase I from {resume_path}", flush=True)
        ckpt  = torch.load(resume_path, map_location=device)
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt["model"], strict=True)
        if ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            try: scaler.load_state_dict(ckpt["scaler"])
            except: pass
        start_epoch    = int(ckpt.get("epoch",0)) + 1
        best_f1        = max(best_f1, float(ckpt.get("best_metric", -1)))
        best_threshold = float(ckpt.get("best_threshold", best_threshold or 0.5))
        no_improve     = max(0, start_epoch - best_epoch - 1)
        print(f"  resume: start_epoch={start_epoch}  best_f1={best_f1:.6f}", flush=True)

    if start_epoch > args.phase1_epochs:
        print(f"Phase I already done (epoch {start_epoch-1} >= {args.phase1_epochs}).", flush=True)
        return ckpt_dir / "phase1_best.pt"

    for epoch in range(start_epoch, args.phase1_epochs + 1):
        model.train()
        tm, fm, dm = AverageMeter(), AverageMeter(), AverageMeter()
        print(f"\nPhase I epoch {epoch}/{args.phase1_epochs} | LR={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        it = tqdm(train_loader, desc=f"p1-{epoch}") if tqdm and sys.stderr.isatty() else train_loader
        for step, batch in enumerate(it, 1):
            pre = batch["pre"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ctx = (autocast(device_type=device.type, enabled=args.amp and device.type=="cuda")
                   if USE_TORCH_AMP else autocast(enabled=args.amp and device.type=="cuda"))
            with ctx:
                logits = model(pre)
                loss, focal, dice = criterion(logits, loc)
                loss = args.loc_loss_weight * loss
            backward_step(loss, model, optimizer, scaler, args)
            bs = pre.size(0); tm.update(loss.item(),bs); fm.update(focal.item(),bs); dm.update(dice.item(),bs)
            if step % 20 == 0 or step == len(train_loader):
                print(f"  P1 {epoch}/{args.phase1_epochs} step {step}/{len(train_loader)} "
                      f"loss={tm.avg:.4f} focal={fm.avg:.4f} dice={dm.avg:.4f}", flush=True)
        scheduler.step()

        th, vres = scan_phase1_thresholds(model, val_loader, device, args.thresholds,
                                          scr_dir / f"phase1_epoch_{epoch:03d}_scan.csv")
        vf1 = float(vres["localization_f1"])
        history.append({"epoch":epoch,"lr":optimizer.param_groups[0]["lr"],"train_loss":tm.avg,
                         "val_best_threshold":th,"val_localization_f1":vf1})
        print(f"Phase I Epoch {epoch:03d} | train_loss={tm.avg:.4f} | val_loc_f1={vf1:.6f} | thr={th:.2f}", flush=True)

        if vf1 > best_f1:
            best_f1 = vf1; best_epoch = epoch; best_threshold = th; no_improve = 0
            save_checkpoint(ckpt_dir/"phase1_best.pt", model, optimizer, scheduler, scaler,
                            epoch, best_f1, args, {"best_threshold": best_threshold})
            print(f"  ✓ Saved best | epoch={epoch} loc_f1={best_f1:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} ep(s). Best epoch={best_epoch}", flush=True)

        save_checkpoint(ckpt_dir/"phase1_last.pt", model, optimizer, scheduler, scaler,
                        epoch, best_f1, args, {"best_threshold": best_threshold})
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(ckpt_dir/f"phase1_epoch_{epoch:03d}.pt", model, optimizer,
                            scheduler, scaler, epoch, best_f1, args, {"best_threshold": best_threshold})
        json.dump(history, open(out_dir/"history_phase1.json","w",encoding="utf-8"), indent=2)

        if no_improve >= args.early_stopping_patience:
            print(f"Phase I early stopping at epoch {epoch}.", flush=True); break

    print(f"Phase I done. Best epoch={best_epoch} F1={best_f1:.6f} thr={best_threshold:.2f}", flush=True)
    return ckpt_dir / "phase1_best.pt"


# ─────────────────────────────────────────────
# Phase II training
# ─────────────────────────────────────────────
def train_phase2(args, device, phase1_ckpt: Optional[Path]) -> Path:
    print("\n================ PHASE II: DAMAGE CLASSIFICATION ================", flush=True)
    train_loader, val_loader, _, train_ds = make_loaders(args)
    print(f"Train={len(train_loader.dataset)}  Val={len(val_loader.dataset)}", flush=True)

    model = HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size).to(device)
    if phase1_ckpt and phase1_ckpt.exists():
        load_phase1_backbone_into_phase2(model, phase1_ckpt, device)
    else:
        print("WARNING: Phase I checkpoint not found — training Phase II from scratch.", flush=True)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    cw        = make_class_weights(train_ds, minor_boost=args.minor_damage_boost).to(device)
    criterion = MulticlassFocalDiceLoss(class_weights=cw, gamma=args.focal_gamma, ignore_index=255).to(device)
    # Aux localization loss shares BinaryFocalDiceLoss with a mild pos_weight
    loc_pw_p2   = make_loc_pos_weight(train_ds).to(device)
    aux_crit    = BinaryFocalDiceLoss(pos_weight=loc_pw_p2, gamma=args.focal_gamma).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # FIX 5 — shorter warmup for Phase II (3 ep default), then cosine over remaining epochs
    p2_warmup = min(3, args.phase2_epochs // 5)
    scheduler = make_scheduler(optimizer, args.phase2_epochs, warmup_epochs=p2_warmup)
    scaler    = make_scaler(args, device)

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    scr_dir  = out_dir / "scores";      scr_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute rare-damage indices for CutMix (done once, not per batch)
    print("Computing rare-damage indices for CutMix…", flush=True)
    rare_indices = train_ds.get_rare_damage_indices()
    print(f"  Rare-damage tiles: {len(rare_indices)}/{len(train_ds)}", flush=True)

    best_score = best_epoch = 0; no_improve = 0; history = []

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        tm, fm, dm = AverageMeter(), AverageMeter(), AverageMeter()
        print(f"\nPhase II epoch {epoch}/{args.phase2_epochs} | LR={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        it = tqdm(train_loader, desc=f"p2-{epoch}") if tqdm and sys.stderr.isatty() else train_loader

        for step, batch in enumerate(it, 1):
            pre    = batch["pre"].to(device, non_blocking=True)
            post   = batch["post"].to(device, non_blocking=True)
            target = batch["target5"].to(device, non_blocking=True)
            loc    = batch["loc"].to(device, non_blocking=True)

            # FIX 7/10 — CutMix: paste rare-damage crops into images, damage labels, and aux loc labels
            pre, post, target, loc = cutmix_batch(pre, post, target, loc, rare_indices, train_ds,
                                                  p=args.cutmix_prob)

            optimizer.zero_grad(set_to_none=True)
            ctx = (autocast(device_type=device.type, enabled=args.amp and device.type=="cuda")
                   if USE_TORCH_AMP else autocast(enabled=args.amp and device.type=="cuda"))
            with ctx:
                logits_cls, aux_loc = model(pre, post)
                loss_cls, focal, dice = criterion(logits_cls, target)
                # FIX 8 — auxiliary localization loss
                loss_aux, _, _ = aux_crit(aux_loc, loc)
                loss = args.cls_loss_weight * loss_cls + args.aux_loc_weight * loss_aux

            backward_step(loss, model, optimizer, scaler, args)
            bs = pre.size(0); tm.update(loss.item(),bs); fm.update(focal.item(),bs); dm.update(dice.item(),bs)
            if step % 20 == 0 or step == len(train_loader):
                print(f"  P2 {epoch}/{args.phase2_epochs} step {step}/{len(train_loader)} "
                      f"loss={tm.avg:.4f} focal={fm.avg:.4f} dice={dm.avg:.4f}", flush=True)

        scheduler.step()
        vres  = evaluate_phase2(model, val_loader, device)
        vs    = float(vres["score"])
        row   = {"epoch":epoch,"lr":optimizer.param_groups[0]["lr"],"train_loss":tm.avg,
                 "hold_score":vs,**{k:float(vres[k]) for k in
                 ["localization_f1","damage_f1","damage_f1_no_damage",
                  "damage_f1_minor_damage","damage_f1_major_damage","damage_f1_destroyed"]}}
        history.append(row)
        print(f"Phase II Epoch {epoch:03d} | loss={tm.avg:.4f} | score={vs:.6f} | "
              f"loc={row['localization_f1']:.4f} | no={row['damage_f1_no_damage']:.4f} | "
              f"min={row['damage_f1_minor_damage']:.4f} | maj={row['damage_f1_major_damage']:.4f} | "
              f"dest={row['damage_f1_destroyed']:.4f}", flush=True)

        if vs > best_score:
            best_score = vs; best_epoch = epoch; no_improve = 0
            save_checkpoint(ckpt_dir/"phase2_best.pt", model, optimizer, scheduler,
                            scaler, epoch, best_score, args)
            print(f"  ✓ Saved best | epoch={epoch} score={best_score:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} ep(s). Best epoch={best_epoch}", flush=True)

        save_checkpoint(ckpt_dir/"phase2_last.pt", model, optimizer, scheduler, scaler,
                        epoch, best_score, args)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(ckpt_dir/f"phase2_epoch_{epoch:03d}.pt", model, optimizer,
                            scheduler, scaler, epoch, best_score, args)
        json.dump(history, open(out_dir/"history_phase2.json","w",encoding="utf-8"), indent=2)
        if no_improve >= args.early_stopping_patience:
            print(f"Phase II early stopping at epoch {epoch}.", flush=True); break

    print(f"Phase II done. Best epoch={best_epoch} score={best_score:.6f}", flush=True)
    return ckpt_dir / "phase2_best.pt"


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────
def test_phase2(args, device, checkpoint_path: Path) -> None:
    print("\n================ TESTING ================", flush=True)
    _, _, test_loader, _ = make_loaders(args)
    model = HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    ckpt      = load_model_weights(model, checkpoint_path, device)
    best_ep   = int(ckpt.get("epoch", -1))
    results   = evaluate_phase2(model, test_loader, device)

    out_dir = Path(args.output_dir); scr_dir = out_dir / "scores"; scr_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(scr_dir/"scores_xbd_test_hrtbda.json","w",encoding="utf-8"), indent=2)
    splits_text = "+".join(args.train_split) if isinstance(args.train_split,list) else str(args.train_split)
    lines = [
        f"Experiment: HRTBDA-v2 xBD {splits_text} -> {args.val_split} -> {args.test_split}",
        f"Best Phase II epoch selected on hold: {best_ep}",
        f"Localization F1: {results['localization_f1']:.6f}",
        f"No Damage F1:    {results['damage_f1_no_damage']:.6f}",
        f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}",
        f"Major Damage F1: {results['damage_f1_major_damage']:.6f}",
        f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}",
        f"Damage F1:       {results['damage_f1']:.6f}",
        f"Overall Score:   {results['score']:.6f}",
    ]
    for path in [scr_dir/"scores_xbd_test_hrtbda.txt", scr_dir/"summary_hrtbda.txt"]:
        open(path,"w",encoding="utf-8").write("\n".join(lines)+"\n")
        print(f"Wrote: {path}", flush=True)
    print("\n".join(lines), flush=True)


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("HRTBDA v2")
    p.add_argument("--phase", default="both", choices=["both","phase1","phase2","test"])
    p.add_argument("--resume-phase1-from", type=str, default="")
    p.add_argument("--xbd-root", type=str, required=True)
    p.add_argument("--train-split", nargs="+", default=["tier3"])
    p.add_argument("--val-split",  type=str, default="hold")
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--phase1-epochs", type=int, default=150)
    p.add_argument("--phase2-epochs", type=int, default=80)    # FIX 5: raised from 30
    p.add_argument("--warmup-epochs", type=int, default=10)    # FIX 4: new param
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size",    type=int, default=1024)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--base-channels",    type=int,   default=48)
    p.add_argument("--decoder-channels", type=int,   default=128)
    p.add_argument("--window-size",      type=int,   default=8)
    p.add_argument("--save-every",              type=int,   default=1)
    p.add_argument("--early-stopping-patience", type=int,   default=999)
    p.add_argument("--max-grad-norm",   type=float, default=1.0)
    p.add_argument("--focal-gamma",     type=float, default=2.0)
    p.add_argument("--loc-loss-weight", type=float, default=1.0)
    p.add_argument("--cls-loss-weight", type=float, default=1.0)
    p.add_argument("--aux-loc-weight",  type=float, default=0.3)   # FIX 8: aux head weight
    p.add_argument("--minor-damage-boost", type=float, default=3.0)  # FIX 3: class weight boost
    p.add_argument("--cutmix-prob",     type=float, default=0.3)     # FIX 7: CutMix probability
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir/"scores").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== HRTBDA v2 =====", flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)
    print("Backbone: 4-stage HRNet + DCSwin + CSF | Fixes: focal-pt, inv-freq weights, "
          "warmup-cosine LR, CutMix, aux-loc head, scale-jitter aug", flush=True)
    print("=====================", flush=True)

    ckpt_dir = out_dir / "checkpoints"
    if args.phase == "phase1":
        train_phase1(args, device)
    elif args.phase == "phase2":
        train_phase2(args, device, ckpt_dir / "phase1_best.pt")
    elif args.phase == "test":
        test_phase2(args, device, ckpt_dir / "phase2_best.pt")
    else:  # both
        p1 = train_phase1(args, device)
        p2 = train_phase2(args, device, p1)
        test_phase2(args, device, p2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
