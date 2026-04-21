from __future__ import annotations

import argparse
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class PairedSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    pre_target_path: Path
    post_target_path: Path


class BaseXBDStyleDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool):
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        self.targets_dir = self.split_root / "targets"
        self.image_size = int(image_size)
        self.training = bool(training)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images dir not found: {self.images_dir}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.split_root}")

    def _collect_samples(self) -> List[PairedSample]:
        post_images: List[Path] = []
        for ext in IMG_EXTS:
            post_images.extend(self.images_dir.glob(f"*_post_disaster{ext}"))
        post_images = sorted(post_images)
        samples: List[PairedSample] = []
        for post_path in post_images:
            prefix = post_path.stem.replace("_post_disaster", "")
            ext = post_path.suffix
            pre_path = self.images_dir / f"{prefix}_pre_disaster{ext}"
            pre_tgt = self.targets_dir / f"{prefix}_pre_disaster_target.png"
            post_tgt = self.targets_dir / f"{prefix}_post_disaster_target.png"
            if pre_path.exists() and pre_tgt.exists() and post_tgt.exists():
                samples.append(PairedSample(prefix, pre_path, post_path, pre_tgt, post_tgt))
        return samples

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    @staticmethod
    def _build_damage_target(loc: np.ndarray, dmg: np.ndarray) -> np.ndarray:
        loc_bin = loc > 0
        target = np.full(loc.shape, 255, dtype=np.uint8)
        target[(dmg == 1) & loc_bin] = 0
        target[(dmg == 2) & loc_bin] = 1
        target[(dmg == 3) & loc_bin] = 2
        target[(dmg == 4) & loc_bin] = 3
        return target

    def _resize(self, pre: np.ndarray, post: np.ndarray, loc: Optional[np.ndarray], dmg_target: Optional[np.ndarray]):
        if pre.shape[:2] != (self.image_size, self.image_size):
            pre = cv2.resize(pre, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if post.shape[:2] != (self.image_size, self.image_size):
            post = cv2.resize(post, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if loc is not None and loc.shape[:2] != (self.image_size, self.image_size):
            loc = cv2.resize(loc, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if dmg_target is not None and dmg_target.shape[:2] != (self.image_size, self.image_size):
            dmg_target = cv2.resize(dmg_target, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return pre, post, loc, dmg_target

    def _norm_rgb(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        x = (x - self._mean) / self._std
        return x


class XBDStyleLabeledDataset(BaseXBDStyleDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool):
        super().__init__(root, split, image_size, training)
        if not self.targets_dir.exists():
            raise FileNotFoundError(f"Expected targets dir not found: {self.targets_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        pre = self._read_rgb(s.pre_image_path)
        post = self._read_rgb(s.post_image_path)
        loc = self._read_mask(s.pre_target_path)
        dmg = self._read_mask(s.post_target_path)
        loc_bin = (loc > 0).astype(np.float32)
        dmg_target = self._build_damage_target(loc, dmg)
        pre, post, loc_bin, dmg_target = self._resize(pre, post, loc_bin, dmg_target)
        return {
            "pre": torch.from_numpy(self._norm_rgb(pre)).float(),
            "post": torch.from_numpy(self._norm_rgb(post)).float(),
            "loc": torch.from_numpy(loc_bin).float(),
            "dmg": torch.from_numpy(dmg_target).long(),
            "stem": s.stem,
        }

    def get_localization_pixel_counts(self) -> Tuple[int, int]:
        pos, neg = 0, 0
        for s in self.samples:
            loc = self._read_mask(s.pre_target_path) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())
        return pos, neg

    def get_damage_class_counts(self) -> np.ndarray:
        counts = np.zeros(4, dtype=np.int64)
        for s in self.samples:
            loc = self._read_mask(s.pre_target_path)
            dmg = self._read_mask(s.post_target_path)
            target = self._build_damage_target(loc, dmg)
            valid = target != 255
            if valid.any():
                vals, freqs = np.unique(target[valid], return_counts=True)
                for v, f in zip(vals.tolist(), freqs.tolist()):
                    counts[int(v)] += int(f)
        return counts


class XBDStyleUnlabeledDataset(BaseXBDStyleDataset):
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        pre = self._read_rgb(s.pre_image_path)
        post = self._read_rgb(s.post_image_path)
        pre, post, _, _ = self._resize(pre, post, None, None)
        return {
            "pre": torch.from_numpy(self._norm_rgb(pre)).float(),
            "post": torch.from_numpy(self._norm_rgb(post)).float(),
            "stem": s.stem,
        }


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
    def __init__(self, tp: int, fp: int, fn: int, name: str = ""):
        self.tp = int(tp)
        self.fp = int(fp)
        self.fn = int(fn)
        self.name = name

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
        return 0.0 if (p == 0.0 or r == 0.0) else (2.0 * p * r) / (p + r)

    def as_dict(self):
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


class RunningConfusionMatrix:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        y_true = y_true.view(-1).cpu()
        y_pred = y_pred.view(-1).cpu()
        valid = (y_true >= 0) & (y_true < self.num_classes)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if y_true.numel() == 0:
            return
        idx = self.num_classes * y_true + y_pred
        bins = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.matrix += bins.reshape(self.num_classes, self.num_classes)

    def macro_f1(self) -> float:
        cm = self.matrix.float()
        tp = torch.diag(cm)
        precision = tp / (cm.sum(dim=0) + 1e-7)
        recall = tp / (cm.sum(dim=1) + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return float(torch.nanmean(f1))


def harmonic_mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs]
    return len(xs) / sum((x + 1e-6) ** -1 for x in xs)


class BinaryFocalDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = probs * target + (1.0 - probs) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + 1e-7) / (union + 1e-7)).mean()
        return focal + dice, focal, dice


class MulticlassFocalDiceLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = F.cross_entropy(logits, target, reduction="none", ignore_index=self.ignore_index, weight=self.class_weights)
        valid = target != self.ignore_index
        if valid.any():
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            target_safe = target.clone()
            target_safe[~valid] = 0
            pt = probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)
            focal = ((1.0 - pt).pow(self.gamma) * ce)[valid].mean()
        else:
            focal = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        probs = F.softmax(logits, dim=1)
        num_classes = logits.size(1)
        dice_terms = []
        for cls in range(num_classes):
            target_mask = (target == cls).float() * valid.float()
            pred_mask = probs[:, cls] * valid.float()
            inter = (pred_mask * target_mask).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2))
            present = union > 0
            if present.any():
                dice_terms.append(1.0 - ((2.0 * inter[present] + 1e-7) / (union[present] + 1e-7)).mean())
        dice = torch.stack(dice_terms).mean() if dice_terms else torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return focal + dice, focal, dice


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HaarWaveletDecompose(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=2, padding=0, groups=self.channels)


class MWFStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.wavelet = HaarWaveletDecompose(in_ch)
        self.main = nn.Sequential(
            ConvBNAct(in_ch, out_ch // 2, 3, 1),
            ConvBNAct(out_ch // 2, out_ch // 2, 3, 1),
        )
        self.wave = nn.Sequential(
            ConvBNAct(in_ch * 4, out_ch // 2, 1, 1, 0),
            ConvBNAct(out_ch // 2, out_ch // 2, 3, 1),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(out_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.main(x)
        ih = self.wave(self.wavelet(x))
        if ih.shape[-2:] != fx.shape[-2:]:
            ih = F.interpolate(ih, size=fx.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([fx, ih], dim=1))


class AdaptiveAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), channels, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        local = self.local(x)
        q, k, v = self.qkv(local).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        attn = torch.matmul(q, k) / math.sqrt(max(1, c))
        attn = attn.softmax(dim=-1)
        global_feat = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        stripe_h = x.mean(dim=3, keepdim=True)
        stripe_w = x.mean(dim=2, keepdim=True)
        stripe = stripe_h.expand(-1, -1, -1, w) + stripe_w.expand(-1, -1, h, -1)
        out = self.proj(global_feat + local + stripe)
        gate = self.channel_gate(out)
        return self.norm(x + out * gate)


class GMFFN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c1 = channels // 2
        c2 = channels - c1
        self.c1 = c1
        self.local = nn.Sequential(
            ConvBNAct(c1, c1, 3, 1, groups=max(1, math.gcd(c1, c1))),
            ConvBNAct(c1, c1, 3, 1, groups=max(1, math.gcd(c1, c1))),
        )
        self.global_proj = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.c1, x.size(1) - self.c1], dim=1)
        t1 = self.local(x1)
        t2 = self.global_proj(x2)
        if t1.size(1) != t2.size(1):
            if t1.size(1) < t2.size(1):
                t1m = F.pad(t1, (0, 0, 0, 0, 0, t2.size(1) - t1.size(1)))
                mult = t2 * t1m
            else:
                mult = t1[:, :t2.size(1)] * t2
        else:
            mult = t1 * t2
        out = self.proj(torch.cat([t1, mult], dim=1))
        return x + out


class GAMSFBlock(nn.Module):
    def __init__(self, channels: int, repeats: int):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(repeats):
            layers.append(AdaptiveAttention(channels))
            layers.append(GMFFN(channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GAMSFEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, aa_layout: Tuple[int, int, int] = (4, 4, 2), channels: Tuple[int, int, int, int] = (32, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        s2, s3, s4 = aa_layout
        self.stem = MWFStem(in_ch, c1)
        self.down1 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), GAMSFBlock(c2, s2))
        self.down2 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), GAMSFBlock(c3, s3))
        self.down3 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), GAMSFBlock(c4, s4))

    def forward(self, x: torch.Tensor):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        return s1, s2, s3, s4


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch + skip_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class GAMSFStage1(nn.Module):
    def __init__(self, aa_layout: Tuple[int, int, int] = (4, 4, 2), channels: Tuple[int, int, int, int] = (32, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder = GAMSFEncoder(3, aa_layout, channels)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        s1, s2, s3, s4 = self.encoder(x)
        x = self.dec3(s4, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        logits = self.head(x)
        if return_feat:
            return logits, s4
        return logits


class GAMSFStage2(nn.Module):
    def __init__(self, aa_layout: Tuple[int, int, int] = (4, 4, 2), channels: Tuple[int, int, int, int] = (32, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder = GAMSFEncoder(3, aa_layout, channels)
        self.fuse4 = ConvBNAct(c4 * 2, c4, 1, 1, 0)
        self.fuse3 = ConvBNAct(c3 * 2, c3, 1, 1, 0)
        self.fuse2 = ConvBNAct(c2 * 2, c2, 1, 1, 0)
        self.fuse1 = ConvBNAct(c1 * 2, c1, 1, 1, 0)
        self.mask_proj4 = ConvBNAct(c4 + 1, c4, 1, 1, 0)
        self.mask_proj3 = ConvBNAct(c3 + 1, c3, 1, 1, 0)
        self.mask_proj2 = ConvBNAct(c2 + 1, c2, 1, 1, 0)
        self.mask_proj1 = ConvBNAct(c1 + 1, c1, 1, 1, 0)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, 4, kernel_size=1)

    def load_stage1_encoder(self, stage1_model: GAMSFStage1) -> None:
        self.encoder.load_state_dict(stage1_model.encoder.state_dict())

    def _inject_mask(self, feat: torch.Tensor, mask: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        m = F.interpolate(mask.unsqueeze(1), size=feat.shape[-2:], mode="nearest")
        return proj(torch.cat([feat, m], dim=1))

    def forward(self, pre: torch.Tensor, post: torch.Tensor, mask: torch.Tensor, return_feat: bool = False):
        p1, p2, p3, p4 = self.encoder(pre)
        q1, q2, q3, q4 = self.encoder(post)
        f4 = self.fuse4(torch.cat([p4, q4], dim=1))
        f3 = self.fuse3(torch.cat([p3, q3], dim=1))
        f2 = self.fuse2(torch.cat([p2, q2], dim=1))
        f1 = self.fuse1(torch.cat([p1, q1], dim=1))
        f4 = self._inject_mask(f4, mask, self.mask_proj4)
        f3 = self._inject_mask(f3, mask, self.mask_proj3)
        f2 = self._inject_mask(f2, mask, self.mask_proj2)
        f1 = self._inject_mask(f1, mask, self.mask_proj1)
        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        logits = self.head(x)
        if return_feat:
            return logits, f4
        return logits


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        pooled = torch.mean(feat, dim=(2, 3))
        pooled = grad_reverse(pooled, grl_lambda)
        return self.net(pooled).squeeze(1)


def parse_aa_layout(text: str) -> Tuple[int, int, int]:
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("aa-layout must contain exactly 3 integers like 4,4,2")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("GAMSF + DANN UDA two-stage training")
    parser.add_argument("--source-dataset", type=str, required=True, choices=["xbd", "ida", "ian", "irma"])
    parser.add_argument("--target-dataset", type=str, required=True, choices=["xbd", "ida", "ian", "irma"])
    parser.add_argument("--xbd-root", type=str, default="/homes/j244s673/documents/wsu/phd/xview2")
    parser.add_argument("--ida-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet")
    parser.add_argument("--ian-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet")
    parser.add_argument("--irma-root", type=str, default="/homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--stage1-epochs", type=int, default=40)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--loc-threshold", type=float, default=0.5)
    parser.add_argument("--aa-layout", type=str, default="4,4,2")
    parser.add_argument("--stage1-domain-weight", type=float, default=0.05)
    parser.add_argument("--stage2-domain-weight", type=float, default=0.05)
    return parser.parse_args()


def dataset_root_and_splits(name: str, args: argparse.Namespace) -> Tuple[str, str, str, str]:
    if name == "xbd":
        return args.xbd_root, "train", "hold", "test"
    if name == "ida":
        return args.ida_root, "train", "val", "test"
    if name == "ian":
        return args.ian_root, "train", "val", "test"
    if name == "irma":
        return args.irma_root, "train", "val", "test"
    raise ValueError(name)


def compute_grl_lambda(epoch: int, step: int, steps_per_epoch: int, total_epochs: int) -> float:
    progress = ((epoch - 1) * steps_per_epoch + step) / max(1, total_epochs * steps_per_epoch)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_stage1(model: GAMSFStage1, loader: DataLoader, criterion: BinaryFocalDiceLoss, device: torch.device, threshold: float):
    model.eval()
    loss_meter = AverageMeter()
    focal_meter = AverageMeter()
    dice_meter = AverageMeter()
    tp = fp = fn = 0
    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        loc = batch["loc"].to(device, non_blocking=True)
        logits = model(pre).squeeze(1)
        loss, focal, dice_loss = criterion(logits, loc)
        pred = (torch.sigmoid(logits) > threshold).float()
        tp += int(((pred == 1) & (loc == 1)).sum().item())
        fp += int(((pred == 1) & (loc == 0)).sum().item())
        fn += int(((pred == 0) & (loc == 1)).sum().item())
        inter = (pred * loc).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + loc.sum(dim=(1, 2))
        dice_value = ((2.0 * inter + 1e-7) / (union + 1e-7)).mean().item()
        bs = pre.size(0)
        loss_meter.update(loss.item(), bs)
        focal_meter.update(focal.item(), bs)
        dice_meter.update(dice_value, bs)
    return {
        "loss": loss_meter.avg,
        "focal": focal_meter.avg,
        "loc_f1": F1Recorder(tp, fp, fn, "localization").f1,
        "loc_dice": dice_meter.avg,
    }


@torch.no_grad()
def evaluate_pipeline(stage1: GAMSFStage1, stage2: GAMSFStage2, loader: DataLoader, loc_threshold: float, device: torch.device) -> Dict[str, object]:
    stage1.eval()
    stage2.eval()

    loc_tp = loc_fp = loc_fn = 0
    dmg_counts = {
        1: {"tp": 0, "fp": 0, "fn": 0, "name": "no_damage"},
        2: {"tp": 0, "fp": 0, "fn": 0, "name": "minor_damage"},
        3: {"tp": 0, "fp": 0, "fn": 0, "name": "major_damage"},
        4: {"tp": 0, "fp": 0, "fn": 0, "name": "destroyed"},
    }
    loc_dice_meter = AverageMeter()
    conf = RunningConfusionMatrix(num_classes=4)

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        dmg_true_raw = batch["dmg"].to(device, non_blocking=True).long()

        loc_logits = stage1(pre).squeeze(1)
        loc_pred = (torch.sigmoid(loc_logits) > loc_threshold).long()
        dmg_logits = stage2(pre, post, loc_pred.float())

        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())
        inter = (loc_pred.float() * loc_true.float()).sum(dim=(1, 2))
        union = loc_pred.float().sum(dim=(1, 2)) + loc_true.float().sum(dim=(1, 2))
        loc_dice_meter.update(((2.0 * inter + 1e-7) / (union + 1e-7)).mean().item(), pre.size(0))

        dmg_pred = torch.argmax(dmg_logits, dim=1) + 1
        dmg_pred = dmg_pred * loc_pred
        valid_gt = (loc_true == 1) & (dmg_true_raw != 255)
        dmg_true = torch.zeros_like(dmg_true_raw)
        dmg_true[valid_gt] = dmg_true_raw[valid_gt] + 1
        dp = dmg_pred[valid_gt]
        dt = dmg_true[valid_gt]
        if dt.numel() > 0:
            conf.update(dt - 1, dp - 1)
        for c in [1, 2, 3, 4]:
            tp = ((dp == c) & (dt == c)).sum()
            fp = ((dp == c) & (dt != c)).sum()
            fn = ((dp != c) & (dt == c)).sum()
            dmg_counts[c]["tp"] += int(tp.item())
            dmg_counts[c]["fp"] += int(fp.item())
            dmg_counts[c]["fn"] += int(fn.item())

    loc_f1 = F1Recorder(loc_tp, loc_fp, loc_fn, "localization")
    no_damage = F1Recorder(dmg_counts[1]["tp"], dmg_counts[1]["fp"], dmg_counts[1]["fn"], "no_damage")
    minor = F1Recorder(dmg_counts[2]["tp"], dmg_counts[2]["fp"], dmg_counts[2]["fn"], "minor_damage")
    major = F1Recorder(dmg_counts[3]["tp"], dmg_counts[3]["fp"], dmg_counts[3]["fn"], "major_damage")
    destroyed = F1Recorder(dmg_counts[4]["tp"], dmg_counts[4]["fp"], dmg_counts[4]["fn"], "destroyed")
    damage_f1 = harmonic_mean([no_damage.f1, minor.f1, major.f1, destroyed.f1])
    score = 0.3 * loc_f1.f1 + 0.7 * damage_f1
    return {
        "score": score,
        "localization_f1": loc_f1.f1,
        "localization_dice": loc_dice_meter.avg,
        "damage_f1": damage_f1,
        "damage_macro_f1": conf.macro_f1(),
        "damage_f1_no_damage": no_damage.f1,
        "damage_f1_minor_damage": minor.f1,
        "damage_f1_major_damage": major.f1,
        "damage_f1_destroyed": destroyed.f1,
        "details": {
            "localization": loc_f1.as_dict(),
            "no_damage": no_damage.as_dict(),
            "minor_damage": minor.as_dict(),
            "major_damage": major.as_dict(),
            "destroyed": destroyed.as_dict(),
        },
    }


def save_checkpoint(save_path: Path, model: nn.Module, optimizer, scheduler, scaler, epoch: int, best_score: float, best_epoch: int, args: argparse.Namespace, domain_disc: Optional[nn.Module] = None):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "args": vars(args),
    }
    if domain_disc is not None:
        state["domain_disc"] = domain_disc.state_dict()
    torch.save(state, save_path)


def write_target_test_outputs(results: Dict[str, object], output_dir: Path, target_name: str) -> None:
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    json_path = scores_dir / f"scores_{target_name}_test.json"
    txt_path = scores_dir / f"scores_{target_name}_test.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Localization F1: {results['localization_f1']:.6f}\n")
        f.write(f"Localization Dice: {results['localization_dice']:.6f}\n")
        f.write(f"No Damage F1:    {results['damage_f1_no_damage']:.6f}\n")
        f.write(f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}\n")
        f.write(f"Major Damage F1: {results['damage_f1_major_damage']:.6f}\n")
        f.write(f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}\n")
        f.write(f"Damage F1:       {results['damage_f1']:.6f}\n")
        f.write(f"Overall Score:   {results['score']:.6f}\n")


def main() -> None:
    args = parse_args()
    if args.source_dataset == args.target_dataset:
        raise ValueError("source_dataset and target_dataset must be different")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage1" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "stage2" / "checkpoints").mkdir(parents=True, exist_ok=True)

    with open(output_dir / "uda_config.json", "w", encoding="utf-8") as f:
        json.dump({"source": args.source_dataset, "target": args.target_dataset, **vars(args)}, f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    aa_layout = parse_aa_layout(args.aa_layout)
    print(f"Using device: {device}", flush=True)
    print(f"Source dataset: {args.source_dataset}", flush=True)
    print(f"Target dataset: {args.target_dataset}", flush=True)

    src_root, src_train_split, src_val_split, _ = dataset_root_and_splits(args.source_dataset, args)
    tgt_root, tgt_train_split, tgt_val_split, tgt_test_split = dataset_root_and_splits(args.target_dataset, args)

    src_train = XBDStyleLabeledDataset(src_root, src_train_split, args.img_size, True)
    src_val = XBDStyleLabeledDataset(src_root, src_val_split, args.img_size, False)
    tgt_train_u = XBDStyleUnlabeledDataset(tgt_root, tgt_train_split, args.img_size, True)
    tgt_val_u = XBDStyleUnlabeledDataset(tgt_root, tgt_val_split, args.img_size, True)
    tgt_test = XBDStyleLabeledDataset(tgt_root, tgt_test_split, args.img_size, False)

    src_train_loader = DataLoader(src_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    src_val_loader = DataLoader(src_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    tgt_u_loader = DataLoader(ConcatDataset([tgt_train_u, tgt_val_u]), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tgt_test_loader = DataLoader(tgt_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    loc_pos, loc_neg = src_train.get_localization_pixel_counts()
    loc_pos_weight = torch.tensor([max(1.0, loc_neg / max(loc_pos, 1))], dtype=torch.float32, device=device)
    dmg_counts = src_train.get_damage_class_counts().astype(np.float64)
    dmg_counts[dmg_counts == 0] = 1.0
    inv = dmg_counts.sum() / dmg_counts
    dmg_class_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32, device=device)

    print(f"Source train samples: {len(src_train)}", flush=True)
    print(f"Source val samples:   {len(src_val)}", flush=True)
    print(f"Target unlabeled train+val samples: {len(tgt_train_u) + len(tgt_val_u)}", flush=True)
    print(f"Target test samples:  {len(tgt_test)}", flush=True)

    # Stage 1
    stage1 = GAMSFStage1(aa_layout=aa_layout).to(device)
    stage1_disc = DomainDiscriminator(128).to(device)
    stage1_optimizer = torch.optim.AdamW(list(stage1.parameters()) + list(stage1_disc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    stage1_scheduler = torch.optim.lr_scheduler.MultiStepLR(stage1_optimizer, milestones=sorted(set(max(1, int(args.stage1_epochs * x)) for x in (0.5, 0.75, 0.9))), gamma=0.5)
    stage1_scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda") if USE_TORCH_AMP else GradScaler(enabled=args.amp and device.type == "cuda")
    loc_criterion = BinaryFocalDiceLoss(pos_weight=loc_pos_weight).to(device)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)

    stage1_best_score = -1.0
    stage1_best_epoch = 0
    stage1_wait = 0
    stage1_history: List[Dict[str, float | int]] = []
    tgt_iter = cycle(tgt_u_loader)
    stage1_steps = len(src_train_loader)

    for epoch in range(1, args.stage1_epochs + 1):
        stage1.train()
        stage1_disc.train()
        total_meter = AverageMeter()
        sup_meter = AverageMeter()
        dom_meter = AverageMeter()
        iterator = tqdm(src_train_loader, desc=f"stage1 {epoch}/{args.stage1_epochs}") if (tqdm is not None and sys.stderr.isatty()) else src_train_loader
        for step, src_batch in enumerate(iterator, start=1):
            tgt_batch = next(tgt_iter)
            src_pre = src_batch["pre"].to(device, non_blocking=True)
            src_loc = src_batch["loc"].to(device, non_blocking=True)
            tgt_pre = tgt_batch["pre"].to(device, non_blocking=True)
            grl_lambda = compute_grl_lambda(epoch, step, stage1_steps, args.stage1_epochs)
            stage1_optimizer.zero_grad(set_to_none=True)
            ctx = autocast(device_type=device.type, enabled=args.amp and device.type == "cuda") if USE_TORCH_AMP else autocast(enabled=args.amp and device.type == "cuda")
            with ctx:
                src_logits, src_feat = stage1(src_pre, return_feat=True)
                src_logits = src_logits.squeeze(1)
                _, src_focal, src_dice = loc_criterion(src_logits, src_loc)
                sup_loss = src_focal + src_dice

                tgt_logits, tgt_feat = stage1(tgt_pre, return_feat=True)
                src_dom = stage1_disc(src_feat, grl_lambda)
                tgt_dom = stage1_disc(tgt_feat, grl_lambda)
                src_dom_t = torch.zeros_like(src_dom)
                tgt_dom_t = torch.ones_like(tgt_dom)
                dom_loss = 0.5 * (domain_criterion(src_dom, src_dom_t) + domain_criterion(tgt_dom, tgt_dom_t))
                total_loss = sup_loss + args.stage1_domain_weight * dom_loss
            stage1_scaler.scale(total_loss).backward()
            stage1_scaler.step(stage1_optimizer)
            stage1_scaler.update()
            bs = src_pre.size(0)
            total_meter.update(total_loss.item(), bs)
            sup_meter.update(sup_loss.item(), bs)
            dom_meter.update(dom_loss.item(), bs)
            if tqdm is not None and sys.stderr.isatty():
                iterator.set_postfix(loss=f"{total_meter.avg:.4f}", sup=f"{sup_meter.avg:.4f}", dom=f"{dom_meter.avg:.4f}")
        stage1_scheduler.step()
        src_val_metrics = evaluate_stage1(stage1, src_val_loader, loc_criterion, device, args.loc_threshold)
        tgt_loc_metrics = evaluate_stage1(stage1, tgt_test_loader, loc_criterion, device, args.loc_threshold)
        val_score = src_val_metrics["loc_f1"] + src_val_metrics["loc_dice"]
        row = {
            "epoch": epoch,
            "lr": stage1_optimizer.param_groups[0]["lr"],
            "train_total_loss": total_meter.avg,
            "train_supervised_loss": sup_meter.avg,
            "train_domain_loss": dom_meter.avg,
            "src_val_loc_f1": src_val_metrics["loc_f1"],
            "src_val_loc_dice": src_val_metrics["loc_dice"],
            "src_val_score": val_score,
            "tgt_test_loc_f1": tgt_loc_metrics["loc_f1"],
            "tgt_test_loc_dice": tgt_loc_metrics["loc_dice"],
        }
        stage1_history.append(row)
        print(json.dumps({"stage": 1, **row}, indent=2), flush=True)
        if val_score > stage1_best_score:
            stage1_best_score = float(val_score)
            stage1_best_epoch = epoch
            stage1_wait = 0
            save_checkpoint(output_dir / "stage1" / "checkpoints" / "best.pt", stage1, stage1_optimizer, stage1_scheduler, stage1_scaler, epoch, stage1_best_score, stage1_best_epoch, args, domain_disc=stage1_disc)
        else:
            stage1_wait += 1
        save_checkpoint(output_dir / "stage1" / "checkpoints" / "last.pt", stage1, stage1_optimizer, stage1_scheduler, stage1_scaler, epoch, stage1_best_score, stage1_best_epoch, args, domain_disc=stage1_disc)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(output_dir / "stage1" / "checkpoints" / f"epoch_{epoch:03d}.pt", stage1, stage1_optimizer, stage1_scheduler, stage1_scaler, epoch, stage1_best_score, stage1_best_epoch, args, domain_disc=stage1_disc)
        with open(output_dir / "stage1" / "history.json", "w", encoding="utf-8") as f:
            json.dump(stage1_history, f, indent=2)
        if stage1_wait >= args.early_stopping_patience:
            print(f"Stage 1 early stopping at epoch {epoch}", flush=True)
            break

    stage1_ckpt = torch.load(output_dir / "stage1" / "checkpoints" / "best.pt", map_location=device)
    stage1.load_state_dict(stage1_ckpt["model"])
    stage1_target_metrics = evaluate_stage1(stage1, tgt_test_loader, loc_criterion, device, args.loc_threshold)
    with open(output_dir / "stage1" / "target_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stage1_target_metrics, f, indent=2)

    # Stage 2
    stage2 = GAMSFStage2(aa_layout=aa_layout).to(device)
    stage2.load_stage1_encoder(stage1)
    stage2_disc = DomainDiscriminator(128).to(device)
    stage2_optimizer = torch.optim.AdamW(list(stage2.parameters()) + list(stage2_disc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    stage2_scheduler = torch.optim.lr_scheduler.MultiStepLR(stage2_optimizer, milestones=sorted(set(max(1, int(args.stage2_epochs * x)) for x in (0.5, 0.75, 0.9))), gamma=0.5)
    stage2_scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda") if USE_TORCH_AMP else GradScaler(enabled=args.amp and device.type == "cuda")
    dmg_criterion = MulticlassFocalDiceLoss(class_weights=dmg_class_weights, ignore_index=255).to(device)

    stage2_best_score = -1.0
    stage2_best_epoch = 0
    stage2_wait = 0
    stage2_history: List[Dict[str, float | int]] = []
    tgt_iter2 = cycle(tgt_u_loader)
    stage2_steps = len(src_train_loader)

    for epoch in range(1, args.stage2_epochs + 1):
        stage2.train()
        stage2_disc.train()
        total_meter = AverageMeter()
        sup_meter = AverageMeter()
        dom_meter = AverageMeter()
        iterator = tqdm(src_train_loader, desc=f"stage2 {epoch}/{args.stage2_epochs}") if (tqdm is not None and sys.stderr.isatty()) else src_train_loader
        for step, src_batch in enumerate(iterator, start=1):
            tgt_batch = next(tgt_iter2)
            src_pre = src_batch["pre"].to(device, non_blocking=True)
            src_post = src_batch["post"].to(device, non_blocking=True)
            src_loc = src_batch["loc"].to(device, non_blocking=True)
            src_dmg = src_batch["dmg"].to(device, non_blocking=True)

            tgt_pre = tgt_batch["pre"].to(device, non_blocking=True)
            tgt_post = tgt_batch["post"].to(device, non_blocking=True)
            with torch.no_grad():
                tgt_loc_logits = stage1(tgt_pre).squeeze(1)
                tgt_loc = (torch.sigmoid(tgt_loc_logits) > args.loc_threshold).float()

            grl_lambda = compute_grl_lambda(epoch, step, stage2_steps, args.stage2_epochs)
            stage2_optimizer.zero_grad(set_to_none=True)
            ctx = autocast(device_type=device.type, enabled=args.amp and device.type == "cuda") if USE_TORCH_AMP else autocast(enabled=args.amp and device.type == "cuda")
            with ctx:
                src_logits, src_feat = stage2(src_pre, src_post, src_loc, return_feat=True)
                sup_loss, _, _ = dmg_criterion(src_logits, src_dmg)

                _, tgt_feat = stage2(tgt_pre, tgt_post, tgt_loc, return_feat=True)
                src_dom = stage2_disc(src_feat, grl_lambda)
                tgt_dom = stage2_disc(tgt_feat, grl_lambda)
                src_dom_t = torch.zeros_like(src_dom)
                tgt_dom_t = torch.ones_like(tgt_dom)
                dom_loss = 0.5 * (domain_criterion(src_dom, src_dom_t) + domain_criterion(tgt_dom, tgt_dom_t))
                total_loss = sup_loss + args.stage2_domain_weight * dom_loss
            stage2_scaler.scale(total_loss).backward()
            stage2_scaler.step(stage2_optimizer)
            stage2_scaler.update()
            bs = src_pre.size(0)
            total_meter.update(total_loss.item(), bs)
            sup_meter.update(sup_loss.item(), bs)
            dom_meter.update(dom_loss.item(), bs)
            if tqdm is not None and sys.stderr.isatty():
                iterator.set_postfix(loss=f"{total_meter.avg:.4f}", sup=f"{sup_meter.avg:.4f}", dom=f"{dom_meter.avg:.4f}")
        stage2_scheduler.step()
        src_val_pipeline = evaluate_pipeline(stage1, stage2, src_val_loader, args.loc_threshold, device)
        tgt_test_pipeline = evaluate_pipeline(stage1, stage2, tgt_test_loader, args.loc_threshold, device)
        val_score = src_val_pipeline["score"]
        row = {
            "epoch": epoch,
            "lr": stage2_optimizer.param_groups[0]["lr"],
            "train_total_loss": total_meter.avg,
            "train_supervised_loss": sup_meter.avg,
            "train_domain_loss": dom_meter.avg,
            "src_val_localization_f1": src_val_pipeline["localization_f1"],
            "src_val_damage_f1": src_val_pipeline["damage_f1"],
            "src_val_score": val_score,
            "tgt_test_localization_f1": tgt_test_pipeline["localization_f1"],
            "tgt_test_damage_f1": tgt_test_pipeline["damage_f1"],
            "tgt_test_no_damage_f1": tgt_test_pipeline["damage_f1_no_damage"],
            "tgt_test_minor_damage_f1": tgt_test_pipeline["damage_f1_minor_damage"],
            "tgt_test_major_damage_f1": tgt_test_pipeline["damage_f1_major_damage"],
            "tgt_test_destroyed_f1": tgt_test_pipeline["damage_f1_destroyed"],
        }
        stage2_history.append(row)
        print(json.dumps({"stage": 2, **row}, indent=2), flush=True)
        if val_score > stage2_best_score:
            stage2_best_score = float(val_score)
            stage2_best_epoch = epoch
            stage2_wait = 0
            save_checkpoint(output_dir / "stage2" / "checkpoints" / "best.pt", stage2, stage2_optimizer, stage2_scheduler, stage2_scaler, epoch, stage2_best_score, stage2_best_epoch, args, domain_disc=stage2_disc)
        else:
            stage2_wait += 1
        save_checkpoint(output_dir / "stage2" / "checkpoints" / "last.pt", stage2, stage2_optimizer, stage2_scheduler, stage2_scaler, epoch, stage2_best_score, stage2_best_epoch, args, domain_disc=stage2_disc)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(output_dir / "stage2" / "checkpoints" / f"epoch_{epoch:03d}.pt", stage2, stage2_optimizer, stage2_scheduler, stage2_scaler, epoch, stage2_best_score, stage2_best_epoch, args, domain_disc=stage2_disc)
        with open(output_dir / "stage2" / "history.json", "w", encoding="utf-8") as f:
            json.dump(stage2_history, f, indent=2)
        if stage2_wait >= args.early_stopping_patience:
            print(f"Stage 2 early stopping at epoch {epoch}", flush=True)
            break

    stage2_ckpt = torch.load(output_dir / "stage2" / "checkpoints" / "best.pt", map_location=device)
    stage2.load_state_dict(stage2_ckpt["model"])
    final_results = evaluate_pipeline(stage1, stage2, tgt_test_loader, args.loc_threshold, device)
    with open(output_dir / "target_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    write_target_test_outputs(final_results, output_dir, args.target_dataset)

    print("\n===== FINAL TARGET TEST RESULTS =====", flush=True)
    print(json.dumps(final_results, indent=2), flush=True)


if __name__ == "__main__":
    main()
