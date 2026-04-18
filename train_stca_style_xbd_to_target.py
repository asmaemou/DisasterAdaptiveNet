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
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

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
TARGET_NAMES = ("ida", "ian", "rescuenet")


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class F1Recorder:
    def __init__(self, tp: int, fp: int, fn: int, name: str):
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

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def harmonic_mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs]
    return len(xs) / sum((x + 1e-6) ** -1 for x in xs)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + 1e-7) / (union + 1e-7)).mean()
        return bce, dice


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return GradientReversalFunction.apply(x, alpha)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNet34UNetSTCA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight
        backbone.conv1 = new_conv

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 64, 64)

        self.local_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.damage_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )

        self.grl = GradientReversal()
        self.domain_classifier = DomainClassifier(512, 256)

    def encode(self, x: torch.Tensor):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

    def decode(self, feats):
        x0, x1, x2, x3, x4 = feats
        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)
        d0 = F.interpolate(d0, scale_factor=2, mode="bilinear", align_corners=False)
        return d0

    def forward(self, x: torch.Tensor, grl_alpha: Optional[float] = None):
        feats = self.encode(x)
        dec = self.decode(feats)
        loc = self.local_head(dec)
        dmg = self.damage_head(dec)
        seg_logits = torch.cat([loc, dmg], dim=1)
        dom_logits = None
        if grl_alpha is not None:
            pooled = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
            pooled = self.grl(pooled, grl_alpha)
            dom_logits = self.domain_classifier(pooled)
        return seg_logits, dom_logits


class BasePairDataset(Dataset):
    def __init__(self, image_size: int, training: bool) -> None:
        self.image_size = int(image_size)
        self.training = bool(training)
        self.mean3 = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std3 = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

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
    def _build_damage_target_from_standard_mask(loc: np.ndarray, dmg: np.ndarray) -> np.ndarray:
        loc_bin = loc > 0
        target = np.full(loc.shape, 255, dtype=np.uint8)
        target[(dmg == 1) & loc_bin] = 0
        target[(dmg == 2) & loc_bin] = 1
        target[(dmg == 3) & loc_bin] = 2
        target[(dmg == 4) & loc_bin] = 3
        return target

    def _resize_rgb(self, x: np.ndarray) -> np.ndarray:
        if x.shape[:2] != (self.image_size, self.image_size):
            x = cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return x

    def _resize_mask(self, x: np.ndarray) -> np.ndarray:
        if x.shape[:2] != (self.image_size, self.image_size):
            x = cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return x

    def _normalize3(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return (x - self.mean3) / self.std3

    def _augment(self, pre: np.ndarray, post: Optional[np.ndarray], loc: Optional[np.ndarray], dmg: Optional[np.ndarray]):
        if not self.training:
            return pre, post, loc, dmg
        if np.random.rand() < 0.5:
            pre = np.flip(pre, axis=1).copy()
            if post is not None:
                post = np.flip(post, axis=1).copy()
            if loc is not None:
                loc = np.flip(loc, axis=1).copy()
            if dmg is not None:
                dmg = np.flip(dmg, axis=1).copy()
        if np.random.rand() < 0.5:
            pre = np.flip(pre, axis=0).copy()
            if post is not None:
                post = np.flip(post, axis=0).copy()
            if loc is not None:
                loc = np.flip(loc, axis=0).copy()
            if dmg is not None:
                dmg = np.flip(dmg, axis=0).copy()
        k = np.random.randint(0, 4)
        if k:
            pre = np.rot90(pre, k=k).copy()
            if post is not None:
                post = np.rot90(post, k=k).copy()
            if loc is not None:
                loc = np.rot90(loc, k=k).copy()
            if dmg is not None:
                dmg = np.rot90(dmg, k=k).copy()
        return pre, post, loc, dmg


@dataclass(frozen=True)
class PairSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    pre_target_path: Optional[Path]
    post_target_path: Optional[Path]


class XBDLikeLabeledDataset(BasePairDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool):
        super().__init__(image_size, training)
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        self.targets_dir = self.split_root / "targets"
        if not self.images_dir.exists() or not self.targets_dir.exists():
            raise FileNotFoundError(f"Expected xBD-like pair layout under {self.split_root}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No paired labeled samples found under {self.split_root}")

    def _collect_samples(self) -> List[PairSample]:
        post_images: List[Path] = []
        for pattern in ["*_post_disaster.png", "*_post_disaster.jpg", "*_post_disaster.jpeg", "*_post_disaster.tif", "*_post_disaster.tiff", "*_post_disaster.bmp"]:
            post_images.extend(self.images_dir.glob(pattern))
        post_images = sorted(post_images)
        samples: List[PairSample] = []
        for post_path in post_images:
            prefix = post_path.stem.replace("_post_disaster", "")
            ext = post_path.suffix
            pre_path = self.images_dir / f"{prefix}_pre_disaster{ext}"
            pre_tgt = self.targets_dir / f"{prefix}_pre_disaster_target.png"
            post_tgt = self.targets_dir / f"{prefix}_post_disaster_target.png"
            if pre_path.exists() and pre_tgt.exists() and post_tgt.exists():
                samples.append(PairSample(prefix, pre_path, post_path, pre_tgt, post_tgt))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        pre = self._resize_rgb(self._read_rgb(s.pre_image_path))
        post = self._resize_rgb(self._read_rgb(s.post_image_path))
        loc = self._resize_mask(self._read_mask(s.pre_target_path))
        dmg = self._resize_mask(self._read_mask(s.post_target_path))
        dmg_target = self._build_damage_target_from_standard_mask(loc, dmg)
        pre, post, loc, dmg_target = self._augment(pre, post, loc, dmg_target)
        loc = (loc > 0).astype(np.float32)
        pre_t = torch.from_numpy(self._normalize3(pre)).float()
        post_t = torch.from_numpy(self._normalize3(post)).float()
        img6 = torch.cat([pre_t, post_t], dim=0)
        return {
            "img": img6,
            "pre": pre_t,
            "post": post_t,
            "loc": torch.from_numpy(loc).float(),
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
            target = self._build_damage_target_from_standard_mask(loc, dmg)
            valid = target != 255
            if valid.any():
                vals, freqs = np.unique(target[valid], return_counts=True)
                for v, f in zip(vals.tolist(), freqs.tolist()):
                    counts[int(v)] += int(f)
        return counts


class XBDLikePreOnlyDataset(BasePairDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool):
        super().__init__(image_size, training)
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images dir not found: {self.images_dir}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No pre-disaster images found under {self.split_root}")

    def _collect_samples(self) -> List[PairSample]:
        pre_images: List[Path] = []
        for pattern in ["*_pre_disaster.png", "*_pre_disaster.jpg", "*_pre_disaster.jpeg", "*_pre_disaster.tif", "*_pre_disaster.tiff", "*_pre_disaster.bmp"]:
            pre_images.extend(self.images_dir.glob(pattern))
        pre_images = sorted(pre_images)
        samples: List[PairSample] = []
        for pre_path in pre_images:
            prefix = pre_path.stem.replace("_pre_disaster", "")
            post_candidates = list(self.images_dir.glob(f"{prefix}_post_disaster.*"))
            if post_candidates:
                samples.append(PairSample(prefix, pre_path, post_candidates[0], None, None))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        pre = self._resize_rgb(self._read_rgb(s.pre_image_path))
        pre, _, _, _ = self._augment(pre, None, None, None)
        pre_t = torch.from_numpy(self._normalize3(pre)).float()
        return {"pre": pre_t, "stem": s.stem}


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def compute_domain_alpha(epoch: int, step: int, steps_per_epoch: int, total_epochs: int) -> float:
    progress = ((epoch - 1) * steps_per_epoch + step) / max(1, total_epochs * steps_per_epoch)
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def compute_supervised_losses(
    seg_logits: torch.Tensor,
    loc: torch.Tensor,
    dmg: torch.Tensor,
    loc_criterion: BCEDiceLoss,
    dmg_criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logit_loc = seg_logits[:, 0]
    logit_dmg = seg_logits[:, 1:5]
    loc_bce, loc_dice_loss = loc_criterion(logit_loc, loc)
    valid_dmg = dmg != 255
    if valid_dmg.any():
        dmg_ce = dmg_criterion(logit_dmg, dmg)
    else:
        dmg_ce = torch.tensor(0.0, device=device, dtype=logit_loc.dtype)
    total = args.loc_bce_weight * loc_bce + args.loc_dice_weight * loc_dice_loss + args.dmg_ce_weight * dmg_ce
    return total, loc_bce, loc_dice_loss, dmg_ce


def aggregate_counts(datasets: List[object]) -> Tuple[torch.Tensor, torch.Tensor]:
    loc_pos, loc_neg = 0, 0
    dmg_counts = np.zeros(4, dtype=np.int64)
    for ds in datasets:
        p, n = ds.get_localization_pixel_counts()
        loc_pos += int(p)
        loc_neg += int(n)
        dmg_counts += ds.get_damage_class_counts()
    loc_pos_weight = torch.tensor([max(1.0, loc_neg / max(loc_pos, 1))], dtype=torch.float32)
    dmg_counts = dmg_counts.astype(np.float64)
    dmg_counts[dmg_counts == 0] = 1.0
    inv = dmg_counts.sum() / dmg_counts
    dmg_class_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32)
    return loc_pos_weight, dmg_class_weights


@torch.no_grad()
def evaluate_source_validation(model, loader, loc_criterion, dmg_criterion, device, args):
    model.eval()
    loss_meter = AverageMeter()
    loc_bce_meter = AverageMeter()
    loc_dice_loss_meter = AverageMeter()
    loc_dice_meter = AverageMeter()
    dmg_ce_meter = AverageMeter()
    dmg_acc_meter = AverageMeter()
    conf = RunningConfusionMatrix(4)
    use_tqdm = tqdm is not None and sys.stderr.isatty()
    iterator = tqdm(loader, desc="src_val", leave=False) if use_tqdm else loader
    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc = batch["loc"].to(device, non_blocking=True)
        dmg = batch["dmg"].to(device, non_blocking=True)
        seg_logits, _ = model(img, grl_alpha=None)
        loss, loc_bce, loc_dice_loss, dmg_ce = compute_supervised_losses(seg_logits, loc, dmg, loc_criterion, dmg_criterion, device, args)
        logit_loc = seg_logits[:, 0]
        logit_dmg = seg_logits[:, 1:5]
        loc_pred = (torch.sigmoid(logit_loc) > args.loc_threshold).float()
        inter = (loc_pred * loc).sum(dim=(1, 2))
        union = loc_pred.sum(dim=(1, 2)) + loc.sum(dim=(1, 2))
        loc_dice = ((2.0 * inter + 1e-7) / (union + 1e-7)).mean().item()
        dmg_pred = torch.argmax(logit_dmg, dim=1)
        valid = dmg != 255
        if valid.any():
            dmg_acc = (dmg_pred[valid] == dmg[valid]).float().mean().item()
            conf.update(dmg[valid], dmg_pred[valid])
            dmg_ce_value = dmg_ce.item()
        else:
            dmg_acc = 0.0
            dmg_ce_value = 0.0
        bs = img.size(0)
        loss_meter.update(loss.item(), bs)
        loc_bce_meter.update(loc_bce.item(), bs)
        loc_dice_loss_meter.update(loc_dice_loss.item(), bs)
        loc_dice_meter.update(loc_dice, bs)
        dmg_ce_meter.update(dmg_ce_value, bs)
        dmg_acc_meter.update(dmg_acc, bs)
    return {
        "loss": loss_meter.avg,
        "loc_bce": loc_bce_meter.avg,
        "loc_dice_loss": loc_dice_loss_meter.avg,
        "loc_dice": loc_dice_meter.avg,
        "dmg_ce": dmg_ce_meter.avg,
        "dmg_acc": dmg_acc_meter.avg,
        "dmg_macro_f1": conf.macro_f1(),
    }


@torch.no_grad()
def evaluate_target_test_f1(model, loader, device, loc_threshold):
    model.eval()
    loc_tp, loc_fp, loc_fn = 0, 0, 0
    dmg_counts = {
        1: {"tp": 0, "fp": 0, "fn": 0, "name": "no_damage"},
        2: {"tp": 0, "fp": 0, "fn": 0, "name": "minor_damage"},
        3: {"tp": 0, "fp": 0, "fn": 0, "name": "major_damage"},
        4: {"tp": 0, "fp": 0, "fn": 0, "name": "destroyed"},
    }
    use_tqdm = tqdm is not None and sys.stderr.isatty()
    iterator = tqdm(loader, desc="target_test", leave=False) if use_tqdm else loader
    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        dmg_true_raw = batch["dmg"].to(device, non_blocking=True).long()
        seg_logits, _ = model(img, grl_alpha=None)
        loc_logits = seg_logits[:, 0]
        dmg_logits = seg_logits[:, 1:5]
        loc_pred = (torch.sigmoid(loc_logits) > loc_threshold).long()
        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())
        dmg_pred = torch.argmax(dmg_logits, dim=1) + 1
        dmg_pred = dmg_pred * loc_pred
        valid_gt = (loc_true == 1) & (dmg_true_raw != 255)
        dmg_true = torch.zeros_like(dmg_true_raw)
        dmg_true[valid_gt] = dmg_true_raw[valid_gt] + 1
        dp = dmg_pred[valid_gt]
        dt = dmg_true[valid_gt]
        for cls in [1, 2, 3, 4]:
            tp = ((dp == cls) & (dt == cls)).sum()
            fp = ((dp == cls) & (dt != cls)).sum()
            fn = ((dp != cls) & (dt == cls)).sum()
            dmg_counts[cls]["tp"] += int(tp.item())
            dmg_counts[cls]["fp"] += int(fp.item())
            dmg_counts[cls]["fn"] += int(fn.item())
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
        "damage_f1": damage_f1,
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


def save_checkpoint(save_path: Path, model, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "args": vars(args),
    }, save_path)


def write_target_test_outputs(results: Dict[str, object], output_dir: Path, target_name: str) -> None:
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    json_path = scores_dir / f"scores_{target_name}_test.json"
    txt_path = scores_dir / f"scores_{target_name}_test.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Localization F1: {results['localization_f1']:.6f}\n")
        f.write(f"No Damage F1:    {results['damage_f1_no_damage']:.6f}\n")
        f.write(f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}\n")
        f.write(f"Major Damage F1: {results['damage_f1_major_damage']:.6f}\n")
        f.write(f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}\n")
        f.write(f"Damage F1:       {results['damage_f1']:.6f}\n")
        f.write(f"Overall Score:   {results['score']:.6f}\n")


def detect_target_support(target_name: str, target_root: Path, target_train_split: str) -> Tuple[bool, str]:
    if target_name in {"ida", "ian"}:
        return True, "xBD-like paired pre/post target available"
    split_root = target_root / target_train_split / "images"
    if not split_root.exists():
        return False, f"Target image directory not found: {split_root}"
    has_pre = any(split_root.glob("*_pre_disaster.*"))
    if has_pre:
        return True, "paired pre/post target appears available"
    return False, (
        "Current RescueNet directory does not appear to contain target pre-disaster images in xBD-like form. "
        "This STCA-style script needs target pre-disaster images."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("STCA-style xBD-to-target adaptation using target pre-disaster images only")
    parser.add_argument("--target-dataset", type=str, required=True, choices=list(TARGET_NAMES))
    parser.add_argument("--xbd-root", type=str, default="/homes/j244s673/documents/wsu/phd/xview2")
    parser.add_argument("--ida-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet")
    parser.add_argument("--ian-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet")
    parser.add_argument("--rescuenet-root", type=str, default="/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd")
    parser.add_argument("--xbd-train-split", type=str, default="train")
    parser.add_argument("--xbd-val-split", type=str, default="hold")
    parser.add_argument("--target-train-split", type=str, default="train")
    parser.add_argument("--target-val-split", type=str, default="val")
    parser.add_argument("--target-test-split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--loc-threshold", type=float, default=0.5)
    parser.add_argument("--loc-bce-weight", type=float, default=1.0)
    parser.add_argument("--loc-dice-weight", type=float, default=1.0)
    parser.add_argument("--dmg-ce-weight", type=float, default=1.0)
    parser.add_argument("--domain-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    target_root = {
        "ida": Path(args.ida_root),
        "ian": Path(args.ian_root),
        "rescuenet": Path(args.rescuenet_root),
    }[args.target_dataset]

    supported, reason = detect_target_support(args.target_dataset, target_root, args.target_train_split)
    if not supported:
        print(f"SKIPPING target {args.target_dataset}: {reason}", flush=True)
        with open(output_dir / "stca_skip.json", "w", encoding="utf-8") as f:
            json.dump({"target": args.target_dataset, "status": "skipped", "reason": reason}, f, indent=2)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Target dataset: {args.target_dataset}", flush=True)
    print(f"STCA-style note: this script is a practical target-pre-only adaptation baseline, not an exact paper reproduction.", flush=True)

    print("Loading xBD source train...", flush=True)
    src_train = XBDLikeLabeledDataset(args.xbd_root, args.xbd_train_split, args.img_size, True)
    print("Loading xBD source val...", flush=True)
    src_val = XBDLikeLabeledDataset(args.xbd_root, args.xbd_val_split, args.img_size, False)
    print(f"Loading {args.target_dataset} target pre-only train...", flush=True)
    tgt_pre_train = XBDLikePreOnlyDataset(target_root, args.target_train_split, args.img_size, True)
    print(f"Loading {args.target_dataset} target pre-only val...", flush=True)
    tgt_pre_val = XBDLikePreOnlyDataset(target_root, args.target_val_split, args.img_size, True)
    print(f"Loading {args.target_dataset} target test...", flush=True)
    tgt_test = XBDLikeLabeledDataset(target_root, args.target_test_split, args.img_size, False)

    src_train_loader = DataLoader(src_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    src_val_loader = DataLoader(src_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    tgt_pre_loader = DataLoader(torch.utils.data.ConcatDataset([tgt_pre_train, tgt_pre_val]), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tgt_test_loader = DataLoader(tgt_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    loc_pos_weight, dmg_class_weights = aggregate_counts([src_train])
    print(f"Source xBD train samples: {len(src_train)} | xBD val samples: {len(src_val)}", flush=True)
    print(f"Target pre-only samples: {len(tgt_pre_train)} train + {len(tgt_pre_val)} val", flush=True)
    print(f"Target test samples: {len(tgt_test)}", flush=True)
    print(f"Localization pos_weight: {loc_pos_weight.tolist()}", flush=True)
    print(f"Damage class weights: {dmg_class_weights.tolist()}", flush=True)

    model = ResNet34UNetSTCA().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    milestones = sorted(set(max(1, int(args.epochs * x)) for x in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    if USE_TORCH_AMP:
        scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    else:
        scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    loc_criterion = BCEDiceLoss(pos_weight=loc_pos_weight.to(device)).to(device)
    dmg_criterion = nn.CrossEntropyLoss(weight=dmg_class_weights.to(device), ignore_index=255).to(device)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float | int]] = []
    tgt_iter = cycle(tgt_pre_loader)
    steps_per_epoch = len(src_train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
        sup_meter = AverageMeter()
        loc_bce_meter = AverageMeter()
        loc_dice_meter = AverageMeter()
        dmg_ce_meter = AverageMeter()
        domain_src_meter = AverageMeter()
        domain_stca_meter = AverageMeter()
        total_meter = AverageMeter()

        use_tqdm = tqdm is not None and sys.stderr.isatty()
        iterator = tqdm(src_train_loader, desc=f"train {epoch}/{args.epochs}") if use_tqdm else src_train_loader

        for step, src_batch in enumerate(iterator, start=1):
            tgt_batch = next(tgt_iter)
            src_img = src_batch["img"].to(device, non_blocking=True)
            src_pre = src_batch["pre"].to(device, non_blocking=True)
            src_post = src_batch["post"].to(device, non_blocking=True)
            src_loc = src_batch["loc"].to(device, non_blocking=True)
            src_dmg = src_batch["dmg"].to(device, non_blocking=True)
            tgt_pre = tgt_batch["pre"].to(device, non_blocking=True)

            # STCA-style synthetic pair: target pre-disaster + source post-disaster.
            stca_img = torch.cat([tgt_pre, src_post], dim=1)
            grl_alpha = compute_domain_alpha(epoch, step, steps_per_epoch, args.epochs)
            optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    src_seg_logits, src_domain_logits = model(src_img, grl_alpha=grl_alpha)
                    _, stca_domain_logits = model(stca_img, grl_alpha=grl_alpha)
                    sup_total, loc_bce, loc_dice, dmg_ce = compute_supervised_losses(src_seg_logits, src_loc, src_dmg, loc_criterion, dmg_criterion, device, args)
                    src_domain_targets = torch.zeros((src_img.size(0), 1), device=device)
                    stca_domain_targets = torch.ones((stca_img.size(0), 1), device=device)
                    domain_src_loss = domain_criterion(src_domain_logits, src_domain_targets)
                    domain_stca_loss = domain_criterion(stca_domain_logits, stca_domain_targets)
                    total_loss = sup_total + args.domain_weight * (domain_src_loss + domain_stca_loss)
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    src_seg_logits, src_domain_logits = model(src_img, grl_alpha=grl_alpha)
                    _, stca_domain_logits = model(stca_img, grl_alpha=grl_alpha)
                    sup_total, loc_bce, loc_dice, dmg_ce = compute_supervised_losses(src_seg_logits, src_loc, src_dmg, loc_criterion, dmg_criterion, device, args)
                    src_domain_targets = torch.zeros((src_img.size(0), 1), device=device)
                    stca_domain_targets = torch.ones((stca_img.size(0), 1), device=device)
                    domain_src_loss = domain_criterion(src_domain_logits, src_domain_targets)
                    domain_stca_loss = domain_criterion(stca_domain_logits, stca_domain_targets)
                    total_loss = sup_total + args.domain_weight * (domain_src_loss + domain_stca_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = src_img.size(0)
            sup_meter.update(sup_total.item(), bs)
            loc_bce_meter.update(loc_bce.item(), bs)
            loc_dice_meter.update(loc_dice.item(), bs)
            dmg_ce_meter.update(dmg_ce.item(), bs)
            domain_src_meter.update(domain_src_loss.item(), bs)
            domain_stca_meter.update(domain_stca_loss.item(), bs)
            total_meter.update(total_loss.item(), bs)

            if use_tqdm:
                iterator.set_postfix(
                    loss=f"{total_meter.avg:.4f}",
                    sup=f"{sup_meter.avg:.4f}",
                    doms=f"{domain_src_meter.avg:.4f}",
                    domstca=f"{domain_stca_meter.avg:.4f}",
                )
            elif step % 20 == 0 or step == len(src_train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | Step {step}/{len(src_train_loader)} | "
                    f"loss={total_meter.avg:.4f} | sup={sup_meter.avg:.4f} | "
                    f"dom_src={domain_src_meter.avg:.4f} | dom_stca={domain_stca_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()
        val_metrics = evaluate_source_validation(model, src_val_loader, loc_criterion, dmg_criterion, device, args)
        val_score = val_metrics["loc_dice"] + val_metrics["dmg_macro_f1"]
        row: Dict[str, float | int] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_total_loss": total_meter.avg,
            "train_supervised_loss": sup_meter.avg,
            "train_loc_bce": loc_bce_meter.avg,
            "train_loc_dice_loss": loc_dice_meter.avg,
            "train_dmg_ce": dmg_ce_meter.avg,
            "train_domain_src": domain_src_meter.avg,
            "train_domain_stca": domain_stca_meter.avg,
            "src_val_loss": val_metrics["loss"],
            "src_val_loc_dice": val_metrics["loc_dice"],
            "src_val_dmg_acc": val_metrics["dmg_acc"],
            "src_val_dmg_macro_f1": val_metrics["dmg_macro_f1"],
            "src_val_score": val_score,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_total={row['train_total_loss']:.4f} | src_val_score={row['src_val_score']:.4f} | "
            f"src_val_loc_dice={row['src_val_loc_dice']:.4f} | src_val_dmg_macro_f1={row['src_val_dmg_macro_f1']:.4f}",
            flush=True,
        )

        improved = val_score > best_score
        if improved:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
            print(f"Saved new best checkpoint at epoch {epoch} with source-val score={best_score:.4f}", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s). Best epoch so far: {best_epoch} | best_score={best_score:.4f}", flush=True)

        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)

        row["best_score_so_far"] = best_score
        row["best_epoch_so_far"] = best_epoch
        row["epochs_without_improvement"] = epochs_without_improvement
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. No source-validation improvement for {args.early_stopping_patience} consecutive epochs.",
                flush=True,
            )
            break

    print(f"Evaluating best checkpoint on {args.target_dataset} test split...", flush=True)
    best_ckpt = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    target_results = evaluate_target_test_f1(model, tgt_test_loader, device, args.loc_threshold)
    print(json.dumps(target_results, indent=2), flush=True)
    write_target_test_outputs(target_results, output_dir, args.target_dataset)
    with open(output_dir / "target_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(target_results, f, indent=2)
    print("Done.", flush=True)
    print(f"Best epoch: {best_ckpt.get('best_epoch', 'unknown')}", flush=True)
    print(f"Best source-val score: {best_ckpt.get('best_score', 'unknown')}", flush=True)


if __name__ == "__main__":
    main()
