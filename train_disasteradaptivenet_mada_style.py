from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from utils.models import DisasterAdaptiveNet

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
DATASET_NAMES = ("xbd", "ida", "ian", "rescuenet")


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_sources(s: str) -> List[str]:
    vals = [x.strip().lower() for x in s.split(",") if x.strip()]
    bad = [x for x in vals if x not in DATASET_NAMES]
    if bad:
        raise argparse.ArgumentTypeError(f"Invalid source names: {bad}")
    if len(set(vals)) != len(vals):
        raise argparse.ArgumentTypeError("Duplicate source names are not allowed")
    return vals


def resize_rgb_and_masks(images: List[np.ndarray], masks: List[np.ndarray], image_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    out_imgs, out_masks = [], []
    for img in images:
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(img)
    for mask in masks:
        if mask.shape[:2] != (image_size, image_size):
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        out_masks.append(mask)
    return out_imgs, out_masks


def resize_images_only(images: List[np.ndarray], image_size: int) -> List[np.ndarray]:
    out_imgs = []
    for img in images:
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(img)
    return out_imgs


def apply_shared_augmentations(images: List[np.ndarray], masks: List[np.ndarray], training: bool) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not training:
        return images, masks
    if np.random.rand() < 0.5:
        images = [np.flip(x, axis=1).copy() for x in images]
        masks = [np.flip(x, axis=1).copy() for x in masks]
    if np.random.rand() < 0.5:
        images = [np.flip(x, axis=0).copy() for x in images]
        masks = [np.flip(x, axis=0).copy() for x in masks]
    k = np.random.randint(0, 4)
    if k:
        images = [np.rot90(x, k=k).copy() for x in images]
        masks = [np.rot90(x, k=k).copy() for x in masks]
    return images, masks


def apply_image_augmentations(images: List[np.ndarray], training: bool) -> List[np.ndarray]:
    if not training:
        return images
    if np.random.rand() < 0.5:
        images = [np.flip(x, axis=1).copy() for x in images]
    if np.random.rand() < 0.5:
        images = [np.flip(x, axis=0).copy() for x in images]
    k = np.random.randint(0, 4)
    if k:
        images = [np.rot90(x, k=k).copy() for x in images]
    return images


class BaseDamageDataset(Dataset):
    def __init__(self, image_size: int, training: bool, conditioning_id: int = 0, source_name: str = "") -> None:
        self.image_size = int(image_size)
        self.training = bool(training)
        self.conditioning_id = int(conditioning_id)
        self.source_name = source_name
        self._mean = np.array([0.485, 0.456, 0.406] * 2, dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225] * 2, dtype=np.float32)[:, None, None]

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
    def _build_damage_target_from_standard_mask(loc: np.ndarray, dmg: np.ndarray) -> np.ndarray:
        loc_bin = loc > 0
        target = np.full(loc.shape, 255, dtype=np.uint8)
        target[(dmg == 1) & loc_bin] = 0
        target[(dmg == 2) & loc_bin] = 1
        target[(dmg == 3) & loc_bin] = 2
        target[(dmg == 4) & loc_bin] = 3
        return target

    def _normalize(self, images: List[np.ndarray]) -> np.ndarray:
        arr = np.concatenate([x.astype(np.float32) / 255.0 for x in images], axis=2)
        arr = arr.transpose(2, 0, 1)
        return (arr - self._mean) / self._std

    def _finalize_labeled(self, images: List[np.ndarray], loc: np.ndarray, dmg: np.ndarray, stem: str) -> Dict[str, torch.Tensor | str]:
        images, masks = resize_rgb_and_masks(images, [loc, dmg], self.image_size)
        images, masks = apply_shared_augmentations(images, masks, self.training)
        loc, dmg = masks
        loc = (loc > 0).astype(np.float32)
        arr = self._normalize(images)
        return {
            "img": torch.from_numpy(arr).float(),
            "loc": torch.from_numpy(loc).float(),
            "dmg": torch.from_numpy(dmg).long(),
            "cond_id": torch.tensor([self.conditioning_id], dtype=torch.long),
            "stem": stem,
            "source_name": self.source_name,
        }

    def _finalize_unlabeled(self, images: List[np.ndarray], stem: str) -> Dict[str, torch.Tensor | str]:
        images = resize_images_only(images, self.image_size)
        images = apply_image_augmentations(images, self.training)
        arr = self._normalize(images)
        return {
            "img": torch.from_numpy(arr).float(),
            "cond_id": torch.tensor([self.conditioning_id], dtype=torch.long),
            "stem": stem,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class XBDSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    pre_target_path: Optional[Path]
    post_target_path: Optional[Path]


class XBDLikeLabeledDataset(BaseDamageDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, conditioning_id: int = 0, source_name: str = ""):
        super().__init__(image_size, training, conditioning_id, source_name)
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        self.targets_dir = self.split_root / "targets"
        if not self.images_dir.exists() or not self.targets_dir.exists():
            raise FileNotFoundError(f"Expected xBD-like directories under {self.split_root}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No labeled xBD-like samples found under {self.split_root}")

    def _collect_samples(self) -> List[XBDSample]:
        posts: List[Path] = []
        for pattern in ["*_post_disaster.png", "*_post_disaster.jpg", "*_post_disaster.jpeg", "*_post_disaster.tif", "*_post_disaster.tiff", "*_post_disaster.bmp"]:
            posts.extend(self.images_dir.glob(pattern))
        posts = sorted(posts)
        samples: List[XBDSample] = []
        for post in posts:
            prefix = post.stem.replace("_post_disaster", "")
            pre = self.images_dir / f"{prefix}_pre_disaster{post.suffix}"
            pre_t = self.targets_dir / f"{prefix}_pre_disaster_target.png"
            post_t = self.targets_dir / f"{prefix}_post_disaster_target.png"
            if pre.exists() and pre_t.exists() and post_t.exists():
                samples.append(XBDSample(prefix, pre, post, pre_t, post_t))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        pre = self._read_rgb(s.pre_image_path)
        post = self._read_rgb(s.post_image_path)
        loc = self._read_mask(s.pre_target_path)
        dmg = self._read_mask(s.post_target_path)
        target = self._build_damage_target_from_standard_mask(loc, dmg)
        return self._finalize_labeled([pre, post], loc, target, s.stem)

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
            tgt = self._build_damage_target_from_standard_mask(loc, dmg)
            valid = tgt != 255
            if valid.any():
                vals, freqs = np.unique(tgt[valid], return_counts=True)
                for v, f in zip(vals.tolist(), freqs.tolist()):
                    counts[int(v)] += int(f)
        return counts


class XBDLikeUnlabeledDataset(BaseDamageDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, conditioning_id: int = 0, source_name: str = ""):
        super().__init__(image_size, training, conditioning_id, source_name)
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images dir under {self.split_root}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No unlabeled xBD-like pairs found under {self.split_root}")

    def _collect_samples(self) -> List[XBDSample]:
        posts: List[Path] = []
        for pattern in ["*_post_disaster.png", "*_post_disaster.jpg", "*_post_disaster.jpeg", "*_post_disaster.tif", "*_post_disaster.tiff", "*_post_disaster.bmp"]:
            posts.extend(self.images_dir.glob(pattern))
        posts = sorted(posts)
        samples: List[XBDSample] = []
        for post in posts:
            prefix = post.stem.replace("_post_disaster", "")
            pre = self.images_dir / f"{prefix}_pre_disaster{post.suffix}"
            if pre.exists():
                samples.append(XBDSample(prefix, pre, post, None, None))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        pre = self._read_rgb(s.pre_image_path)
        post = self._read_rgb(s.post_image_path)
        return self._finalize_unlabeled([pre, post], s.stem)


@dataclass(frozen=True)
class RescueSample:
    stem: str
    image_path: Path
    loc_path: Optional[Path]
    dmg_path: Optional[Path]


class RescueNetLabeledDataset(BaseDamageDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, conditioning_id: int = 0, source_name: str = ""):
        super().__init__(image_size, training, conditioning_id, source_name)
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        self.loc_dir = self.split_root / "masks" / "localization"
        self.dmg_dir = self.split_root / "masks" / "damage"
        if not self.images_dir.exists() or not self.loc_dir.exists() or not self.dmg_dir.exists():
            raise FileNotFoundError(f"Expected RescueNet directories under {self.split_root}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No labeled RescueNet samples found under {self.split_root}")

    def _collect_samples(self) -> List[RescueSample]:
        out = []
        for p in sorted([x for x in self.images_dir.iterdir() if is_img(x)], key=lambda x: x.stem):
            loc = self.loc_dir / f"{p.stem}.png"
            dmg = self.dmg_dir / f"{p.stem}.png"
            if loc.exists() and dmg.exists():
                out.append(RescueSample(p.stem, p, loc, dmg))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._read_rgb(s.image_path)
        loc = self._read_mask(s.loc_path)
        dmg = self._read_mask(s.dmg_path)
        target = self._build_damage_target_from_standard_mask(loc, dmg)
        return self._finalize_labeled([img, img.copy()], loc, target, s.stem)

    def get_localization_pixel_counts(self) -> Tuple[int, int]:
        pos, neg = 0, 0
        for s in self.samples:
            loc = self._read_mask(s.loc_path) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())
        return pos, neg

    def get_damage_class_counts(self) -> np.ndarray:
        counts = np.zeros(4, dtype=np.int64)
        for s in self.samples:
            loc = self._read_mask(s.loc_path)
            dmg = self._read_mask(s.dmg_path)
            tgt = self._build_damage_target_from_standard_mask(loc, dmg)
            valid = tgt != 255
            if valid.any():
                vals, freqs = np.unique(tgt[valid], return_counts=True)
                for v, f in zip(vals.tolist(), freqs.tolist()):
                    counts[int(v)] += int(f)
        return counts


class RescueNetUnlabeledDataset(BaseDamageDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, conditioning_id: int = 0, source_name: str = ""):
        super().__init__(image_size, training, conditioning_id, source_name)
        self.root = Path(root)
        self.images_dir = self.root / split / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images dir under {self.root / split}")
        self.samples = sorted([x for x in self.images_dir.iterdir() if is_img(x)], key=lambda x: x.stem)
        if not self.samples:
            raise RuntimeError(f"No unlabeled RescueNet images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        p = self.samples[idx]
        img = self._read_rgb(p)
        return self._finalize_unlabeled([img, img.copy()], p.stem)


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
        return 0.0 if p == 0.0 or r == 0.0 else 2.0 * p * r / (p + r)

    def as_dict(self):
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": self.precision, "recall": self.recall, "f1": self.f1}


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


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class ConditionalDomainClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_domains: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains),
        )

    def forward(self, feat: torch.Tensor, probs: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        pooled_feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        pooled_prob = F.adaptive_avg_pool2d(probs, 1).flatten(1)
        cond = torch.cat([pooled_feat, pooled_prob], dim=1)
        cond = grad_reverse(cond, grl_lambda)
        return self.net(cond)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Practical MADA-style training with DisasterAdaptiveNet")
    p.add_argument("--sources", type=parse_sources, required=True, help="Comma-separated source domains, e.g. xbd,ida or xbd,ida,ian")
    p.add_argument("--target", type=str, required=True, choices=list(DATASET_NAMES))
    p.add_argument("--xbd-root", type=str, default="/homes/j244s673/documents/wsu/phd/xview2")
    p.add_argument("--ida-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet")
    p.add_argument("--ian-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet")
    p.add_argument("--rescuenet-root", type=str, default="/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd")
    p.add_argument("--translated-xbd-root", type=str, default="")
    p.add_argument("--translated-ida-root", type=str, default="")
    p.add_argument("--translated-ian-root", type=str, default="")
    p.add_argument("--translated-rescuenet-root", type=str, default="")
    p.add_argument("--xbd-train-split", type=str, default="train")
    p.add_argument("--xbd-val-split", type=str, default="hold")
    p.add_argument("--xbd-test-split", type=str, default="test")
    p.add_argument("--ida-train-split", type=str, default="train")
    p.add_argument("--ida-val-split", type=str, default="val")
    p.add_argument("--ida-test-split", type=str, default="test")
    p.add_argument("--ian-train-split", type=str, default="train")
    p.add_argument("--ian-val-split", type=str, default="val")
    p.add_argument("--ian-test-split", type=str, default="test")
    p.add_argument("--rescuenet-train-split", type=str, default="train")
    p.add_argument("--rescuenet-val-split", type=str, default="val")
    p.add_argument("--rescuenet-test-split", type=str, default="test")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--conditioning-id", type=int, default=0)
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--loc-threshold", type=float, default=0.5)
    p.add_argument("--loc-bce-weight", type=float, default=1.0)
    p.add_argument("--loc-dice-weight", type=float, default=1.0)
    p.add_argument("--dmg-ce-weight", type=float, default=1.0)
    p.add_argument("--domain-weight", type=float, default=0.15)
    return p.parse_args()


def get_root(args: argparse.Namespace, name: str, translated: bool = False) -> str:
    mapping = {
        "xbd": args.translated_xbd_root if translated and args.translated_xbd_root else args.xbd_root,
        "ida": args.translated_ida_root if translated and args.translated_ida_root else args.ida_root,
        "ian": args.translated_ian_root if translated and args.translated_ian_root else args.ian_root,
        "rescuenet": args.translated_rescuenet_root if translated and args.translated_rescuenet_root else args.rescuenet_root,
    }
    return mapping[name]


def get_split(args: argparse.Namespace, name: str, which: str) -> str:
    return {
        "xbd": {"train": args.xbd_train_split, "val": args.xbd_val_split, "test": args.xbd_test_split},
        "ida": {"train": args.ida_train_split, "val": args.ida_val_split, "test": args.ida_test_split},
        "ian": {"train": args.ian_train_split, "val": args.ian_val_split, "test": args.ian_test_split},
        "rescuenet": {"train": args.rescuenet_train_split, "val": args.rescuenet_val_split, "test": args.rescuenet_test_split},
    }[name][which]


def is_rescuenet(name: str) -> bool:
    return name == "rescuenet"


def build_labeled_dataset(args: argparse.Namespace, name: str, split: str, training: bool, use_translated: bool) -> Dataset:
    root = get_root(args, name, translated=use_translated)
    if is_rescuenet(name):
        return RescueNetLabeledDataset(root, split, args.img_size, training, args.conditioning_id, name)
    return XBDLikeLabeledDataset(root, split, args.img_size, training, args.conditioning_id, name)


def build_unlabeled_dataset(args: argparse.Namespace, name: str, split: str, training: bool) -> Dataset:
    root = get_root(args, name, translated=False)
    if is_rescuenet(name):
        return RescueNetUnlabeledDataset(root, split, args.img_size, training, args.conditioning_id, name)
    return XBDLikeUnlabeledDataset(root, split, args.img_size, training, args.conditioning_id, name)


def make_model(device: torch.device) -> nn.Module:
    cfg = SimpleNamespace(MODEL=SimpleNamespace(OUT_CHANNELS=5), DATASET=SimpleNamespace(CONDITIONING_KEY={"generic": 0}))
    model = DisasterAdaptiveNet(cfg)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model.to(device)


def cycle_loader(loader: DataLoader) -> Iterable[dict]:
    while True:
        for batch in loader:
            yield batch


def compute_domain_lambda(epoch: int, step: int, steps_per_epoch: int, total_epochs: int) -> float:
    progress = ((epoch - 1) * steps_per_epoch + step) / max(1, total_epochs * steps_per_epoch)
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


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


def compute_supervised_losses(logits: torch.Tensor, loc: torch.Tensor, dmg: torch.Tensor, loc_criterion: BCEDiceLoss, dmg_criterion: nn.Module, device: torch.device, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logit_loc = logits[:, 0]
    logit_dmg = logits[:, 1:5]
    loc_bce, loc_dice = loc_criterion(logit_loc, loc)
    valid = dmg != 255
    if valid.any():
        dmg_ce = dmg_criterion(logit_dmg, dmg)
    else:
        dmg_ce = torch.tensor(0.0, device=device, dtype=logit_loc.dtype)
    total = args.loc_bce_weight * loc_bce + args.loc_dice_weight * loc_dice + args.dmg_ce_weight * dmg_ce
    return total, loc_bce, loc_dice, dmg_ce


@torch.no_grad()
def evaluate_source_validation(model: nn.Module, loader: DataLoader, loc_criterion: BCEDiceLoss, dmg_criterion: nn.Module, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    loss_meter, loc_bce_meter, loc_dice_meter, loc_dice_score_meter, dmg_ce_meter, dmg_acc_meter = [AverageMeter() for _ in range(6)]
    conf = RunningConfusionMatrix(4)
    iterator = tqdm(loader, desc="src_val", leave=False) if tqdm is not None and sys.stderr.isatty() else loader
    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc = batch["loc"].to(device, non_blocking=True)
        dmg = batch["dmg"].to(device, non_blocking=True)
        cond = batch["cond_id"].to(device, non_blocking=True)
        logits = model(img, cond)
        loss, loc_bce, loc_dice, dmg_ce = compute_supervised_losses(logits, loc, dmg, loc_criterion, dmg_criterion, device, args)
        loc_pred = (torch.sigmoid(logits[:, 0]) > args.loc_threshold).float()
        inter = (loc_pred * loc).sum(dim=(1, 2))
        union = loc_pred.sum(dim=(1, 2)) + loc.sum(dim=(1, 2))
        loc_dice_score = ((2.0 * inter + 1e-7) / (union + 1e-7)).mean().item()
        dmg_pred = torch.argmax(logits[:, 1:5], dim=1)
        valid = dmg != 255
        dmg_acc = (dmg_pred[valid] == dmg[valid]).float().mean().item() if valid.any() else 0.0
        if valid.any():
            conf.update(dmg[valid], dmg_pred[valid])
        bs = img.size(0)
        loss_meter.update(loss.item(), bs)
        loc_bce_meter.update(loc_bce.item(), bs)
        loc_dice_meter.update(loc_dice.item(), bs)
        loc_dice_score_meter.update(loc_dice_score, bs)
        dmg_ce_meter.update(dmg_ce.item(), bs)
        dmg_acc_meter.update(dmg_acc, bs)
    return {"loss": loss_meter.avg, "loc_bce": loc_bce_meter.avg, "loc_dice_loss": loc_dice_meter.avg, "loc_dice": loc_dice_score_meter.avg, "dmg_ce": dmg_ce_meter.avg, "dmg_acc": dmg_acc_meter.avg, "dmg_macro_f1": conf.macro_f1()}


@torch.no_grad()
def evaluate_target_test_f1(model: nn.Module, loader: DataLoader, device: torch.device, loc_threshold: float) -> Dict[str, object]:
    model.eval()
    loc_tp, loc_fp, loc_fn = 0, 0, 0
    dmg_counts = {1: {"tp": 0, "fp": 0, "fn": 0}, 2: {"tp": 0, "fp": 0, "fn": 0}, 3: {"tp": 0, "fp": 0, "fn": 0}, 4: {"tp": 0, "fp": 0, "fn": 0}}
    iterator = tqdm(loader, desc="target_test", leave=False) if tqdm is not None and sys.stderr.isatty() else loader
    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        dmg_true_raw = batch["dmg"].to(device, non_blocking=True).long()
        cond = batch["cond_id"].to(device, non_blocking=True)
        logits = model(img, cond)
        loc_pred = (torch.sigmoid(logits[:, 0]) > loc_threshold).long()
        dmg_pred = (torch.argmax(logits[:, 1:5], dim=1) + 1) * loc_pred
        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())
        valid_gt = (loc_true == 1) & (dmg_true_raw != 255)
        dmg_true = torch.zeros_like(dmg_true_raw)
        dmg_true[valid_gt] = dmg_true_raw[valid_gt] + 1
        dp, dt = dmg_pred[valid_gt], dmg_true[valid_gt]
        for cls in [1,2,3,4]:
            dmg_counts[cls]["tp"] += int(((dp == cls) & (dt == cls)).sum().item())
            dmg_counts[cls]["fp"] += int(((dp == cls) & (dt != cls)).sum().item())
            dmg_counts[cls]["fn"] += int(((dp != cls) & (dt == cls)).sum().item())
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
        "details": {"localization": loc_f1.as_dict(), "no_damage": no_damage.as_dict(), "minor_damage": minor.as_dict(), "major_damage": major.as_dict(), "destroyed": destroyed.as_dict()},
    }


def save_checkpoint(path: Path, model: nn.Module, domain_disc: nn.Module, optimizer: torch.optim.Optimizer, scheduler, scaler: GradScaler, epoch: int, best_score: float, best_epoch: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "domain_disc": domain_disc.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict() if scaler else None, "best_score": best_score, "best_epoch": best_epoch, "args": vars(args)}, path)


def write_scores(results: Dict[str, object], output_dir: Path, target_name: str) -> None:
    sdir = output_dir / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    jpath = sdir / f"scores_{target_name}_test.json"
    tpath = sdir / f"scores_{target_name}_test.txt"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(f"Localization F1: {results['localization_f1']:.6f}\n")
        f.write(f"No Damage F1:    {results['damage_f1_no_damage']:.6f}\n")
        f.write(f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}\n")
        f.write(f"Major Damage F1: {results['damage_f1_major_damage']:.6f}\n")
        f.write(f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}\n")
        f.write(f"Damage F1:       {results['damage_f1']:.6f}\n")
        f.write(f"Overall Score:   {results['score']:.6f}\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.target in args.sources:
        raise ValueError("Target cannot also be listed as a source")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    with open(output_dir / "mada_config.json", "w", encoding="utf-8") as f:
        json.dump({"sources": args.sources, "target": args.target, **vars(args)}, f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Sources: {args.sources}", flush=True)
    print(f"Target: {args.target}", flush=True)

    src_train_datasets = [build_labeled_dataset(args, s, get_split(args, s, "train"), True, use_translated=True) for s in args.sources]
    src_val_datasets = [build_labeled_dataset(args, s, get_split(args, s, "val"), False, use_translated=True) for s in args.sources]
    src_names = list(args.sources)
    num_domains = len(src_names) + 1  # + target
    domain_label_of = {name: i for i, name in enumerate(src_names)}
    target_domain_label = len(src_names)

    target_unlabeled_parts: List[Dataset] = []
    seen = set()
    for split in [get_split(args, args.target, "train"), get_split(args, args.target, "val")]:
        if split not in seen:
            target_unlabeled_parts.append(build_unlabeled_dataset(args, args.target, split, True))
            seen.add(split)
    target_test_dataset = build_labeled_dataset(args, args.target, get_split(args, args.target, "test"), False, use_translated=False)

    concat_train = ConcatDataset(src_train_datasets)
    weights = []
    for ds in src_train_datasets:
        w = 1.0 / max(1, len(ds))
        weights.extend([w] * len(ds))
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
    src_train_loader = DataLoader(concat_train, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    src_val_loader = DataLoader(ConcatDataset(src_val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    tgt_u_loader = DataLoader(ConcatDataset(target_unlabeled_parts), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tgt_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    loc_pos_weight, dmg_class_weights = aggregate_counts(src_train_datasets)
    print(f"Localization pos_weight: {loc_pos_weight.tolist()}", flush=True)
    print(f"Damage class weights: {dmg_class_weights.tolist()}", flush=True)
    for name, ds in zip(src_names, src_train_datasets):
        print(f"Source train [{name}]: {len(ds)}", flush=True)
    print(f"Target unlabeled total [{args.target}]: {sum(len(x) for x in target_unlabeled_parts)}", flush=True)
    print(f"Target test [{args.target}]: {len(target_test_dataset)}", flush=True)

    model = make_model(device)
    # pooled logits (5) + pooled probs (5)
    domain_disc = ConditionalDomainClassifier(feature_dim=10, num_domains=num_domains).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(domain_disc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    milestones = sorted(set(max(1, int(args.epochs * x)) for x in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    if USE_TORCH_AMP:
        scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    else:
        scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    loc_criterion = BCEDiceLoss(pos_weight=loc_pos_weight.to(device)).to(device)
    dmg_criterion = nn.CrossEntropyLoss(weight=dmg_class_weights.to(device), ignore_index=255).to(device)
    domain_criterion = nn.CrossEntropyLoss().to(device)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float | int]] = []
    tgt_iter = cycle_loader(tgt_u_loader)
    steps_per_epoch = len(src_train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        domain_disc.train()
        print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
        sup_meter, dom_meter, dom_acc_meter, total_meter = [AverageMeter() for _ in range(4)]
        iterator = tqdm(src_train_loader, desc=f"train {epoch}/{args.epochs}") if tqdm is not None and sys.stderr.isatty() else src_train_loader
        for step, src_batch in enumerate(iterator, start=1):
            tgt_batch = next(tgt_iter)
            src_img = src_batch["img"].to(device, non_blocking=True)
            src_loc = src_batch["loc"].to(device, non_blocking=True)
            src_dmg = src_batch["dmg"].to(device, non_blocking=True)
            src_cond = src_batch["cond_id"].to(device, non_blocking=True)
            src_domain = torch.tensor([domain_label_of[x] for x in src_batch["source_name"]], device=device, dtype=torch.long)

            tgt_img = tgt_batch["img"].to(device, non_blocking=True)
            tgt_cond = tgt_batch["cond_id"].to(device, non_blocking=True)
            tgt_domain = torch.full((tgt_img.size(0),), target_domain_label, device=device, dtype=torch.long)

            grl_lambda = compute_domain_lambda(epoch, step, steps_per_epoch, args.epochs)
            optimizer.zero_grad(set_to_none=True)
            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    src_logits = model(src_img, src_cond)
                    tgt_logits = model(tgt_img, tgt_cond)
                    sup_total, _, _, _ = compute_supervised_losses(src_logits, src_loc, src_dmg, loc_criterion, dmg_criterion, device, args)

                    src_probs = torch.softmax(src_logits, dim=1)
                    tgt_probs = torch.softmax(tgt_logits, dim=1)
                    src_dom_logits = domain_disc(src_logits, src_probs, grl_lambda)
                    tgt_dom_logits = domain_disc(tgt_logits, tgt_probs, grl_lambda)
                    dom_src_loss = domain_criterion(src_dom_logits, src_domain)
                    dom_tgt_loss = domain_criterion(tgt_dom_logits, tgt_domain)
                    dom_loss = 0.5 * (dom_src_loss + dom_tgt_loss)
                    total_loss = sup_total + args.domain_weight * grl_lambda * dom_loss
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    src_logits = model(src_img, src_cond)
                    tgt_logits = model(tgt_img, tgt_cond)
                    sup_total, _, _, _ = compute_supervised_losses(src_logits, src_loc, src_dmg, loc_criterion, dmg_criterion, device, args)
                    src_probs = torch.softmax(src_logits, dim=1)
                    tgt_probs = torch.softmax(tgt_logits, dim=1)
                    src_dom_logits = domain_disc(src_logits, src_probs, grl_lambda)
                    tgt_dom_logits = domain_disc(tgt_logits, tgt_probs, grl_lambda)
                    dom_src_loss = domain_criterion(src_dom_logits, src_domain)
                    dom_tgt_loss = domain_criterion(tgt_dom_logits, tgt_domain)
                    dom_loss = 0.5 * (dom_src_loss + dom_tgt_loss)
                    total_loss = sup_total + args.domain_weight * grl_lambda * dom_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                dom_pred = torch.cat([torch.argmax(src_dom_logits, dim=1), torch.argmax(tgt_dom_logits, dim=1)])
                dom_true = torch.cat([src_domain, tgt_domain])
                dom_acc = (dom_pred == dom_true).float().mean().item()
            bs = src_img.size(0)
            sup_meter.update(sup_total.item(), bs)
            dom_meter.update(dom_loss.item(), bs)
            dom_acc_meter.update(dom_acc, bs)
            total_meter.update(total_loss.item(), bs)
            if tqdm is not None and sys.stderr.isatty():
                iterator.set_postfix(loss=f"{total_meter.avg:.4f}", sup=f"{sup_meter.avg:.4f}", dom=f"{dom_meter.avg:.4f}", dacc=f"{dom_acc_meter.avg:.4f}", grl=f"{grl_lambda:.3f}")
            elif step % 20 == 0 or step == steps_per_epoch:
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{steps_per_epoch} | loss={total_meter.avg:.4f} | sup={sup_meter.avg:.4f} | dom={dom_meter.avg:.4f} | dacc={dom_acc_meter.avg:.4f} | grl={grl_lambda:.3f}", flush=True)

        scheduler.step()
        val_metrics = evaluate_source_validation(model, src_val_loader, loc_criterion, dmg_criterion, device, args)
        val_score = val_metrics["loc_dice"] + val_metrics["dmg_macro_f1"]
        row: Dict[str, float | int] = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_total_loss": total_meter.avg, "train_supervised_loss": sup_meter.avg, "train_domain_loss": dom_meter.avg, "train_domain_acc": dom_acc_meter.avg, "src_val_loss": val_metrics["loss"], "src_val_loc_dice": val_metrics["loc_dice"], "src_val_dmg_acc": val_metrics["dmg_acc"], "src_val_dmg_macro_f1": val_metrics["dmg_macro_f1"], "src_val_score": val_score}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_total={row['train_total_loss']:.4f} | train_domain={row['train_domain_loss']:.4f} | src_val_score={row['src_val_score']:.4f}", flush=True)
        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
            print(f"Saved new best checkpoint at epoch {epoch} with source-val score={best_score:.4f}", flush=True)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s). Best epoch so far: {best_epoch} | best_score={best_score:.4f}", flush=True)

        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
        row["best_score_so_far"] = best_score
        row["best_epoch_so_far"] = best_epoch
        row["epochs_without_improvement"] = epochs_without_improvement
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}. No source-validation improvement for {args.early_stopping_patience} consecutive epochs.", flush=True)
            break

    print(f"Evaluating best checkpoint on target test split: {args.target}", flush=True)
    ckpt = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    target_results = evaluate_target_test_f1(model, tgt_test_loader, device, args.loc_threshold)
    print(json.dumps(target_results, indent=2), flush=True)
    write_scores(target_results, output_dir, args.target)
    with open(output_dir / "target_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(target_results, f, indent=2)
    print("Done.", flush=True)
    print(f"Best epoch: {ckpt.get('best_epoch', 'unknown')}", flush=True)
    print(f"Best source-val score: {ckpt.get('best_score', 'unknown')}", flush=True)


if __name__ == "__main__":
    main()
