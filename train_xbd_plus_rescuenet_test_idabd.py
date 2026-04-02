from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
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


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_rgb_and_masks(
    image_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    image_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    out_imgs = []
    out_masks = []

    for img in image_list:
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(img)

    for m in mask_list:
        if m.shape[:2] != (image_size, image_size):
            m = cv2.resize(m, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        out_masks.append(m)

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


class BaseDamageDataset(Dataset):
    def __init__(self, image_size: int, training: bool, conditioning_id: int = 0) -> None:
        self.image_size = int(image_size)
        self.training = bool(training)
        self.conditioning_id = int(conditioning_id)
        self._mean = np.array([0.485, 0.456, 0.406] * 2, dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225] * 2, dtype=np.float32)[:, None, None]

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

    def _finalize_item(
        self,
        image_list: List[np.ndarray],
        loc: np.ndarray,
        dmg_target: np.ndarray,
        stem: str,
        source_name: str,
    ) -> Dict[str, torch.Tensor | str]:
        image_list, mask_list = resize_rgb_and_masks(image_list, [loc, dmg_target], self.image_size)
        image_list, mask_list = apply_shared_augmentations(image_list, mask_list, self.training)
        loc, dmg_target = mask_list

        loc = (loc > 0).astype(np.float32)

        img_cat = np.concatenate([x.astype(np.float32) / 255.0 for x in image_list], axis=2)
        img_cat = img_cat.transpose(2, 0, 1)
        img_cat = (img_cat - self._mean) / self._std

        return {
            "img": torch.from_numpy(img_cat).float(),
            "loc": torch.from_numpy(loc).float(),
            "dmg": torch.from_numpy(dmg_target).long(),
            "cond_id": torch.tensor([self.conditioning_id], dtype=torch.long),
            "stem": stem,
            "source_name": source_name,
        }


@dataclass(frozen=True)
class XBDSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    pre_target_path: Path
    post_target_path: Path


class XBDOriginalDataset(BaseDamageDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, conditioning_id: int = 0):
        super().__init__(image_size=image_size, training=training, conditioning_id=conditioning_id)
        self.root = Path(root)
        self.split = split
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

    def _collect_samples(self) -> List[XBDSample]:
        post_images = []
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
            stem = post_path.name
            if "_post_disaster" not in stem:
                continue

            prefix = stem.split("_post_disaster")[0]
            ext = post_path.suffix

            pre_path = self.images_dir / f"{prefix}_pre_disaster{ext}"
            pre_tgt = self.targets_dir / f"{prefix}_pre_disaster_target.png"
            post_tgt = self.targets_dir / f"{prefix}_post_disaster_target.png"

            if not pre_path.exists():
                continue
            if not pre_tgt.exists() or not post_tgt.exists():
                continue

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

    def __getitem__(self, index: int):
        s = self.samples[index]
        pre = self._read_rgb(s.pre_image_path)
        post = self._read_rgb(s.post_image_path)
        loc = self._read_mask(s.pre_target_path)
        dmg = self._read_mask(s.post_target_path)

        dmg_target = self._build_damage_target_from_standard_mask(loc, dmg)
        return self._finalize_item([pre, post], loc, dmg_target, s.stem, "xbd_style")

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


@dataclass(frozen=True)
class RescueSample:
    stem: str
    image_path: Path
    loc_path: Path
    dmg_path: Path


class RescueNetXBDDataset(BaseDamageDataset):
    """
    Supports either:
      split/images/... and split/masks/{localization,damage}/...
    or images directly under split/ with masks under split/masks/...
    """

    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, conditioning_id: int = 0):
        super().__init__(image_size=image_size, training=training, conditioning_id=conditioning_id)
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split
        self.loc_dir = self.split_root / "masks" / "localization"
        self.dmg_dir = self.split_root / "masks" / "damage"

        if not self.loc_dir.exists():
            raise FileNotFoundError(f"Expected localization dir not found: {self.loc_dir}")
        if not self.dmg_dir.exists():
            raise FileNotFoundError(f"Expected damage dir not found: {self.dmg_dir}")

        candidate_img_root = self.split_root / "images"
        self.image_root = candidate_img_root if candidate_img_root.exists() else self.split_root

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No RescueNet-xBD samples found under {self.split_root}")

    def _collect_samples(self) -> List[RescueSample]:
        image_files = []

        for p in self.image_root.rglob("*"):
            if not is_img(p):
                continue
            if "masks" in p.parts:
                continue
            image_files.append(p)

        image_files = sorted(image_files)
        samples: List[RescueSample] = []

        for image_path in image_files:
            rel = image_path.relative_to(self.image_root)
            loc_path = self.loc_dir / rel
            dmg_path = self.dmg_dir / rel

            if not loc_path.exists():
                stem_matches = list(self.loc_dir.rglob(image_path.name))
                if len(stem_matches) == 1:
                    loc_path = stem_matches[0]

            if not dmg_path.exists():
                stem_matches = list(self.dmg_dir.rglob(image_path.name))
                if len(stem_matches) == 1:
                    dmg_path = stem_matches[0]

            if not loc_path.exists() or not dmg_path.exists():
                continue

            samples.append(
                RescueSample(
                    stem=image_path.stem,
                    image_path=image_path,
                    loc_path=loc_path,
                    dmg_path=dmg_path,
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        image = self._read_rgb(s.image_path)
        loc = self._read_mask(s.loc_path)
        dmg = self._read_mask(s.dmg_path)

        dmg_target = self._build_damage_target_from_standard_mask(loc, dmg)
        return self._finalize_item([image, image.copy()], loc, dmg_target, s.stem, "rescuenet_xbd")

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
            target = self._build_damage_target_from_standard_mask(loc, dmg)
            valid = target != 255
            if valid.any():
                vals, freqs = np.unique(target[valid], return_counts=True)
                for v, f in zip(vals.tolist(), freqs.tolist()):
                    counts[int(v)] += int(f)
        return counts


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

    def as_dict(self):
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
    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + 1e-7) / (union + 1e-7)).mean()
        return bce, dice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train on xBD + RescueNet-xBD, test on IDA-BD")
    parser.add_argument("--xbd-root", type=str, default="/homes/j244s673/documents/wsu/phd/xview2")
    parser.add_argument("--rescuenet-root", type=str, default="/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd")
    parser.add_argument("--target-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet")
    parser.add_argument("--xbd-train-split", type=str, default="train")
    parser.add_argument("--xbd-val-split", type=str, default="hold")
    parser.add_argument("--rescuenet-train-split", type=str, default="train")
    parser.add_argument("--rescuenet-val-split", type=str, default="val")
    parser.add_argument("--target-test-split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="output/xbd_plus_rescuenet_to_idabd")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--conditioning-id", type=int, default=0)
    parser.add_argument("--loc-threshold", type=float, default=0.5)
    parser.add_argument("--loc-bce-weight", type=float, default=1.0)
    parser.add_argument("--loc-dice-weight", type=float, default=1.0)
    parser.add_argument("--dmg-ce-weight", type=float, default=1.0)
    return parser.parse_args()


def make_model(device: torch.device) -> nn.Module:
    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(OUT_CHANNELS=5),
        DATASET=SimpleNamespace(CONDITIONING_KEY={"generic": 0}),
    )
    model = DisasterAdaptiveNet(cfg)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    model.to(device)
    return model


def make_balanced_concat_loader(datasets: List[Dataset], batch_size: int, num_workers: int) -> DataLoader:
    concat_ds = ConcatDataset(datasets)
    weights = []
    for ds in datasets:
        w = 1.0 / max(len(ds), 1)
        weights.extend([w] * len(ds))
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
    return DataLoader(
        concat_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_eval_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


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


def compute_losses(
    logits: torch.Tensor,
    loc: torch.Tensor,
    dmg: torch.Tensor,
    loc_criterion: BCEDiceLoss,
    dmg_criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
):
    logit_loc = logits[:, 0]
    logit_dmg = logits[:, 1:5]

    loc_bce, loc_dice_loss = loc_criterion(logit_loc, loc)

    valid_dmg = dmg != 255
    if valid_dmg.any():
        dmg_ce = dmg_criterion(logit_dmg, dmg)
        has_valid_damage = True
    else:
        dmg_ce = torch.tensor(0.0, device=device, dtype=logit_loc.dtype)
        has_valid_damage = False

    loss = (
        args.loc_bce_weight * loc_bce
        + args.loc_dice_weight * loc_dice_loss
        + args.dmg_ce_weight * dmg_ce
    )
    return loss, loc_bce, loc_dice_loss, dmg_ce, has_valid_damage


@torch.no_grad()
def evaluate_source_validation(
    model: nn.Module,
    loader: DataLoader,
    loc_criterion: BCEDiceLoss,
    dmg_criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    loc_bce_meter = AverageMeter()
    loc_dice_loss_meter = AverageMeter()
    loc_dice_meter = AverageMeter()
    dmg_ce_meter = AverageMeter()
    dmg_acc_meter = AverageMeter()
    conf = RunningConfusionMatrix(num_classes=4)

    use_tqdm = tqdm is not None and sys.stderr.isatty()
    iterator = tqdm(loader, desc="val", leave=False) if use_tqdm else loader

    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc = batch["loc"].to(device, non_blocking=True)
        dmg = batch["dmg"].to(device, non_blocking=True)
        cond_id = batch["cond_id"].to(device, non_blocking=True)

        logits = model(img, cond_id)
        loss, loc_bce, loc_dice_loss, dmg_ce, _ = compute_losses(
            logits, loc, dmg, loc_criterion, dmg_criterion, device, args
        )

        logit_loc = logits[:, 0]
        logit_dmg = logits[:, 1:5]

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
def evaluate_target_test_f1(model: nn.Module, loader: DataLoader, device: torch.device, loc_threshold: float) -> Dict[str, object]:
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
        cond_id = batch["cond_id"].to(device, non_blocking=True)

        logits = model(img, cond_id)
        loc_logits = logits[:, 0]
        dmg_logits = logits[:, 1:5]

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

        for c in [1, 2, 3, 4]:
            tp = ((dp == c) & (dt == c)).sum()
            fp = ((dp == c) & (dt != c)).sum()
            fn = ((dp != c) & (dt == c)).sum()
            dmg_counts[c]["tp"] += int(tp.item())
            dmg_counts[c]["fp"] += int(fp.item())
            dmg_counts[c]["fn"] += int(fn.item())

    loc_f1 = F1Recorder(loc_tp, loc_fp, loc_fn, "localization")
    no_damage_f1 = F1Recorder(dmg_counts[1]["tp"], dmg_counts[1]["fp"], dmg_counts[1]["fn"], "no_damage")
    minor_damage_f1 = F1Recorder(dmg_counts[2]["tp"], dmg_counts[2]["fp"], dmg_counts[2]["fn"], "minor_damage")
    major_damage_f1 = F1Recorder(dmg_counts[3]["tp"], dmg_counts[3]["fp"], dmg_counts[3]["fn"], "major_damage")
    destroyed_f1 = F1Recorder(dmg_counts[4]["tp"], dmg_counts[4]["fp"], dmg_counts[4]["fn"], "destroyed")

    damage_f1s = [no_damage_f1.f1, minor_damage_f1.f1, major_damage_f1.f1, destroyed_f1.f1]
    damage_f1 = harmonic_mean(damage_f1s)
    score = 0.3 * loc_f1.f1 + 0.7 * damage_f1

    return {
        "score": score,
        "localization_f1": loc_f1.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no_damage_f1.f1,
        "damage_f1_minor_damage": minor_damage_f1.f1,
        "damage_f1_major_damage": major_damage_f1.f1,
        "damage_f1_destroyed": destroyed_f1.f1,
        "details": {
            "localization": loc_f1.as_dict(),
            "no_damage": no_damage_f1.as_dict(),
            "minor_damage": minor_damage_f1.as_dict(),
            "major_damage": major_damage_f1.as_dict(),
            "destroyed": destroyed_f1.as_dict(),
        },
    }


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "args": vars(args),
    }
    torch.save(state, save_path)


def write_target_test_outputs(results: Dict[str, object], output_dir: Path) -> None:
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    json_path = scores_dir / "scores_idabd_test.json"
    txt_path = scores_dir / "scores_idabd_test.txt"

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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    xbd_train = XBDOriginalDataset(args.xbd_root, args.xbd_train_split, args.img_size, True, args.conditioning_id)
    xbd_val = XBDOriginalDataset(args.xbd_root, args.xbd_val_split, args.img_size, False, args.conditioning_id)

    rescuenet_train = RescueNetXBDDataset(args.rescuenet_root, args.rescuenet_train_split, args.img_size, True, args.conditioning_id)
    rescuenet_val = RescueNetXBDDataset(args.rescuenet_root, args.rescuenet_val_split, args.img_size, False, args.conditioning_id)

    target_test = XBDOriginalDataset(args.target_root, args.target_test_split, args.img_size, False, args.conditioning_id)

    train_loader = make_balanced_concat_loader([xbd_train, rescuenet_train], args.batch_size, args.num_workers)
    val_loader = make_eval_loader(ConcatDataset([xbd_val, rescuenet_val]), args.batch_size, args.num_workers)
    target_test_loader = make_eval_loader(target_test, args.batch_size, args.num_workers)

    loc_pos_weight, dmg_class_weights = aggregate_counts([xbd_train, rescuenet_train])

    print(f"xBD train samples: {len(xbd_train)} | xBD val samples: {len(xbd_val)}", flush=True)
    print(f"RescueNet-xBD train samples: {len(rescuenet_train)} | RescueNet-xBD val samples: {len(rescuenet_val)}", flush=True)
    print(f"IDA-BD test samples: {len(target_test)}", flush=True)
    print(f"Localization pos_weight: {loc_pos_weight.tolist()}", flush=True)
    print(f"Damage class weights: {dmg_class_weights.tolist()}", flush=True)

    model = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    milestones = sorted(set(max(1, int(args.epochs * x)) for x in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    if USE_TORCH_AMP:
        scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    else:
        scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    loc_pos_weight = loc_pos_weight.to(device)
    dmg_class_weights = dmg_class_weights.to(device)
    loc_criterion = BCEDiceLoss(pos_weight=loc_pos_weight).to(device)
    dmg_criterion = nn.CrossEntropyLoss(weight=dmg_class_weights, ignore_index=255).to(device)

    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"Starting epoch {epoch}/{args.epochs}", flush=True)

        loss_meter = AverageMeter()
        loc_bce_meter = AverageMeter()
        loc_dice_loss_meter = AverageMeter()
        dmg_ce_meter = AverageMeter()

        use_tqdm = tqdm is not None and sys.stderr.isatty()
        iterator = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}") if use_tqdm else train_loader

        for step, batch in enumerate(iterator, start=1):
            img = batch["img"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            dmg = batch["dmg"].to(device, non_blocking=True)
            cond_id = batch["cond_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    logits = model(img, cond_id)
                    loss, loc_bce, loc_dice_loss, dmg_ce, has_valid_damage = compute_losses(
                        logits, loc, dmg, loc_criterion, dmg_criterion, device, args
                    )
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(img, cond_id)
                    loss, loc_bce, loc_dice_loss, dmg_ce, has_valid_damage = compute_losses(
                        logits, loc, dmg, loc_criterion, dmg_criterion, device, args
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = img.size(0)
            loss_meter.update(loss.item(), bs)
            loc_bce_meter.update(loc_bce.item(), bs)
            loc_dice_loss_meter.update(loc_dice_loss.item(), bs)
            dmg_ce_meter.update(dmg_ce.item() if has_valid_damage else 0.0, bs)

            if use_tqdm:
                iterator.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    loc_bce=f"{loc_bce_meter.avg:.4f}",
                    loc_dice=f"{loc_dice_loss_meter.avg:.4f}",
                    dmg_ce=f"{dmg_ce_meter.avg:.4f}",
                )
            elif step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | "
                    f"loss={loss_meter.avg:.4f} | loc_bce={loc_bce_meter.avg:.4f} | "
                    f"loc_dice={loc_dice_loss_meter.avg:.4f} | dmg_ce={dmg_ce_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()

        val_metrics = evaluate_source_validation(model, val_loader, loc_criterion, dmg_criterion, device, args)
        val_score = val_metrics["loc_dice"] + val_metrics["dmg_macro_f1"]

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_meter.avg,
            "train_loc_bce": loc_bce_meter.avg,
            "train_loc_dice_loss": loc_dice_loss_meter.avg,
            "train_dmg_ce": dmg_ce_meter.avg,
            "val_loss": val_metrics["loss"],
            "val_loc_dice": val_metrics["loc_dice"],
            "val_dmg_acc": val_metrics["dmg_acc"],
            "val_dmg_macro_f1": val_metrics["dmg_macro_f1"],
            "val_score": val_score,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_loc_dice={row['val_loc_dice']:.4f} | "
            f"val_dmg_acc={row['val_dmg_acc']:.4f} | val_dmg_macro_f1={row['val_dmg_macro_f1']:.4f}",
            flush=True,
        )

        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_score,
            args,
        )

        if val_score > best_score:
            best_score = val_score
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                args,
            )
            print(f"Saved new best checkpoint with source-val score={best_score:.4f}", flush=True)

        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                args,
            )

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Evaluating best checkpoint on IDA-BD test split...", flush=True)
    best_ckpt = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    target_results = evaluate_target_test_f1(model, target_test_loader, device, args.loc_threshold)
    print(json.dumps(target_results, indent=2), flush=True)

    write_target_test_outputs(target_results, output_dir)
    with open(output_dir / "target_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(target_results, f, indent=2)


if __name__ == "__main__":
    main()