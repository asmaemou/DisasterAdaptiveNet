#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make sure imports work when this script is called from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
        """
        Creates 4-class damage target:
          0 = no damage
          1 = minor damage
          2 = major damage
          3 = destroyed
          255 = ignore / non-building
        """
        loc_bin = loc > 0
        target = np.full(loc.shape, 255, dtype=np.uint8)

        target[(dmg == 1) & loc_bin] = 0
        target[(dmg == 2) & loc_bin] = 1
        target[(dmg == 3) & loc_bin] = 2
        target[(dmg == 4) & loc_bin] = 3

        return target

    def _normalize_images(self, image_list: List[np.ndarray]) -> np.ndarray:
        img_cat = np.concatenate([x.astype(np.float32) / 255.0 for x in image_list], axis=2)
        img_cat = img_cat.transpose(2, 0, 1)
        img_cat = (img_cat - self._mean) / self._std
        return img_cat

    def _finalize_item(
        self,
        image_list: List[np.ndarray],
        loc: np.ndarray,
        dmg_target: np.ndarray,
        stem: str,
        split_name: str,
    ) -> Dict[str, torch.Tensor | str]:
        image_list, mask_list = resize_rgb_and_masks(
            image_list=image_list,
            mask_list=[loc, dmg_target],
            image_size=self.image_size,
        )

        image_list, mask_list = apply_shared_augmentations(
            image_list=image_list,
            mask_list=mask_list,
            training=self.training,
        )

        loc, dmg_target = mask_list
        loc = (loc > 0).astype(np.float32)

        img_cat = self._normalize_images(image_list)

        return {
            "img": torch.from_numpy(img_cat).float(),
            "loc": torch.from_numpy(loc).float(),
            "dmg": torch.from_numpy(dmg_target).long(),
            "cond_id": torch.tensor([self.conditioning_id], dtype=torch.long),
            "stem": stem,
            "split_name": split_name,
        }


@dataclass(frozen=True)
class XBDSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    pre_target_path: Path
    post_target_path: Path


class XBDOriginalDataset(BaseDamageDataset):
    """
    Expects this structure:

      root/
        tier3/
          images/
          targets/
        hold/
          images/
          targets/
        test/
          images/
          targets/

    File pattern:
      *_pre_disaster.png
      *_post_disaster.png
      *_pre_disaster_target.png
      *_post_disaster_target.png
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        training: bool,
        conditioning_id: int = 0,
    ):
        super().__init__(
            image_size=image_size,
            training=training,
            conditioning_id=conditioning_id,
        )

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
            if "_post_disaster" not in post_path.name:
                continue

            prefix = post_path.stem.replace("_post_disaster", "")
            ext = post_path.suffix

            pre_path = self.images_dir / f"{prefix}_pre_disaster{ext}"
            pre_tgt = self.targets_dir / f"{prefix}_pre_disaster_target.png"
            post_tgt = self.targets_dir / f"{prefix}_post_disaster_target.png"

            if not pre_path.exists():
                continue
            if not pre_tgt.exists():
                continue
            if not post_tgt.exists():
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
        sample = self.samples[index]

        pre = self._read_rgb(sample.pre_image_path)
        post = self._read_rgb(sample.post_image_path)

        loc = self._read_mask(sample.pre_target_path)
        dmg = self._read_mask(sample.post_target_path)

        dmg_target = self._build_damage_target_from_standard_mask(loc, dmg)

        return self._finalize_item(
            image_list=[pre, post],
            loc=loc,
            dmg_target=dmg_target,
            stem=sample.stem,
            split_name=self.split,
        )

    def get_localization_pixel_counts(self) -> Tuple[int, int]:
        pos = 0
        neg = 0

        for sample in self.samples:
            loc = self._read_mask(sample.pre_target_path) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())

        return pos, neg

    def get_damage_class_counts(self) -> np.ndarray:
        counts = np.zeros(4, dtype=np.int64)

        for sample in self.samples:
            loc = self._read_mask(sample.pre_target_path)
            dmg = self._read_mask(sample.post_target_path)
            target = self._build_damage_target_from_standard_mask(loc, dmg)

            valid = target != 255

            if valid.any():
                vals, freqs = np.unique(target[valid], return_counts=True)
                for value, freq in zip(vals.tolist(), freqs.tolist()):
                    counts[int(value)] += int(freq)

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


def harmonic_mean(values: List[float]) -> float:
    values = [float(x) for x in values]
    return len(values) / sum((x + 1e-6) ** -1 for x in values)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

        dice_loss = 1.0 - ((2.0 * intersection + 1e-7) / (union + 1e-7)).mean()

        return bce, dice_loss


def make_model(device: torch.device) -> nn.Module:
    cfg = argparse.Namespace()
    cfg.MODEL = argparse.Namespace(OUT_CHANNELS=5)
    cfg.DATASET = argparse.Namespace(CONDITIONING_KEY={"generic": 0})

    model = DisasterAdaptiveNet(cfg)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    model.to(device)
    return model


def make_train_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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


def aggregate_counts(dataset: XBDOriginalDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    loc_pos, loc_neg = dataset.get_localization_pixel_counts()

    loc_pos_weight = torch.tensor(
        [max(1.0, loc_neg / max(loc_pos, 1))],
        dtype=torch.float32,
    )

    damage_counts = dataset.get_damage_class_counts().astype(np.float64)
    damage_counts[damage_counts == 0] = 1.0

    inv = damage_counts.sum() / damage_counts
    damage_class_weights = torch.tensor(
        inv / inv.sum() * len(inv),
        dtype=torch.float32,
    )

    return loc_pos_weight, damage_class_weights


def compute_supervised_losses(
    logits: torch.Tensor,
    loc: torch.Tensor,
    dmg: torch.Tensor,
    loc_criterion: BCEDiceLoss,
    dmg_criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loc_logits = logits[:, 0]
    dmg_logits = logits[:, 1:5]

    loc_bce, loc_dice_loss = loc_criterion(loc_logits, loc)

    valid_damage = dmg != 255

    if valid_damage.any():
        dmg_ce = dmg_criterion(dmg_logits, dmg)
    else:
        dmg_ce = torch.tensor(0.0, device=device, dtype=loc_logits.dtype)

    total = (
        args.loc_bce_weight * loc_bce
        + args.loc_dice_weight * loc_dice_loss
        + args.dmg_ce_weight * dmg_ce
    )

    return total, loc_bce, loc_dice_loss, dmg_ce


@torch.no_grad()
def evaluate_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loc_threshold: float,
) -> Dict[str, object]:
    model.eval()

    loc_tp, loc_fp, loc_fn = 0, 0, 0

    damage_counts = {
        1: {"tp": 0, "fp": 0, "fn": 0},
        2: {"tp": 0, "fp": 0, "fn": 0},
        3: {"tp": 0, "fp": 0, "fn": 0},
        4: {"tp": 0, "fp": 0, "fn": 0},
    }

    iterator = tqdm(loader, desc="eval", leave=False) if (tqdm is not None and sys.stderr.isatty()) else loader

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

        pred_valid = dmg_pred[valid_gt]
        true_valid = dmg_true[valid_gt]

        for cls in [1, 2, 3, 4]:
            tp = ((pred_valid == cls) & (true_valid == cls)).sum()
            fp = ((pred_valid == cls) & (true_valid != cls)).sum()
            fn = ((pred_valid != cls) & (true_valid == cls)).sum()

            damage_counts[cls]["tp"] += int(tp.item())
            damage_counts[cls]["fp"] += int(fp.item())
            damage_counts[cls]["fn"] += int(fn.item())

    loc_f1 = F1Recorder(loc_tp, loc_fp, loc_fn)
    no_damage_f1 = F1Recorder(damage_counts[1]["tp"], damage_counts[1]["fp"], damage_counts[1]["fn"])
    minor_damage_f1 = F1Recorder(damage_counts[2]["tp"], damage_counts[2]["fp"], damage_counts[2]["fn"])
    major_damage_f1 = F1Recorder(damage_counts[3]["tp"], damage_counts[3]["fp"], damage_counts[3]["fn"])
    destroyed_f1 = F1Recorder(damage_counts[4]["tp"], damage_counts[4]["fp"], damage_counts[4]["fn"])

    damage_f1 = harmonic_mean([
        no_damage_f1.f1,
        minor_damage_f1.f1,
        major_damage_f1.f1,
        destroyed_f1.f1,
    ])

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


@torch.no_grad()
def scan_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: List[float],
    output_csv: Path,
) -> Tuple[float, Dict[str, object]]:
    best_threshold = thresholds[0]
    best_results: Dict[str, object] = {}
    best_score = -1.0

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
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
        ])

        for threshold in thresholds:
            results = evaluate_f1(model, loader, device, threshold)
            score = float(results["score"])

            writer.writerow([
                threshold,
                results["score"],
                results["localization_f1"],
                results["damage_f1"],
                results["damage_f1_no_damage"],
                results["damage_f1_minor_damage"],
                results["damage_f1_major_damage"],
                results["damage_f1_destroyed"],
            ])

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_results = results

    return best_threshold, best_results


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    epoch: int,
    best_score: float,
    best_threshold: float,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    torch.save({
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "best_threshold": best_threshold,
        "args": vars(args),
    }, save_path)


def load_model_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    return checkpoint


def write_final_outputs(results: Dict[str, object], output_dir: Path, best_epoch: int, best_threshold: float) -> None:
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    json_path = scores_dir / "scores_xbd_test.json"
    txt_path = scores_dir / "scores_xbd_test.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Best epoch selected on hold: {best_epoch}\n")
        f.write(f"Best localization threshold selected on hold: {best_threshold:.4f}\n")
        f.write(f"Localization F1: {results['localization_f1']:.6f}\n")
        f.write(f"No Damage F1:    {results['damage_f1_no_damage']:.6f}\n")
        f.write(f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}\n")
        f.write(f"Major Damage F1: {results['damage_f1_major_damage']:.6f}\n")
        f.write(f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}\n")
        f.write(f"Damage F1:       {results['damage_f1']:.6f}\n")
        f.write(f"Overall Score:   {results['score']:.6f}\n")

    summary_path = scores_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Experiment: supervised DisasterAdaptiveNet xBD tier3 -> hold -> test\n")
        f.write(f"Best epoch selected on hold: {best_epoch}\n")
        f.write(f"Best localization threshold selected on hold: {best_threshold:.4f}\n")
        f.write(f"Localization F1: {results['localization_f1']:.6f}\n")
        f.write(f"No Damage F1:    {results['damage_f1_no_damage']:.6f}\n")
        f.write(f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}\n")
        f.write(f"Major Damage F1: {results['damage_f1_major_damage']:.6f}\n")
        f.write(f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}\n")
        f.write(f"Damage F1:       {results['damage_f1']:.6f}\n")
        f.write(f"Overall Score:   {results['score']:.6f}\n")

    print(f"Wrote final JSON: {json_path}", flush=True)
    print(f"Wrote final TXT:  {txt_path}", flush=True)
    print(f"Wrote summary:    {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Supervised DisasterAdaptiveNet on xBD tier3/hold/test")

    parser.add_argument("--xbd-root", type=str, required=True)
    parser.add_argument("--train-split", type=str, default="tier3")
    parser.add_argument("--val-split", type=str, default="hold")
    parser.add_argument("--test-split", type=str, default="test")

    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--img-size", type=int, required=True)

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--conditioning-id", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    parser.add_argument("--loc-bce-weight", type=float, default=1.0)
    parser.add_argument("--loc-dice-weight", type=float, default=1.0)
    parser.add_argument("--dmg-ce-weight", type=float, default=1.0)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        help="Localization thresholds scanned on hold. Best one is used on test.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    scores_dir = output_dir / "scores"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== SUPERVISED DISASTERADAPTIVENET TRAINING =====", flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"xBD root:     {args.xbd_root}", flush=True)
    print(f"Train split:  {args.train_split}", flush=True)
    print(f"Val split:    {args.val_split}", flush=True)
    print(f"Test split:   {args.test_split}", flush=True)
    print(f"Output dir:   {output_dir}", flush=True)
    print("No domain adaptation is used.", flush=True)
    print("No target/unlabeled dataset is used.", flush=True)
    print("Model: DisasterAdaptiveNet", flush=True)
    print("===================================================", flush=True)

    train_ds = XBDOriginalDataset(
        root=args.xbd_root,
        split=args.train_split,
        image_size=args.img_size,
        training=True,
        conditioning_id=args.conditioning_id,
    )

    val_ds = XBDOriginalDataset(
        root=args.xbd_root,
        split=args.val_split,
        image_size=args.img_size,
        training=False,
        conditioning_id=args.conditioning_id,
    )

    test_ds = XBDOriginalDataset(
        root=args.xbd_root,
        split=args.test_split,
        image_size=args.img_size,
        training=False,
        conditioning_id=args.conditioning_id,
    )

    train_loader = make_train_loader(train_ds, args.batch_size, args.num_workers)
    val_loader = make_eval_loader(val_ds, args.batch_size, args.num_workers)
    test_loader = make_eval_loader(test_ds, args.batch_size, args.num_workers)

    print(f"Train samples: {len(train_ds)}", flush=True)
    print(f"Val samples:   {len(val_ds)}", flush=True)
    print(f"Test samples:  {len(test_ds)}", flush=True)

    loc_pos_weight, dmg_class_weights = aggregate_counts(train_ds)

    model = make_model(device)

    print("===== MODEL =====", flush=True)
    print(type(model), flush=True)
    print(model, flush=True)
    print("=================", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    milestones = sorted(set(max(1, int(args.epochs * x)) for x in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.5,
    )

    if USE_TORCH_AMP:
        scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    else:
        scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    loc_pos_weight = loc_pos_weight.to(device)
    dmg_class_weights = dmg_class_weights.to(device)

    loc_criterion = BCEDiceLoss(pos_weight=loc_pos_weight).to(device)
    dmg_criterion = nn.CrossEntropyLoss(
        weight=dmg_class_weights,
        ignore_index=255,
    ).to(device)

    print(f"Localization pos_weight: {loc_pos_weight.detach().cpu().numpy().tolist()}", flush=True)
    print(f"Damage class weights:    {dmg_class_weights.detach().cpu().numpy().tolist()}", flush=True)

    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: List[Dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        print(f"\nStarting epoch {epoch}/{args.epochs}", flush=True)

        total_meter = AverageMeter()
        loc_bce_meter = AverageMeter()
        loc_dice_meter = AverageMeter()
        dmg_ce_meter = AverageMeter()

        iterator = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}") if (tqdm is not None and sys.stderr.isatty()) else train_loader

        for step, batch in enumerate(iterator, start=1):
            img = batch["img"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            dmg = batch["dmg"].to(device, non_blocking=True)
            cond_id = batch["cond_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    logits = model(img, cond_id)
                    total_loss, loc_bce, loc_dice_loss, dmg_ce = compute_supervised_losses(
                        logits=logits,
                        loc=loc,
                        dmg=dmg,
                        loc_criterion=loc_criterion,
                        dmg_criterion=dmg_criterion,
                        device=device,
                        args=args,
                    )
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(img, cond_id)
                    total_loss, loc_bce, loc_dice_loss, dmg_ce = compute_supervised_losses(
                        logits=logits,
                        loc=loc,
                        dmg=dmg,
                        loc_criterion=loc_criterion,
                        dmg_criterion=dmg_criterion,
                        device=device,
                        args=args,
                    )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = img.size(0)

            total_meter.update(total_loss.item(), batch_size)
            loc_bce_meter.update(loc_bce.item(), batch_size)
            loc_dice_meter.update(loc_dice_loss.item(), batch_size)
            dmg_ce_meter.update(dmg_ce.item(), batch_size)

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"loss={total_meter.avg:.4f} | "
                    f"loc_bce={loc_bce_meter.avg:.4f} | "
                    f"loc_dice_loss={loc_dice_meter.avg:.4f} | "
                    f"dmg_ce={dmg_ce_meter.avg:.4f}",
                    flush=True,
                )

        scheduler.step()

        threshold_scan_csv = scores_dir / f"epoch_{epoch:03d}_hold_threshold_scan.csv"

        epoch_best_threshold, val_results = scan_thresholds(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=args.thresholds,
            output_csv=threshold_scan_csv,
        )

        val_score = float(val_results["score"])

        row: Dict[str, float | int] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_meter.avg,
            "train_loc_bce": loc_bce_meter.avg,
            "train_loc_dice_loss": loc_dice_meter.avg,
            "train_dmg_ce": dmg_ce_meter.avg,
            "hold_best_threshold": epoch_best_threshold,
            "hold_score": val_score,
            "hold_localization_f1": float(val_results["localization_f1"]),
            "hold_no_damage_f1": float(val_results["damage_f1_no_damage"]),
            "hold_minor_damage_f1": float(val_results["damage_f1_minor_damage"]),
            "hold_major_damage_f1": float(val_results["damage_f1_major_damage"]),
            "hold_destroyed_f1": float(val_results["damage_f1_destroyed"]),
            "hold_damage_f1": float(val_results["damage_f1"]),
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"hold_score={row['hold_score']:.6f} | "
            f"hold_threshold={row['hold_best_threshold']:.2f} | "
            f"hold_loc_f1={row['hold_localization_f1']:.6f} | "
            f"hold_damage_f1={row['hold_damage_f1']:.6f}",
            flush=True,
        )

        improved = val_score > best_score

        if improved:
            best_score = val_score
            best_epoch = epoch
            best_threshold = epoch_best_threshold
            epochs_without_improvement = 0

            save_checkpoint(
                save_path=checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_threshold=best_threshold,
                args=args,
            )

            print(
                f"Saved best checkpoint at epoch {epoch} | "
                f"hold_score={best_score:.6f} | "
                f"threshold={best_threshold:.2f}",
                flush=True,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Best epoch={best_epoch} | best_hold_score={best_score:.6f}",
                flush=True,
            )

        save_checkpoint(
            save_path=checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            best_threshold=best_threshold,
            args=args,
        )

        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(
                save_path=checkpoints_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_threshold=best_threshold,
                args=args,
            )

        row["best_score_so_far"] = best_score
        row["best_epoch_so_far"] = best_epoch
        row["best_threshold_so_far"] = best_threshold
        row["epochs_without_improvement"] = epochs_without_improvement

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"No hold improvement for {args.early_stopping_patience} epochs.",
                flush=True,
            )
            break

    print("\nEvaluating best checkpoint on xBD test split...", flush=True)

    best_checkpoint = load_model_state(
        model=model,
        checkpoint_path=checkpoints_dir / "best.pt",
        device=device,
    )

    final_threshold = float(best_checkpoint["best_threshold"])
    final_epoch = int(best_checkpoint["epoch"])

    test_results = evaluate_f1(
        model=model,
        loader=test_loader,
        device=device,
        loc_threshold=final_threshold,
    )

    print(json.dumps(test_results, indent=2), flush=True)

    write_final_outputs(
        results=test_results,
        output_dir=output_dir,
        best_epoch=final_epoch,
        best_threshold=final_threshold,
    )

    print("Done.", flush=True)
    print(f"Best epoch selected on hold: {final_epoch}", flush=True)
    print(f"Best threshold selected on hold: {final_threshold}", flush=True)
    print(f"Best hold score: {best_checkpoint.get('best_score', 'unknown')}", flush=True)


if __name__ == "__main__":
    main()