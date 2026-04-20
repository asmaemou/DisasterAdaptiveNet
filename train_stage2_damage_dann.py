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
from torch.utils.data import ConcatDataset, DataLoader, Dataset
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


def resize_pair_and_masks(pre: np.ndarray, post: np.ndarray, loc: Optional[np.ndarray], dmg: Optional[np.ndarray], image_size: int):
    if pre.shape[:2] != (image_size, image_size):
        pre = cv2.resize(pre, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if post.shape[:2] != (image_size, image_size):
        post = cv2.resize(post, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if loc is not None and loc.shape[:2] != (image_size, image_size):
        loc = cv2.resize(loc, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    if dmg is not None and dmg.shape[:2] != (image_size, image_size):
        dmg = cv2.resize(dmg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return pre, post, loc, dmg


def apply_pair_aug(pre: np.ndarray, post: np.ndarray, loc: Optional[np.ndarray], dmg: Optional[np.ndarray], training: bool):
    return pre, post, loc, dmg


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


class Stage2DamageNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        dmg_logits = self.decoder(feat)
        return dmg_logits, feat


class Stage2DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, feat: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        pooled = torch.mean(feat, dim=(2, 3))
        pooled = grad_reverse(pooled, grl_lambda)
        return self.net(pooled).squeeze(1)


class BaseStage2Dataset(Dataset):
    def __init__(self, image_size: int, training: bool, source_name: str):
        self.image_size = int(image_size)
        self.training = bool(training)
        self.source_name = source_name
        self._mean = np.array([0.485, 0.456, 0.406] * 2 + [0.5], dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225] * 2 + [0.5], dtype=np.float32)[:, None, None]

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

    def _normalize(self, pre: np.ndarray, post: np.ndarray, loc_mask: np.ndarray) -> np.ndarray:
        loc_mask = loc_mask.astype(np.float32)[..., None]
        x = np.concatenate([pre.astype(np.float32) / 255.0, post.astype(np.float32) / 255.0, loc_mask], axis=2)
        x = x.transpose(2, 0, 1)
        x = (x - self._mean) / self._std
        return x

    def _finalize_labeled(self, pre: np.ndarray, post: np.ndarray, loc: np.ndarray, dmg: np.ndarray, stem: str):
        pre, post, loc, dmg = resize_pair_and_masks(pre, post, loc, dmg, self.image_size)
        pre, post, loc, dmg = apply_pair_aug(pre, post, loc, dmg, self.training)
        loc = (loc > 0).astype(np.float32)
        dmg_target = self._build_damage_target(loc, dmg)
        return {
            "img": torch.from_numpy(self._normalize(pre, post, loc)).float(),
            "loc": torch.from_numpy(loc).float(),
            "dmg": torch.from_numpy(dmg_target).long(),
            "stem": stem,
            "source_name": self.source_name,
        }

    def _finalize_unlabeled(self, pre: np.ndarray, post: np.ndarray, loc: np.ndarray, stem: str):
        pre, post, loc, _ = resize_pair_and_masks(pre, post, loc, None, self.image_size)
        pre, post, loc, _ = apply_pair_aug(pre, post, loc, None, self.training)
        loc = (loc > 0).astype(np.float32)
        return {
            "img": torch.from_numpy(self._normalize(pre, post, loc)).float(),
            "stem": stem,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class PairedSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    pre_target_path: Path
    post_target_path: Path


class XBDStyleStage2LabeledDataset(BaseStage2Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, source_name: str):
        super().__init__(image_size=image_size, training=training, source_name=source_name)
        self.root = Path(root)
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

    def _collect_samples(self) -> List[PairedSample]:
        post_images = []
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        pre = self._read_rgb(s.pre_image_path)
        post = self._read_rgb(s.post_image_path)
        loc = self._read_mask(s.pre_target_path)
        dmg = self._read_mask(s.post_target_path)
        return self._finalize_labeled(pre, post, loc, dmg, s.stem)

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


class XBDStyleStage2UnlabeledDataset(BaseStage2Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, source_name: str, stage1_model, device):
        super().__init__(image_size=image_size, training=training, source_name=source_name)
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images dir not found: {self.images_dir}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.split_root}")
        self.stage1_model = stage1_model
        self.device = device
        self._mean3 = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self._std3 = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def _collect_samples(self) -> List[Tuple[str, Path, Path]]:
        post_images = []
        for ext in IMG_EXTS:
            post_images.extend(self.images_dir.glob(f"*_post_disaster{ext}"))
        post_images = sorted(post_images)
        samples = []
        for post_path in post_images:
            prefix = post_path.stem.replace("_post_disaster", "")
            ext = post_path.suffix
            pre_path = self.images_dir / f"{prefix}_pre_disaster{ext}"
            if pre_path.exists():
                samples.append((prefix, pre_path, post_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _predict_loc_mask(self, pre: np.ndarray) -> np.ndarray:
        img = pre
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        x = (x - self._mean3) / self._std3
        x = torch.from_numpy(x).unsqueeze(0).float().to(self.device)
        cond_id = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.stage1_model(x, cond_id)[:, 0]
            pred = (torch.sigmoid(logits) > 0.5).float()
        return pred.squeeze(0).cpu().numpy()

    def __getitem__(self, index: int):
        stem, pre_path, post_path = self.samples[index]
        pre = self._read_rgb(pre_path)
        post = self._read_rgb(post_path)
        loc_pred = self._predict_loc_mask(pre)
        return self._finalize_unlabeled(pre, post, loc_pred, stem)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 2 damage classification with domain adaptation")
    parser.add_argument("--source-dataset", type=str, required=True, choices=["xbd", "ida", "ian", "irma"])
    parser.add_argument("--target-dataset", type=str, required=True, choices=["xbd", "ida", "ian", "irma"])
    parser.add_argument("--stage1-checkpoint", type=str, required=True)
    parser.add_argument("--xbd-root", type=str, default="/homes/j244s673/documents/wsu/phd/xview2")
    parser.add_argument("--ida-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet")
    parser.add_argument("--ian-root", type=str, default="/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet")
    parser.add_argument("--irma-root", type=str, default="/homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
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
    parser.add_argument("--dmg-ce-weight", type=float, default=1.0)
    parser.add_argument("--domain-weight", type=float, default=0.05)
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


def load_stage1_model(ckpt_path: str, device: torch.device):
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(OUT_CHANNELS=5),
        DATASET=SimpleNamespace(CONDITIONING_KEY={"generic": 0}),
    )
    model = DisasterAdaptiveNet(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_grl_lambda(epoch: int, step: int, steps_per_epoch: int, total_epochs: int) -> float:
    progress = ((epoch - 1) * steps_per_epoch + step) / max(1, total_epochs * steps_per_epoch)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def evaluate_source_validation(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    conf = RunningConfusionMatrix(4)
    use_tqdm = tqdm is not None and sys.stderr.isatty()
    iterator = tqdm(loader, desc="eval", leave=False) if use_tqdm else loader
    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        dmg = batch["dmg"].to(device, non_blocking=True)
        logits, _ = model(img)
        valid = dmg != 255
        if valid.any():
            loss = criterion(logits, dmg)
            pred = torch.argmax(logits, dim=1)
            acc = (pred[valid] == dmg[valid]).float().mean().item()
            conf.update(dmg[valid], pred[valid])
        else:
            loss = torch.tensor(0.0, device=device)
            acc = 0.0
        bs = img.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)
    return {"loss": loss_meter.avg, "acc": acc_meter.avg, "macro_f1": conf.macro_f1()}


def evaluate_target_test_f1(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
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
        logits, _ = model(img)
        dmg_pred = torch.argmax(logits, dim=1) + 1
        loc_pred = (batch["img"][:, -1] > 0).long().to(device)
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
    no_damage = F1Recorder(dmg_counts[1]["tp"], dmg_counts[1]["fp"], dmg_counts[1]["fn"], "no_damage")
    minor = F1Recorder(dmg_counts[2]["tp"], dmg_counts[2]["fp"], dmg_counts[2]["fn"], "minor_damage")
    major = F1Recorder(dmg_counts[3]["tp"], dmg_counts[3]["fp"], dmg_counts[3]["fn"], "major_damage")
    destroyed = F1Recorder(dmg_counts[4]["tp"], dmg_counts[4]["fp"], dmg_counts[4]["fn"], "destroyed")
    damage_f1 = harmonic_mean([no_damage.f1, minor.f1, major.f1, destroyed.f1])
    return {
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no_damage.f1,
        "damage_f1_minor_damage": minor.f1,
        "damage_f1_major_damage": major.f1,
        "damage_f1_destroyed": destroyed.f1,
        "details": {
            "no_damage": no_damage.as_dict(),
            "minor_damage": minor.as_dict(),
            "major_damage": major.as_dict(),
            "destroyed": destroyed.as_dict(),
        },
    }


def save_checkpoint(save_path: Path, model: nn.Module, domain_disc: nn.Module, optimizer, scheduler, scaler,
                    epoch: int, best_score: float, best_epoch: int, args: argparse.Namespace):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "domain_disc": domain_disc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "args": vars(args),
    }, save_path)


def main() -> None:
    args = parse_args()
    if args.source_dataset == args.target_dataset:
        raise ValueError("source_dataset and target_dataset must be different")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stage1_model = load_stage1_model(args.stage1_checkpoint, device)

    src_root, src_train_split, src_val_split, _ = dataset_root_and_splits(args.source_dataset, args)
    tgt_root, tgt_train_split, tgt_val_split, tgt_test_split = dataset_root_and_splits(args.target_dataset, args)
    print(f"Using device: {device}", flush=True)
    print(f"Source: {args.source_dataset} | root={src_root}", flush=True)
    print(f"Target: {args.target_dataset} | root={tgt_root}", flush=True)
    print(f"Stage1 checkpoint: {args.stage1_checkpoint}", flush=True)

    src_train = XBDStyleStage2LabeledDataset(src_root, src_train_split, args.img_size, True, args.source_dataset)
    src_val = XBDStyleStage2LabeledDataset(src_root, src_val_split, args.img_size, False, args.source_dataset)
    tgt_train_u = XBDStyleStage2UnlabeledDataset(tgt_root, tgt_train_split, args.img_size, True, args.target_dataset, stage1_model, device)
    tgt_val_u = XBDStyleStage2UnlabeledDataset(tgt_root, tgt_val_split, args.img_size, True, args.target_dataset, stage1_model, device)
    tgt_test = XBDStyleStage2LabeledDataset(tgt_root, tgt_test_split, args.img_size, False, args.target_dataset)

    src_train_loader = DataLoader(src_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    src_val_loader = DataLoader(src_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    tgt_u_loader = DataLoader(ConcatDataset([tgt_train_u, tgt_val_u]), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tgt_test_loader = DataLoader(tgt_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    dmg_counts = src_train.get_damage_class_counts().astype(np.float64)
    dmg_counts[dmg_counts == 0] = 1.0
    inv = dmg_counts.sum() / dmg_counts
    dmg_class_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32)

    model = Stage2DamageNet().to(device)
    domain_disc = Stage2DomainDiscriminator(64).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(domain_disc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    milestones = sorted(set(max(1, int(args.epochs * x)) for x in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda") if USE_TORCH_AMP else GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(weight=dmg_class_weights.to(device), ignore_index=255).to(device)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    def cycle(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch
    tgt_iter = cycle(tgt_u_loader)
    steps_per_epoch = len(src_train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        domain_disc.train()
        total_meter = AverageMeter()
        sup_meter = AverageMeter()
        dom_meter = AverageMeter()
        use_tqdm = tqdm is not None and sys.stderr.isatty()
        iterator = tqdm(src_train_loader, desc=f"train {epoch}/{args.epochs}") if use_tqdm else src_train_loader
        for step, src_batch in enumerate(iterator, start=1):
            tgt_batch = next(tgt_iter)
            src_img = src_batch["img"].to(device, non_blocking=True)
            src_dmg = src_batch["dmg"].to(device, non_blocking=True)
            tgt_img = tgt_batch["img"].to(device, non_blocking=True)
            grl_lambda = compute_grl_lambda(epoch, step, steps_per_epoch, args.epochs)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=USE_TORCH_AMP and args.amp and device.type == "cuda") if USE_TORCH_AMP else autocast(enabled=args.amp and device.type == "cuda"):
                src_logits, src_feat = model(src_img)
                tgt_logits, tgt_feat = model(tgt_img)
                valid = src_dmg != 255
                if valid.any():
                    sup_loss = args.dmg_ce_weight * criterion(src_logits, src_dmg)
                else:
                    sup_loss = torch.tensor(0.0, device=device, dtype=src_logits.dtype)
                src_dom_logits = domain_disc(src_feat, grl_lambda)
                tgt_dom_logits = domain_disc(tgt_feat, grl_lambda)
                src_dom_targets = torch.zeros_like(src_dom_logits)
                tgt_dom_targets = torch.ones_like(tgt_dom_logits)
                dom_loss = 0.5 * (domain_criterion(src_dom_logits, src_dom_targets) + domain_criterion(tgt_dom_logits, tgt_dom_targets))
                total_loss = sup_loss + args.domain_weight * dom_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = src_img.size(0)
            total_meter.update(total_loss.item(), bs)
            sup_meter.update(sup_loss.item(), bs)
            dom_meter.update(dom_loss.item(), bs)
            if use_tqdm:
                iterator.set_postfix(loss=f"{total_meter.avg:.4f}", sup=f"{sup_meter.avg:.4f}", dom=f"{dom_meter.avg:.4f}")

        scheduler.step()
        src_val_metrics = evaluate_source_validation(model, src_val_loader, criterion, device)
        tgt_test_metrics = evaluate_target_test_f1(model, tgt_test_loader, device)
        val_score = src_val_metrics["acc"] + src_val_metrics["macro_f1"]
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_total_loss": total_meter.avg,
            "train_supervised_loss": sup_meter.avg,
            "train_domain_loss": dom_meter.avg,
            "src_val_acc": src_val_metrics["acc"],
            "src_val_macro_f1": src_val_metrics["macro_f1"],
            "src_val_score": val_score,
            "tgt_test_damage_f1": tgt_test_metrics["damage_f1"],
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
        else:
            epochs_without_improvement += 1
        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if epochs_without_improvement >= args.early_stopping_patience:
            break

    ckpt = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    tgt_test_metrics = evaluate_target_test_f1(model, tgt_test_loader, device)
    with open(output_dir / "target_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tgt_test_metrics, f, indent=2)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    with open(scores_dir / f"scores_{args.target_dataset}_stage2_target_test.txt", "w", encoding="utf-8") as f:
        f.write(f"No Damage F1:    {tgt_test_metrics['damage_f1_no_damage']:.6f}\n")
        f.write(f"Minor Damage F1: {tgt_test_metrics['damage_f1_minor_damage']:.6f}\n")
        f.write(f"Major Damage F1: {tgt_test_metrics['damage_f1_major_damage']:.6f}\n")
        f.write(f"Destroyed F1:    {tgt_test_metrics['damage_f1_destroyed']:.6f}\n")
        f.write(f"Damage F1:       {tgt_test_metrics['damage_f1']:.6f}\n")


if __name__ == "__main__":
    main()
