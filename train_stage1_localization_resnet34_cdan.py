from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_rgb_and_mask(image: np.ndarray, mask: Optional[np.ndarray], image_size: int):
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if mask is not None and mask.shape[:2] != (image_size, image_size):
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def apply_shared_aug(image: np.ndarray, mask: Optional[np.ndarray], training: bool):
    # Intentionally no augmentation for this stage.
    return image, mask


class BaseLocalizationDataset(Dataset):
    def __init__(self, image_size: int, training: bool, source_name: str):
        self.image_size = int(image_size)
        self.training = bool(training)
        self.source_name = source_name
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

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

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = (image - self._mean) / self._std
        return image

    def _finalize_labeled(self, image: np.ndarray, loc: np.ndarray, stem: str):
        image, loc = resize_rgb_and_mask(image, loc, self.image_size)
        image, loc = apply_shared_aug(image, loc, self.training)
        loc = (loc > 0).astype(np.float32)
        return {
            "img": torch.from_numpy(self._normalize(image)).float(),
            "loc": torch.from_numpy(loc).float(),
            "stem": stem,
            "source_name": self.source_name,
        }

    def _finalize_unlabeled(self, image: np.ndarray, stem: str):
        image, _ = resize_rgb_and_mask(image, None, self.image_size)
        image, _ = apply_shared_aug(image, None, self.training)
        return {
            "img": torch.from_numpy(self._normalize(image)).float(),
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


class XBDStyleLocalizationLabeledDataset(BaseLocalizationDataset):
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
        loc = self._read_mask(s.pre_target_path)
        return self._finalize_labeled(pre, loc, s.stem)

    def get_localization_pixel_counts(self) -> Tuple[int, int]:
        pos, neg = 0, 0
        for s in self.samples:
            loc = self._read_mask(s.pre_target_path) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())
        return pos, neg


class XBDStyleLocalizationUnlabeledDataset(BaseLocalizationDataset):
    def __init__(self, root: str | Path, split: str, image_size: int, training: bool, source_name: str):
        super().__init__(image_size=image_size, training=training, source_name=source_name)
        self.root = Path(root)
        self.split_root = self.root / split
        self.images_dir = self.split_root / "images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Expected images dir not found: {self.images_dir}")
        self.pre_images = self._collect_images()
        if not self.pre_images:
            raise RuntimeError(f"No pre-disaster images found under {self.split_root}")

    def _collect_images(self) -> List[Path]:
        pre_images = []
        for ext in IMG_EXTS:
            pre_images.extend(self.images_dir.glob(f"*_pre_disaster{ext}"))
        return sorted(pre_images)

    def __len__(self) -> int:
        return len(self.pre_images)

    def __getitem__(self, index: int):
        pre_path = self.pre_images[index]
        stem = pre_path.stem.replace("_pre_disaster", "")
        pre = self._read_rgb(pre_path)
        return self._finalize_unlabeled(pre, stem)


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
    def __init__(self, tp: int, fp: int, fn: int) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return 0.0 if d == 0 else self.tp / d

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return 0.0 if d == 0 else self.tp / d

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 0.0 if (p == 0.0 or r == 0.0) else 2.0 * p * r / (p + r)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + 1e-7) / (union + 1e-7)).mean()
        return bce, dice


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


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Stage1ResNet34UNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
        if pretrained:
            print(f"using weights from {weights}", flush=True)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.relu(self.bn1(self.conv1(x)))   # H/2
        x1 = self.layer1(self.maxpool(x0))        # H/4
        x2 = self.layer2(x1)                      # H/8
        x3 = self.layer3(x2)                      # H/16
        x4 = self.layer4(x3)                      # H/32

        d3 = self.up4(x4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up2(d2, x1)
        d0 = self.up1(d1, x0)
        logits = self.final_up(d0).squeeze(1)
        return logits, x4


class Stage1CDANDiscriminator(nn.Module):
    def __init__(self, feat_dim: int = 512, prob_dim: int = 2) -> None:
        super().__init__()
        self.in_dim = feat_dim * prob_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def make_condition(self, feat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # feat: [B, C], probs: [B, 2]
        cond = torch.bmm(probs.unsqueeze(2), feat.unsqueeze(1)).reshape(feat.size(0), -1)
        return cond

    def forward(self, feat_map: torch.Tensor, loc_logits: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        p_build = torch.sigmoid(loc_logits).mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        probs = torch.cat([1.0 - p_build, p_build], dim=1)
        cond = self.make_condition(feat, probs)
        cond = grad_reverse(cond, grl_lambda)
        return self.net(cond).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 1 localization with ResNet34 U-Net + CDAN")
    parser.add_argument("--source-dataset", type=str, required=True, choices=["xbd", "ida", "ian", "irma"])
    parser.add_argument("--target-dataset", type=str, required=True, choices=["xbd", "ida", "ian", "irma"])
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
    parser.add_argument("--loc-threshold", type=float, default=0.5)
    parser.add_argument("--loc-bce-weight", type=float, default=1.0)
    parser.add_argument("--loc-dice-weight", type=float, default=1.0)
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


def compute_grl_lambda(epoch: int, step: int, steps_per_epoch: int, total_epochs: int) -> float:
    progress = ((epoch - 1) * steps_per_epoch + step) / max(1, total_epochs * steps_per_epoch)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def evaluate_localization(model: nn.Module, loader: DataLoader, criterion: BCEDiceLoss, device: torch.device, args: argparse.Namespace):
    model.eval()
    loss_meter = AverageMeter()
    bce_meter = AverageMeter()
    dice_meter = AverageMeter()
    f1_tp = f1_fp = f1_fn = 0

    use_tqdm = tqdm is not None and sys.stderr.isatty()
    iterator = tqdm(loader, desc="eval", leave=False) if use_tqdm else loader

    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc = batch["loc"].to(device, non_blocking=True)
        logits, _ = model(img)
        bce, dice_loss = criterion(logits, loc)
        loss = args.loc_bce_weight * bce + args.loc_dice_weight * dice_loss
        pred = (torch.sigmoid(logits) > args.loc_threshold).float()
        f1_tp += int(((pred == 1) & (loc == 1)).sum().item())
        f1_fp += int(((pred == 1) & (loc == 0)).sum().item())
        f1_fn += int(((pred == 0) & (loc == 1)).sum().item())

        inter = (pred * loc).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + loc.sum(dim=(1, 2))
        dice = ((2.0 * inter + 1e-7) / (union + 1e-7)).mean().item()
        bs = img.size(0)
        loss_meter.update(loss.item(), bs)
        bce_meter.update(bce.item(), bs)
        dice_meter.update(dice, bs)

    return {
        "loss": loss_meter.avg,
        "loc_bce": bce_meter.avg,
        "loc_dice": dice_meter.avg,
        "loc_f1": F1Recorder(f1_tp, f1_fp, f1_fn).f1,
    }


def save_checkpoint(save_path: Path, model: nn.Module, domain_disc: nn.Module, optimizer, scheduler, scaler,
                    epoch: int, best_score: float, best_epoch: int, args: argparse.Namespace) -> None:
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
    print(f"Using device: {device}", flush=True)

    src_root, src_train_split, src_val_split, _ = dataset_root_and_splits(args.source_dataset, args)
    tgt_root, tgt_train_split, tgt_val_split, tgt_test_split = dataset_root_and_splits(args.target_dataset, args)
    print(f"Source: {args.source_dataset} | root={src_root}", flush=True)
    print(f"Target: {args.target_dataset} | root={tgt_root}", flush=True)
    print("Stage-1 uses ResNet34 U-Net with CDAN-style localization adaptation.", flush=True)

    src_train = XBDStyleLocalizationLabeledDataset(src_root, src_train_split, args.img_size, True, args.source_dataset)
    src_val = XBDStyleLocalizationLabeledDataset(src_root, src_val_split, args.img_size, False, args.source_dataset)
    tgt_train_u = XBDStyleLocalizationUnlabeledDataset(tgt_root, tgt_train_split, args.img_size, True, args.target_dataset)
    tgt_val_u = XBDStyleLocalizationUnlabeledDataset(tgt_root, tgt_val_split, args.img_size, True, args.target_dataset)
    tgt_test = XBDStyleLocalizationLabeledDataset(tgt_root, tgt_test_split, args.img_size, False, args.target_dataset)

    src_train_loader = DataLoader(src_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    src_val_loader = DataLoader(src_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    tgt_u_loader = DataLoader(ConcatDataset([tgt_train_u, tgt_val_u]), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tgt_test_loader = DataLoader(tgt_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    loc_pos, loc_neg = src_train.get_localization_pixel_counts()
    loc_pos_weight = torch.tensor([max(1.0, loc_neg / max(loc_pos, 1))], dtype=torch.float32)

    model = Stage1ResNet34UNet(pretrained=True).to(device)
    domain_disc = Stage1CDANDiscriminator(512, 2).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(domain_disc.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    milestones = sorted(set(max(1, int(args.epochs * x)) for x in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda") if USE_TORCH_AMP else GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = BCEDiceLoss(pos_weight=loc_pos_weight.to(device)).to(device)
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
            src_loc = src_batch["loc"].to(device, non_blocking=True)
            tgt_img = tgt_batch["img"].to(device, non_blocking=True)
            grl_lambda = compute_grl_lambda(epoch, step, steps_per_epoch, args.epochs)
            optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    src_logits, src_feat = model(src_img)
                    tgt_logits, tgt_feat = model(tgt_img)
                    loc_bce, loc_dice = criterion(src_logits, src_loc)
                    sup_loss = args.loc_bce_weight * loc_bce + args.loc_dice_weight * loc_dice
                    src_dom_logits = domain_disc(src_feat, src_logits, grl_lambda)
                    tgt_dom_logits = domain_disc(tgt_feat, tgt_logits, grl_lambda)
                    src_dom_targets = torch.zeros_like(src_dom_logits)
                    tgt_dom_targets = torch.ones_like(tgt_dom_logits)
                    dom_loss = 0.5 * (
                        domain_criterion(src_dom_logits, src_dom_targets) +
                        domain_criterion(tgt_dom_logits, tgt_dom_targets)
                    )
                    total_loss = sup_loss + args.domain_weight * dom_loss
            else:
                with autocast(enabled=args.amp and device.type == "cuda"):
                    src_logits, src_feat = model(src_img)
                    tgt_logits, tgt_feat = model(tgt_img)
                    loc_bce, loc_dice = criterion(src_logits, src_loc)
                    sup_loss = args.loc_bce_weight * loc_bce + args.loc_dice_weight * loc_dice
                    src_dom_logits = domain_disc(src_feat, src_logits, grl_lambda)
                    tgt_dom_logits = domain_disc(tgt_feat, tgt_logits, grl_lambda)
                    src_dom_targets = torch.zeros_like(src_dom_logits)
                    tgt_dom_targets = torch.ones_like(tgt_dom_logits)
                    dom_loss = 0.5 * (
                        domain_criterion(src_dom_logits, src_dom_targets) +
                        domain_criterion(tgt_dom_logits, tgt_dom_targets)
                    )
                    total_loss = sup_loss + args.domain_weight * dom_loss

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch={epoch}, step={step}: "
                    f"total={total_loss.item()}, sup={sup_loss.item()}, dom={dom_loss.item()}"
                )

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
        src_val_metrics = evaluate_localization(model, src_val_loader, criterion, device, args)
        tgt_test_metrics = evaluate_localization(model, tgt_test_loader, criterion, device, args)
        val_score = src_val_metrics["loc_f1"] + src_val_metrics["loc_dice"]
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_total_loss": total_meter.avg,
            "train_supervised_loss": sup_meter.avg,
            "train_domain_loss": dom_meter.avg,
            "src_val_loc_f1": src_val_metrics["loc_f1"],
            "src_val_loc_dice": src_val_metrics["loc_dice"],
            "src_val_score": val_score,
            "tgt_test_loc_f1": tgt_test_metrics["loc_f1"],
            "tgt_test_loc_dice": tgt_test_metrics["loc_dice"],
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
            print(f"Saved new best checkpoint at epoch {epoch} with source-val score={best_score:.4f}", flush=True)
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Best epoch so far: {best_epoch} | best_score={best_score:.4f}",
                flush=True,
            )

        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, domain_disc, optimizer, scheduler, scaler, epoch, best_score, best_epoch, args)

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if epochs_without_improvement >= args.early_stopping_patience:
            break

    ckpt = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    tgt_test_metrics = evaluate_localization(model, tgt_test_loader, criterion, device, args)
    with open(output_dir / "target_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tgt_test_metrics, f, indent=2)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    with open(scores_dir / f"scores_{args.target_dataset}_stage1_target_test.txt", "w", encoding="utf-8") as f:
        f.write(f"Localization F1:   {tgt_test_metrics['loc_f1']:.6f}\n")
        f.write(f"Localization Dice: {tgt_test_metrics['loc_dice']:.6f}\n")
        f.write(f"Loss:              {tgt_test_metrics['loss']:.6f}\n")


if __name__ == "__main__":
    main()
