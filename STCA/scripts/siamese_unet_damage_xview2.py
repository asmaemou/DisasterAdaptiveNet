#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = 255


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class XView2Dataset(Dataset):
    def __init__(self, split_dir: str, mode: str = "train", crop_size: int = 512, stride: int = 256):
        self.split_dir = Path(split_dir)
        self.mode = mode
        self.crop_size = crop_size
        self.stride = stride
        self.image_dir = self.split_dir / "images"
        self.target_dir = self.split_dir / "targets"

        if not self.image_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(f"Missing images/targets under {self.split_dir}")

        self.pre_masks = sorted(self.target_dir.glob("*_pre_disaster_target.png"))
        if not self.pre_masks:
            self.pre_masks = sorted(self.target_dir.glob("*_pre_*_target.png"))
        if not self.pre_masks:
            raise FileNotFoundError(f"No pre target png files found in {self.target_dir}")

        self.samples: List[Dict[str, Path]] = []
        for pre_m in self.pre_masks:
            post_name = pre_m.name.replace("_pre_disaster_target.png", "_post_disaster_target.png")
            if post_name == pre_m.name:
                post_name = pre_m.name.replace("_pre_", "_post_")
            post_m = self.target_dir / post_name
            pre_img = self.image_dir / pre_m.name.replace("_target.png", ".png")
            post_img = self.image_dir / post_m.name.replace("_target.png", ".png")
            if post_m.exists() and pre_img.exists() and post_img.exists():
                self.samples.append(dict(pre_img=pre_img, post_img=post_img, pre_mask=pre_m, post_mask=post_m))

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.split_dir}")

        print(f"[dataset] split={self.split_dir.name} mode={self.mode} samples={len(self.samples)}")
        self.tiles = self._build_tiles()

    def _build_tiles(self) -> Optional[List[Tuple[int, int, int]]]:
        if self.mode == "eval":
            return None
        tiles: List[Tuple[int, int, int]] = []
        for idx in range(len(self.samples)):
            for y in range(0, 1024 - self.crop_size + 1, self.stride):
                for x in range(0, 1024 - self.crop_size + 1, self.stride):
                    tiles.append((idx, x, y))
        return tiles

    def __len__(self) -> int:
        return len(self.samples) if self.mode == "eval" else len(self.tiles)

    def _load_sample(self, sample: Dict[str, Path]):
        pre = np.array(Image.open(sample["pre_img"]).convert("RGB"))
        post = np.array(Image.open(sample["post_img"]).convert("RGB"))
        pre_mask = np.array(Image.open(sample["pre_mask"]))
        post_mask = np.array(Image.open(sample["post_mask"]))
        return pre, post, pre_mask, post_mask, sample["pre_img"].name

    def _augment(self, pre, post, pre_mask, post_mask):
        if random.random() < 0.5:
            pre = np.flip(pre, axis=1).copy()
            post = np.flip(post, axis=1).copy()
            pre_mask = np.flip(pre_mask, axis=1).copy()
            post_mask = np.flip(post_mask, axis=1).copy()
        if random.random() < 0.5:
            pre = np.flip(pre, axis=0).copy()
            post = np.flip(post, axis=0).copy()
            pre_mask = np.flip(pre_mask, axis=0).copy()
            post_mask = np.flip(post_mask, axis=0).copy()
        k = random.randint(0, 3)
        if k:
            pre = np.rot90(pre, k).copy()
            post = np.rot90(post, k).copy()
            pre_mask = np.rot90(pre_mask, k).copy()
            post_mask = np.rot90(post_mask, k).copy()
        return pre, post, pre_mask, post_mask

    def __getitem__(self, idx):
        if self.mode == "eval":
            sample = self.samples[idx]
            pre, post, pre_mask, post_mask, name = self._load_sample(sample)
        else:
            img_idx, x, y = self.tiles[idx]
            sample = self.samples[img_idx]
            pre, post, pre_mask, post_mask, name = self._load_sample(sample)
            s = self.crop_size
            pre = pre[y:y+s, x:x+s]
            post = post[y:y+s, x:x+s]
            pre_mask = pre_mask[y:y+s, x:x+s]
            post_mask = post_mask[y:y+s, x:x+s]
            if self.mode == "train":
                pre, post, pre_mask, post_mask = self._augment(pre, post, pre_mask, post_mask)

        pre_t = torch.from_numpy(pre).permute(2, 0, 1).float() / 255.0
        post_t = torch.from_numpy(post).permute(2, 0, 1).float() / 255.0
        return {
            "pre": pre_t,
            "post": post_t,
            "loc_mask": torch.from_numpy((pre_mask > 0).astype(np.uint8)),
            "dam_mask": torch.from_numpy(post_mask.astype(np.uint8)),
            "image_filename": name,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, widths: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.blocks.append(ConvBlock(c, w))
            self.pools.append(nn.MaxPool2d(2))
            c = w
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2)
        self.widths = widths

    def forward(self, x):
        feats = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            feats.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        return feats, x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNetDamage(nn.Module):
    def __init__(self, widths: Tuple[int, ...] = (32, 64, 128, 256), dam_classes: int = 5):
        super().__init__()
        self.encoder = Encoder(3, widths)
        bottleneck_ch = widths[-1] * 2

        self.loc_up4 = UpBlock(bottleneck_ch, widths[-1], widths[-1])
        self.loc_up3 = UpBlock(widths[-1], widths[-2], widths[-2])
        self.loc_up2 = UpBlock(widths[-2], widths[-3], widths[-3])
        self.loc_up1 = UpBlock(widths[-3], widths[-4], widths[-4])
        self.loc_head = nn.Conv2d(widths[0], 1, kernel_size=1)

        dam_bottleneck_in = bottleneck_ch * 3
        self.dam_reduce = nn.Conv2d(dam_bottleneck_in, bottleneck_ch, kernel_size=1)
        self.dam_up4 = UpBlock(bottleneck_ch, widths[-1] * 3, widths[-1])
        self.dam_up3 = UpBlock(widths[-1], widths[-2] * 3, widths[-2])
        self.dam_up2 = UpBlock(widths[-2], widths[-3] * 3, widths[-3])
        self.dam_up1 = UpBlock(widths[-3], widths[-4] * 3, widths[-4])
        self.dam_head = nn.Conv2d(widths[0], dam_classes, kernel_size=1)

    def forward(self, pre, post):
        pre_skips, pre_b = self.encoder(pre)
        post_skips, post_b = self.encoder(post)

        x_loc = pre_b
        x_loc = self.loc_up4(x_loc, pre_skips[-1])
        x_loc = self.loc_up3(x_loc, pre_skips[-2])
        x_loc = self.loc_up2(x_loc, pre_skips[-3])
        x_loc = self.loc_up1(x_loc, pre_skips[-4])
        loc_logits = self.loc_head(x_loc)

        x_dam = torch.cat([pre_b, post_b, torch.abs(pre_b - post_b)], dim=1)
        x_dam = self.dam_reduce(x_dam)
        fused_skips = [torch.cat([a, b, torch.abs(a - b)], dim=1) for a, b in zip(pre_skips, post_skips)]
        x_dam = self.dam_up4(x_dam, fused_skips[-1])
        x_dam = self.dam_up3(x_dam, fused_skips[-2])
        x_dam = self.dam_up2(x_dam, fused_skips[-3])
        x_dam = self.dam_up1(x_dam, fused_skips[-4])
        dam_logits = self.dam_head(x_dam)

        return loc_logits, dam_logits


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def binary_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def class_f1(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    valid = gt != IGNORE_INDEX
    pred_c = (pred == cls) & valid
    gt_c = (gt == cls) & valid
    tp = np.logical_and(pred_c, gt_c).sum()
    fp = np.logical_and(pred_c, ~gt_c).sum()
    fn = np.logical_and(~pred_c, gt_c).sum()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loc_threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    loc_scores, reg_scores, min_scores, maj_scores, des_scores = [], [], [], [], []
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        gt_loc = batch["loc_mask"].numpy()
        gt_dam = batch["dam_mask"].numpy()
        loc_logits, dam_logits = model(pre, post)
        if loc_logits.shape[-2:] != gt_loc.shape[-2:]:
            loc_logits = F.interpolate(loc_logits, size=gt_loc.shape[-2:], mode="bilinear", align_corners=False)
        if dam_logits.shape[-2:] != gt_dam.shape[-2:]:
            dam_logits = F.interpolate(dam_logits, size=gt_dam.shape[-2:], mode="bilinear", align_corners=False)

        loc_pred = (torch.sigmoid(loc_logits) > loc_threshold).cpu().numpy()[:, 0]
        dam_pred = dam_logits.argmax(dim=1).cpu().numpy()
        dam_pred = dam_pred.copy()
        dam_pred[loc_pred == 0] = 0

        for i in range(loc_pred.shape[0]):
            loc_scores.append(binary_f1(loc_pred[i], gt_loc[i] > 0))
            reg_scores.append(class_f1(dam_pred[i], gt_dam[i], 1))
            min_scores.append(class_f1(dam_pred[i], gt_dam[i], 2))
            maj_scores.append(class_f1(dam_pred[i], gt_dam[i], 3))
            des_scores.append(class_f1(dam_pred[i], gt_dam[i], 4))

    f1_loc = float(np.mean(loc_scores)) * 100.0
    f1_regular = float(np.mean(reg_scores)) * 100.0
    f1_minor = float(np.mean(min_scores)) * 100.0
    f1_major = float(np.mean(maj_scores)) * 100.0
    f1_destroyed = float(np.mean(des_scores)) * 100.0
    f1_dam = 0.0 if min(f1_regular, f1_minor, f1_major, f1_destroyed) == 0 else (
        f1_regular * f1_minor * f1_major * f1_destroyed
    ) ** 0.25
    f1_avg = 0.3 * f1_loc + 0.7 * f1_dam

    return {
        "f1_avg": f1_avg,
        "f1_loc": f1_loc,
        "f1_dam": f1_dam,
        "f1_regular": f1_regular,
        "f1_minor": f1_minor,
        "f1_major": f1_major,
        "f1_destroyed": f1_destroyed,
    }


def main():
    ap = argparse.ArgumentParser(description="Siamese U-Net baseline for damage detection")
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--train-split", default="tier3")
    ap.add_argument("--val-root", default=None)
    ap.add_argument("--val-split", default="test")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--eval-split", default="hold")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--loc-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / "checkpoints"
    metrics_dir = work_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_root = Path(args.train_root)
    val_root = Path(args.val_root) if args.val_root else train_root
    eval_root = Path(args.eval_root) if args.eval_root else train_root
    train_dir = train_root / args.train_split
    val_dir = val_root / args.val_split
    eval_dir = eval_root / args.eval_split

    print(f"[paths] train={train_dir}")
    print(f"[paths] val={val_dir}")
    print(f"[paths] eval={eval_dir}")

    train_ds = XView2Dataset(str(train_dir), mode="train", crop_size=args.crop_size, stride=args.stride)
    val_ds = XView2Dataset(str(val_dir), mode="eval")
    eval_ds = XView2Dataset(str(eval_dir), mode="eval")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model = SiameseUNetDamage().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            loc_gt = batch["loc_mask"].to(device).float().unsqueeze(1)
            dam_gt = batch["dam_mask"].to(device).long()

            loc_logits, dam_logits = model(pre, post)
            if loc_logits.shape[-2:] != loc_gt.shape[-2:]:
                loc_logits = F.interpolate(loc_logits, size=loc_gt.shape[-2:], mode="bilinear", align_corners=False)
            if dam_logits.shape[-2:] != dam_gt.shape[-2:]:
                dam_logits = F.interpolate(dam_logits, size=dam_gt.shape[-2:], mode="bilinear", align_corners=False)

            loc_loss = F.binary_cross_entropy_with_logits(loc_logits, loc_gt) + dice_loss_from_logits(loc_logits, loc_gt)
            masked_dam_gt = dam_gt.clone()
            masked_dam_gt[loc_gt[:, 0] == 0] = IGNORE_INDEX
            dam_loss = F.cross_entropy(dam_logits, masked_dam_gt, ignore_index=IGNORE_INDEX)
            loss = loc_loss + dam_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        val_metrics = evaluate(model, val_loader, device, args.loc_threshold)
        print(f"[epoch={epoch}] loss={running/max(1,len(train_loader)):.4f} val_f1_avg={val_metrics['f1_avg']:.2f} val_f1_loc={val_metrics['f1_loc']:.2f} val_f1_dam={val_metrics['f1_dam']:.2f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
        if val_metrics["f1_avg"] > best_val:
            best_val = val_metrics["f1_avg"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, ckpt_dir / "best.pt")

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    val_metrics = evaluate(model, val_loader, device, args.loc_threshold)
    eval_metrics = evaluate(model, eval_loader, device, args.loc_threshold)
    save_json(metrics_dir / "val.json", val_metrics)
    save_json(metrics_dir / "eval.json", eval_metrics)
    print("val", val_metrics)
    print("eval", eval_metrics)


if __name__ == "__main__":
    main()
