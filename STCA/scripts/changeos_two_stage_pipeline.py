#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import ever as er
import torchange as tc
from torchange.models.changeos import ChangeOS
from torchange.configs.changeos import cos_r18, cos_r34, cos_r50, cos_r101, cos_swint

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


class XView2ChangeOSDataset(Dataset):
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

        pre_t = torch.from_numpy(pre).permute(2, 0, 1).float()
        post_t = torch.from_numpy(post).permute(2, 0, 1).float()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        pre_t = (pre_t - mean) / std
        post_t = (post_t - mean) / std
        x = torch.cat([pre_t, post_t], dim=0)
        y = {
            "loc_mask": torch.from_numpy((pre_mask > 0).astype(np.uint8)),
            "dam_mask": torch.from_numpy(post_mask.astype(np.uint8)),
            "image_filename": name,
        }
        return x, y


def make_model(backbone: str) -> ChangeOS:
    cfg_mod = {
        "r18": cos_r18,
        "r34": cos_r34,
        "r50": cos_r50,
        "r101": cos_r101,
        "swint": cos_swint,
    }[backbone]
    return er.builder.make_model(cfg_mod.config["model"])


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def upsample_like(logits: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_hw:
        return logits
    return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)


def get_forward_outputs(model: ChangeOS, x: torch.Tensor):
    feats = tc.bitemporal_forward(model.encoder, x)
    t1_features, st_features = model.decoder(*feats)
    loc_logit = model.head.loc_cls(t1_features)
    dam_logit = model.head.dam_cls(st_features)
    return loc_logit, dam_logit


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
def evaluate_two_stage(model: ChangeOS, loader: DataLoader, device: torch.device, loc_threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    loc_scores, reg_scores, min_scores, maj_scores, des_scores = [], [], [], [], []

    for x, y in loader:
        x = x.to(device)
        gt_loc = y["loc_mask"].numpy()
        gt_dam = y["dam_mask"].numpy()

        loc_logit, dam_logit = get_forward_outputs(model, x)
        loc_logit = upsample_like(loc_logit, gt_loc.shape[-2:])
        dam_logit = upsample_like(dam_logit, gt_dam.shape[-2:])

        loc_prob = torch.sigmoid(loc_logit)
        loc_pred = (loc_prob > loc_threshold).long().cpu().numpy()

        if loc_pred.ndim == 4 and loc_pred.shape[1] == 1:
            loc_pred = loc_pred[:, 0]
        elif loc_pred.ndim == 4 and loc_pred.shape[-1] == 1:
            loc_pred = loc_pred[..., 0]
        elif loc_pred.ndim != 3:
            raise RuntimeError(f"Unexpected loc_pred shape: {loc_pred.shape}")

        dam_pred = dam_logit.argmax(dim=1).cpu().numpy()
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


def freeze_for_stage2(model: ChangeOS) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False
    if hasattr(model.decoder, "loc_neck"):
        for p in model.decoder.loc_neck.parameters():
            p.requires_grad = False
    if hasattr(model.head, "loc_cls"):
        for p in model.head.loc_cls.parameters():
            p.requires_grad = False
    if hasattr(model.decoder, "dam_neck"):
        for p in model.decoder.dam_neck.parameters():
            p.requires_grad = True
    if hasattr(model.head, "dam_cls"):
        for p in model.head.dam_cls.parameters():
            p.requires_grad = True
    if hasattr(model.decoder, "fuse_conv"):
        for p in model.decoder.fuse_conv.parameters():
            p.requires_grad = True


def build_optimizer(model: ChangeOS, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-stage ChangeOS pipeline: stage1 localization, stage2 damage on detected buildings")
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-split", default="tier3")
    ap.add_argument("--val-split", default="test")
    ap.add_argument("--eval-split", default="hold")
    ap.add_argument("--backbone", choices=["r18", "r34", "r50", "r101", "swint"], default="r18")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--epochs-loc", type=int, default=10)
    ap.add_argument("--epochs-dam", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-loc", type=float, default=3e-4)
    ap.add_argument("--lr-dam", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loc-threshold", type=float, default=0.5)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / "checkpoints"
    metrics_dir = work_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    root = Path(args.root)
    train_dir = root / args.train_split
    val_dir = root / args.val_split
    eval_dir = root / args.eval_split
    print(f"[paths] train={train_dir}")
    print(f"[paths] val={val_dir}")
    print(f"[paths] eval={eval_dir}")

    train_ds = XView2ChangeOSDataset(str(train_dir), mode="train", crop_size=args.crop_size, stride=args.stride)
    val_ds = XView2ChangeOSDataset(str(val_dir), mode="eval")
    eval_ds = XView2ChangeOSDataset(str(eval_dir), mode="eval")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

    model = make_model(args.backbone).to(device)

    optimizer = build_optimizer(model, args.lr_loc, args.weight_decay)
    best_loc = -1.0
    print("=== Stage 1: building localization ===")
    for epoch in range(1, args.epochs_loc + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            loc_gt = y["loc_mask"].to(device).float().unsqueeze(1)
            loc_logit, _ = get_forward_outputs(model, x)
            loc_logit = upsample_like(loc_logit, loc_gt.shape[-2:])
            loss_bce = F.binary_cross_entropy_with_logits(loc_logit, loc_gt)
            loss_dice = dice_loss_from_logits(loc_logit, loc_gt)
            loss = loss_bce + loss_dice
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        val_metrics = evaluate_two_stage(model, val_loader, device, args.loc_threshold)
        print(f"[loc] epoch={epoch} loss={running/max(1,len(train_loader)):.4f} val_f1_loc={val_metrics['f1_loc']:.2f} val_f1_dam={val_metrics['f1_dam']:.2f} val_f1_avg={val_metrics['f1_avg']:.2f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "stage1_last.pt")
        if val_metrics["f1_loc"] > best_loc:
            best_loc = val_metrics["f1_loc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, ckpt_dir / "stage1_best.pt")

    best_stage1 = torch.load(ckpt_dir / "stage1_best.pt", map_location=device)
    model.load_state_dict(best_stage1["model"])
    stage1_test = evaluate_two_stage(model, val_loader, device, args.loc_threshold)
    stage1_hold = evaluate_two_stage(model, eval_loader, device, args.loc_threshold)
    save_json(metrics_dir / "stage1_test.json", stage1_test)
    save_json(metrics_dir / "stage1_hold.json", stage1_hold)
    print("stage1_test", stage1_test)
    print("stage1_hold", stage1_hold)

    freeze_for_stage2(model)
    optimizer = build_optimizer(model, args.lr_dam, args.weight_decay)
    best_avg = -1.0
    print("=== Stage 2: damage classification on detected buildings ===")
    for epoch in range(1, args.epochs_dam + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            loc_gt = y["loc_mask"].to(device).float().unsqueeze(1)
            dam_gt = y["dam_mask"].to(device).long()

            with torch.no_grad():
                loc_logit, _ = get_forward_outputs(model, x)
                loc_logit = upsample_like(loc_logit, dam_gt.shape[-2:])
                loc_prob = torch.sigmoid(loc_logit)
                roi_pred = (loc_prob > args.loc_threshold).squeeze(1)

            _, dam_logit = get_forward_outputs(model, x)
            dam_logit = upsample_like(dam_logit, dam_gt.shape[-2:])

            masked_target = dam_gt.clone()
            masked_target[~roi_pred] = IGNORE_INDEX

            if (masked_target != IGNORE_INDEX).sum().item() == 0:
                gt_roi = (loc_gt.squeeze(1) > 0)
                masked_target = dam_gt.clone()
                masked_target[~gt_roi] = IGNORE_INDEX

            loss = F.cross_entropy(dam_logit, masked_target, ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        val_metrics = evaluate_two_stage(model, val_loader, device, args.loc_threshold)
        print(f"[dam] epoch={epoch} loss={running/max(1,len(train_loader)):.4f} val_f1_loc={val_metrics['f1_loc']:.2f} val_f1_dam={val_metrics['f1_dam']:.2f} val_f1_avg={val_metrics['f1_avg']:.2f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "stage2_last.pt")
        if val_metrics["f1_avg"] > best_avg:
            best_avg = val_metrics["f1_avg"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, ckpt_dir / "stage2_best.pt")

    best_stage2 = torch.load(ckpt_dir / "stage2_best.pt", map_location=device)
    model.load_state_dict(best_stage2["model"])
    final_test = evaluate_two_stage(model, val_loader, device, args.loc_threshold)
    final_hold = evaluate_two_stage(model, eval_loader, device, args.loc_threshold)
    save_json(metrics_dir / "stage2_test.json", final_test)
    save_json(metrics_dir / "stage2_hold.json", final_hold)
    print("stage2_test", final_test)
    print("stage2_hold", final_hold)


if __name__ == "__main__":
    main()
