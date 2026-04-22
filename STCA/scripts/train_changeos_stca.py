#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

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
        self.pre_masks = sorted(self.target_dir.glob("*_pre_*_target.png"))
        if not self.pre_masks:
            raise FileNotFoundError(f"No pre-disaster target png files found in {self.target_dir}")
        self.samples: List[Dict[str, Path]] = []
        for pre_m in self.pre_masks:
            post_m = self.target_dir / pre_m.name.replace("_pre_", "_post_")
            pre_img = self.image_dir / pre_m.name.replace("_target.png", "")
            post_img = self.image_dir / post_m.name.replace("_target.png", "")
            if not post_m.exists() or not pre_img.exists() or not post_img.exists():
                continue
            self.samples.append(dict(pre_img=pre_img, post_img=post_img, pre_mask=pre_m, post_mask=post_m))
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.split_dir}")

        self.tiles = self._build_tiles()

    def _build_tiles(self) -> Optional[List[Tuple[int, int, int]]]:
        if self.mode == "eval":
            return None
        tiles = []
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
        # random crop is already done before augment when mode=train
        if random.random() < 0.5:
            pre = np.flip(pre, axis=1).copy(); post = np.flip(post, axis=1).copy()
            pre_mask = np.flip(pre_mask, axis=1).copy(); post_mask = np.flip(post_mask, axis=1).copy()
        if random.random() < 0.5:
            pre = np.flip(pre, axis=0).copy(); post = np.flip(post, axis=0).copy()
            pre_mask = np.flip(pre_mask, axis=0).copy(); post_mask = np.flip(post_mask, axis=0).copy()
        k = random.randint(0, 3)
        if k:
            pre = np.rot90(pre, k).copy(); post = np.rot90(post, k).copy()
            pre_mask = np.rot90(pre_mask, k).copy(); post_mask = np.rot90(post_mask, k).copy()
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

        pre_t = torch.from_numpy(pre).permute(2,0,1).float()
        post_t = torch.from_numpy(post).permute(2,0,1).float()
        # official ChangeOS normalization values from inference SDK
        mean = torch.tensor([123.675,116.28,103.53]).view(3,1,1)
        std = torch.tensor([58.395,57.12,57.375]).view(3,1,1)
        pre_t = (pre_t - mean) / std
        post_t = (post_t - mean) / std
        x = torch.cat([pre_t, post_t], dim=0)
        y = {
            "masks": [torch.from_numpy((pre_mask > 0).astype(np.uint8)), torch.from_numpy(post_mask.astype(np.uint8))],
            "image_filename": name,
        }
        return x, y


class XView2TargetPreDataset(Dataset):
    def __init__(self, split_dir: str, crop_size: int = 512, stride: int = 256):
        self.ds = XView2ChangeOSDataset(split_dir, mode="train", crop_size=crop_size, stride=stride)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x[:3], y["image_filename"]


def make_model(backbone: str) -> ChangeOS:
    cfg_mod = {
        "r18": cos_r18,
        "r34": cos_r34,
        "r50": cos_r50,
        "r101": cos_r101,
        "swint": cos_swint,
    }[backbone]
    model = er.builder.make_model(cfg_mod.config["model"])
    return model


def reduce_losses(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for v in loss_dict.values():
        total = v if total is None else total + v
    if total is None:
        raise RuntimeError("Empty loss dict")
    return total


def extract_single_features(model: ChangeOS, img: torch.Tensor, branch: str) -> torch.Tensor:
    features = model.encoder(img)
    if branch == "loc":
        return model.decoder.loc_neck(features)
    if branch == "dam":
        return model.decoder.dam_neck(features)
    raise ValueError(branch)


def downsample_mask(mask: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float().unsqueeze(1), size=hw, mode="nearest").squeeze(1).long()


def sample_indices_from_scores(scores: torch.Tensor, valid: torch.Tensor, n: int) -> torch.Tensor:
    flat_scores = scores.flatten()
    flat_valid = valid.flatten()
    idx = torch.nonzero(flat_valid, as_tuple=False).flatten()
    if idx.numel() == 0:
        return idx
    vals = flat_scores[idx]
    k = min(n, idx.numel())
    order = torch.topk(vals, k=k, largest=True).indices
    return idx[order]


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes='present') -> torch.Tensor:
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = range(C) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return torch.tensor(0.0, device=probas.device)
    return torch.mean(torch.stack(losses))


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore: int = 255) -> torch.Tensor:
    B, C, H, W = probas.shape
    probas = probas.permute(0,2,3,1).reshape(-1, C)
    labels = labels.reshape(-1)
    valid = labels != ignore
    probas = probas[valid]
    labels = labels[valid]
    return lovasz_softmax_flat(probas, labels)


def compute_pair_loss(model: ChangeOS, src_post: torch.Tensor, src_post_mask: torch.Tensor,
                      tgt_pre: torch.Tensor, threshold: float, src_samples: int, tgt_samples: int,
                      stca_loss: str = "lovasz") -> torch.Tensor:
    device = src_post.device
    src_feat = extract_single_features(model, src_post, branch="dam")
    tgt_feat = extract_single_features(model, tgt_pre, branch="loc")
    tgt_loc_logit = model.head.loc_cls(tgt_feat, upsample=False)
    tgt_prob = torch.sigmoid(tgt_loc_logit)

    src_mask_ds = downsample_mask(src_post_mask, src_feat.shape[-2:])
    losses: List[torch.Tensor] = []

    B, C, Hs, Ws = src_feat.shape
    Ht, Wt = tgt_feat.shape[-2:]
    for b in range(B):
        src_valid = src_mask_ds[b] > 1  # focus on damaged classes only
        tgt_valid = (tgt_prob[b, 0] > threshold)
        src_idx = sample_indices_from_scores(src_mask_ds[b].float(), src_valid, src_samples)
        tgt_idx = sample_indices_from_scores(tgt_prob[b, 0], tgt_valid, tgt_samples)
        if src_idx.numel() == 0 or tgt_idx.numel() == 0:
            continue

        src_vec = src_feat[b].reshape(C, -1)[:, src_idx]  # [C, k]
        tgt_vec = tgt_feat[b].reshape(C, -1)[:, tgt_idx]  # [C, m]
        k = src_vec.shape[1]
        m = tgt_vec.shape[1]
        src_lab = src_mask_ds[b].reshape(-1)[src_idx]  # [k]

        pair_cat = torch.cat([
            tgt_vec.unsqueeze(2).expand(C, m, k).permute(0, 2, 1),
            src_vec.unsqueeze(1).expand(C, k, m),
        ], dim=0).unsqueeze(0)  # [1, 2C, k, m]
        pair_fused = model.decoder.fuse_conv(pair_cat)
        logits = model.head.dam_cls(pair_fused, upsample=False)  # [1, 5, k, m]
        labels = src_lab[:, None].expand(k, m).unsqueeze(0).to(device)

        if stca_loss == "lovasz":
            prob = torch.softmax(logits, dim=1)
            loss = lovasz_softmax(prob, labels, ignore=IGNORE_INDEX)
        else:
            loss = F.cross_entropy(logits, labels.long(), ignore_index=IGNORE_INDEX)
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def binary_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2*tp + fp + fn
    return 0.0 if denom == 0 else (2*tp) / denom


def class_f1(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    valid = gt != IGNORE_INDEX
    pred_c = (pred == cls) & valid
    gt_c = (gt == cls) & valid
    tp = np.logical_and(pred_c, gt_c).sum()
    fp = np.logical_and(pred_c, ~gt_c).sum()
    fn = np.logical_and(~pred_c, gt_c).sum()
    denom = 2*tp + fp + fn
    return 0.0 if denom == 0 else (2*tp) / denom


@torch.no_grad()
def evaluate(model: ChangeOS, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loc_scores, reg_scores, min_scores, maj_scores, des_scores = [], [], [], [], []
    for x, y in loader:
        x = x.to(device)
        features = tc.bitemporal_forward(model.encoder, x)
        t1_features, st_features = model.decoder(*features)
        loc_logit = model.head.loc_cls(t1_features)
        dam_logit = model.head.dam_cls(st_features)
        loc_pred, dam_pred = model.head.object_based_infer(loc_logit, dam_logit, logit=True)
        loc_pred = loc_pred.cpu().numpy().squeeze(1)
        dam_pred = dam_pred.cpu().numpy()
        gt_loc = y["masks"][0].numpy()
        gt_dam = y["masks"][1].numpy()
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
    f1_dam = (f1_regular * f1_minor * f1_major * f1_destroyed)
    f1_dam = 0.0 if min(f1_regular, f1_minor, f1_major, f1_destroyed) == 0 else f1_dam ** 0.25
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


def save_json(path: Path, obj: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-root", required=True, help="Root containing split/images and split/targets")
    ap.add_argument("--source-split", default="tier3")
    ap.add_argument("--target-split", default="train")
    ap.add_argument("--val-split", default="test")
    ap.add_argument("--eval-split", default="hold")
    ap.add_argument("--backbone", choices=["r18","r34","r50","r101","swint"], default="r18")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--epochs-source", type=int, default=10)
    ap.add_argument("--epochs-stca", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--stca-weight", type=float, default=0.2)
    ap.add_argument("--target-threshold", type=float, default=0.5)
    ap.add_argument("--target-feature-samples", type=int, default=256)
    ap.add_argument("--source-feature-samples", type=int, default=256)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stca-loss", choices=["ce","lovasz"], default="lovasz")
    args = ap.parse_args()

    seed_everything(args.seed)
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / "checkpoints"
    metrics_dir = work_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    src_dir = os.path.join(args.prepared_root, args.source_split)
    tgt_dir = os.path.join(args.prepared_root, args.target_split)
    val_dir = os.path.join(args.prepared_root, args.val_split)
    eval_dir = os.path.join(args.prepared_root, args.eval_split)

    train_ds = XView2ChangeOSDataset(src_dir, mode="train", crop_size=args.crop_size, stride=args.stride)
    tgt_ds = XView2TargetPreDataset(tgt_dir, crop_size=args.crop_size, stride=args.stride)
    val_ds = XView2ChangeOSDataset(val_dir, mode="eval")
    eval_ds = XView2ChangeOSDataset(eval_dir, mode="eval")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

    best_source = -1.0
    print("=== Stage 1: supervised ChangeOS training on source ===")
    for epoch in range(1, args.epochs_source + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            yy = {"masks": [y["masks"][0].to(device), y["masks"][1].to(device)]}
            loss_dict = model(x, yy)
            loss = reduce_losses(loss_dict)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        val_metrics = evaluate(model, val_loader, device)
        print(f"[source] epoch={epoch} loss={running/max(1,len(train_loader)):.4f} val_f1_avg={val_metrics['f1_avg']:.2f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "source_last.pt")
        if val_metrics["f1_avg"] > best_source:
            best_source = val_metrics["f1_avg"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, ckpt_dir / "source_best.pt")

    source_test = evaluate(model, val_loader, device)
    source_hold = evaluate(model, eval_loader, device)
    save_json(metrics_dir / "source_test.json", source_test)
    save_json(metrics_dir / "source_hold.json", source_hold)
    print("source_test", source_test)
    print("source_hold", source_hold)

    # reload best source before STCA adaptation
    best_ckpt = torch.load(ckpt_dir / "source_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    print("=== Stage 2: STCA adaptation with ChangeOS backbone ===")
    best_stca = -1.0
    for epoch in range(1, args.epochs_stca + 1):
        model.train()
        running = 0.0
        tgt_iter = iter(tgt_loader)
        for x, y in train_loader:
            try:
                tgt_pre, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_pre, _ = next(tgt_iter)
            x = x.to(device)
            tgt_pre = tgt_pre.to(device)
            src_post = x[:, 3:]
            src_post_mask = y["masks"][1].to(device)
            stca_loss = compute_pair_loss(
                model, src_post, src_post_mask, tgt_pre,
                threshold=args.target_threshold,
                src_samples=args.source_feature_samples,
                tgt_samples=args.target_feature_samples,
                stca_loss=args.stca_loss,
            )
            loss = args.stca_weight * stca_loss
            if not torch.isfinite(loss) or float(loss.item()) == 0.0:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        val_metrics = evaluate(model, val_loader, device)
        print(f"[stca] epoch={epoch} loss={running/max(1,len(train_loader)):.4f} val_f1_avg={val_metrics['f1_avg']:.2f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "stca_last.pt")
        if val_metrics["f1_avg"] > best_stca:
            best_stca = val_metrics["f1_avg"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics}, ckpt_dir / "stca_best.pt")

    best_ckpt = torch.load(ckpt_dir / "stca_best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    stca_test = evaluate(model, val_loader, device)
    stca_hold = evaluate(model, eval_loader, device)
    save_json(metrics_dir / "stca_test.json", stca_test)
    save_json(metrics_dir / "stca_hold.json", stca_hold)
    print("stca_test", stca_test)
    print("stca_hold", stca_hold)


if __name__ == "__main__":
    main()
