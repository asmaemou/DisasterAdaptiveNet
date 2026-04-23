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
            raise FileNotFoundError(f"No pre-disaster targets found in {self.target_dir}")

        self.samples: List[Dict[str, Path]] = []
        for pre_m in self.pre_masks:
            post_name = pre_m.name.replace("_pre_disaster_target.png", "_post_disaster_target.png")
            if post_name == pre_m.name:
                post_name = pre_m.name.replace("_pre_", "_post_")
            post_m = self.target_dir / post_name

            pre_img = self.image_dir / pre_m.name.replace("_target.png", ".png")
            post_img = self.image_dir / post_m.name.replace("_target.png", ".png")

            if post_m.exists() and pre_img.exists() and post_img.exists():
                self.samples.append(
                    {
                        "pre_img": pre_img,
                        "post_img": post_img,
                        "pre_mask": pre_m,
                        "post_mask": post_m,
                    }
                )

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


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep_prob)
        return x * rnd / keep_prob


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        x_tokens = x.flatten(2).transpose(1, 2)
        q = self.q(x_tokens).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_sr = self.sr(x)
            x_sr = x_sr.flatten(2).transpose(1, 2)
            x_sr = self.norm(x_sr)
        else:
            x_sr = x_tokens

        kv = self.kv(x_sr).reshape(b, x_sr.shape[1], 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out


class MixFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MixFFN(dim, mlp_ratio)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MiTStage(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, depth: int, num_heads: int, sr_ratio: int, patch_size: int, stride: int, drop_path_rates: List[float]):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_ch, embed_dim, patch_size, stride)
        self.blocks = nn.Sequential(*[
            MiTBlock(embed_dim, num_heads, sr_ratio, 4.0, drop_path_rates[i])
            for i in range(depth)
        ])
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class MixTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        embed_dims: Tuple[int, int, int, int] = (32, 64, 160, 256),
        depths: Tuple[int, int, int, int] = (3, 4, 6, 3),
        num_heads: Tuple[int, int, int, int] = (1, 2, 5, 8),
        sr_ratios: Tuple[int, int, int, int] = (8, 4, 2, 1),
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        cur = 0
        self.stage1 = MiTStage(in_ch, embed_dims[0], depths[0], num_heads[0], sr_ratios[0], 7, 4, dpr[cur:cur + depths[0]])
        cur += depths[0]
        self.stage2 = MiTStage(embed_dims[0], embed_dims[1], depths[1], num_heads[1], sr_ratios[1], 3, 2, dpr[cur:cur + depths[1]])
        cur += depths[1]
        self.stage3 = MiTStage(embed_dims[1], embed_dims[2], depths[2], num_heads[2], sr_ratios[2], 3, 2, dpr[cur:cur + depths[2]])
        cur += depths[2]
        self.stage4 = MiTStage(embed_dims[2], embed_dims[3], depths[3], num_heads[3], sr_ratios[3], 3, 2, dpr[cur:cur + depths[3]])

    def forward(self, x):
        feats = []
        x = self.stage1(x); feats.append(x)
        x = self.stage2(x); feats.append(x)
        x = self.stage3(x); feats.append(x)
        x = self.stage4(x); feats.append(x)
        return feats


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class MultitemporalAdaptiveFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.loc_attn = ChannelAttention(channels)
        self.dam_attn = ChannelAttention(channels)

    def forward(self, t1, t2):
        z = torch.cat([t1, t2, torch.abs(t1 - t2)], dim=1)
        z = self.proj(z)
        return self.loc_attn(z), self.dam_attn(z)


class CrossLayerFusion(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]):
        target_hw = feats[0].shape[-2:]
        ups = []
        for feat in feats:
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            ups.append(feat)
        return self.fuse(torch.cat(ups, dim=1))


class DamFormer(nn.Module):
    def __init__(
        self,
        embed_dims: Tuple[int, int, int, int] = (32, 64, 160, 256),
        depths: Tuple[int, int, int, int] = (3, 4, 6, 3),
        num_heads: Tuple[int, int, int, int] = (1, 2, 5, 8),
        sr_ratios: Tuple[int, int, int, int] = (8, 4, 2, 1),
        decoder_dim: int = 128,
        dam_classes: int = 5,
    ):
        super().__init__()
        self.encoder = MixTransformerEncoder(
            in_ch=3,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
        )
        self.fusions = nn.ModuleList([MultitemporalAdaptiveFusion(c) for c in embed_dims])
        self.loc_fuse = CrossLayerFusion(list(embed_dims), decoder_dim)
        self.dam_fuse = CrossLayerFusion(list(embed_dims), decoder_dim)
        self.loc_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.dam_head = nn.Conv2d(decoder_dim, dam_classes, kernel_size=1)

    def forward(self, pre, post):
        t1_feats = self.encoder(pre)
        t2_feats = self.encoder(post)

        loc_feats = []
        dam_feats = []
        for fusion, f1, f2 in zip(self.fusions, t1_feats, t2_feats):
            loc_f, dam_f = fusion(f1, f2)
            loc_feats.append(loc_f)
            dam_feats.append(dam_f)

        loc_fused = self.loc_fuse(loc_feats)
        dam_fused = self.dam_fuse(dam_feats)
        dam_fused = dam_fused + loc_fused

        return self.loc_head(loc_fused), self.dam_head(dam_fused)


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes="present") -> torch.Tensor:
    if probas.numel() == 0:
        return probas * 0.0
    c_count = probas.size(1)
    losses = []
    class_to_sum = range(c_count) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
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
    b, c, h, w = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, c)
    labels = labels.reshape(-1)
    valid = labels != ignore
    probas = probas[valid]
    labels = labels[valid]
    return lovasz_softmax_flat(probas, labels)


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
    ap = argparse.ArgumentParser(description="DamFormer-style building damage assessment on xView2-style data")
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--train-split", default="tier3")
    ap.add_argument("--val-root", default=None)
    ap.add_argument("--val-split", default="test")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--eval-split", default="hold")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=5e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
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

    model = DamFormer().to(device)
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
            dam_ce = F.cross_entropy(dam_logits, dam_gt, ignore_index=IGNORE_INDEX)
            dam_lovasz = lovasz_softmax(torch.softmax(dam_logits, dim=1), dam_gt, ignore=IGNORE_INDEX)
            dam_loss = dam_ce + dam_lovasz
            loss = loc_loss + args.alpha * dam_loss

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
