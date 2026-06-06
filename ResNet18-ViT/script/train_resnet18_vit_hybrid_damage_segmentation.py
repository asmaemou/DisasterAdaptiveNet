#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser("ResNet18-ViT Hybrid Damage Segmentation")

    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--train-split", nargs="+", default=["train"])
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vit-dim", type=int, default=192)
    parser.add_argument("--vit-depth", type=int, default=4)
    parser.add_argument("--vit-heads", type=int, default=6)
    parser.add_argument("--vit-patch-size", type=int, default=32)

    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--class-weight-scan-limit", type=int, default=2000)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_target_path(targets_dir: Path, post_path: Path):
    stem = post_path.stem
    suffixes = [post_path.suffix, ".png", ".tif", ".tiff", ".jpg", ".jpeg"]

    candidate_stems = [
        stem,
        stem.replace("_post_disaster", ""),
        stem.replace("_post_disaster", "_target"),
        stem.replace("_post_disaster", "_damage"),
        stem.replace("_post_disaster", "_mask"),
        stem.replace("_post_disaster", "_label"),
    ]

    for cstem in candidate_stems:
        for ext in suffixes:
            p = targets_dir / f"{cstem}{ext}"
            if p.exists():
                return p

    if "_post_disaster" in stem:
        prefix = stem.split("_post_disaster")[0]
        matches = []
        for ext in IMAGE_EXTS:
            matches.extend(targets_dir.glob(prefix + "*" + ext))
        matches = sorted([p for p in matches if p.is_file()])
        if len(matches) == 1:
            return matches[0]

    return None


def collect_samples(root: Path, splits):
    samples = []

    for split in splits:
        images_dir = root / split / "images"
        targets_dir = root / split / "targets"

        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {images_dir}")
        if not targets_dir.exists():
            raise FileNotFoundError(f"Missing targets directory: {targets_dir}")

        post_images = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "_post_disaster" in p.name
        )

        split_samples = []
        missing = 0

        for post_path in post_images:
            pre_path = images_dir / post_path.name.replace("_post_disaster", "_pre_disaster")
            target_path = find_target_path(targets_dir, post_path)

            if not pre_path.exists() or target_path is None:
                missing += 1
                continue

            split_samples.append((pre_path, post_path, target_path))

        samples.extend(split_samples)

        print(
            f"Split={split} | post images={len(post_images)} | usable samples={len(split_samples)} | missing={missing}",
            flush=True,
        )

    if len(samples) == 0:
        raise RuntimeError(f"No usable paired samples found under {root}")

    return samples


class XBDStyleDataset(Dataset):
    def __init__(self, samples, img_size, train):
        self.samples = samples
        self.img_size = img_size
        self.train = train
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(6, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(6, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_path, post_path, target_path = self.samples[idx]

        pre = Image.open(pre_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        post = Image.open(post_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(target_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

        if self.train:
            if random.random() < 0.5:
                pre = pre.transpose(Image.FLIP_LEFT_RIGHT)
                post = post.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                pre = pre.transpose(Image.FLIP_TOP_BOTTOM)
                post = post.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        pre = np.asarray(pre, dtype=np.float32) / 255.0
        post = np.asarray(post, dtype=np.float32) / 255.0

        x = np.concatenate([pre, post], axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        x = (x - self.mean) / self.std

        y = np.asarray(mask, dtype=np.int64)
        y = np.clip(y, 0, 4)
        y = torch.from_numpy(y).long()

        return {"image": x, "mask": y}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18ViTHybridDamageSeg(nn.Module):
    def __init__(
        self,
        num_classes=5,
        vit_dim=192,
        vit_depth=4,
        vit_heads=6,
        vit_patch_size=32,
    ):
        super().__init__()

        resnet = torchvision.models.resnet18(weights=None)

        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            6,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.patch_embed = nn.Conv2d(
            6,
            vit_dim,
            kernel_size=vit_patch_size,
            stride=vit_patch_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=vit_heads,
            dim_feedforward=vit_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.vit = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)
        self.vit_norm = nn.LayerNorm(vit_dim)

        self.fuse = ConvBlock(512 + vit_dim, 256)
        self.dec3 = ConvBlock(256 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.dec0 = ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        input_hw = x.shape[-2:]

        x0 = self.relu(self.bn1(self.conv1(x)))  # H/2
        x1 = self.maxpool(x0)

        c1 = self.layer1(x1)  # H/4
        c2 = self.layer2(c1)  # H/8
        c3 = self.layer3(c2)  # H/16
        c4 = self.layer4(c3)  # H/32

        vt = self.patch_embed(x)
        b, d, h, w = vt.shape
        tokens = vt.flatten(2).transpose(1, 2)
        tokens = self.vit(tokens)
        tokens = self.vit_norm(tokens)
        vt = tokens.transpose(1, 2).reshape(b, d, h, w)
        vt = F.interpolate(vt, size=c4.shape[-2:], mode="bilinear", align_corners=False)

        y = self.fuse(torch.cat([c4, vt], dim=1))

        y = F.interpolate(y, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, c3], dim=1))

        y = F.interpolate(y, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, c2], dim=1))

        y = F.interpolate(y, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, c1], dim=1))

        y = F.interpolate(y, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec0(torch.cat([y, x0], dim=1))

        y = F.interpolate(y, size=input_hw, mode="bilinear", align_corners=False)
        return self.head(y)


def compute_class_weights(samples, scan_limit):
    counts = np.zeros(5, dtype=np.float64)

    selected = samples
    if scan_limit > 0 and len(samples) > scan_limit:
        selected = random.sample(samples, scan_limit)

    for _, _, target_path in selected:
        mask = Image.open(target_path).convert("L")
        arr = np.asarray(mask, dtype=np.int64)
        arr = np.clip(arr, 0, 4)
        counts += np.bincount(arr.reshape(-1), minlength=5)[:5]

    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(freq + 1e-6)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.1, 10.0)
    weights[0] = min(weights[0], 0.25)

    print("Class counts:", counts.astype(np.int64).tolist(), flush=True)
    print("Class weights:", weights.tolist(), flush=True)

    return torch.tensor(weights, dtype=torch.float32)


def f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return float((2 * tp) / denom)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()

    tp = np.zeros(5, dtype=np.float64)
    fp = np.zeros(5, dtype=np.float64)
    fn = np.zeros(5, dtype=np.float64)

    loc_tp = 0.0
    loc_fp = 0.0
    loc_fn = 0.0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)

        pred = torch.argmax(logits, dim=1)

        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()

        for c in range(5):
            p = pred_np == c
            t = y_np == c
            tp[c] += np.logical_and(p, t).sum()
            fp[c] += np.logical_and(p, np.logical_not(t)).sum()
            fn[c] += np.logical_and(np.logical_not(p), t).sum()

        pb = pred > 0
        tb = y > 0

        loc_tp += torch.logical_and(pb, tb).sum().item()
        loc_fp += torch.logical_and(pb, torch.logical_not(tb)).sum().item()
        loc_fn += torch.logical_and(torch.logical_not(pb), tb).sum().item()

    class_f1 = [f1(tp[c], fp[c], fn[c]) for c in range(5)]

    loc_f1 = f1(loc_tp, loc_fp, loc_fn)
    no_damage = class_f1[1]
    minor = class_f1[2]
    major = class_f1[3]
    destroyed = class_f1[4]
    macro = float(np.mean([no_damage, minor, major, destroyed]))
    score = 0.5 * loc_f1 + 0.5 * macro

    return {
        "localization_f1": loc_f1,
        "no_damage_f1": no_damage,
        "minor_damage_f1": minor,
        "major_damage_f1": major,
        "destroyed_f1": destroyed,
        "macro_f1": macro,
        "selection_score": score,
    }


def metric_line(name, m):
    return (
        f"{name} | loc={m['localization_f1']:.6f} | "
        f"no={m['no_damage_f1']:.6f} | minor={m['minor_damage_f1']:.6f} | "
        f"major={m['major_damage_f1']:.6f} | destroyed={m['destroyed_f1']:.6f} | "
        f"macro={m['macro_f1']:.6f} | score={m['selection_score']:.6f}"
    )


def adjust_lr(optimizer, epoch, step_ratio, args):
    current = epoch + step_ratio

    if args.warmup_epochs > 0 and current < args.warmup_epochs:
        lr = args.lr * max(current / args.warmup_epochs, 1e-6)
    else:
        denom = max(args.epochs - args.warmup_epochs, 1)
        progress = min(max((current - args.warmup_epochs) / denom, 0.0), 1.0)
        lr = 0.5 * args.lr * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def load_checkpoint(model, ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}", flush=True)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean = {}
    for k, v in state.items():
        clean[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"Loaded checkpoint | missing={len(missing)} | unexpected={len(unexpected)}", flush=True)


def save_results(output_dir, best_epoch, best_val, test_metrics, args):
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "ResNet18-ViT Hybrid Damage Segmentation",
        "best_epoch": best_epoch,
        "best_validation": best_val,
        "test": test_metrics,
        "args": vars(args),
    }

    json_path = scores_dir / "scores_resnet18_vit_hybrid_damage.json"
    txt_path = scores_dir / "summary_resnet18_vit_hybrid_damage.txt"

    json_path.write_text(json.dumps(result, indent=2))

    txt = f"""Experiment: ResNet18-ViT Hybrid Damage Segmentation
Best validation epoch: {best_epoch}

Test Localization F1: {test_metrics['localization_f1']:.6f}
No Damage F1:         {test_metrics['no_damage_f1']:.6f}
Minor Damage F1:      {test_metrics['minor_damage_f1']:.6f}
Major Damage F1:      {test_metrics['major_damage_f1']:.6f}
Destroyed F1:         {test_metrics['destroyed_f1']:.6f}
Macro-F1:             {test_metrics['macro_f1']:.6f}
Selection Score:      {test_metrics['selection_score']:.6f}
"""

    txt_path.write_text(txt)

    print(txt, flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.dataset_root)

    print("===== ResNet18-ViT Hybrid Damage Segmentation =====", flush=True)
    print(f"Dataset root: {root}", flush=True)
    print(f"Train split: {args.train_split}", flush=True)
    print(f"Val split: {args.val_split}", flush=True)
    print(f"Test split: {args.test_split}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print("===================================================", flush=True)

    train_samples = collect_samples(root, args.train_split)
    val_samples = collect_samples(root, [args.val_split])
    test_samples = collect_samples(root, [args.test_split])

    print(f"# Train: {len(train_samples)}", flush=True)
    print(f"# Val:   {len(val_samples)}", flush=True)
    print(f"# Test:  {len(test_samples)}", flush=True)

    train_ds = XBDStyleDataset(train_samples, args.img_size, train=True)
    val_ds = XBDStyleDataset(val_samples, args.img_size, train=False)
    test_ds = XBDStyleDataset(test_samples, args.img_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    model = ResNet18ViTHybridDamageSeg(
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        vit_patch_size=args.vit_patch_size,
    ).to(device)

    if args.resume_checkpoint:
        load_checkpoint(model, Path(args.resume_checkpoint), device)

    class_weights = compute_class_weights(train_samples, args.class_weight_scan_limit).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    best_epoch = 0
    best_val = None
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            lr = adjust_lr(optimizer, epoch - 1, (step - 1) / max(len(train_loader), 1), args)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / max(args.grad_accum_steps, 1)

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * max(args.grad_accum_steps, 1)

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | step {step:04d}/{len(train_loader)} | "
                    f"lr={lr:.8f} | loss={running_loss / step:.6f}",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device, use_amp)
        elapsed = (time.time() - start) / 60.0

        print(
            f"Epoch {epoch:03d} | train_loss={running_loss / len(train_loader):.6f} | "
            f"time={elapsed:.1f} min | {metric_line('val', val_metrics)}",
            flush=True,
        )

        if val_metrics["selection_score"] > best_score:
            best_score = val_metrics["selection_score"]
            best_epoch = epoch
            best_val = val_metrics

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_metric": best_score,
                    "best_val": best_val,
                    "args": vars(args),
                },
                best_path,
            )

            print(f"Saved best checkpoint: {best_path}", flush=True)

    print(f"Training done. Best epoch={best_epoch}, best score={best_score:.6f}", flush=True)

    load_checkpoint(model, best_path, device)

    test_metrics = evaluate(model, test_loader, device, use_amp)
    print(metric_line("test", test_metrics), flush=True)

    save_results(output_dir, best_epoch, best_val, test_metrics, args)


if __name__ == "__main__":
    main()