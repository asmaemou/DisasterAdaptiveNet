#!/usr/bin/env python3
"""Paper-faithful DamFormer reproduction for xBD-format paired imagery.

Architecture follows Chen et al., IGARSS 2022:
  weight-shared MiT-B2 Siamese encoder -> task-specific multitemporal
  adaptive fusion -> lightweight All-MLP localization/damage decoders.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_xbd_hrtbda_v5_swin_pretrained_cascade as legacy


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random.floor() / keep


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel: int, stride: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, channels, kernel_size=kernel, stride=stride,
            padding=kernel // 2,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.projection(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width


class EfficientAttention(nn.Module):
    """MiT spatial-reduction self-attention from SegFormer."""

    def __init__(self, channels: int, heads: int, sr_ratio: int):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("MiT channels must be divisible by attention heads")
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.query = nn.Linear(channels, channels)
        self.key_value = nn.Linear(channels, channels * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(channels)
        self.projection = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        query = self.query(x).reshape(batch, tokens, self.heads, channels // self.heads).permute(0, 2, 1, 3)
        reduced = x
        if self.sr_ratio > 1:
            reduced = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).reshape(batch, channels, -1).transpose(1, 2)
            reduced = self.sr_norm(reduced)
        key_value = self.key_value(reduced).reshape(
            batch, -1, 2, self.heads, channels // self.heads
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.projection(output)


class DepthwiseConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        return self.conv(x).flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        hidden = channels * expansion
        self.fc1 = nn.Linear(channels, hidden)
        self.depthwise = DepthwiseConv(hidden)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.depthwise(x, height, width)
        return self.fc2(self.activation(x))


class MiTBlock(nn.Module):
    def __init__(self, channels: int, heads: int, sr_ratio: int, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = EfficientAttention(channels, heads, sr_ratio)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = MixFFN(channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        return x + self.drop_path(self.ffn(self.norm2(x), height, width))


class MixTransformerB2(nn.Module):
    """Self-contained SegFormer MiT-B2 encoder: depths [3,4,6,3]."""

    def __init__(self, drop_path_rate: float = 0.1):
        super().__init__()
        channels = [64, 128, 320, 512]
        depths = [3, 4, 6, 3]
        heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        self.channels = channels
        self.patch_embeddings = nn.ModuleList([
            OverlapPatchEmbed(3, channels[0], 7, 4),
            OverlapPatchEmbed(channels[0], channels[1], 3, 2),
            OverlapPatchEmbed(channels[1], channels[2], 3, 2),
            OverlapPatchEmbed(channels[2], channels[3], 3, 2),
        ])
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        for channels_i, depth, heads_i, sr_ratio in zip(channels, depths, heads, sr_ratios):
            self.stages.append(nn.ModuleList([
                MiTBlock(channels_i, heads_i, sr_ratio, rates[offset + index])
                for index in range(depth)
            ]))
            self.norms.append(nn.LayerNorm(channels_i))
            offset += depth
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        x = image
        for patch, blocks, norm in zip(self.patch_embeddings, self.stages, self.norms):
            x, height, width = patch(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            batch, _, channels = x.shape
            feature = x.transpose(1, 2).reshape(batch, channels, height, width)
            outputs.append(feature)
            x = feature
        return outputs


class ChannelAttention(nn.Module):
    """CBAM channel attention used by DamFormer's adaptive fusion."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + maximum)


class AdaptiveFusion(nn.Module):
    """Concatenation, convolution, and channel attention from Sec. 2.1."""

    def __init__(self, channels: int):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = ChannelAttention(channels)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return self.channel_attention(self.merge(torch.cat((pre, post), dim=1)))


class AllMLPDecoder(nn.Module):
    """SegFormer-style multi-level projection and 1x1 cross-layer fusion."""

    def __init__(self, channels: List[int], width: int, classes: int):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.cross_layer_fusion = nn.Sequential(
            nn.Conv2d(width * len(channels), width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(width, classes, 1)

    def features(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        output_size = inputs[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, inputs):
            feature = projection(feature)
            if feature.shape[-2:] != output_size:
                feature = F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        return self.cross_layer_fusion(torch.cat(projected, dim=1))

    def classify(self, feature: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.classifier(feature), size=output_size, mode="bilinear", align_corners=False)


class DamFormer(nn.Module):
    """Dual-task Siamese Transformer architecture shown in paper Fig. 1."""

    def __init__(self, decoder_width: int):
        super().__init__()
        self.encoder = MixTransformerB2()
        channels = self.encoder.channels

        # Separate attention modules yield task-specific localization and
        # classification features, as described at the end of Sec. 2.1.
        self.loc_fusion = nn.ModuleList([AdaptiveFusion(c) for c in channels])
        self.damage_fusion = nn.ModuleList([AdaptiveFusion(c) for c in channels])
        self.loc_decoder = AllMLPDecoder(channels, decoder_width, classes=1)
        self.damage_decoder = AllMLPDecoder(channels, decoder_width, classes=5)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The same encoder instance is called twice: weights are shared.
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        loc_features = [m(a, b) for m, a, b in zip(self.loc_fusion, pre_features, post_features)]
        damage_features = [m(a, b) for m, a, b in zip(self.damage_fusion, pre_features, post_features)]
        loc_feature = self.loc_decoder.features(loc_features)
        damage_feature = self.damage_decoder.features(damage_features)
        # Paper Sec. 2.2: localization multi-level feature is added to the
        # classification sub-network before the damage classifier.
        damage_feature = damage_feature + loc_feature
        size = pre.shape[-2:]
        loc = self.loc_decoder.classify(loc_feature, size).squeeze(1)
        damage = self.damage_decoder.classify(damage_feature, size)
        return {"loc": loc, "damage": damage}


def make_loaders(args: argparse.Namespace):
    # DamFormer uses full image pairs; rare-class crop sampling is intentionally
    # absent because it is not part of the published architecture/training loss.
    train_ds = legacy.XBDHRTBDADataset(
        args.data_root, args.train_split, args.img_size, training=True,
        crop_size=args.img_size, crop_candidate_count=1, extra_photometric=False,
    )
    val_ds = legacy.XBDHRTBDADataset(args.data_root, args.val_split, args.img_size, training=False)
    test_ds = legacy.XBDHRTBDADataset(args.data_root, args.test_split, args.img_size, training=False)
    common = dict(num_workers=args.num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, **common),
    )


def localization_loss(logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2))
    denominator = probability.sum((1, 2)) + target.sum((1, 2))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice, bce, dice


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    pixels = gt_sorted.numel()
    total_positive = gt_sorted.sum()
    intersection = total_positive - gt_sorted.float().cumsum(0)
    union = total_positive + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-7)
    if pixels > 1:
        # Do not update overlapping slices in place. PyTorch 2.11 rejects
        # that older Lovasz implementation pattern because the right-hand
        # slice aliases the tensor being written.
        jaccard = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
    return jaccard


def lovasz_softmax_flat(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if probabilities.numel() == 0:
        return probabilities.sum() * 0.0
    losses = []
    for cls in range(probabilities.shape[1]):
        foreground = (labels == cls).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, cls]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground_sorted)))
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0.0


def lovasz_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, probabilities.shape[1])
    return lovasz_softmax_flat(probabilities, labels.reshape(-1))


def damage_loss(logits: torch.Tensor, target5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.float()
    cross_entropy = F.cross_entropy(logits, target5.long())
    lovasz = lovasz_softmax(logits, target5.long())
    return cross_entropy + lovasz, cross_entropy, lovasz


def empty_counts() -> Dict:
    return {"loc_tp": 0, "loc_fp": 0, "loc_fn": 0, **{c: {"tp": 0, "fp": 0, "fn": 0} for c in range(1, 5)}}


def update_counts(pred: torch.Tensor, loc_pred: torch.Tensor, loc_true: torch.Tensor, target5: torch.Tensor, counts: Dict) -> None:
    counts["loc_tp"] += int(((loc_pred == 1) & (loc_true == 1)).sum())
    counts["loc_fp"] += int(((loc_pred == 1) & (loc_true == 0)).sum())
    counts["loc_fn"] += int(((loc_pred == 0) & (loc_true == 1)).sum())
    valid = (target5 >= 1) & (target5 <= 4)
    predicted, truth = pred[valid], target5[valid]
    for cls in range(1, 5):
        counts[cls]["tp"] += int(((predicted == cls) & (truth == cls)).sum())
        counts[cls]["fp"] += int(((predicted == cls) & (truth != cls)).sum())
        counts[cls]["fn"] += int(((predicted != cls) & (truth == cls)).sum())


def summarize(counts: Dict, threshold: float) -> Dict[str, float]:
    loc = legacy.F1Recorder(counts["loc_tp"], counts["loc_fp"], counts["loc_fn"])
    classes = [legacy.F1Recorder(**counts[c]) for c in range(1, 5)]
    damage = legacy.harmonic_mean([record.f1 for record in classes])
    return {
        "score": 0.3 * loc.f1 + 0.7 * damage,
        "localization_f1": loc.f1,
        "damage_f1": damage,
        "damage_f1_no_damage": classes[0].f1,
        "damage_f1_minor_damage": classes[1].f1,
        "damage_f1_major_damage": classes[2].f1,
        "damage_f1_destroyed": classes[3].f1,
        "localization_threshold": threshold,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, thresholds: List[float]) -> Dict[str, float]:
    model.eval()
    cached = []
    for batch in loader:
        output = model(batch["pre"].to(device), batch["post"].to(device))
        cached.append((
            torch.sigmoid(output["loc"].float()).cpu(),
            output["damage"].float().argmax(1).cpu(),
            batch["loc"].long(), batch["target5"].long(),
        ))
    best = None
    for threshold in thresholds:
        counts = empty_counts()
        for loc_probability, damage_prediction, loc_true, target5 in cached:
            loc_prediction = (loc_probability >= threshold).long()
            # The classification decoder contains a background class. Masking
            # it again with localization matches the two-task final prediction.
            prediction = damage_prediction * loc_prediction
            update_counts(prediction, loc_prediction, loc_true, target5, counts)
        result = summarize(counts, threshold)
        if best is None or result["score"] > best["score"]:
            best = result
    return best


def print_result(label: str, result: Dict[str, float]) -> None:
    print(
        f"{label} | score={result['score']:.6f} | loc={result['localization_f1']:.6f} | "
        f"damage={result['damage_f1']:.6f} | no={result['damage_f1_no_damage']:.6f} | "
        f"minor={result['damage_f1_minor_damage']:.6f} | major={result['damage_f1_major_damage']:.6f} | "
        f"destroyed={result['damage_f1_destroyed']:.6f} | loc_th={result['localization_threshold']:.2f}",
        flush=True,
    )


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, result: Dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        "best_metric": result["score"], "best_results": result, "args": vars(args),
    }, path)


def train(args: argparse.Namespace, device: torch.device) -> Path:
    train_loader, val_loader, _ = make_loaders(args)
    model = DamFormer(args.decoder_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output = Path(args.output_dir)
    best, best_epoch, history = None, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = legacy.AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            pre, post = batch["pre"].to(device), batch["post"].to(device)
            loc, target5 = batch["loc"].to(device), batch["target5"].to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                prediction = model(pre, post)
            loc_total, loc_bce, loc_dice = localization_loss(prediction["loc"], loc)
            dam_total, dam_ce, dam_lovasz = damage_loss(prediction["damage"], target5)
            loss = loc_total + args.damage_weight * dam_total
            components = {"total": loss, "loc_bce": loc_bce, "loc_dice": loc_dice, "damage_ce": dam_ce, "lovasz": dam_lovasz}
            bad = {name: float(value.detach()) for name, value in components.items() if not torch.isfinite(value).all()}
            if bad:
                raise FloatingPointError(f"Non-finite loss epoch={epoch}, step={step}: {bad}")
            scaler.scale(loss / args.grad_accum_steps).backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            meter.update(float(loss.detach()), pre.shape[0])
            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={meter.avg:.4f} loc={float(loc_total.detach()):.4f} "
                    f"damage={float(dam_total.detach()):.4f}", flush=True,
                )

        progress = epoch / max(args.epochs, 1)
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        result = evaluate(model, val_loader, device, args.thresholds)
        print_result(f"Validation epoch {epoch}", result)
        history.append({"epoch": epoch, "train_loss": meter.avg, "lr": lr, **result})
        (output / "history.json").write_text(json.dumps(history, indent=2))
        save_checkpoint(output / "checkpoints" / "last.pt", model, optimizer, epoch, result, args)
        if best is None or result["score"] > best["score"]:
            best, best_epoch = result, epoch
            save_checkpoint(output / "checkpoints" / "best.pt", model, optimizer, epoch, result, args)
            print(f"Saved new best DamFormer checkpoint at epoch {epoch}.", flush=True)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping after no improvement since epoch {best_epoch}.", flush=True)
            break
    return output / "checkpoints" / "best.pt"


def test(args: argparse.Namespace, device: torch.device, checkpoint: Path) -> Dict[str, float]:
    _, _, test_loader = make_loaders(args)
    model = DamFormer(args.decoder_width).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    threshold = float(saved["best_results"]["localization_threshold"])
    result = evaluate(model, test_loader, device, [threshold])
    result["checkpoint_epoch"] = int(saved["epoch"])
    print_result("FINAL TEST", result)
    scores = Path(args.output_dir) / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / "test_results.json").write_text(json.dumps(result, indent=2))
    (scores / "summary.txt").write_text("\n".join(f"{key}: {value}" for key, value in result.items()) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Paper-faithful DamFormer training")
    parser.add_argument("--phase", choices=["train_test", "train", "test"], default="train_test")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--decoder-width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--damage-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65, 0.75])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy.set_seed(args.seed)
    output = Path(args.output_dir)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | Architecture: paper-faithful DamFormer (Siamese MiT-B2) | "
        f"Encoder initialization: SegFormer MiT initialization | Data: {args.data_root}", flush=True,
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else output / "checkpoints" / "best.pt"
    if args.phase in {"train", "train_test"}:
        checkpoint = train(args, device)
    if args.phase in {"test", "train_test"}:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        test(args, device, checkpoint)


if __name__ == "__main__":
    main()
