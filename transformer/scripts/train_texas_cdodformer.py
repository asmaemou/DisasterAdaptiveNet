#!/usr/bin/env python3
"""CDOD-Former: Cross-Disaster Ordinal Damage Transformer.

DamFormer-inspired design with a weight-shared Siamese Swin encoder, explicit
alignment-aware local/global temporal reasoning, task-specific feature adapters,
dual decoders, localization-conditioned damage reasoning, ordinal consistency,
and boundary supervision. Validation chooses inference calibration; test labels
are used exactly once after the best checkpoint/configuration is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_texas_swin_damageformer as common
import train_xbd_hrtbda_v5_swin_pretrained_cascade as legacy


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(average + maximum)


class GlobalSubsampleCrossAttention(nn.Module):
    """Twins/PVT-style global temporal context at bounded token complexity."""
    def __init__(self, channels: int, heads: int, grid: int):
        super().__init__()
        self.grid = int(grid)
        self.pre_to_post = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.post_to_pre = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_pre = nn.LayerNorm(channels)
        self.norm_post = nn.LayerNorm(channels)
        self.project = nn.Sequential(
            common.ConvGNAct(channels * 3, channels, 1),
            common.ConvGNAct(channels, channels, 3),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        height, width = pre.shape[-2:]
        gh, gw = min(self.grid, height), min(self.grid, width)
        pre_pool = F.adaptive_avg_pool2d(pre, (gh, gw))
        post_pool = F.adaptive_avg_pool2d(post, (gh, gw))
        pre_tokens = self.norm_pre(pre_pool.flatten(2).transpose(1, 2))
        post_tokens = self.norm_post(post_pool.flatten(2).transpose(1, 2))
        pre_context, _ = self.pre_to_post(pre_tokens, post_tokens, post_tokens, need_weights=False)
        post_context, _ = self.post_to_pre(post_tokens, pre_tokens, pre_tokens, need_weights=False)
        pre_context = pre_context.transpose(1, 2).reshape(pre.shape[0], pre.shape[1], gh, gw)
        post_context = post_context.transpose(1, 2).reshape(post.shape[0], post.shape[1], gh, gw)
        context = self.project(torch.cat((pre_context, post_context, torch.abs(post_context - pre_context)), 1))
        return F.interpolate(context, (height, width), mode="bilinear", align_corners=False)


class LocalGlobalTemporalFusion(nn.Module):
    """Learned alignment + local cross-attention + global subsampled context."""
    def __init__(self, channels: int, heads: int, window: int, grid: int, max_offset: float):
        super().__init__()
        self.local = common.TemporalCrossAttention(channels, heads, window, max_offset)
        self.global_context = GlobalSubsampleCrossAttention(channels, heads, grid)
        self.merge = nn.Sequential(
            common.ConvGNAct(channels * 4, channels, 1),
            common.ConvGNAct(channels, channels, 3),
            ChannelGate(channels),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        local = self.local(pre, post)
        global_context = self.global_context(pre, post)
        return self.merge(torch.cat((local, global_context, torch.abs(post - pre), pre * post), 1))


class TaskAdapter(nn.Module):
    """DamFormer-style task-specific adaptation after shared temporal reasoning."""
    def __init__(self, channels: int):
        super().__init__()
        self.adapter = nn.Sequential(
            common.ConvGNAct(channels, channels, 3), ChannelGate(channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.scale * self.adapter(feature)


class GlobalBuildingReasoner(nn.Module):
    """Transformer reasoning over coarse building/change tokens."""
    def __init__(self, channels: int, heads: int, grid: int = 14):
        super().__init__()
        self.grid = int(grid)
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=channels * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.project = common.ConvGNAct(channels, channels, 3)

    def forward(self, feature: torch.Tensor, building_probability: torch.Tensor) -> torch.Tensor:
        height, width = feature.shape[-2:]
        weighted = feature * (1.0 + building_probability)
        pooled = F.adaptive_avg_pool2d(weighted, (self.grid, self.grid))
        tokens = self.encoder(pooled.flatten(2).transpose(1, 2))
        context = tokens.transpose(1, 2).reshape(feature.shape[0], feature.shape[1], self.grid, self.grid)
        context = F.interpolate(context, (height, width), mode="bilinear", align_corners=False)
        return feature + self.project(context)


class CDODFormer(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.backbone = legacy.SwinPretrainedBackbone(
            variant=args.swin_variant, pretrained=True, img_size=args.img_size,
            patch_size=4, window_size=7,
        )
        width = args.decoder_channels
        self.project = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, width, 1, bias=False), nn.GroupNorm(common.group_count(width), width), nn.GELU())
            for c in self.backbone.channels
        ])
        self.temporal = nn.ModuleList([
            LocalGlobalTemporalFusion(
                width, args.temporal_heads, args.temporal_window,
                args.global_grid, 4.0 if index < 2 else 2.0,
            ) for index in range(4)
        ])
        self.localization_adapters = nn.ModuleList([TaskAdapter(width) for _ in range(4)])
        self.damage_adapters = nn.ModuleList([TaskAdapter(width) for _ in range(4)])
        self.localization_decoder = common.ProgressiveDecoder(width)
        self.damage_decoder = common.ProgressiveDecoder(width)
        self.localization_head = nn.Sequential(common.ConvGNAct(width, width // 2), nn.Conv2d(width // 2, 1, 1))
        self.boundary_head = nn.Sequential(common.ConvGNAct(width, width // 2), nn.Conv2d(width // 2, 1, 1))
        self.cross_task = nn.Sequential(
            common.ConvGNAct(width * 2 + 1, width, 1), common.ConvGNAct(width, width, 3),
        )
        self.reasoner = GlobalBuildingReasoner(width, args.temporal_heads, args.reasoner_grid)
        self.damage_head = nn.Sequential(common.ConvGNAct(width, width), nn.Conv2d(width, 4, 1))
        self.ordinal_head = nn.Sequential(common.ConvGNAct(width, width // 2), nn.Conv2d(width // 2, 3, 1))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Dict[str, torch.Tensor]:
        pre_features = [projection(x) for projection, x in zip(self.project, self.backbone(pre))]
        post_features = [projection(x) for projection, x in zip(self.project, self.backbone(post))]
        temporal = [module(a, b) for module, a, b in zip(self.temporal, pre_features, post_features)]
        localization_features = [module(x) for module, x in zip(self.localization_adapters, temporal)]
        damage_features = [module(x) for module, x in zip(self.damage_adapters, temporal)]
        localization_feature = self.localization_decoder(localization_features)
        loc_low = self.localization_head(localization_feature)
        boundary_low = self.boundary_head(localization_feature)
        damage_feature = self.damage_decoder(damage_features)
        loc_for_damage = torch.sigmoid(loc_low)
        damage_feature = self.cross_task(torch.cat((damage_feature, localization_feature, loc_for_damage), 1))
        damage_feature = self.reasoner(damage_feature, loc_for_damage)
        output_size = pre.shape[-2:]
        return {
            "loc": F.interpolate(loc_low, output_size, mode="bilinear", align_corners=False).squeeze(1),
            "boundary": F.interpolate(boundary_low, output_size, mode="bilinear", align_corners=False).squeeze(1),
            "damage": F.interpolate(self.damage_head(damage_feature), output_size, mode="bilinear", align_corners=False),
            "ordinal": F.interpolate(self.ordinal_head(damage_feature), output_size, mode="bilinear", align_corners=False),
        }


def make_loaders(args: argparse.Namespace):
    train_ds = legacy.XBDHRTBDADataset(
        args.data_root, args.train_split, args.img_size, training=True,
        crop_size=args.crop_size, crop_candidate_count=args.crop_candidates,
        crop_class_weights=(1.0, args.minor_crop_weight, args.major_crop_weight, args.destroyed_crop_weight),
        extra_photometric=True,
    )
    val_ds = legacy.XBDHRTBDADataset(args.data_root, args.val_split, args.img_size, training=False)
    test_ds = legacy.XBDHRTBDADataset(args.data_root, args.test_split, args.img_size, training=False)
    kwargs = dict(num_workers=args.num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs),
        DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, **kwargs),
        train_ds,
    )


def ordinal_distribution(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits.float())
    q1 = cumulative[:, 0]
    q2 = torch.minimum(q1, cumulative[:, 1])
    q3 = torch.minimum(q2, cumulative[:, 2])
    probability = torch.stack((1 - q1, q1 - q2, q2 - q3, q3), 1).clamp_min(1e-7)
    return probability / probability.sum(1, keepdim=True).clamp_min(1e-7)


def ordinal_consistency(damage_logits: torch.Tensor, ordinal_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != 255
    if not valid.any():
        return damage_logits.sum() * 0.0
    probability = torch.softmax(damage_logits.float(), 1)
    cumulative = torch.stack((1 - probability[:, 0], probability[:, 2] + probability[:, 3], probability[:, 3]), 1)
    difference = (cumulative - torch.sigmoid(ordinal_logits.float())).square()
    return (difference * valid.unsqueeze(1)).sum() / (valid.sum().clamp_min(1) * 3)


def boundary_target(localization: torch.Tensor) -> torch.Tensor:
    mask = localization.float().unsqueeze(1)
    dilation = F.max_pool2d(mask, 3, stride=1, padding=1)
    erosion = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (dilation - erosion).clamp(0, 1).squeeze(1)


@torch.no_grad()
def cache_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    cached = []
    for batch in loader:
        output = model(batch["pre"].to(device), batch["post"].to(device))
        categorical = torch.softmax(output["damage"].float(), 1).cpu()
        ordinal = ordinal_distribution(output["ordinal"]).cpu()
        loc = torch.sigmoid(output["loc"].float()).cpu()
        for index in range(loc.shape[0]):
            cached.append((loc[index], categorical[index], ordinal[index], batch["loc"][index].long(), batch["target5"][index].long()))
    return cached


def evaluate_cached(cached, thresholds: Sequence[float], blends: Sequence[float], votes: Sequence[int]):
    best_result = best_config = None
    for threshold, blend, vote in product(thresholds, blends, votes):
        counts = {"loc_tp": 0, "loc_fp": 0, "loc_fn": 0, **{c: {"tp": 0, "fp": 0, "fn": 0} for c in range(1, 5)}}
        for loc_probability, categorical, ordinal, loc_true, target5 in cached:
            loc = (loc_probability >= threshold).long()
            probability = (1.0 - blend) * categorical + blend * ordinal
            if vote:
                prediction = common.object_vote(probability.unsqueeze(0), loc.unsqueeze(0), 1)[0]
            else:
                prediction = (probability.argmax(0) + 1) * loc
            common.update_counts(prediction, loc, loc_true, target5, counts)
        result = common.finalize_counts(counts, threshold)
        result["damage_macro_f1"] = float(np.mean([
            result["damage_f1_no_damage"], result["damage_f1_minor_damage"],
            result["damage_f1_major_damage"], result["damage_f1_destroyed"],
        ]))
        if best_result is None or result["score"] > best_result["score"]:
            best_result = result
            best_config = {"localization_threshold": threshold, "ordinal_blend": blend, "object_vote": bool(vote)}
    return best_result, best_config


def evaluate(model, loader, device, args, fixed_config=None):
    cached = cache_predictions(model, loader, device)
    if fixed_config is None:
        return evaluate_cached(cached, args.thresholds, args.ordinal_blends, args.object_vote)
    result, _ = evaluate_cached(
        cached, [fixed_config["localization_threshold"]],
        [fixed_config["ordinal_blend"]], [int(fixed_config["object_vote"])],
    )
    return result, fixed_config


def make_optimizer(model: CDODFormer, args: argparse.Namespace):
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    new = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    return torch.optim.AdamW(
        [{"params": backbone, "lr": args.backbone_lr}, {"params": new, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )


def save_checkpoint(path: Path, model, optimizer, epoch: int, result: Dict, config: Dict, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "CDODFormer", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "best_metric": result["score"], "best_results": result,
        "inference_config": config, "args": vars(args),
    }, path)


def train(args, device, train_loader, val_loader, train_ds) -> Path:
    model = CDODFormer(args).to(device)
    model.set_backbone_trainable(False)
    optimizer = make_optimizer(model, args)
    localization_criterion = legacy.BinaryFocalDiceLoss(legacy.make_loc_pos_weight(train_ds).to(device), gamma=2.0).to(device)
    class_weights = common.damage_weights(train_ds).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output = Path(args.output_dir)
    history, best_result, best_epoch = [], None, 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            model.set_backbone_trainable(True)
            print("Unfroze shared Swin encoder.", flush=True)
        model.train()
        meter = legacy.AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            pre, post = batch["pre"].to(device), batch["post"].to(device)
            loc, target5 = batch["loc"].to(device), batch["target5"].to(device)
            target = common.damage_target(target5)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                output_prediction = model(pre, post)
            loc_loss, _, _ = localization_criterion(output_prediction["loc"].float(), loc.float())
            ce = common.safe_cross_entropy(output_prediction["damage"].float(), target, class_weights)
            dice = common.soft_dice_loss(output_prediction["damage"].float(), target, class_weights)
            ordinal = common.ordinal_loss(output_prediction["ordinal"].float(), target)
            consistency = ordinal_consistency(output_prediction["damage"], output_prediction["ordinal"], target)
            instance = common.instance_consistency_loss(output_prediction["damage"].float(), target5, args.min_instance_pixels)
            boundary = F.binary_cross_entropy_with_logits(output_prediction["boundary"].float(), boundary_target(loc))
            loss = (
                loc_loss + ce + args.dice_weight * dice + args.ordinal_weight * ordinal
                + args.consistency_weight * consistency + args.instance_weight * instance
                + args.boundary_weight * boundary
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
            scaler.scale(loss / args.grad_accum_steps).backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            meter.update(float(loss.detach()), pre.shape[0])
            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss={meter.avg:.4f}", flush=True)
        progress = epoch / max(args.epochs, 1)
        for group, initial in zip(optimizer.param_groups, (args.backbone_lr, args.lr)):
            group["lr"] = initial * 0.5 * (1 + math.cos(math.pi * progress))
        validation, config = evaluate(model, val_loader, device, args)
        common.print_result(f"Validation epoch {epoch}", validation)
        print(f"Validation-selected inference: {config}", flush=True)
        history.append({"epoch": epoch, "train_loss": meter.avg, "inference_config": config, **validation})
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if best_result is None or validation["score"] > best_result["score"]:
            best_result, best_epoch = validation, epoch
            save_checkpoint(output / "checkpoints" / "best.pt", model, optimizer, epoch, validation, config, args)
            print(f"Saved new best CDOD-Former at epoch {epoch}.", flush=True)
        save_checkpoint(output / "checkpoints" / "last.pt", model, optimizer, epoch, validation, config, args)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping: no validation improvement since epoch {best_epoch}.", flush=True)
            break
    return output / "checkpoints" / "best.pt"


def test(args, device, loader, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CDODFormer(args).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    result, config = evaluate(model, loader, device, args, checkpoint["inference_config"])
    result["checkpoint_epoch"] = int(checkpoint["epoch"])
    common.print_result("FINAL UNSEEN TEXAS TEST", result)
    report = {
        "architecture": "CDODFormer", "selection_split": args.val_split,
        "test_split": args.test_split, "inference_config": config, "test": result,
    }
    score_dir = Path(args.output_dir) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / "test_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (score_dir / "summary.txt").write_text("\n".join(f"{key}: {value}" for key, value in result.items()) + "\n")


def parse_args():
    parser = argparse.ArgumentParser("CDOD-Former Texas experiment")
    parser.add_argument("--phase", choices=["train_test", "train", "test"], default="train_test")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=896)
    parser.add_argument("--crop-size", type=int, default=672)
    parser.add_argument("--crop-candidates", type=int, default=48)
    parser.add_argument("--minor-crop-weight", type=float, default=20.0)
    parser.add_argument("--major-crop-weight", type=float, default=20.0)
    parser.add_argument("--destroyed-crop-weight", type=float, default=8.0)
    parser.add_argument("--swin-variant", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--decoder-channels", type=int, default=192)
    parser.add_argument("--temporal-heads", type=int, default=6)
    parser.add_argument("--temporal-window", type=int, default=7)
    parser.add_argument("--global-grid", type=int, default=12)
    parser.add_argument("--reasoner-grid", type=int, default=14)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--ordinal-weight", type=float, default=0.35)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--instance-weight", type=float, default=0.75)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--min-instance-pixels", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65, 0.75])
    parser.add_argument("--ordinal-blends", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--object-vote", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    legacy.set_seed(args.seed)
    if args.decoder_channels % args.temporal_heads:
        raise ValueError("--decoder-channels must be divisible by --temporal-heads")
    if args.img_size % 224 or args.crop_size % 224:
        raise ValueError("--img-size and --crop-size must be multiples of 224 for this Swin-Tiny runtime")
    output = Path(args.output_dir)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device} | Architecture=CDOD-Former | Data={args.data_root}", flush=True)
    train_loader, val_loader, test_loader, train_ds = make_loaders(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output / "checkpoints" / "best.pt"
    if args.phase in {"train", "train_test"}:
        checkpoint = train(args, device, train_loader, val_loader, train_ds)
    if args.phase in {"test", "train_test"}:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        test(args, device, test_loader, checkpoint)


if __name__ == "__main__":
    main()
