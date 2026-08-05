#!/usr/bin/env python3
"""SwinDamageFormer: instance-aware, ordinal, cross-temporal BDA on Texas."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_xbd_hrtbda_v5_swin_pretrained_cascade as legacy


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(group_count(cout), cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def warp_with_flow(x: torch.Tensor, flow: torch.Tensor, max_offset: float) -> torch.Tensor:
    """Warp x with bounded pixel offsets predicted at x's feature resolution."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    bounded = torch.tanh(flow) * float(max_offset)
    dx = bounded[:, 0] * (2.0 / max(w - 1, 1))
    dy = bounded[:, 1] * (2.0 / max(h - 1, 1))
    grid = base + torch.stack((dx, dy), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def partition_windows(x: torch.Tensor, window: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
    b, c, h, w = x.shape
    ph = (window - h % window) % window
    pw = (window - w % window) % window
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    hp, wp = x.shape[-2:]
    tokens = (
        x.view(b, c, hp // window, window, wp // window, window)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(-1, window * window, c)
    )
    return tokens, (b, h, w, hp, wp)


def reverse_windows(tokens: torch.Tensor, shape: Tuple[int, int, int, int, int], window: int) -> torch.Tensor:
    b, h, w, hp, wp = shape
    c = tokens.shape[-1]
    x = (
        tokens.view(b, hp // window, wp // window, window, window, c)
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .view(b, c, hp, wp)
    )
    return x[:, :, :h, :w]


class TemporalCrossAttention(nn.Module):
    """Local alignment + bidirectional window cross-attention + explicit change cues."""
    def __init__(self, channels: int, heads: int, window: int, max_offset: float):
        super().__init__()
        self.window = int(window)
        self.max_offset = float(max_offset)
        self.pre_norm = nn.GroupNorm(group_count(channels), channels)
        self.post_norm = nn.GroupNorm(group_count(channels), channels)
        self.offset = nn.Sequential(
            ConvGNAct(channels * 3, channels, 3),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.pre_to_post = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.post_to_pre = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 6, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, pre: torch.Tensor, post: torch.Tensor, building: torch.Tensor | None = None) -> torch.Tensor:
        flow = self.offset(torch.cat((pre, post, torch.abs(post - pre)), dim=1))
        post_aligned = warp_with_flow(post, flow, self.max_offset)
        pn = self.pre_norm(pre)
        qn = self.post_norm(post_aligned)
        pt, shape = partition_windows(pn, self.window)
        qt, _ = partition_windows(qn, self.window)
        p_cross, _ = self.pre_to_post(pt, qt, qt, need_weights=False)
        q_cross, _ = self.post_to_pre(qt, pt, pt, need_weights=False)
        p_cross = reverse_windows(p_cross, shape, self.window)
        q_cross = reverse_windows(q_cross, shape, self.window)
        cues = torch.cat((pre, post_aligned, torch.abs(post_aligned - pre), pre * post_aligned, p_cross, q_cross), dim=1)
        change = self.fuse(cues)
        change = change + self.gate(torch.cat((pre, post_aligned), dim=1)) * (p_cross + q_cross)
        if building is not None:
            bmap = F.interpolate(building, size=change.shape[-2:], mode="bilinear", align_corners=False)
            change = change * (1.0 + torch.sigmoid(bmap))
        return change


class ProgressiveDecoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.refine3 = nn.Sequential(ConvGNAct(channels * 2, channels), ConvGNAct(channels, channels))
        self.refine2 = nn.Sequential(ConvGNAct(channels * 2, channels), ConvGNAct(channels, channels))
        self.refine1 = nn.Sequential(ConvGNAct(channels * 2, channels), ConvGNAct(channels, channels))

    @staticmethod
    def up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = feats
        x = self.refine3(torch.cat((self.up(f4, f3), f3), dim=1))
        x = self.refine2(torch.cat((self.up(x, f2), f2), dim=1))
        return self.refine1(torch.cat((self.up(x, f1), f1), dim=1))


class SwinDamageFormer(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.backbone = legacy.SwinPretrainedBackbone(
            variant=args.swin_variant,
            pretrained=True,
            img_size=args.img_size,
            patch_size=4,
            window_size=7,
        )
        width = args.decoder_channels
        self.project = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, width, 1, bias=False), nn.GroupNorm(group_count(width), width), nn.GELU())
            for c in self.backbone.channels
        ])
        self.loc_decoder = ProgressiveDecoder(width)
        self.loc_head = nn.Sequential(ConvGNAct(width, width // 2), nn.Conv2d(width // 2, 1, 1))
        heads = args.temporal_heads
        self.temporal = nn.ModuleList([
            TemporalCrossAttention(width, heads, args.temporal_window, max_offset=4.0 if i < 2 else 2.0)
            for i in range(4)
        ])
        self.damage_decoder = ProgressiveDecoder(width)
        self.damage_head = nn.Sequential(ConvGNAct(width, width), nn.Conv2d(width, 4, 1))
        self.ordinal_head = nn.Sequential(ConvGNAct(width, width // 2), nn.Conv2d(width // 2, 3, 1))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> Dict[str, torch.Tensor]:
        pre_raw = self.backbone(pre)
        post_raw = self.backbone(post)
        pre_feats = [proj(feat) for proj, feat in zip(self.project, pre_raw)]
        post_feats = [proj(feat) for proj, feat in zip(self.project, post_raw)]
        loc_feat = self.loc_decoder(pre_feats)
        loc_low = self.loc_head(loc_feat)
        loc_logits = F.interpolate(loc_low, size=pre.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        building = loc_low.detach() if self.training else loc_low
        changes = [block(a, b, building) for block, a, b in zip(self.temporal, pre_feats, post_feats)]
        damage_feat = self.damage_decoder(changes)
        damage = F.interpolate(self.damage_head(damage_feat), size=pre.shape[-2:], mode="bilinear", align_corners=False)
        ordinal = F.interpolate(self.ordinal_head(damage_feat), size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return {"loc": loc_logits, "damage": damage, "ordinal": ordinal}


def rare_tile_sample_weights(dataset: legacy.XBDHRTBDADataset, args: argparse.Namespace) -> torch.Tensor:
    """Oversample tiles containing rare damage without changing test/val distributions."""
    weights = []
    tile_counts = {"minor": 0, "major": 0, "destroyed": 0}
    for sample in dataset.samples:
        localization = dataset._read_mask(sample.pre_target_path)
        damage = dataset._read_mask(sample.post_target_path)
        target = dataset._target5_from_masks(localization, damage)
        has_minor = bool((target == 2).any())
        has_major = bool((target == 3).any())
        has_destroyed = bool((target == 4).any())
        tile_counts["minor"] += int(has_minor)
        tile_counts["major"] += int(has_major)
        tile_counts["destroyed"] += int(has_destroyed)
        # Add rather than multiply so a tile containing several rare classes
        # remains important without receiving an extreme sampling probability.
        weight = (
            1.0
            + args.minor_tile_weight * has_minor
            + args.major_tile_weight * has_major
            + args.destroyed_tile_weight * has_destroyed
        )
        weights.append(weight)
    result = torch.tensor(weights, dtype=torch.double)
    result /= result.mean().clamp_min(1e-12)
    print(f"Rare-damage tile counts: {tile_counts}", flush=True)
    print(
        f"Tile sampling weights: min={result.min():.3f}, mean={result.mean():.3f}, "
        f"max={result.max():.3f}", flush=True,
    )
    return result


def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, legacy.XBDHRTBDADataset]:
    train_ds = legacy.XBDHRTBDADataset(
        args.data_root, args.train_split, args.img_size, training=True,
        crop_size=args.crop_size, crop_candidate_count=args.crop_candidates,
        crop_class_weights=(1.0, args.minor_crop_weight, args.major_crop_weight, 4.0),
        extra_photometric=True,
    )
    val_ds = legacy.XBDHRTBDADataset(args.data_root, args.val_split, args.img_size, training=False)
    test_ds = legacy.XBDHRTBDADataset(args.data_root, args.test_split, args.img_size, training=False)
    common = dict(num_workers=args.num_workers, pin_memory=False)
    sampler = None
    if args.rare_tile_sampling:
        sampler = WeightedRandomSampler(
            rare_tile_sample_weights(train_ds, args),
            num_samples=len(train_ds), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler is None, drop_last=True, **common,
    )
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **common)
    test = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **common)
    return train, val, test, train_ds


def damage_weights(dataset: legacy.XBDHRTBDADataset) -> torch.Tensor:
    counts = dataset.class5_counts()[1:5].astype(np.float64)
    weights = 1.0 / np.sqrt(counts / counts.sum() + 1e-12)
    weights /= weights.mean()
    print(f"Damage counts [no,minor,major,destroyed]: {counts.astype(int).tolist()}", flush=True)
    print(f"Damage weights: {weights.tolist()}", flush=True)
    return torch.tensor(weights, dtype=torch.float32)


def damage_target(target5: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(target5, 255)
    valid = (target5 >= 1) & (target5 <= 4)
    target[valid] = target5[valid] - 1
    return target


def safe_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Cross entropy that returns a differentiable zero for all-ignore crops."""
    if not (target != 255).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, weight=weights, ignore_index=255)


def class_balanced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    """Foreground-only focal CE with weights applied after focal modulation.

    Computing pt from unweighted log-probability avoids class weights changing
    the focal difficulty term, a common implementation error.
    """
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0.0
    logits_valid = logits.permute(0, 2, 3, 1)[valid]
    target_valid = target[valid]
    log_probability = F.log_softmax(logits_valid.float(), dim=1)
    nll = F.nll_loss(log_probability, target_valid, reduction="none")
    smooth = -log_probability.mean(dim=1)
    ce = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    pt = torch.exp(-nll).clamp(1e-7, 1.0)
    focal = (1.0 - pt).pow(gamma)
    class_weight = weights[target_valid]
    return (focal * ce * class_weight).sum() / class_weight.sum().clamp_min(1e-7)


class ModelEMA:
    """Exponential moving average used for stable validation and checkpoints."""
    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            current = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                value.copy_(current)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    valid = target != 255
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe, 4).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    probs = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
    norm = weights / weights.sum()
    return (dice * norm).sum()


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0.0
    safe = target.masked_fill(~valid, 0)
    levels = torch.stack(((safe >= 1), (safe >= 2), (safe >= 3)), dim=1).float()
    raw = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")
    return (raw * valid.unsqueeze(1)).sum() / (valid.sum().clamp_min(1) * 3)


def instance_consistency_loss(logits: torch.Tensor, target5: torch.Tensor, min_pixels: int = 8) -> torch.Tensor:
    """Classify connected ground-truth damage regions from their mean pixel probability."""
    probs = torch.softmax(logits.float(), dim=1)
    losses: List[torch.Tensor] = []
    for bi in range(target5.shape[0]):
        target_np = target5[bi].detach().cpu().numpy().astype(np.uint8)
        for raw_class in range(1, 5):
            nlabels, labels = cv2.connectedComponents((target_np == raw_class).astype(np.uint8), connectivity=8)
            for component in range(1, nlabels):
                mask_np = labels == component
                if int(mask_np.sum()) < min_pixels:
                    continue
                mask = torch.from_numpy(mask_np).to(device=logits.device)
                mean_prob = probs[bi, :, mask].mean(dim=1).clamp_min(1e-7)
                losses.append(-torch.log(mean_prob[raw_class - 1]))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def object_vote(prob: torch.Tensor, loc: torch.Tensor, min_pixels: int) -> torch.Tensor:
    """Assign one mean-probability damage class to each predicted building component."""
    output = torch.zeros(loc.shape, dtype=torch.long, device=prob.device)
    for bi in range(loc.shape[0]):
        loc_np = loc[bi].detach().cpu().numpy().astype(np.uint8)
        nlabels, labels = cv2.connectedComponents(loc_np, connectivity=8)
        for component in range(1, nlabels):
            mask_np = labels == component
            if int(mask_np.sum()) < min_pixels:
                continue
            mask = torch.from_numpy(mask_np).to(prob.device)
            cls = int(prob[bi, :, mask].mean(dim=1).argmax().item()) + 1
            output[bi][mask] = cls
    return output


def update_counts(pred: torch.Tensor, loc_pred: torch.Tensor, loc_true: torch.Tensor, target5: torch.Tensor, counts: Dict) -> None:
    counts["loc_tp"] += int(((loc_pred == 1) & (loc_true == 1)).sum())
    counts["loc_fp"] += int(((loc_pred == 1) & (loc_true == 0)).sum())
    counts["loc_fn"] += int(((loc_pred == 0) & (loc_true == 1)).sum())
    valid = (target5 >= 1) & (target5 <= 4)
    pv, tv = pred[valid], target5[valid]
    for cls in range(1, 5):
        counts[cls]["tp"] += int(((pv == cls) & (tv == cls)).sum())
        counts[cls]["fp"] += int(((pv == cls) & (tv != cls)).sum())
        counts[cls]["fn"] += int(((pv != cls) & (tv == cls)).sum())


def finalize_counts(counts: Dict, threshold: float) -> Dict[str, float]:
    loc = legacy.F1Recorder(counts["loc_tp"], counts["loc_fp"], counts["loc_fn"])
    rec = [legacy.F1Recorder(counts[c]["tp"], counts[c]["fp"], counts[c]["fn"]) for c in range(1, 5)]
    damage = legacy.harmonic_mean([x.f1 for x in rec])
    macro = float(np.mean([x.f1 for x in rec]))
    return {
        "score": 0.3 * loc.f1 + 0.7 * damage,
        "localization_f1": loc.f1,
        "damage_f1": damage,
        "damage_macro_f1": macro,
        "macro_composite_score": 0.3 * loc.f1 + 0.7 * macro,
        "damage_f1_no_damage": rec[0].f1,
        "damage_f1_minor_damage": rec[1].f1,
        "damage_f1_major_damage": rec[2].f1,
        "damage_f1_destroyed": rec[3].f1,
        "predicted_damage_pixels": {
            str(class_id): int(counts[class_id]["tp"] + counts[class_id]["fp"])
            for class_id in range(1, 5)
        },
        "true_damage_pixels": {
            str(class_id): int(counts[class_id]["tp"] + counts[class_id]["fn"])
            for class_id in range(1, 5)
        },
        "localization_threshold": threshold,
    }


def ordinal_distribution(logits: torch.Tensor) -> torch.Tensor:
    """Convert three cumulative ordinal logits into four class probabilities."""
    cumulative = torch.sigmoid(logits.float())
    q1 = cumulative[:, 0]
    q2 = torch.minimum(q1, cumulative[:, 1])
    q3 = torch.minimum(q2, cumulative[:, 2])
    probability = torch.stack((1.0 - q1, q1 - q2, q2 - q3, q3), dim=1)
    probability = probability.clamp_min(1e-7)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-7)


INFERENCE_CLASS_BOOSTS = (
    (1.0, 1.0, 1.0, 1.0),
    (1.0, 1.25, 1.25, 1.0),
    (1.0, 1.5, 1.25, 1.0),
    (1.0, 1.25, 1.5, 1.0),
    (1.0, 1.5, 1.5, 1.1),
)


def inference_config_from_result(result: Dict) -> Dict:
    return {
        "localization_threshold": float(result.get("localization_threshold", 0.5)),
        "ordinal_blend": float(result.get("ordinal_blend", 0.0)),
        "object_vote": bool(result.get("object_vote", True)),
        "class_boosts": tuple(float(value) for value in result.get("class_boosts", (1, 1, 1, 1))),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: List[float],
    min_pixels: int,
    fixed_config: Dict | None = None,
) -> Dict[str, float]:
    model.eval()
    cached = []
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        out = model(pre, post)
        cached.append((
            torch.sigmoid(out["loc"].float()).cpu(),
            torch.softmax(out["damage"].float(), 1).cpu(),
            ordinal_distribution(out["ordinal"]).cpu(),
            batch["loc"].long(), batch["target5"].long(),
        ))
    if fixed_config is None:
        # Localization threshold does not depend on the damage calibration.
        # Select it first by localization F1, reducing the damage grid from
        # 150 full-resolution passes to 30 per validation epoch.
        threshold_scores = []
        for threshold in thresholds:
            tp = fp = fn = 0
            for loc_prob, _, _, loc_true, _ in cached:
                prediction = loc_prob > threshold
                truth = loc_true == 1
                tp += int((prediction & truth).sum())
                fp += int((prediction & ~truth).sum())
                fn += int((~prediction & truth).sum())
            threshold_scores.append((legacy.F1Recorder(tp, fp, fn).f1, float(threshold)))
        selected_threshold = max(threshold_scores)[1]
        configurations = [
            {
                "localization_threshold": threshold,
                "ordinal_blend": blend,
                "object_vote": vote,
                "class_boosts": boosts,
            }
            for threshold in (selected_threshold,)
            for blend in (0.0, 0.25, 0.5)
            for vote in (False, True)
            for boosts in INFERENCE_CLASS_BOOSTS
        ]
    else:
        configurations = [fixed_config]
    best = None
    for config in configurations:
        threshold = float(config["localization_threshold"])
        blend = float(config["ordinal_blend"])
        boosts = torch.tensor(config["class_boosts"], dtype=torch.float32).view(1, 4, 1, 1)
        counts = {"loc_tp": 0, "loc_fp": 0, "loc_fn": 0, **{c: {"tp": 0, "fp": 0, "fn": 0} for c in range(1, 5)}}
        for loc_prob, damage_prob, ordinal_prob, loc_true, target5 in cached:
            loc_pred = (loc_prob > threshold).long()
            probability = ((1.0 - blend) * damage_prob + blend * ordinal_prob) * boosts
            probability /= probability.sum(dim=1, keepdim=True).clamp_min(1e-7)
            if bool(config["object_vote"]):
                pred = object_vote(probability, loc_pred, min_pixels)
            else:
                pred = (probability.argmax(dim=1) + 1) * loc_pred
            update_counts(pred, loc_pred, loc_true, target5, counts)
        result = finalize_counts(counts, threshold)
        result.update({
            "ordinal_blend": blend,
            "object_vote": bool(config["object_vote"]),
            "class_boosts": [float(value) for value in config["class_boosts"]],
        })
        # Official harmonic score remains the primary criterion. Macro damage
        # and localization break ties when a rare class has zero F1, avoiding
        # the previous localization-only selection collapse.
        ranking = (result["score"], result["damage_macro_f1"], result["localization_f1"])
        if best is None or ranking > best[0]:
            best = (ranking, result)
    assert best is not None
    return best[1]


def make_optimizer(model: SwinDamageFormer, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    backbone = [p for p in model.parameters() if id(p) in backbone_ids]
    new_modules = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [{"params": backbone, "lr": args.backbone_lr}, {"params": new_modules, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )


def save_checkpoint(path: Path, model: SwinDamageFormer, optimizer, epoch: int, best: Dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metric": best["score"], "best_results": best, "args": vars(args)}, path)


def print_result(label: str, result: Dict[str, float]) -> None:
    print(
        f"{label} | score={result['score']:.6f} | loc={result['localization_f1']:.6f} | "
        f"damage={result['damage_f1']:.6f} | no={result['damage_f1_no_damage']:.6f} | "
        f"minor={result['damage_f1_minor_damage']:.6f} | major={result['damage_f1_major_damage']:.6f} | "
        f"destroyed={result['damage_f1_destroyed']:.6f} | macro={result.get('damage_macro_f1', 0.0):.6f} | "
        f"loc_th={result['localization_threshold']:.2f} | ordinal_blend={result.get('ordinal_blend', 0.0):.2f} | "
        f"object_vote={result.get('object_vote', True)} | boosts={result.get('class_boosts', [1, 1, 1, 1])}",
        flush=True,
    )


def train(args: argparse.Namespace, device: torch.device) -> Path:
    train_loader, val_loader, _, train_ds = make_loaders(args)
    model = SwinDamageFormer(args).to(device)
    if args.init_checkpoint:
        initialization = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = initialization.get("model", initialization)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Initialization checkpoint is not architecture-compatible: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        print(
            f"Initialized the complete SwinDamageFormer from {args.init_checkpoint} "
            f"(stored epoch={initialization.get('epoch', 'unknown')}).",
            flush=True,
        )
    model.set_backbone_trainable(False)
    optimizer = make_optimizer(model, args)
    ema = ModelEMA(model, args.ema_decay)
    loc_weight = legacy.make_loc_pos_weight(train_ds).to(device)
    loc_criterion = legacy.BinaryFocalDiceLoss(loc_weight, gamma=2.0).to(device)
    cls_weight = damage_weights(train_ds).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output = Path(args.output_dir)
    history, best_result, best_epoch = [], None, 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            model.set_backbone_trainable(True)
            print("Unfroze Swin backbone for differential-LR training.", flush=True)
        if epoch <= args.warmup_epochs:
            lr_scale = epoch / max(args.warmup_epochs, 1)
        else:
            cosine_progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        for group, base_lr in zip(optimizer.param_groups, (args.backbone_lr, args.lr)):
            group["lr"] = base_lr * lr_scale
        print(
            f"Epoch {epoch}: decoder_lr={optimizer.param_groups[1]['lr']:.8f}, "
            f"backbone_lr={optimizer.param_groups[0]['lr']:.8f}", flush=True,
        )
        model.train()
        meter = legacy.AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            loc = batch["loc"].to(device)
            target5 = batch["target5"].to(device)
            target = damage_target(target5)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(pre, post)
            # Keep the memory-heavy forward pass in AMP, but calculate losses
            # in float32. This avoids underflow in rare-class probabilities.
            loc_logits = out["loc"].float()
            damage_logits = out["damage"].float()
            ordinal_logits = out["ordinal"].float()
            loc_loss, _, _ = loc_criterion(loc_logits, loc.float())
            ce = class_balanced_focal_loss(
                damage_logits, target, cls_weight,
                gamma=args.damage_focal_gamma,
                label_smoothing=args.damage_label_smoothing,
            )
            dice = soft_dice_loss(damage_logits, target, cls_weight)
            ordinal = ordinal_loss(ordinal_logits, target)
            instance = instance_consistency_loss(damage_logits, target5, args.min_instance_pixels)
            with torch.amp.autocast("cuda", enabled=False):
                loss = loc_loss + ce + 0.5 * dice + args.ordinal_weight * ordinal + args.instance_weight * instance
            components = {
                "total": loss, "localization": loc_loss, "cross_entropy": ce,
                "dice": dice, "ordinal": ordinal, "instance": instance,
            }
            bad = {name: float(value.detach().item()) for name, value in components.items() if not torch.isfinite(value).all()}
            if bad:
                valid_pixels = int((target != 255).sum().item())
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, step={step}, valid_damage_pixels={valid_pixels}: {bad}"
                )
            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = step % args.grad_accum_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            meter.update(float(loss.item()), pre.shape[0])
            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss={meter.avg:.4f}", flush=True)

        val = evaluate(ema.module, val_loader, device, args.thresholds, args.min_instance_pixels)
        # A predeclared balanced selection score remains informative when the
        # official harmonic damage score is near zero because one rare class
        # is temporarily absent. The official score is still reported intact.
        val["checkpoint_selection_score"] = 0.5 * val["score"] + 0.5 * val["macro_composite_score"]
        print_result(f"Validation epoch {epoch}", val)
        print(f"Checkpoint selection score={val['checkpoint_selection_score']:.6f}", flush=True)
        row = {
            "epoch": epoch, "train_loss": meter.avg,
            "decoder_lr": optimizer.param_groups[1]["lr"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "validation_model": f"EMA(decay={args.ema_decay})", **val,
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        save_checkpoint(output / "checkpoints" / "last.pt", ema.module, optimizer, epoch, val, args)
        if best_result is None or val["checkpoint_selection_score"] > best_result["checkpoint_selection_score"]:
            best_result, best_epoch = val, epoch
            save_checkpoint(output / "checkpoints" / "best.pt", ema.module, optimizer, epoch, val, args)
            print(f"Saved best checkpoint at epoch {epoch}.", flush=True)
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping: no improvement since epoch {best_epoch}.", flush=True)
            break
    return output / "checkpoints" / "best.pt"


def test(args: argparse.Namespace, device: torch.device, checkpoint: Path) -> Dict[str, float]:
    _, _, test_loader, _ = make_loaders(args)
    model = SwinDamageFormer(args).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    saved_result = ckpt.get("best_results", {})
    config = inference_config_from_result(saved_result)
    result = evaluate(
        model, test_loader, device, [config["localization_threshold"]],
        args.min_instance_pixels, fixed_config=config,
    )
    result["checkpoint_epoch"] = int(ckpt.get("epoch", -1))
    print_result("FINAL TEST", result)
    scores = Path(args.output_dir) / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / "test_results.json").write_text(json.dumps(result, indent=2))
    (scores / "summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in result.items()) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SwinDamageFormer Texas building damage assessment")
    p.add_argument("--phase", choices=["train_test", "train", "test"], default="train_test")
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument(
        "--init-checkpoint", default="",
        help="Initialize every model weight from a compatible SwinDamageFormer checkpoint before training.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--freeze-backbone-epochs", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=896)
    p.add_argument("--crop-size", type=int, default=672)
    p.add_argument("--crop-candidates", type=int, default=32)
    p.add_argument("--minor-crop-weight", type=float, default=16.0)
    p.add_argument("--major-crop-weight", type=float, default=16.0)
    p.add_argument("--rare-tile-sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--minor-tile-weight", type=float, default=4.0)
    p.add_argument("--major-tile-weight", type=float, default=3.0)
    p.add_argument("--destroyed-tile-weight", type=float, default=2.0)
    p.add_argument("--swin-variant", default="swin_tiny_patch4_window7_224")
    p.add_argument("--decoder-channels", type=int, default=192)
    p.add_argument("--temporal-heads", type=int, default=6)
    p.add_argument("--temporal-window", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ordinal-weight", type=float, default=0.25)
    p.add_argument("--instance-weight", type=float, default=0.5)
    p.add_argument("--damage-focal-gamma", type=float, default=2.0)
    p.add_argument("--damage-label-smoothing", type=float, default=0.02)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--min-instance-pixels", type=int, default=8)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65, 0.75])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    legacy.set_seed(args.seed)
    if args.decoder_channels % args.temporal_heads != 0:
        raise ValueError("--decoder-channels must be divisible by --temporal-heads")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1")
    if args.img_size % 224 != 0 or args.crop_size % 224 != 0:
        raise ValueError("--img-size and --crop-size must be multiples of 224 for Swin-Tiny")
    output = Path(args.output_dir)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Architecture: SwinDamageFormer | Data: {args.data_root}", flush=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output / "checkpoints" / "best.pt"
    if args.phase in {"train", "train_test"}:
        checkpoint = train(args, device)
    if args.phase in {"test", "train_test"}:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        test(args, device, checkpoint)


if __name__ == "__main__":
    main()
