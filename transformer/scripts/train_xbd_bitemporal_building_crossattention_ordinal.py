#!/usr/bin/env python3
"""Bi-temporal building-guided CNN-Swin cross-attention with ordinal damage.

Protocol: xBD train+tier3 -> hold model/threshold selection -> untouched test.
The model returns eight logits: localization, four categorical damage logits,
and three cumulative ordinal logits. Existing evaluation consumes channels
0:5; the custom training loss additionally supervises channels 5:8.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable


class WindowedBidirectionalCrossAttention(nn.Module):
    """Memory-bounded pre/post cross-attention in non-overlapping windows."""

    def __init__(self, channels: int, heads: int = 6, window_size: int = 7):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.window_size = window_size
        self.pre_norm = nn.LayerNorm(channels)
        self.post_norm = nn.LayerNorm(channels)
        self.post_queries_pre = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.pre_queries_post = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.pre_scale = nn.Parameter(torch.zeros(1))
        self.post_scale = nn.Parameter(torch.zeros(1))
        self.output_norm = nn.GroupNorm(16, channels)

    def _windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.reshape(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
        return x, (b, h, w, hp, wp)

    def _restore(self, windows: torch.Tensor, shape) -> torch.Tensor:
        b, h, w, hp, wp = shape
        ws, c = self.window_size, windows.shape[-1]
        x = windows.reshape(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
        return x[:, :, :h, :w]

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        pre_w, shape = self._windows(pre)
        post_w, post_shape = self._windows(post)
        if shape != post_shape:
            raise RuntimeError("Pre/post cross-attention features are not spatially aligned")

        pre_n = self.pre_norm(pre_w)
        post_n = self.post_norm(post_w)
        post_delta, _ = self.post_queries_pre(post_n, pre_n, pre_n, need_weights=False)
        pre_delta, _ = self.pre_queries_post(pre_n, post_n, post_n, need_weights=False)

        pre_out = pre + self.pre_scale * self._restore(pre_delta, shape)
        post_out = post + self.post_scale * self._restore(post_delta, shape)
        return self.output_norm(pre_out), self.output_norm(post_out)


class BuildingGuidedCrossAttentionOrdinalNet(stable.ResNet34SwinFiLMGated):
    def __init__(self, image_size: int = 896, width: int = 96):
        super().__init__(image_size=image_size, width=width)

        # This architecture uses temporal cross-attention rather than FiLM or
        # convex gated fusion; inherited pretrained encoders/FPNs are retained.
        del self.res_film
        del self.swin_film
        del self.gate
        del self.head
        del self.res_temporal
        del self.swin_temporal

        self.res_projection = nn.Sequential(
            nn.Conv2d(48, width, 1, bias=False), stable.norm(width), nn.GELU()
        )
        self.res_cross_attention = WindowedBidirectionalCrossAttention(width, heads=6, window_size=7)
        self.swin_cross_attention = WindowedBidirectionalCrossAttention(width, heads=6, window_size=7)
        self.res_change = nn.Sequential(
            nn.Conv2d(width * 4, width, 3, padding=1, bias=False),
            stable.norm(width), nn.GELU(),
        )
        self.swin_change = nn.Sequential(
            nn.Conv2d(width * 4, width, 3, padding=1, bias=False),
            stable.norm(width), nn.GELU(),
        )
        self.hybrid_fusion = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            stable.norm(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 1, bias=False),
            stable.norm(width),
            nn.GELU(),
        )
        self.localization_head = nn.Conv2d(width, 1, 1)
        self.damage_refine = nn.Sequential(
            nn.Conv2d(width + 1, width, 3, padding=1, bias=False),
            stable.norm(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            stable.norm(width),
            nn.GELU(),
        )
        self.damage_head = nn.Conv2d(width, 4, 1)
        self.ordinal_head = nn.Conv2d(width, 3, 1)

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        del condition  # Cross-attention model is intentionally hazard-agnostic.
        pre, post = images[:, :3], images[:, 3:]
        output_size = images.shape[-2:]
        fusion_size = (max(1, output_size[0] // 4), max(1, output_size[1] // 4))

        res_pre = F.interpolate(
            self.resnet_unet.forward_once(pre), size=fusion_size,
            mode="bilinear", align_corners=False,
        )
        res_post = F.interpolate(
            self.resnet_unet.forward_once(post), size=fusion_size,
            mode="bilinear", align_corners=False,
        )
        res_pre = self.res_projection(res_pre)
        res_post = self.res_projection(res_post)
        res_pre, res_post = self.res_cross_attention(res_pre, res_post)
        res_change = self.res_change(self.temporal(res_pre, res_post))

        swin_pre = self.swin_fpn(self.swin(pre), fusion_size)
        swin_post = self.swin_fpn(self.swin(post), fusion_size)
        swin_pre, swin_post = self.swin_cross_attention(swin_pre, swin_post)
        swin_change = self.swin_change(self.temporal(swin_pre, swin_post))

        fused = self.hybrid_fusion(torch.cat([res_change, swin_change], dim=1))
        fused = fused + self.refine(fused)

        localization = self.localization_head(fused)
        building_attention = torch.sigmoid(localization)
        guided = fused * (1.0 + building_attention)
        guided = self.damage_refine(torch.cat([guided, building_attention], dim=1))
        damage = self.damage_head(guided)
        ordinal = self.ordinal_head(guided)

        logits = torch.cat([localization, damage, ordinal], dim=1)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != 255
    if not bool(valid.any()):
        return logits.sum() * 0.0
    safe = target.clamp(0, 3)
    thresholds = torch.stack([safe > 0, safe > 1, safe > 2], dim=1).to(logits.dtype)
    valid3 = valid[:, None].expand_as(thresholds)
    bce = F.binary_cross_entropy_with_logits(logits, thresholds, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * thresholds + (1.0 - probability) * (1.0 - thresholds)
    focal = ((1.0 - pt).pow(2.0) * bce)[valid3].mean()

    # Enforce P(damage>0) >= P(damage>1) >= P(damage>2).
    monotonic = F.relu(probability[:, 1] - probability[:, 0]).mean()
    monotonic = monotonic + F.relu(probability[:, 2] - probability[:, 1]).mean()
    return focal + 0.1 * monotonic


def compute_losses(logits, loc, dmg, loc_criterion, dmg_criterion, device, args):
    loc_bce, loc_dice = loc_criterion(logits[:, 0], loc)
    weights = dmg_criterion.weight.to(device=device, dtype=logits.dtype)
    damage_focal, damage_dice = stable.focal_dice_damage_loss(logits[:, 1:5], dmg, weights)
    ordered = ordinal_loss(logits[:, 5:8], dmg)

    damage_total = damage_focal + damage_dice + 0.30 * ordered
    total = (
        args.loc_bce_weight * loc_bce
        + args.loc_dice_weight * loc_dice
        + args.dmg_ce_weight * damage_total
    )
    if not torch.isfinite(total):
        raise FloatingPointError(
            "Non-finite cross-attention/ordinal loss: "
            f"loc_bce={float(loc_bce.detach()):.6g}, "
            f"loc_dice={float(loc_dice.detach()):.6g}, "
            f"damage_focal={float(damage_focal.detach()):.6g}, "
            f"damage_dice={float(damage_dice.detach()):.6g}, "
            f"ordinal={float(ordered.detach()):.6g}"
        )
    return total, loc_bce, loc_dice, damage_total


def make_model(device: torch.device) -> nn.Module:
    model = BuildingGuidedCrossAttentionOrdinalNet(image_size=896, width=96)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model.to(device)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        print("AMP autocast dtype: bfloat16", flush=True)
    runner.XBDOriginalDataset = stable.MultiSplitHazardDataset
    runner.make_model = make_model
    runner.compute_supervised_losses = compute_losses
    runner.aggregate_counts = stable.stable_aggregate_counts
    runner.torch.optim.AdamW = stable.ClippedAdamW
    runner.main()
