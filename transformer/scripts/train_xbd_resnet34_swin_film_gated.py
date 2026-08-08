#!/usr/bin/env python3
"""Train an isolated ResNet34 + Swin-Tiny + FiLM gated-fusion experiment.

The data protocol is xBD train+tier3 -> hold model selection -> untouched test.
This module deliberately reuses the mature loader, checkpointing, threshold scan,
and metric implementation from train_xbd_supervised_disasteradaptivenet.py while
replacing only the dataset conditioning, architecture, and damage loss.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
from transformer.scripts.train_xbd_hrtbda_v5_swin_pretrained_cascade import (
    SwinPretrainedBackbone,
)
from utils.models import Res34_Unet_Loc


HAZARDS = ("volcanic", "earthquake", "wildfire", "flood", "storm")
BASE_XBD_DATASET = runner.XBDOriginalDataset


def hazard_id(stem: str) -> int:
    """Map standard xBD event names to the five paper conditioning groups."""
    name = stem.lower()
    rules = (
        ("volcanic", ("volcano", "eruption")),
        ("earthquake", ("earthquake", "tsunami")),
        # xBD contains both North-American "wildfire"/"fire" names and the
        # Australian event name "pinery-bushfire". They share one FiLM class.
        ("wildfire", ("wildfire", "bushfire", "-fire", "fire-")),
        ("flood", ("flood",)),
        ("storm", ("hurricane", "tornado", "cyclone", "typhoon")),
    )
    for label, tokens in rules:
        if any(token in name for token in tokens):
            return HAZARDS.index(label)
    raise RuntimeError(
        f"Unknown xBD hazard for '{stem}'. Add an explicit mapping instead of "
        "silently assigning an incorrect FiLM condition."
    )


class MultiSplitHazardDataset(BASE_XBD_DATASET):
    """Supports --train-split train+tier3 and assigns per-image hazard IDs."""

    def __init__(self, root, split, image_size, training, conditioning_id=0):
        split_names = [item.strip() for item in str(split).split("+") if item.strip()]
        if not split_names:
            raise ValueError("At least one split is required")
        super().__init__(root, split_names[0], image_size, training, conditioning_id)
        if len(split_names) > 1:
            samples = list(self.samples)
            for extra in split_names[1:]:
                other = BASE_XBD_DATASET(
                    root=root,
                    split=extra,
                    image_size=image_size,
                    training=training,
                    conditioning_id=conditioning_id,
                )
                samples.extend(other.samples)
            stems = [sample.stem for sample in samples]
            if len(stems) != len(set(stems)):
                raise RuntimeError("Duplicate stems found across requested training splits")
            self.samples = samples
            self.split = "+".join(split_names)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["cond_id"] = torch.tensor([hazard_id(str(item["stem"]))], dtype=torch.long)
        return item


class IdentityFiLM(nn.Module):
    """Hazard-conditioned affine modulation, initialized as the identity."""

    def __init__(self, conditions: int, channels: int, hidden: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(conditions, hidden)
        self.affine = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels)
        )
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.reshape(-1).long()
        gamma, beta = self.affine(self.embedding(condition)).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return feature * (1.0 + gamma) + beta


def norm(channels: int) -> nn.GroupNorm:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SwinFPN(nn.Module):
    def __init__(self, channels: List[int], width: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, width, 1, bias=False), norm(width), nn.GELU()) for c in channels]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(width * len(channels), width, 3, padding=1, bias=False),
            norm(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            norm(width),
            nn.GELU(),
        )

    def forward(self, features: List[torch.Tensor], output_size) -> torch.Tensor:
        target = features[0].shape[-2:]
        projected = [
            F.interpolate(layer(x), size=target, mode="bilinear", align_corners=False)
            for layer, x in zip(self.projections, features)
        ]
        return F.interpolate(
            self.refine(torch.cat(projected, dim=1)),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class ResNet34SwinFiLMGated(nn.Module):
    """Shared Siamese CNN/Transformer encoders with FiLM and learned fusion."""

    def __init__(self, image_size: int = 896, width: int = 96):
        super().__init__()
        # Paper-style shared ResNet34 U-Net. forward_once returns 48 channels.
        cfg = type("Cfg", (), {})()
        cfg.MODEL = type("ModelCfg", (), {"OUT_CHANNELS": 5})()
        self.resnet_unet = Res34_Unet_Loc(cfg)

        self.swin = SwinPretrainedBackbone(
            variant="swin_tiny_patch4_window7_224",
            pretrained=True,
            img_size=image_size,
        )
        self.swin_fpn = SwinFPN(self.swin.channels, width)

        self.res_film = IdentityFiLM(len(HAZARDS), 48)
        self.swin_film = IdentityFiLM(len(HAZARDS), width)

        self.res_temporal = nn.Sequential(
            nn.Conv2d(48 * 4, width, 3, padding=1, bias=False), norm(width), nn.GELU()
        )
        self.swin_temporal = nn.Sequential(
            nn.Conv2d(width * 4, width, 3, padding=1, bias=False), norm(width), nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            norm(width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False), norm(width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False), norm(width), nn.GELU(),
        )
        self.head = nn.Conv2d(width, 5, 1)

    @staticmethod
    def temporal(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return torch.cat([pre, post, torch.abs(post - pre), pre * post], dim=1)

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pre, post = images[:, :3], images[:, 3:]
        size = images.shape[-2:]

        # Perform temporal interaction and cross-backbone fusion at stride 4.
        # The earlier full-resolution implementation created several
        # [B, 384, 896, 896] activation graphs and could wait indefinitely in
        # CUDA/cuDNN with ~25 GB allocated but no active kernels. Predicting at
        # stride 4 and upsampling only five logits preserves pixel-aligned
        # segmentation while reducing fusion activation area by 16x.
        fusion_size = (max(1, size[0] // 4), max(1, size[1] // 4))

        res_pre = self.resnet_unet.forward_once(pre)
        res_post = self.res_film(self.resnet_unet.forward_once(post), condition)
        res_pre = F.interpolate(res_pre, size=fusion_size, mode="bilinear", align_corners=False)
        res_post = F.interpolate(res_post, size=fusion_size, mode="bilinear", align_corners=False)
        res_pair = self.res_temporal(self.temporal(res_pre, res_post))

        swin_pre = self.swin_fpn(self.swin(pre), fusion_size)
        swin_post = self.swin_film(self.swin_fpn(self.swin(post), fusion_size), condition)
        swin_pair = self.swin_temporal(self.temporal(swin_pre, swin_post))

        gate = self.gate(torch.cat([res_pair, swin_pair], dim=1))
        fused = gate * swin_pair + (1.0 - gate) * res_pair
        logits = self.head(fused + self.refine(fused))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def make_model(device: torch.device) -> nn.Module:
    # The sbatch uses 896. Keeping this explicit also makes incompatible
    # checkpoint construction fail early rather than silently changing Swin.
    model = ResNet34SwinFiLMGated(image_size=896, width=96)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model.to(device)


def focal_dice_damage_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    valid = target != 255
    # Some xBD tiles contain no annotated building pixels. An empty masked
    # tensor followed by .mean() is NaN. Such tiles still provide valid
    # localization supervision, so contribute a differentiable zero to the
    # damage objective instead of being discarded or corrupting training.
    if not bool(valid.any()):
        zero = logits.sum() * 0.0
        return zero, zero

    safe = target.clamp(0, 3)
    one_hot = F.one_hot(safe, num_classes=4).permute(0, 3, 1, 2).to(logits.dtype)
    valid4 = valid[:, None].expand_as(one_hot)
    bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * one_hot + (1.0 - probability) * (1.0 - one_hot)
    focal = ((1.0 - pt).pow(2.0) * bce * weights[None, :, None, None])[valid4].mean()

    probability = probability * valid[:, None]
    truth = one_hot * valid[:, None]
    intersection = (probability * truth).sum((0, 2, 3))
    denominator = probability.sum((0, 2, 3)) + truth.sum((0, 2, 3))
    dice = ((1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)) * weights).mean()
    return focal, dice


def compute_losses(logits, loc, dmg, loc_criterion, dmg_criterion, device, args):
    loc_bce, loc_dice = loc_criterion(logits[:, 0], loc)
    class_weights = dmg_criterion.weight.to(device=device, dtype=logits.dtype)
    dmg_focal, dmg_dice = focal_dice_damage_loss(logits[:, 1:5], dmg, class_weights)
    damage = dmg_focal + dmg_dice
    total = args.loc_bce_weight * loc_bce + args.loc_dice_weight * loc_dice + args.dmg_ce_weight * damage
    if not torch.isfinite(total):
        raise FloatingPointError(
            "Non-finite loss detected; stopping before corrupting the checkpoint: "
            f"loc_bce={float(loc_bce.detach()):.6g}, "
            f"loc_dice={float(loc_dice.detach()):.6g}, "
            f"damage_focal={float(dmg_focal.detach()):.6g}, "
            f"damage_dice={float(dmg_dice.detach()):.6g}"
        )
    return total, loc_bce, loc_dice, damage


if __name__ == "__main__":
    runner.XBDOriginalDataset = MultiSplitHazardDataset
    runner.make_model = make_model
    runner.compute_supervised_losses = compute_losses
    runner.main()
