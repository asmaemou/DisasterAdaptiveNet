#!/usr/bin/env python3
"""Component ablations for the xBD RS-BDNet experiment.

Every variant reuses the full model's dataset, optimization, validation
threshold scan, checkpoint selection, early stopping, and test evaluation.
Only the component named by ``--ablation`` is changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal as full
import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable


ABLATIONS = (
    "full",
    "resnet_only",
    "swin_only",
    "no_cross_attention",
    "no_building_guidance",
    "no_ordinal",
)


def _consume_ablation_argument() -> str:
    """Read our argument and leave the shared runner's arguments untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return str(args.ablation)


ABLATION = _consume_ablation_argument()


class IdentityPair(nn.Module):
    """Bypass pre/post cross-attention without changing tensor shapes."""

    def forward(
        self, pre: torch.Tensor, post: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return pre, post


class RSBDNetAblation(full.BuildingGuidedCrossAttentionOrdinalNet):
    """RS-BDNet with exactly one requested component ablated."""

    def __init__(self, ablation: str, image_size: int = 896, width: int = 96):
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")
        super().__init__(image_size=image_size, width=width)
        self.ablation = ablation
        self.width = width

        if ablation == "no_cross_attention":
            self.res_cross_attention = IdentityPair()
            self.swin_cross_attention = IdentityPair()

        if ablation == "resnet_only":
            del self.swin
            del self.swin_fpn
            del self.swin_cross_attention
            del self.swin_change
            self.hybrid_fusion = self._single_branch_fusion(width)

        if ablation == "swin_only":
            del self.resnet_unet
            del self.res_projection
            del self.res_cross_attention
            del self.res_change
            self.hybrid_fusion = self._single_branch_fusion(width)

        if ablation == "no_building_guidance":
            # The damage head receives only the fused representation: no
            # multiplicative probability weighting and no probability channel.
            self.damage_refine = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                stable.norm(width),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                stable.norm(width),
                nn.GELU(),
            )

        if ablation == "no_ordinal":
            del self.ordinal_head

    @staticmethod
    def _single_branch_fusion(width: int) -> nn.Module:
        """Match the full fusion depth while accepting one branch."""
        return nn.Sequential(
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            stable.norm(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 1, bias=False),
            stable.norm(width),
            nn.GELU(),
        )

    def _resnet_change(
        self, pre: torch.Tensor, post: torch.Tensor, fusion_size
    ) -> torch.Tensor:
        res_pre = F.interpolate(
            self.resnet_unet.forward_once(pre),
            size=fusion_size,
            mode="bilinear",
            align_corners=False,
        )
        res_post = F.interpolate(
            self.resnet_unet.forward_once(post),
            size=fusion_size,
            mode="bilinear",
            align_corners=False,
        )
        res_pre = self.res_projection(res_pre)
        res_post = self.res_projection(res_post)
        res_pre, res_post = self.res_cross_attention(res_pre, res_post)
        return self.res_change(self.temporal(res_pre, res_post))

    def _swin_change(
        self, pre: torch.Tensor, post: torch.Tensor, fusion_size
    ) -> torch.Tensor:
        swin_pre = self.swin_fpn(self.swin(pre), fusion_size)
        swin_post = self.swin_fpn(self.swin(post), fusion_size)
        swin_pre, swin_post = self.swin_cross_attention(swin_pre, swin_post)
        return self.swin_change(self.temporal(swin_pre, swin_post))

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        del condition
        pre, post = images[:, :3], images[:, 3:]
        output_size = images.shape[-2:]
        fusion_size = (
            max(1, output_size[0] // 4),
            max(1, output_size[1] // 4),
        )

        if self.ablation == "resnet_only":
            fusion_input = self._resnet_change(pre, post, fusion_size)
        elif self.ablation == "swin_only":
            fusion_input = self._swin_change(pre, post, fusion_size)
        else:
            res_change = self._resnet_change(pre, post, fusion_size)
            swin_change = self._swin_change(pre, post, fusion_size)
            fusion_input = torch.cat([res_change, swin_change], dim=1)

        fused = self.hybrid_fusion(fusion_input)
        fused = fused + self.refine(fused)

        localization = self.localization_head(fused)
        if self.ablation == "no_building_guidance":
            damage_features = self.damage_refine(fused)
        else:
            building_probability = torch.sigmoid(localization)
            guided = fused * (1.0 + building_probability)
            damage_features = self.damage_refine(
                torch.cat([guided, building_probability], dim=1)
            )

        damage = self.damage_head(damage_features)
        if self.ablation == "no_ordinal":
            # Preserve the eight-channel interface used by the common runner;
            # these channels have no parameters and are excluded from the loss.
            ordinal = damage.new_zeros(
                damage.shape[0], 3, damage.shape[2], damage.shape[3]
            )
        else:
            ordinal = self.ordinal_head(damage_features)

        logits = torch.cat([localization, damage, ordinal], dim=1)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


def compute_losses(logits, loc, dmg, loc_criterion, dmg_criterion, device, args):
    loc_bce, loc_dice = loc_criterion(logits[:, 0], loc)
    weights = dmg_criterion.weight.to(device=device, dtype=logits.dtype)
    damage_focal, damage_dice = stable.focal_dice_damage_loss(
        logits[:, 1:5], dmg, weights
    )
    if ABLATION == "no_ordinal":
        ordered = logits[:, 5:8].sum() * 0.0
    else:
        ordered = full.ordinal_loss(logits[:, 5:8], dmg)

    damage_total = damage_focal + damage_dice + 0.30 * ordered
    total = (
        args.loc_bce_weight * loc_bce
        + args.loc_dice_weight * loc_dice
        + args.dmg_ce_weight * damage_total
    )
    if not torch.isfinite(total):
        raise FloatingPointError(
            f"Non-finite loss for ablation={ABLATION}: "
            f"loc_bce={float(loc_bce.detach()):.6g}, "
            f"loc_dice={float(loc_dice.detach()):.6g}, "
            f"damage_focal={float(damage_focal.detach()):.6g}, "
            f"damage_dice={float(damage_dice.detach()):.6g}, "
            f"ordinal={float(ordered.detach()):.6g}"
        )
    return total, loc_bce, loc_dice, damage_total


def make_model(device: torch.device) -> nn.Module:
    model = RSBDNetAblation(ABLATION, image_size=896, width=96)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Ablation variant: {ABLATION}", flush=True)
    print(f"Parameters: total={total:,}; trainable={trainable:,}", flush=True)
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
