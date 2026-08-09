#!/usr/bin/env python3
"""ResNet34-Swin-T concatenation with residual multi-scale Inception fusion.

This is a controlled ablation of train_xbd_resnet34_swin_concat.py. The only
model change is the fusion operator after CNN/Transformer concatenation.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable
from transformer.scripts.train_xbd_resnet34_swin_concat import ResNet34SwinConcatenation


def conv_norm_act(in_channels: int, out_channels: int, kernel_size: int, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
        stable.norm(out_channels),
        nn.GELU(),
    )


class ResidualInceptionFusion(nn.Module):
    """Four-scale Inception block with a projected residual shortcut."""

    def __init__(self, in_channels: int = 192, out_channels: int = 96, branch_channels: int = 48):
        super().__init__()
        # Fine channel mixing and small-building detail.
        self.branch_1x1 = conv_norm_act(in_channels, branch_channels, 1)

        # Ordinary local structural context.
        self.branch_3x3 = nn.Sequential(
            conv_norm_act(in_channels, branch_channels, 1),
            conv_norm_act(branch_channels, branch_channels, 3, padding=1),
        )

        # Larger effective receptive field for widespread collapse/debris.
        self.branch_dilated = nn.Sequential(
            conv_norm_act(in_channels, branch_channels, 1),
            conv_norm_act(branch_channels, branch_channels, 3, padding=2, dilation=2),
        )

        # Neighborhood context without losing the original output geometry.
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_norm_act(in_channels, branch_channels, 1),
        )

        merged_channels = branch_channels * 4
        self.merge = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, 1, bias=False),
            stable.norm(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            stable.norm(out_channels),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branches = torch.cat(
            [
                self.branch_1x1(features),
                self.branch_3x3(features),
                self.branch_dilated(features),
                self.branch_pool(features),
            ],
            dim=1,
        )
        return self.activation(self.shortcut(features) + self.merge(branches))


class ResNet34SwinInceptionFusion(ResNet34SwinConcatenation):
    """The validated concatenation network with only its fusion block changed."""

    def __init__(self, image_size: int = 896, width: int = 96):
        super().__init__(image_size=image_size, width=width)
        self.concat_fusion = ResidualInceptionFusion(
            in_channels=width * 2,
            out_channels=width,
            branch_channels=width // 2,
        )


def make_model(device: torch.device) -> nn.Module:
    model = ResNet34SwinInceptionFusion(image_size=896, width=96)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model.to(device)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        print("AMP autocast dtype: bfloat16", flush=True)
    runner.XBDOriginalDataset = stable.MultiSplitHazardDataset
    runner.make_model = make_model
    runner.compute_supervised_losses = stable.compute_losses
    runner.aggregate_counts = stable.stable_aggregate_counts
    runner.torch.optim.AdamW = stable.ClippedAdamW
    runner.main()
