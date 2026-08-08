#!/usr/bin/env python3
"""ResNet34 + Swin-Tiny concatenation ablation on xBD.

This reuses the same data protocol, temporal features, loss, model selection,
and evaluation as the gated-FiLM experiment, but deliberately removes both
FiLM and the gate. The two branches are fused by concatenation + convolution.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as hybrid


class ResNet34SwinConcatenation(hybrid.ResNet34SwinFiLMGated):
    """Siamese ResNet34 and Swin-Tiny fused by channel concatenation."""

    def __init__(self, image_size: int = 896, width: int = 96):
        super().__init__(image_size=image_size, width=width)

        # This requested baseline has neither FiLM nor a learned gate.
        del self.gate
        del self.res_film
        del self.swin_film
        self.concat_fusion = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            hybrid.norm(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 1, bias=False),
            hybrid.norm(width),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pre, post = images[:, :3], images[:, 3:]
        size = images.shape[-2:]
        fusion_size = (max(1, size[0] // 4), max(1, size[1] // 4))

        res_pre = self.resnet_unet.forward_once(pre)
        res_post = self.resnet_unet.forward_once(post)
        res_pre = F.interpolate(res_pre, size=fusion_size, mode="bilinear", align_corners=False)
        res_post = F.interpolate(res_post, size=fusion_size, mode="bilinear", align_corners=False)
        res_pair = self.res_temporal(self.temporal(res_pre, res_post))

        swin_pre = self.swin_fpn(self.swin(pre), fusion_size)
        swin_post = self.swin_fpn(self.swin(post), fusion_size)
        swin_pair = self.swin_temporal(self.temporal(swin_pre, swin_post))

        fused = self.concat_fusion(torch.cat([res_pair, swin_pair], dim=1))
        logits = self.head(fused + self.refine(fused))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def make_model(device: torch.device) -> nn.Module:
    model = ResNet34SwinConcatenation(image_size=896, width=96)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model.to(device)


if __name__ == "__main__":
    runner.XBDOriginalDataset = hybrid.MultiSplitHazardDataset
    runner.make_model = make_model
    runner.compute_supervised_losses = hybrid.compute_losses
    runner.main()
