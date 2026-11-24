#!/usr/bin/env python3
"""RS-BDNet parameter-free cross-branch mean fusion ablation.

Retains the original architecture, loss, dataset and training runner;
only the learned hybrid_fusion module is replaced. No original files edited.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal as full
import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable


class MeanFusion(nn.Module):
    """Map concatenated [ResNet, Swin] change features from 2W to W."""
    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2 * self.width:
            raise ValueError(f"Expected N,{2*self.width},H,W; got {tuple(x.shape)}")
        res, swin = x.split(self.width, dim=1)
        return 0.5 * (res + swin)


class RSBDNetSimpleFusion(full.BuildingGuidedCrossAttentionOrdinalNet):
    def __init__(self, image_size: int = 896, width: int = 96):
        super().__init__(image_size=image_size, width=width)
        self.hybrid_fusion = MeanFusion(width)


def make_model(device: torch.device) -> nn.Module:
    model = RSBDNetSimpleFusion(image_size=896, width=96)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Ablation variant: simple_fusion", flush=True)
    print(f"Parameters: total={total:,}; trainable={trainable:,}", flush=True)
    assert sum(p.numel() for p in model.hybrid_fusion.parameters()) == 0
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-loader", choices=("xbd_multisplit", "standard"),
                        default="xbd_multisplit")
    parser.add_argument("--check-fusion", action="store_true",
                        help="CPU-only shape/value/gradient check; no training")
    opts, remaining = parser.parse_known_args()
    if opts.check_fusion:
        x = torch.randn(2, 192, 7, 7, requires_grad=True)
        module = MeanFusion(96)
        y = module(x)
        assert y.shape == (2, 96, 7, 7)
        torch.testing.assert_close(y, (x[:, :96] + x[:, 96:]) / 2)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 0.5))
        print("PASS: shape, arithmetic mean, both branch gradients; zero fusion parameters")
        return
    sys.argv = [sys.argv[0], *remaining]
    if torch.cuda.is_available():
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        print("AMP autocast dtype: bfloat16", flush=True)
    runner.XBDOriginalDataset = (stable.MultiSplitHazardDataset
        if opts.dataset_loader == "xbd_multisplit" else stable.BASE_XBD_DATASET)
    runner.make_model = make_model
    runner.compute_supervised_losses = full.compute_losses
    runner.aggregate_counts = stable.stable_aggregate_counts
    runner.torch.optim.AdamW = stable.ClippedAdamW
    runner.main()


if __name__ == "__main__":
    main()
