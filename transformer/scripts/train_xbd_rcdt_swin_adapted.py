#!/usr/bin/env python3
"""RCDT-SwinT adapted from binary change detection to xBD damage mapping.

Paper-derived components:
  * weight-sharing Siamese Swin-T backbone and light top-down FPN;
  * RCAM offset cross-attention: cosine similarity and Q - attention(Q,K)V;
  * class-query cross-attention + FFN over three coarse-to-fine scales;
  * FCM high-resolution constraint and class-embedding dot product;
  * multi-scale auxiliary supervision.

Adaptation: five independent output embeddings are used for building
localization and four damage classes. This is not an exact reproduction of the
paper's binary-change output task.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable
from transformer.scripts.train_xbd_hrtbda_v5_swin_pretrained_cascade import SwinPretrainedBackbone


def sine_position_2d(height: int, width: int, channels: int, device, dtype) -> torch.Tensor:
    if channels % 4:
        raise ValueError("2D sine position channels must be divisible by four")
    quarter = channels // 4
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(1, quarter - 1)))
    y = y[:, None] * omega[None, :] * (2.0 * math.pi)
    x = x[:, None] * omega[None, :] * (2.0 * math.pi)
    y_embed = torch.cat([y.sin(), y.cos()], dim=1)[:, None, :].expand(-1, width, -1)
    x_embed = torch.cat([x.sin(), x.cos()], dim=1)[None, :, :].expand(height, -1, -1)
    position = torch.cat([y_embed, x_embed], dim=2).reshape(1, height * width, channels)
    return position.to(dtype=dtype)


class LightFPN(nn.Module):
    """Paper-style 1x1 projection, top-down addition, 3x3 conv + GroupNorm."""

    def __init__(self, channels: List[int], width: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1, bias=False),
                    stable.norm(width),
                    nn.GELU(),
                )
                for _ in channels
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        lateral = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for index in range(len(lateral) - 2, -1, -1):
            lateral[index] = lateral[index] + F.interpolate(
                lateral[index + 1], size=lateral[index].shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return [layer(feature) for layer, feature in zip(self.output, lateral)]


class OffsetCrossAttention(nn.Module):
    """RCDT Equation 2: cosine attention followed by subtraction."""

    def __init__(self, channels: int, dropout: float = 0.2, max_side: int = 56):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        self.max_side = int(max_side)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if max(h, w) <= self.max_side:
            return x
        scale = self.max_side / max(h, w)
        size = (max(1, round(h * scale)), max(1, round(w * scale)))
        return F.adaptive_avg_pool2d(x, size)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        before = self._reduce(before)
        after = self._reduce(after)
        if before.shape[-2:] != after.shape[-2:]:
            raise RuntimeError("RCDT before/after feature geometry mismatch")
        b, c, h, w = before.shape
        before_tokens = before.flatten(2).transpose(1, 2)
        after_tokens = after.flatten(2).transpose(1, 2)
        position = sine_position_2d(h, w, c, before.device, before.dtype)

        query = self.query(before_tokens + position)
        key = self.key(after_tokens + position)
        value = self.value(after_tokens + position)
        cosine = torch.matmul(F.normalize(query, dim=-1), F.normalize(key, dim=-1).transpose(-1, -2))
        attention = self.dropout(torch.softmax(cosine, dim=-1))
        relational = query - torch.matmul(attention, value)
        return self.norm(relational), (h, w)


class PixelQueryDecoder(nn.Module):
    """RCDT class/pixel-query cross-attention followed by a three-layer FFN."""

    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.cross = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channels * 4, channels), nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, queries: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        update, _ = self.cross(queries, relation, relation, need_weights=False)
        queries = self.norm1(queries + update)
        return self.norm2(queries + self.ffn(queries))


class FeatureConstraintModule(nn.Module):
    """RCDT FCM: segment embeddings dot high-resolution constrained features."""

    def __init__(self, channels: int, classes: int = 5):
        super().__init__()
        self.segment_mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.constraint = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            stable.norm(channels),
            nn.GELU(),
        )
        self.classes = classes

    def forward(self, queries: torch.Tensor, before_high: torch.Tensor, after_high: torch.Tensor):
        segment = F.normalize(self.segment_mlp(queries), dim=-1)
        constrained = F.normalize(
            self.constraint(torch.cat([before_high, after_high], dim=1)), dim=1
        )
        return torch.einsum("bkc,bchw->bkhw", segment, constrained)


class RCDTSwinDamageAdapted(nn.Module):
    def __init__(self, image_size: int = 896, width: int = 128, classes: int = 5):
        super().__init__()
        self.backbone = SwinPretrainedBackbone(
            variant="swin_tiny_patch4_window7_224", pretrained=True, img_size=image_size
        )
        self.fpn = LightFPN(self.backbone.channels, width)
        self.offset_attention = nn.ModuleList(
            [OffsetCrossAttention(width, dropout=0.2, max_side=56) for _ in range(3)]
        )
        self.query_decoder = nn.ModuleList(
            [PixelQueryDecoder(width, heads=8, dropout=0.2) for _ in range(3)]
        )
        self.class_queries = nn.Parameter(torch.randn(classes, width) * 0.02)
        self.class_position = nn.Parameter(torch.randn(classes, width) * 0.02)
        self.aux_projection = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.fcm = FeatureConstraintModule(width, classes)

    @staticmethod
    def query_map(queries: torch.Tensor, relation: torch.Tensor, shape, projection: nn.Module):
        h, w = shape
        relation_map = relation.transpose(1, 2).reshape(relation.shape[0], -1, h, w)
        embeddings = F.normalize(projection(queries), dim=-1)
        relation_map = F.normalize(relation_map, dim=1)
        return torch.einsum("bkc,bchw->bkhw", embeddings, relation_map)

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        del condition
        before, after = images[:, :3], images[:, 3:]
        output_size = images.shape[-2:]
        before_fpn = self.fpn(self.backbone(before))
        after_fpn = self.fpn(self.backbone(after))

        batch = images.shape[0]
        queries = (self.class_queries + self.class_position)[None].expand(batch, -1, -1)
        auxiliary = []
        # Paper uses three scales, processed coarse-to-fine: 1/32, 1/16, 1/8.
        for layer, feature_index in enumerate((3, 2, 1)):
            relation, shape = self.offset_attention[layer](
                before_fpn[feature_index], after_fpn[feature_index]
            )
            queries = self.query_decoder[layer](queries, relation)
            aux = self.query_map(queries, relation, shape, self.aux_projection[layer])
            auxiliary.append(F.interpolate(aux, size=output_size, mode="bilinear", align_corners=False))

        final = self.fcm(queries, before_fpn[0], after_fpn[0])
        final = F.interpolate(final, size=output_size, mode="bilinear", align_corners=False)
        return torch.cat([final] + auxiliary, dim=1)


def one_scale_loss(logits, loc, dmg, loc_criterion, weights):
    loc_bce, loc_dice = loc_criterion(logits[:, 0], loc)
    damage_focal, damage_dice = stable.focal_dice_damage_loss(logits[:, 1:5], dmg, weights)
    return loc_bce + loc_dice + damage_focal + damage_dice, loc_bce, loc_dice, damage_focal + damage_dice


def compute_losses(logits, loc, dmg, loc_criterion, dmg_criterion, device, args):
    weights = dmg_criterion.weight.to(device=device, dtype=logits.dtype)
    final, loc_bce, loc_dice, damage = one_scale_loss(
        logits[:, :5], loc, dmg, loc_criterion, weights
    )
    auxiliaries = []
    for start in (5, 10, 15):
        aux, _, _, _ = one_scale_loss(logits[:, start:start + 5], loc, dmg, loc_criterion, weights)
        auxiliaries.append(aux)
    auxiliary = torch.stack(auxiliaries).mean()
    total = final + 0.30 * auxiliary
    if not torch.isfinite(total):
        raise FloatingPointError(
            "Non-finite RCDT loss: "
            f"loc_bce={float(loc_bce.detach()):.6g}, "
            f"loc_dice={float(loc_dice.detach()):.6g}, "
            f"damage={float(damage.detach()):.6g}, aux={float(auxiliary.detach()):.6g}"
        )
    return total, loc_bce, loc_dice, damage + 0.30 * auxiliary


def make_model(device: torch.device) -> nn.Module:
    model = RCDTSwinDamageAdapted(image_size=896, width=128, classes=5)
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
