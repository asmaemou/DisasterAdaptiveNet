#!/usr/bin/env python3
"""Unified six-dataset RS-BDNet with dataset-adaptive difference modulation.

This is a separate experiment.  It jointly trains one shared ResNet34/Swin-T
model on xBD and five EBD events.  Model/threshold selection uses the mean
validation score across domains; every test split remains untouched until the
best checkpoint and one global localization threshold have been selected.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as base
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable
from transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal import (
    BuildingGuidedCrossAttentionOrdinalNet,
    ordinal_loss,
)


@dataclass(frozen=True)
class DomainSpec:
    domain_id: int
    slug: str
    label: str
    root: Path
    train_split: str
    val_split: str
    test_split: str


class DomainTaggedDataset(Dataset):
    """Preserve the original sample while replacing conditioning with domain ID."""

    def __init__(self, dataset: Dataset, domain: DomainSpec, tag_id: int | None = None):
        self.dataset = dataset
        self.domain = domain
        self.tag_id = domain.domain_id if tag_id is None else int(tag_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["cond_id"] = torch.tensor([self.tag_id], dtype=torch.long)
        item["domain_slug"] = self.domain.slug
        return item


class DatasetAdaptiveCrossTemporalDifferenceModulation(nn.Module):
    """Small domain embedding that modulates pre/post difference features."""

    def __init__(self, channels: int, num_domains: int, embedding_dim: int, dropout: float, mask_p: float):
        super().__init__()
        self.unknown_id = num_domains
        self.mask_p = float(mask_p)
        self.embedding = nn.Embedding(num_domains + 1, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.affine = nn.ModuleDict()
        self.spatial_gate = nn.ModuleDict()
        for branch in ("resnet", "swin"):
            self.affine[branch] = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.SiLU(),
                nn.Linear(embedding_dim, channels * 2),
            )
            self.spatial_gate[branch] = nn.Conv2d(channels * 3, channels, 1)
            nn.init.zeros_(self.affine[branch][-1].weight)
            nn.init.zeros_(self.affine[branch][-1].bias)
            nn.init.zeros_(self.spatial_gate[branch].weight)
            nn.init.zeros_(self.spatial_gate[branch].bias)

    def domain_embedding(self, domain_id: torch.Tensor) -> torch.Tensor:
        ids = domain_id.reshape(-1).long()
        if self.training and self.mask_p > 0:
            masked = torch.rand(ids.shape, device=ids.device) < self.mask_p
            ids = torch.where(masked, torch.full_like(ids, self.unknown_id), ids)
        return self.dropout(self.embedding(ids))

    def forward(self, pre: torch.Tensor, post: torch.Tensor, embedding: torch.Tensor, branch: str):
        difference = torch.abs(post - pre)
        gamma, beta = self.affine[branch](embedding).chunk(2, dim=1)
        gamma = 0.5 * torch.tanh(gamma)[:, :, None, None]
        beta = 0.1 * torch.tanh(beta)[:, :, None, None]
        # Zero initialization makes both terms exact identities at startup.
        spatial = 2.0 * torch.sigmoid(self.spatial_gate[branch](torch.cat([pre, post, difference], 1)))
        return ((1.0 + gamma) * difference + beta) * spatial


class UnifiedRSBDNet(BuildingGuidedCrossAttentionOrdinalNet):
    def __init__(self, image_size: int, mode: str, embedding_dim: int, dropout: float, mask_p: float):
        super().__init__(image_size=image_size, width=96)
        self.mode = mode
        self.domain_modulation = None
        if mode == "dactdm":
            self.domain_modulation = DatasetAdaptiveCrossTemporalDifferenceModulation(
                channels=96, num_domains=6, embedding_dim=embedding_dim,
                dropout=dropout, mask_p=mask_p,
            )

    @staticmethod
    def temporal_with_difference(pre, post, difference):
        return torch.cat([pre, post, difference, pre * post], dim=1)

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pre, post = images[:, :3], images[:, 3:]
        output_size = images.shape[-2:]
        fusion_size = (max(1, output_size[0] // 4), max(1, output_size[1] // 4))

        res_pre = F.interpolate(self.resnet_unet.forward_once(pre), size=fusion_size, mode="bilinear", align_corners=False)
        res_post = F.interpolate(self.resnet_unet.forward_once(post), size=fusion_size, mode="bilinear", align_corners=False)
        res_pre, res_post = self.res_projection(res_pre), self.res_projection(res_post)
        res_pre, res_post = self.res_cross_attention(res_pre, res_post)

        swin_pre = self.swin_fpn(self.swin(pre), fusion_size)
        swin_post = self.swin_fpn(self.swin(post), fusion_size)
        swin_pre, swin_post = self.swin_cross_attention(swin_pre, swin_post)

        if self.domain_modulation is None:
            res_difference = torch.abs(res_post - res_pre)
            swin_difference = torch.abs(swin_post - swin_pre)
        else:
            embedding = self.domain_modulation.domain_embedding(condition)
            res_difference = self.domain_modulation(res_pre, res_post, embedding, "resnet")
            swin_difference = self.domain_modulation(swin_pre, swin_post, embedding, "swin")

        res_change = self.res_change(self.temporal_with_difference(res_pre, res_post, res_difference))
        swin_change = self.swin_change(self.temporal_with_difference(swin_pre, swin_post, swin_difference))
        fused = self.hybrid_fusion(torch.cat([res_change, swin_change], dim=1))
        fused = fused + self.refine(fused)

        localization = self.localization_head(fused)
        building_attention = torch.sigmoid(localization)
        guided = self.damage_refine(torch.cat([fused * (1.0 + building_attention), building_attention], dim=1))
        logits = torch.cat([localization, self.damage_head(guided), self.ordinal_head(guided)], dim=1)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-parent", type=Path, required=True)
    p.add_argument("--xview2-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("dactdm", "none"), default="dactdm")
    p.add_argument("--exclude-domain", default="none")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--samples-per-epoch", type=int, default=9168)
    p.add_argument("--img-size", type=int, default=896)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--embedding-dropout", type=float, default=0.25)
    p.add_argument("--domain-mask-probability", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", type=Path)
    p.add_argument("--thresholds", type=float, nargs="+", default=[.35, .4, .45, .5, .55, .6, .65, .7, .75])
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def domain_specs(args) -> List[DomainSpec]:
    specs = [
        DomainSpec(0, "xbd", "xBD", args.xview2_root, "train+tier3", "hold", "test"),
        DomainSpec(1, "earthquake_turkey", "Earthquake Turkey", args.data_parent / "earthquake_turkey_preprocessed", "train", "val", "test"),
        DomainSpec(2, "mount_semeru", "Mount Semeru Eruption", args.data_parent / "mount_semeru_eruption_preprocessed", "train", "val", "test"),
        DomainSpec(3, "texas_tornadoes", "Texas Tornadoes", args.data_parent / "texas_tornadoes_preprocessed", "train", "val", "test"),
        DomainSpec(4, "hurricane_delta", "Hurricane Delta", args.data_parent / "hurricane_delta_preprocessed", "train", "val", "test"),
        DomainSpec(5, "pakistan_flooding", "Pakistan Flooding", args.data_parent / "pakistan_flooding_preprocessed", "train", "val", "test"),
    ]
    if args.exclude_domain != "none":
        if sum(s.slug == args.exclude_domain for s in specs) != 1:
            raise ValueError("--exclude-domain must be one of the six domain slugs or 'none'")
    return specs


def make_dataset(spec: DomainSpec, split: str, image_size: int, training: bool, tag_id: int | None = None):
    split_name = getattr(spec, f"{split}_split")
    cls = stable.MultiSplitHazardDataset if spec.slug == "xbd" else base.XBDOriginalDataset
    dataset = cls(root=spec.root, split=split_name, image_size=image_size, training=training, conditioning_id=spec.domain_id)
    return DomainTaggedDataset(dataset, spec, tag_id=tag_id)


def aggregate_training_counts(datasets: Iterable[DomainTaggedDataset]):
    background = buildings = 0
    damage = np.zeros(4, dtype=np.int64)
    for tagged in datasets:
        bg, fg = tagged.dataset.get_localization_pixel_counts()
        background += int(bg); buildings += int(fg)
        damage += tagged.dataset.get_damage_class_counts().astype(np.int64)
    pos_weight = min(10.0, background / max(1, buildings))
    inv = 1.0 / np.sqrt(np.maximum(damage, 1).astype(np.float64))
    weights = inv / inv.mean()
    return torch.tensor([pos_weight], dtype=torch.float32), torch.tensor(weights, dtype=torch.float32), damage


def damage_loss(logits, target, weights):
    focal, dice = stable.focal_dice_damage_loss(logits[:, 1:5], target, weights.to(logits.device, logits.dtype))
    return focal + dice + 0.30 * ordinal_loss(logits[:, 5:8], target)


def f1(tp, fp, fn):
    return (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    loc_tp = loc_fp = loc_fn = 0
    counts = {c: [0, 0, 0] for c in range(1, 5)}
    for batch in loader:
        image = batch["img"].to(device, non_blocking=True)
        cond = batch["cond_id"].to(device, non_blocking=True)
        truth_loc = batch["loc"].numpy() > 0.5
        truth_damage = batch["dmg"].numpy()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(image, cond)
        probability = torch.sigmoid(logits[:, 0]).float().cpu().numpy()
        predicted_loc = probability >= threshold
        predicted_damage = (logits[:, 1:5].argmax(1) + 1).cpu().numpy()
        predicted_damage = predicted_damage * predicted_loc
        loc_tp += int(np.logical_and(predicted_loc, truth_loc).sum())
        loc_fp += int(np.logical_and(predicted_loc, ~truth_loc).sum())
        loc_fn += int(np.logical_and(~predicted_loc, truth_loc).sum())
        valid = truth_damage != 255
        truth_classes = np.where(valid, truth_damage + 1, 0)
        for c in range(1, 5):
            counts[c][0] += int(np.logical_and(predicted_damage == c, truth_classes == c).sum())
            counts[c][1] += int(np.logical_and(predicted_damage == c, truth_classes != c).sum())
            counts[c][2] += int(np.logical_and(predicted_damage != c, truth_classes == c).sum())
    class_f1 = [f1(*counts[c]) for c in range(1, 5)]
    harmonic = 4.0 / sum(1.0 / max(value, 1e-6) for value in class_f1)
    localization = f1(loc_tp, loc_fp, loc_fn)
    return {
        "localization_f1": localization, "no_damage_f1": class_f1[0],
        "minor_damage_f1": class_f1[1], "major_damage_f1": class_f1[2],
        "destroyed_f1": class_f1[3], "macro_damage_f1": float(np.mean(class_f1)),
        "harmonic_damage_f1": harmonic, "official_xview2_score": .3 * localization + .7 * harmonic,
    }


@torch.no_grad()
def evaluate_thresholds(model, loader, device, thresholds):
    """Evaluate every threshold in one inference pass (important for long HPC runs)."""
    model.eval()
    state = {
        float(t): {"loc": [0, 0, 0], "damage": {c: [0, 0, 0] for c in range(1, 5)}}
        for t in thresholds
    }
    for batch in loader:
        image = batch["img"].to(device, non_blocking=True)
        cond = batch["cond_id"].to(device, non_blocking=True)
        truth_loc = batch["loc"].numpy() > 0.5
        truth_damage = batch["dmg"].numpy()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(image, cond)
        probability = torch.sigmoid(logits[:, 0]).float().cpu().numpy()
        damage_class = (logits[:, 1:5].argmax(1) + 1).cpu().numpy()
        valid = truth_damage != 255
        truth_classes = np.where(valid, truth_damage + 1, 0)
        for threshold in state:
            predicted_loc = probability >= threshold
            predicted_damage = damage_class * predicted_loc
            loc = state[threshold]["loc"]
            loc[0] += int(np.logical_and(predicted_loc, truth_loc).sum())
            loc[1] += int(np.logical_and(predicted_loc, ~truth_loc).sum())
            loc[2] += int(np.logical_and(~predicted_loc, truth_loc).sum())
            for c in range(1, 5):
                counts = state[threshold]["damage"][c]
                counts[0] += int(np.logical_and(predicted_damage == c, truth_classes == c).sum())
                counts[1] += int(np.logical_and(predicted_damage == c, truth_classes != c).sum())
                counts[2] += int(np.logical_and(predicted_damage != c, truth_classes == c).sum())
    results = {}
    for threshold, values in state.items():
        class_f1 = [f1(*values["damage"][c]) for c in range(1, 5)]
        harmonic = 4.0 / sum(1.0 / max(value, 1e-6) for value in class_f1)
        localization = f1(*values["loc"])
        results[threshold] = {
            "localization_f1": localization, "no_damage_f1": class_f1[0],
            "minor_damage_f1": class_f1[1], "major_damage_f1": class_f1[2],
            "destroyed_f1": class_f1[3], "macro_damage_f1": float(np.mean(class_f1)),
            "harmonic_damage_f1": harmonic,
            "official_xview2_score": .3 * localization + .7 * harmonic,
        }
    return results


def select_threshold(model, loaders, device, thresholds):
    per_domain = {
        slug: evaluate_thresholds(model, loader, device, thresholds)
        for slug, loader in loaders.items()
    }
    scan = []
    best = None
    for threshold in thresholds:
        threshold = float(threshold)
        domains = {slug: metrics[threshold] for slug, metrics in per_domain.items()}
        mean_score = float(np.mean([v["official_xview2_score"] for v in domains.values()]))
        scan.append({"threshold": threshold, "mean_domain_score": mean_score})
        if best is None or mean_score > best[0]:
            best = (mean_score, threshold, domains)
    return best, scan


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_score, best_threshold, stale, args):
    module = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "model_state_dict": module.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch, "best_score": best_score, "best_threshold": best_threshold,
        "epochs_without_improvement": stale, "args": vars(args),
    }, path)


def main():
    args = arguments()
    base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_specs = domain_specs(args)
    specs = [s for s in all_specs if s.slug != args.exclude_domain]
    held_out = next((s for s in all_specs if s.slug == args.exclude_domain), None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    score_dir = args.output_dir / "scores"; score_dir.mkdir(exist_ok=True)

    train_sets = [make_dataset(s, "train", args.img_size, True) for s in specs]
    val_sets = {s.slug: make_dataset(s, "val", args.img_size, False) for s in specs}
    # In leave-one-domain-out mode, evaluate the excluded domain using ID 6,
    # whose embedding was learned through random ID masking on the five sources.
    test_sets = {
        s.slug: make_dataset(s, "test", args.img_size, False, tag_id=6 if s is held_out else None)
        for s in all_specs
    }
    combined = ConcatDataset(train_sets)
    weights = []
    for dataset in train_sets:
        weights.extend([1.0 / len(dataset)] * len(dataset))
    sampler = WeightedRandomSampler(weights, num_samples=args.samples_per_epoch, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loaders = {k: DataLoader(v, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) for k, v in val_sets.items()}
    test_loaders = {k: DataLoader(v, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) for k, v in test_sets.items()}

    print("===== UNIFIED SIX-DATASET RS-BDNET =====", flush=True)
    print(f"Mode: {args.mode} | device: {device} | balanced samples/epoch: {args.samples_per_epoch}", flush=True)
    for s, train in zip(specs, train_sets):
        print(f"domain_id={s.domain_id} {s.label}: train={len(train)} val={len(val_sets[s.slug])} test={len(test_sets[s.slug])}", flush=True)
    if held_out is not None:
        print(f"LEAVE-ONE-DOMAIN-OUT: {held_out.label} is excluded from train/validation and tested with unknown domain ID=6", flush=True)
    print("Checkpoint/threshold selection: mean validation score across included domains", flush=True)

    loc_weight, damage_weights, damage_counts = aggregate_training_counts(train_sets)
    print(f"Localization pos_weight: {loc_weight.tolist()}", flush=True)
    print(f"Damage counts: {damage_counts.tolist()} | weights: {damage_weights.tolist()}", flush=True)
    model = UnifiedRSBDNet(args.img_size, args.mode, args.embedding_dim, args.embedding_dropout, args.domain_mask_probability).to(device)
    optimizer = stable.ClippedAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .05)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loc_criterion = base.BCEDiceLoss(pos_weight=loc_weight.to(device)).to(device)

    start_epoch, best_score, best_threshold, stale = 1, -1.0, .5, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"]); scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1; best_score = float(state["best_score"])
        best_threshold = float(state["best_threshold"]); stale = int(state.get("epochs_without_improvement", 0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running = 0.0
        for step, batch in enumerate(train_loader, 1):
            image = batch["img"].to(device, non_blocking=True); loc = batch["loc"].to(device, non_blocking=True)
            dmg = batch["dmg"].to(device, non_blocking=True); cond = batch["cond_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                logits = model(image, cond)
                loc_bce, loc_dice = loc_criterion(logits[:, 0], loc)
                dmg_loss = damage_loss(logits, dmg, damage_weights)
                loss = loc_bce + loc_dice + dmg_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += float(loss.detach())
            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | loss={running/step:.4f}", flush=True)
        scheduler.step()
        selected, scan = select_threshold(model, val_loaders, device, args.thresholds)
        val_score, threshold, domain_metrics = selected
        row = {"epoch": epoch, "train_loss": running / len(train_loader), "mean_validation_score": val_score,
               "threshold": threshold, "domain_metrics": domain_metrics}
        history.append(row)
        (score_dir / "history.json").write_text(json.dumps(history, indent=2))
        with (score_dir / f"epoch_{epoch:03d}_threshold_scan.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scan[0].keys()); writer.writeheader(); writer.writerows(scan)
        improved = val_score > best_score
        if improved:
            best_score, best_threshold, stale = val_score, threshold, 0
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_score, best_threshold, stale, args)
        else:
            stale += 1
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_score, best_threshold, stale, args)
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, best_score, best_threshold, stale, args)
        print(f"Epoch {epoch}: mean_val_score={val_score:.6f} threshold={threshold:.2f} best={best_score:.6f} stale={stale}", flush=True)
        if stale >= args.patience:
            print("Early stopping.", flush=True); break

    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"]); best_threshold = float(best["best_threshold"])
    final_domains = {slug: evaluate(model, loader, device, best_threshold) for slug, loader in test_loaders.items()}
    keys = next(iter(final_domains.values())).keys()
    macro = {key: float(np.mean([metrics[key] for metrics in final_domains.values()])) for key in keys}
    result = {"experiment": f"Unified six-dataset RS-BDNet mode={args.mode}", "best_epoch": int(best["epoch"]),
              "validation_selected_threshold": best_threshold, "test_by_domain": final_domains,
              "macro_average_across_domains": macro, "domain_id_map": {s.domain_id: s.label for s in all_specs},
              "leave_one_domain_out": None if held_out is None else held_out.label}
    (score_dir / "final_test_results.json").write_text(json.dumps(result, indent=2))
    lines = [result["experiment"], f"Best epoch: {result['best_epoch']}", f"Global threshold: {best_threshold:.2f}"]
    for slug, metrics in final_domains.items():
        lines.append(f"\n{slug}: " + " | ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
    lines.append("\nMACRO ACROSS DATASETS: " + " | ".join(f"{k}={v:.6f}" for k, v in macro.items()))
    (score_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"DONE: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
