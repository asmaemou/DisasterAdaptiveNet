#!/usr/bin/env python3
"""
Foreground-aware adversarial domain adaptation for HRTBDA v5.

Purpose
-------
Use labeled xBD as the SOURCE domain and unlabeled IDA-BD train images as the
TARGET domain.  The model starts from your xBD-trained HRTBDA v5 checkpoints
and adapts the Phase-II damage feature extractor with a DANN-style domain
adversarial loss.

Important split policy
----------------------
- xBD train+tier3 labels are used for supervised damage training.
- IDA-BD train images are used WITHOUT labels for the domain loss.
- IDA-BD validation labels are used only for model/threshold/dilation selection.
- IDA-BD test labels are used only once at final evaluation.

Model
-----
Phase I is frozen and provides the cascaded building mask.
Phase II is initialized from the xBD v5 multi-label rare-crop checkpoint.
A domain discriminator sees foreground-pooled Phase-II fused features.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.amp import autocast, GradScaler
    USE_TORCH_AMP = True
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast, GradScaler
    USE_TORCH_AMP = False

# Make imports work when this file is placed in transformer/scripts.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the exact HRTBDA v5 architecture, losses, xBD dataset, and metrics.
import train_xbd_hrtbda_v5_multilabel_crop_cascade as v5

# Reuse IDA-BD discovery/splitting utilities from the supervised fine-tune script.
import train_idabd_xbdv5_supervised_finetune as idft


def json_safe_default(o):
    """Make torch/numpy/path objects safe for json.dump."""
    if torch.is_tensor(o):
        if o.numel() == 1:
            return o.detach().cpu().item()
        return o.detach().cpu().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# -----------------------------
# Basic helpers
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


def cycle_loader(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def safe_item(x) -> float:
    try:
        return float(x.detach().cpu().item())
    except Exception:
        return float(x)


# -----------------------------
# IDA-BD unlabeled target dataset
# -----------------------------

class IDABDUnlabeledTargetDataset(Dataset):
    """IDA-BD target train dataset for UDA.

    It returns pre/post images only.  The post mask is loaded only to keep the
    sample discovery/split consistent, but target labels are NOT returned and
    NOT used for training.  Random crops are image-only and label-free.
    """

    def __init__(
        self,
        root: str | Path,
        samples_by_stem: Dict[str, idft.IDABDSample],
        stems: List[str],
        image_size: int,
        crop_size: int = 608,
        extra_photometric: bool = True,
    ):
        self.root = Path(root)
        self.samples_by_stem = samples_by_stem
        self.stems = list(stems)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.extra_photometric = bool(extra_photometric)

    def __len__(self) -> int:
        return len(self.stems)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return (x - idft.IMAGENET_MEAN) / idft.IMAGENET_STD

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        stem = self.stems[index]
        sample = self.samples_by_stem[stem]

        pre = idft.read_rgb(sample.pre_image_path)
        post = idft.read_rgb(sample.post_image_path)

        # Dummy mask lets us reuse the same geometric augmentation utility.
        dummy = np.zeros(pre.shape[:2], dtype=np.uint8)
        [pre, post], [dummy] = idft.resize_img_mask_pair(
            image_list=[pre, post],
            mask_list=[dummy],
            image_size=self.image_size,
        )
        [pre, post], [dummy] = v5.apply_shared_augmentations(
            image_list=[pre, post],
            mask_list=[dummy],
            training=True,
            image_size=self.image_size,
        )

        if self.extra_photometric:
            [pre, post] = v5.apply_extra_photometric_augmentations([pre, post], training=True)

        # Label-free random crop.  This avoids using IDA-BD target masks for
        # rare-crop selection, preserving unsupervised target training.
        if self.crop_size > 0 and self.crop_size < self.image_size:
            h, w = pre.shape[:2]
            cs = int(self.crop_size)
            if h >= cs and w >= cs:
                y0 = random.randint(0, h - cs)
                x0 = random.randint(0, w - cs)
                pre = pre[y0:y0 + cs, x0:x0 + cs]
                post = post[y0:y0 + cs, x0:x0 + cs]

        return {
            "pre": torch.from_numpy(self._normalize(pre)).float(),
            "post": torch.from_numpy(self._normalize(post)).float(),
            "stem": stem,
            "split": "idabd_target_unlabeled",
        }


# -----------------------------
# Domain adversarial pieces
# -----------------------------

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def phase2_forward_with_features(
    model: nn.Module,
    pre: torch.Tensor,
    post: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Forward HRTBDA Phase II and expose fused multi-scale features."""
    m = unwrap_model(model)
    fpre = m.backbone(pre)
    fpost = m.backbone(post)
    fused = [module(a, b) for module, a, b in zip(m.csf, fpre, fpost)]
    damage_logits = m.decoder(fused, output_size=pre.shape[-2:])
    aux_loc = F.interpolate(
        m.aux_loc_head(fpre[0]),
        size=pre.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return damage_logits, aux_loc, fused


def foreground_pool_multiscale_features(
    fused: List[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average-pool each feature scale over foreground mask and concatenate.

    mask: [B,H,W] or [B,1,H,W].  If a sample has no foreground pixels after
    resizing to a scale, fall back to global average pooling for that sample.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()

    pooled_vectors: List[torch.Tensor] = []
    for feat in fused:
        b, c, h, w = feat.shape
        m = F.interpolate(mask, size=(h, w), mode="nearest")
        denom = m.sum(dim=(2, 3)).clamp_min(1.0)  # [B,1]
        masked_pool = (feat * m).sum(dim=(2, 3)) / denom  # [B,C]
        global_pool = feat.mean(dim=(2, 3))

        has_fg = (m.sum(dim=(2, 3)) > 0).float()  # [B,1]
        pooled = masked_pool * has_fg + global_pool * (1.0 - has_fg)
        pooled_vectors.append(pooled)

    return torch.cat(pooled_vectors, dim=1)


def target_entropy_loss(damage_logits: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    """Optional entropy minimization on target foreground pixels."""
    if foreground_mask.dim() == 3:
        foreground_mask = foreground_mask.unsqueeze(1)
    valid = foreground_mask.float()
    denom = valid.sum().clamp_min(1.0)
    p = torch.sigmoid(damage_logits).clamp(1e-6, 1.0 - 1e-6)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return (ent * valid).sum() / (denom * damage_logits.shape[1])


# -----------------------------
# Loaders
# -----------------------------

def make_source_xbd_loader(args: argparse.Namespace) -> Tuple[DataLoader, v5.XBDHRTBDADataset]:
    crop_weights = (
        float(args.crop_weight_no_damage),
        float(args.crop_weight_minor),
        float(args.crop_weight_major),
        float(args.crop_weight_destroyed),
    )
    ds = v5.XBDHRTBDADataset(
        args.xbd_root,
        args.source_train_split,
        args.img_size,
        training=True,
        crop_size=args.phase2_crop_size,
        crop_candidate_count=args.crop_candidate_count,
        crop_class_weights=crop_weights,
        extra_photometric=args.extra_photometric_aug,
    )
    loader = DataLoader(
        ds,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, ds


def make_target_idabd_loader(args: argparse.Namespace) -> Tuple[DataLoader, Dict[str, List[str]]]:
    samples = idft.collect_idabd_samples(args.idabd_root)
    sample_map = {s.stem: s for s in samples}
    splits = idft.prepare_or_load_splits(args, samples)
    ds = IDABDUnlabeledTargetDataset(
        root=args.idabd_root,
        samples_by_stem=sample_map,
        stems=splits["train"],
        image_size=args.img_size,
        crop_size=args.phase2_crop_size,
        extra_photometric=args.extra_photometric_aug,
    )
    loader = DataLoader(
        ds,
        batch_size=args.target_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, splits


def make_idabd_eval_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    samples = idft.collect_idabd_samples(args.idabd_root)
    sample_map = {s.stem: s for s in samples}
    splits = idft.prepare_or_load_splits(args, samples)
    val_ds = idft.IDABDHRTBDADataset(args.idabd_root, sample_map, splits["val"], args.img_size, training=False)
    test_ds = idft.IDABDHRTBDADataset(args.idabd_root, sample_map, splits["test"], args.img_size, training=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return val_loader, test_loader


# -----------------------------
# Checkpoints/model init
# -----------------------------

def load_phase1(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, float, Dict[str, object]]:
    ckpt_path = Path(args.phase1_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase-I checkpoint not found: {ckpt_path}")
    model = v5.HRTBDAPhase1(args.base_channels, args.decoder_channels, args.window_size).to(device)
    meta = v5.load_model_weights(model, ckpt_path, device)
    threshold = float(meta.get("best_threshold", args.phase1_threshold))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("Loaded frozen Phase-I model for target foreground masks and final cascade.", flush=True)
    print(f"Phase-I checkpoint: {ckpt_path}", flush=True)
    print(f"Phase-I epoch: {meta.get('epoch')} | best_metric={meta.get('best_metric')} | threshold={threshold:.2f}", flush=True)
    return model, threshold, meta


def init_phase2(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    ckpt_path = Path(args.phase2_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase-II checkpoint not found: {ckpt_path}")
    model = v5.HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size, num_classes=4).to(device)
    meta = v5.load_model_weights(model, ckpt_path, device)
    print("Initialized Phase-II from xBD v5 checkpoint.", flush=True)
    print(f"Phase-II checkpoint: {ckpt_path}", flush=True)
    print(f"Phase-II source epoch: {meta.get('epoch')} | best_metric={meta.get('best_metric')}", flush=True)
    return model, meta


def save_da_checkpoint(
    path: Path,
    model: nn.Module,
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "epoch": int(epoch),
        "model": unwrap_model(model).state_dict(),
        "domain_discriminator": discriminator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": float(best_score),
        "args": vars(args),
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


# -----------------------------
# Training/eval
# -----------------------------

def make_scaler(args: argparse.Namespace, device: torch.device):
    if USE_TORCH_AMP:
        return GradScaler(device=device.type, enabled=args.amp and device.type == "cuda")
    return GradScaler(enabled=args.amp and device.type == "cuda")


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(epoch_idx: int):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        denom = max(1, epochs - warmup_epochs)
        progress = min(1.0, float(epoch_idx - warmup_epochs) / float(denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_domain_adaptation(args: argparse.Namespace, device: torch.device) -> Tuple[Path, Path]:
    print("\n================ HRTBDA V5 FOREGROUND-AWARE DANN ADAPTATION ================", flush=True)

    source_loader, source_ds = make_source_xbd_loader(args)
    target_loader, _ = make_target_idabd_loader(args)
    val_loader, _ = make_idabd_eval_loaders(args)

    print(f"Source xBD train samples: {len(source_loader.dataset)} | batch={args.source_batch_size}", flush=True)
    print(f"Target IDA-BD train samples: {len(target_loader.dataset)} | batch={args.target_batch_size}", flush=True)
    print(f"Steps per epoch: {args.steps_per_epoch}", flush=True)
    print(f"Phase-II crop size: {args.phase2_crop_size}", flush=True)
    print(f"Source rare-crop candidates: {args.crop_candidate_count}", flush=True)
    print(f"Crop weights [no,minor,major,destroyed]: [{args.crop_weight_no_damage}, {args.crop_weight_minor}, {args.crop_weight_major}, {args.crop_weight_destroyed}]", flush=True)

    phase1_model, phase1_threshold, phase1_meta = load_phase1(args, device)
    phase2_model, phase2_meta = init_phase2(args, device)

    # Domain feature dimension is the sum of HRT backbone channels.
    feature_dim = int(sum(unwrap_model(phase2_model).backbone.channels))
    domain_disc = DomainDiscriminator(feature_dim, hidden_dim=args.domain_hidden_dim, dropout=args.domain_dropout).to(device)

    class_weights = v5.make_damage4_class_weights(source_ds, args=args).to(device)
    criterion_damage = v5.MultilabelDamageFocalDiceLoss(class_weights=class_weights, gamma=args.focal_gamma).to(device)
    loc_pos_weight = v5.make_loc_pos_weight(source_ds).to(device)
    criterion_aux = v5.BinaryFocalDiceLoss(pos_weight=loc_pos_weight, gamma=args.focal_gamma).to(device)
    criterion_domain = nn.BCEWithLogitsLoss().to(device)

    params = list(phase2_model.parameters()) + list(domain_disc.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = make_scheduler(optimizer, args.epochs, warmup_epochs=args.warmup_epochs)
    scaler = make_scaler(args, device)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    scores_dir = output_dir / "scores"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    with open(scores_dir / "adaptation_setup.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": "xBD train+tier3 labeled",
            "target": "IDA-BD train unlabeled",
            "phase1_checkpoint": str(args.phase1_checkpoint),
            "phase1_threshold_loaded": phase1_threshold,
            "phase2_checkpoint_init": str(args.phase2_checkpoint),
            "phase1_meta": phase1_meta,
            "phase2_meta": phase2_meta,
            "args": vars(args),
        }, f, indent=2, default=json_safe_default)

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    history: List[Dict[str, object]] = []
    accumulation_steps = max(1, int(args.grad_accum_steps))

    source_iter = cycle_loader(source_loader)
    target_iter = cycle_loader(target_loader)

    for epoch in range(1, args.epochs + 1):
        phase2_model.train()
        domain_disc.train()
        total_meter = AverageMeter()
        sup_meter = AverageMeter()
        domain_meter = AverageMeter()
        entropy_meter = AverageMeter()
        focal_meter = AverageMeter()
        dice_meter = AverageMeter()
        aux_meter = AverageMeter()
        dom_acc_meter = AverageMeter()

        # Warm up GRL/domain strength to avoid destroying xBD representation early.
        lambda_grl = float(args.lambda_domain) * min(1.0, epoch / float(max(1, args.domain_warmup_epochs)))
        lambda_entropy = float(args.lambda_entropy)

        print(f"\nDA epoch {epoch}/{args.epochs} | LR={optimizer.param_groups[0]['lr']:.8f} | lambda_domain={lambda_grl:.5f} | lambda_entropy={lambda_entropy:.5f}", flush=True)
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, args.steps_per_epoch + 1):
            sb = next(source_iter)
            tb = next(target_iter)

            spre = sb["pre"].to(device, non_blocking=True)
            spost = sb["post"].to(device, non_blocking=True)
            starget5 = sb["target5"].to(device, non_blocking=True)
            sloc = sb["loc"].to(device, non_blocking=True)

            tpre = tb["pre"].to(device, non_blocking=True)
            tpost = tb["post"].to(device, non_blocking=True)

            if USE_TORCH_AMP:
                amp_ctx = autocast(device_type=device.type, enabled=args.amp and device.type == "cuda")
            else:  # pragma: no cover
                amp_ctx = autocast(enabled=args.amp and device.type == "cuda")

            with amp_ctx:
                # Source supervised loss.
                s_damage_logits, s_aux_loc, s_fused = phase2_forward_with_features(phase2_model, spre, spost)
                s_damage_target, s_valid_mask = v5.target5_to_multilabel_damage4(starget5)
                loss_damage, focal, dice = criterion_damage(s_damage_logits, s_damage_target, s_valid_mask)
                loss_aux, _, _ = criterion_aux(s_aux_loc, sloc)
                supervised_loss = args.cls_loss_weight * loss_damage + args.aux_loc_weight * loss_aux

                # Target features and predicted foreground mask from frozen Phase I.
                with torch.no_grad():
                    t_phase1_logits = phase1_model(tpre)
                    t_loc_pred = (torch.sigmoid(t_phase1_logits) > phase1_threshold).float()

                t_damage_logits, _, t_fused = phase2_forward_with_features(phase2_model, tpre, tpost)

                # Foreground-aware domain features.
                s_domain_feat = foreground_pool_multiscale_features(s_fused, sloc)
                t_domain_feat = foreground_pool_multiscale_features(t_fused, t_loc_pred)
                domain_feat = torch.cat([s_domain_feat, t_domain_feat], dim=0)
                domain_labels = torch.cat([
                    torch.ones(s_domain_feat.shape[0], device=device),
                    torch.zeros(t_domain_feat.shape[0], device=device),
                ], dim=0)

                domain_logits = domain_disc(grad_reverse(domain_feat, lambda_grl))
                domain_loss = criterion_domain(domain_logits, domain_labels)

                with torch.no_grad():
                    dom_pred = (torch.sigmoid(domain_logits) > 0.5).float()
                    dom_acc = (dom_pred == domain_labels).float().mean()

                if lambda_entropy > 0:
                    ent_loss = target_entropy_loss(t_damage_logits, t_loc_pred)
                else:
                    ent_loss = t_damage_logits.sum() * 0.0

                total_loss = supervised_loss + args.lambda_domain_loss * domain_loss + lambda_entropy * ent_loss

            if not torch.isfinite(total_loss):
                print(f"WARNING: non-finite loss at epoch={epoch} step={step}; skipping step.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = total_loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % accumulation_steps == 0 or step == args.steps_per_epoch:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = spre.size(0)
            total_meter.update(safe_item(total_loss), bs)
            sup_meter.update(safe_item(supervised_loss), bs)
            domain_meter.update(safe_item(domain_loss), bs)
            entropy_meter.update(safe_item(ent_loss), bs)
            focal_meter.update(safe_item(focal), bs)
            dice_meter.update(safe_item(dice), bs)
            aux_meter.update(safe_item(loss_aux), bs)
            dom_acc_meter.update(safe_item(dom_acc), bs)

            if step % args.log_every == 0 or step == args.steps_per_epoch:
                print(
                    f"DA Epoch {epoch}/{args.epochs} | Step {step}/{args.steps_per_epoch} | "
                    f"loss={total_meter.avg:.4f} sup={sup_meter.avg:.4f} dom={domain_meter.avg:.4f} "
                    f"ent={entropy_meter.avg:.4f} focal={focal_meter.avg:.4f} dice={dice_meter.avg:.4f} "
                    f"aux={aux_meter.avg:.4f} dom_acc={dom_acc_meter.avg:.3f}",
                    flush=True,
                )

        scheduler.step()

        # Validate on IDA-BD validation labels only for model selection.
        val_results = v5.evaluate_phase2_cascade(
            phase1_model=phase1_model,
            phase2_model=phase2_model,
            loader=val_loader,
            device=device,
            phase1_threshold=phase1_threshold,
            postprocess_dilation=args.selection_dilation,
            dilation_kernel=args.dilation_kernel,
        )
        val_score = float(val_results["score"])
        row = {
            "epoch": epoch,
            "train_loss": total_meter.avg,
            "supervised_loss": sup_meter.avg,
            "domain_loss": domain_meter.avg,
            "entropy_loss": entropy_meter.avg,
            "domain_acc": dom_acc_meter.avg,
            "lambda_domain_grl": lambda_grl,
            "val_score": val_score,
            "val_loc_f1": val_results["localization_f1"],
            "val_damage_f1": val_results["damage_f1"],
            "val_no": val_results["damage_f1_no_damage"],
            "val_minor": val_results["damage_f1_minor_damage"],
            "val_major": val_results["damage_f1_major_damage"],
            "val_destroyed": val_results["damage_f1_destroyed"],
        }
        history.append(row)
        print(
            f"DA Epoch {epoch:03d} | train_loss={total_meter.avg:.4f} | dom_acc={dom_acc_meter.avg:.3f} | "
            f"val_score={val_score:.6f} | val_loc={val_results['localization_f1']:.6f} | "
            f"val_damage={val_results['damage_f1']:.6f} | no={val_results['damage_f1_no_damage']:.6f} | "
            f"minor={val_results['damage_f1_minor_damage']:.6f} | major={val_results['damage_f1_major_damage']:.6f} | "
            f"destroyed={val_results['damage_f1_destroyed']:.6f}",
            flush=True,
        )

        with open(scores_dir / "domain_adaptation_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_da_checkpoint(
                ckpt_dir / "phase2_best_da.pt",
                phase2_model,
                domain_disc,
                optimizer,
                epoch,
                best_score,
                args,
                extra={
                    "phase1_checkpoint": str(args.phase1_checkpoint),
                    "phase1_threshold": float(phase1_threshold),
                    "source_phase2_init": str(args.phase2_checkpoint),
                    "val_results": val_results,
                },
            )
            print(f"Saved DA best checkpoint | epoch={epoch} | val_score={best_score:.6f}", flush=True)
        else:
            no_improve += 1
            print(f"DA no improvement for {no_improve} epoch(s). Best epoch={best_epoch}", flush=True)

        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_da_checkpoint(
                ckpt_dir / f"phase2_da_epoch_{epoch:03d}.pt",
                phase2_model,
                domain_disc,
                optimizer,
                epoch,
                best_score,
                args,
            )

        if no_improve >= args.early_stopping_patience:
            print(f"DA early stopping at epoch {epoch}.", flush=True)
            break

    print(f"DA done. Best epoch={best_epoch}, best val score={best_score:.6f}", flush=True)
    return ckpt_dir / "phase2_best_da.pt", Path(args.phase1_checkpoint)


def load_da_phase2_for_eval(args: argparse.Namespace, device: torch.device, ckpt_path: Path) -> nn.Module:
    model = v5.HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size, num_classes=4).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def validation_ablation_and_final_test(
    args: argparse.Namespace,
    device: torch.device,
    phase1_ckpt: Path,
    phase2_ckpt: Path,
) -> None:
    print("\n================ IDA-BD VALIDATION ABLATION + FINAL TEST ================", flush=True)
    val_loader, test_loader = make_idabd_eval_loaders(args)
    phase1_model, phase1_threshold, phase1_meta = load_phase1(args, device)
    phase2_model = load_da_phase2_for_eval(args, device, phase2_ckpt)

    scores_dir = Path(args.output_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for th in args.thresholds:
        for dil in args.dilation_options:
            res = v5.evaluate_phase2_cascade(
                phase1_model=phase1_model,
                phase2_model=phase2_model,
                loader=val_loader,
                device=device,
                phase1_threshold=float(th),
                postprocess_dilation=str(dil),
                dilation_kernel=args.dilation_kernel,
            )
            row = {
                "threshold": float(th),
                "dilation": str(dil),
                "localization_f1": float(res["localization_f1"]),
                "no_damage_f1": float(res["damage_f1_no_damage"]),
                "minor_damage_f1": float(res["damage_f1_minor_damage"]),
                "major_damage_f1": float(res["damage_f1_major_damage"]),
                "destroyed_f1": float(res["damage_f1_destroyed"]),
                "damage_f1": float(res["damage_f1"]),
                "overall_score": float(res["score"]),
            }
            rows.append(row)
            print(
                f"VAL ABLATION | th={th:.2f} dil={dil:10s} | loc={row['localization_f1']:.6f} | "
                f"no={row['no_damage_f1']:.6f} minor={row['minor_damage_f1']:.6f} "
                f"major={row['major_damage_f1']:.6f} destroyed={row['destroyed_f1']:.6f} | "
                f"damage={row['damage_f1']:.6f} overall={row['overall_score']:.6f}",
                flush=True,
            )

    rows_sorted = sorted(rows, key=lambda r: r["overall_score"], reverse=True)
    best = rows_sorted[0]

    with open(scores_dir / "validation_threshold_dilation_ablation_da.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    with open(scores_dir / "validation_threshold_dilation_ablation_da.json", "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2, default=json_safe_default)

    print("\n===== BEST VALIDATION SETTING =====", flush=True)
    print(json.dumps(best, indent=2, default=json_safe_default), flush=True)
    print("===================================", flush=True)

    test_res = v5.evaluate_phase2_cascade(
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=test_loader,
        device=device,
        phase1_threshold=float(best["threshold"]),
        postprocess_dilation=str(best["dilation"]),
        dilation_kernel=args.dilation_kernel,
    )

    summary = {
        "experiment": "xBD-to-IDA-BD unsupervised foreground-aware adversarial domain adaptation HRTBDA v5",
        "source_labeled": "xBD train+tier3",
        "target_unlabeled": "IDA-BD train split",
        "phase1_checkpoint": str(phase1_ckpt),
        "phase1_checkpoint_epoch": phase1_meta.get("epoch"),
        "phase1_checkpoint_best_metric": phase1_meta.get("best_metric"),
        "phase2_checkpoint": str(phase2_ckpt),
        "selected_threshold_from_val": float(best["threshold"]),
        "selected_dilation_from_val": str(best["dilation"]),
        "validation_selected_setting": best,
        "test": {
            "localization_f1": float(test_res["localization_f1"]),
            "no_damage_f1": float(test_res["damage_f1_no_damage"]),
            "minor_damage_f1": float(test_res["damage_f1_minor_damage"]),
            "major_damage_f1": float(test_res["damage_f1_major_damage"]),
            "destroyed_f1": float(test_res["damage_f1_destroyed"]),
            "damage_f1": float(test_res["damage_f1"]),
            "overall_score": float(test_res["score"]),
        },
        "args": vars(args),
    }

    with open(scores_dir / "summary_final_test_selected_by_validation_da.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_safe_default)

    lines = [
        "Experiment: xBD-to-IDA-BD unsupervised foreground-aware adversarial domain adaptation HRTBDA v5",
        f"Source labeled data: xBD {args.source_train_split}",
        "Target unlabeled data: IDA-BD train split",
        f"Phase I checkpoint: {phase1_ckpt}",
        f"Phase I stored epoch: {phase1_meta.get('epoch')}",
        f"Phase I stored best metric: {phase1_meta.get('best_metric')}",
        f"Phase II DA checkpoint: {phase2_ckpt}",
        f"Selected Phase-I threshold from val ablation: {best['threshold']:.2f}",
        f"Selected dilation from val ablation: {best['dilation']}",
        "",
        "Validation-selected setting:",
        f"Val Localization F1: {best['localization_f1']:.6f}",
        f"Val No Damage F1:    {best['no_damage_f1']:.6f}",
        f"Val Minor Damage F1: {best['minor_damage_f1']:.6f}",
        f"Val Major Damage F1: {best['major_damage_f1']:.6f}",
        f"Val Destroyed F1:    {best['destroyed_f1']:.6f}",
        f"Val Damage F1:       {best['damage_f1']:.6f}",
        f"Val Overall Score:   {best['overall_score']:.6f}",
        "",
        "Final test result:",
        f"Test Localization F1 from Phase I mask: {test_res['localization_f1']:.6f}",
        f"No Damage F1:    {test_res['damage_f1_no_damage']:.6f}",
        f"Minor Damage F1: {test_res['damage_f1_minor_damage']:.6f}",
        f"Major Damage F1: {test_res['damage_f1_major_damage']:.6f}",
        f"Destroyed F1:    {test_res['damage_f1_destroyed']:.6f}",
        f"Damage F1:       {test_res['damage_f1']:.6f}",
        f"Overall Score:   {test_res['score']:.6f}",
    ]
    out_txt = scores_dir / "summary_idabd_hrtbda_v5_foreground_dann_selected_test.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote: {out_txt}", flush=True)


# -----------------------------
# Args/main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("xBD -> IDA-BD foreground-aware adversarial domain adaptation for HRTBDA v5")

    p.add_argument("--phase", type=str, default="adapt", choices=["adapt", "test"])

    # Source and target data.
    p.add_argument("--xbd-root", type=str, required=True)
    p.add_argument("--source-train-split", type=str, nargs="+", default=["train", "tier3"])
    p.add_argument("--idabd-root", type=str, required=True)
    p.add_argument("--split-file", type=str, default="")
    p.add_argument("--force-resplit", action="store_true")
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)

    # Checkpoints.
    p.add_argument("--phase1-checkpoint", type=str, required=True, help="Frozen xBD Phase-I checkpoint used for target foreground and final cascade.")
    p.add_argument("--phase2-checkpoint", type=str, required=True, help="xBD v5 Phase-II checkpoint used for initialization, or DA checkpoint for --phase test.")
    p.add_argument("--phase1-threshold", type=float, default=0.5)

    # Output.
    p.add_argument("--output-dir", type=str, required=True)

    # Model.
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)

    # Training.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--source-batch-size", type=int, default=1)
    p.add_argument("--target-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--phase2-crop-size", type=int, default=608)
    p.add_argument("--crop-candidate-count", type=int, default=8)
    p.add_argument("--extra-photometric-aug", action="store_true")

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log-every", type=int, default=50)

    # Supervised damage loss.
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--cls-loss-weight", type=float, default=1.0)
    p.add_argument("--aux-loc-weight", type=float, default=0.2)
    p.add_argument("--minor-damage-boost", type=float, default=1.5)
    p.add_argument("--major-damage-boost", type=float, default=1.5)
    p.add_argument("--destroyed-damage-boost", type=float, default=1.0)
    p.add_argument("--max-damage-class-weight", type=float, default=10.0)

    # Rare-crop weights for labeled xBD source only.
    p.add_argument("--crop-weight-no-damage", type=float, default=1.0)
    p.add_argument("--crop-weight-minor", type=float, default=12.0)
    p.add_argument("--crop-weight-major", type=float, default=12.0)
    p.add_argument("--crop-weight-destroyed", type=float, default=4.0)

    # Domain adaptation.
    p.add_argument("--lambda-domain", type=float, default=0.03, help="Gradient reversal strength after warmup.")
    p.add_argument("--lambda-domain-loss", type=float, default=0.1, help="Multiplier for BCE domain loss term.")
    p.add_argument("--domain-warmup-epochs", type=int, default=5)
    p.add_argument("--lambda-entropy", type=float, default=0.0)
    p.add_argument("--domain-hidden-dim", type=int, default=256)
    p.add_argument("--domain-dropout", type=float, default=0.2)

    # Validation/test selection.
    p.add_argument("--selection-dilation", type=str, default="none")
    p.add_argument("--dilation-kernel", type=int, default=3)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--dilation-options", type=str, nargs="+", default=["none", "minor"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "scores").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== xBD -> IDA-BD FOREGROUND-AWARE ADVERSARIAL DOMAIN ADAPTATION =====", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"xBD source root: {args.xbd_root}", flush=True)
    print(f"xBD source split: {args.source_train_split}", flush=True)
    print(f"IDA-BD target root: {args.idabd_root}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Frozen Phase-I checkpoint: {args.phase1_checkpoint}", flush=True)
    print(f"Phase-II init/checkpoint: {args.phase2_checkpoint}", flush=True)
    print("Training policy: source xBD labels + unlabeled IDA-BD train images; IDA-BD val/test labels only for evaluation.", flush=True)
    print("Architecture: HRTBDA v5 4-branch HRNet-style + DCSwin + CSF + foreground DANN discriminator", flush=True)
    print("=======================================================================", flush=True)

    if args.phase == "adapt":
        phase2_ckpt, phase1_ckpt = train_domain_adaptation(args, device)
        validation_ablation_and_final_test(args, device, phase1_ckpt, phase2_ckpt)
    elif args.phase == "test":
        validation_ablation_and_final_test(args, device, Path(args.phase1_checkpoint), Path(args.phase2_checkpoint))

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
