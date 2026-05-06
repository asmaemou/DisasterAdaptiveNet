#!/usr/bin/env python3
"""
Foreground-aware self-training UDA for HRTBDA v7-MSDF.

Source domain: xBD train+tier3 with labels.
Target domain: IDA-BD train images without labels.
Final eval: IDA-BD test labels are used only for reporting/debug.

Main idea:
  - Start from xBD-trained v7-MSDF Phase I and Phase II.
  - Keep Phase I frozen for target foreground/building masks.
  - Use a Mean-Teacher Phase II model to generate confident target pseudo-labels.
  - Train a student Phase II model using:
      source supervised Focal+Dice+aux loss
      + target pseudo-label consistency loss on confident foreground pixels
      + optional source-to-target style matching for source images.
  - Update teacher with EMA of the student.

This is intended as a stronger UDA baseline than plain DANN for xBD -> IDA-BD.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse flexible IDA-BD loader/evaluator from the direct transfer script.
_THIS_DIR = Path(__file__).resolve().parent
_DIRECT_SCRIPT = _THIS_DIR / "eval_idabd_v7_msdf_direct_transfer.py"
if not _DIRECT_SCRIPT.exists():
    _DIRECT_SCRIPT = Path("transformer/scripts/eval_idabd_v7_msdf_direct_transfer.py")

_spec = importlib.util.spec_from_file_location("idabd_direct", str(_DIRECT_SCRIPT))
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import direct-transfer helper script: {_DIRECT_SCRIPT}")
idabd_direct = importlib.util.module_from_spec(_spec)
sys.modules["idabd_direct"] = idabd_direct
_spec.loader.exec_module(idabd_direct)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def make_v7_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        phase="test",
        resume_phase1=None,
        phase1_checkpoint=str(args.phase1_checkpoint),
        phase2_checkpoint=str(args.phase2_checkpoint),
        xbd_root=str(args.xbd_root),
        train_split=["train", "tier3"],
        val_split="hold",
        test_split="test",
        output_dir=str(args.output_dir),
        phase1_epochs=150,
        phase2_epochs=args.epochs,
        phase1_batch_size=1,
        phase2_batch_size=args.source_batch_size,
        batch_size=args.source_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=1,
        num_workers=args.num_workers,
        img_size=args.img_size,
        phase2_crop_size=args.phase2_crop_size,
        crop_candidate_count=args.crop_candidate_count,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=0,
        save_every=1,
        early_stopping_patience=999,
        focal_gamma=args.focal_gamma,
        loc_loss_weight=1.0,
        cls_loss_weight=1.0,
        aux_loc_weight=args.aux_loc_weight,
        minor_damage_boost=1.5,
        major_damage_boost=1.5,
        max_damage_class_weight=10.0,
        crop_weight_no_damage=1.0,
        crop_weight_minor=12.0,
        crop_weight_major=12.0,
        crop_weight_destroyed=4.0,
        finetune_epochs=0,
        finetune_lr=5e-5,
        postprocess_dilation=args.postprocess_dilation,
        dilation_kernel=args.dilation_kernel,
        phase1_threshold=args.phase1_threshold,
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        amp=args.amp,
        extra_photometric_aug=True,
    )


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def phase1_forward_logits(phase1_model: torch.nn.Module, pre: torch.Tensor) -> torch.Tensor:
    return idabd_direct.phase1_forward_logits(phase1_model, pre)


def get_damage_logits(v7, out):
    return idabd_direct.get_damage_logits(v7, out)


def damage_logits_to_pred(v7, logits: torch.Tensor) -> torch.Tensor:
    return idabd_direct.damage_logits_to_pred(v7, logits)


def make_multilabel_targets(target5: torch.Tensor) -> torch.Tensor:
    chans = [(target5 == c).float() for c in [1, 2, 3, 4]]
    return torch.stack(chans, dim=1)


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float, pos_weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight.view(1, -1, 1, 1),
    )
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    focal = ((1.0 - pt).clamp_min(1e-6) ** gamma) * bce
    return focal.mean()


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = (probs * probs).sum(dim=dims) + (targets * targets).sum(dim=dims)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def source_supervised_loss(v7, phase2_out, target5: torch.Tensor, focal_gamma: float, pos_weight: torch.Tensor, aux_loc_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    damage_logits = get_damage_logits(v7, phase2_out)
    target_ml = make_multilabel_targets(target5)

    focal = focal_bce_with_logits(damage_logits, target_ml, gamma=focal_gamma, pos_weight=pos_weight)
    dice = dice_loss_with_logits(damage_logits, target_ml)

    aux = torch.tensor(0.0, device=damage_logits.device)
    if isinstance(phase2_out, (tuple, list)) and len(phase2_out) >= 2:
        aux_logits = phase2_out[1]
        if aux_logits.ndim == 4 and aux_logits.shape[1] == 1:
            aux_logits = aux_logits[:, 0]
        loc_target = (target5 > 0).float()
        aux = F.binary_cross_entropy_with_logits(aux_logits, loc_target)
    elif isinstance(phase2_out, dict):
        aux_logits = phase2_out.get("aux_loc", None)
        if aux_logits is None:
            aux_logits = phase2_out.get("aux_loc_logits", None)
        if aux_logits is not None:
            if aux_logits.ndim == 4 and aux_logits.shape[1] == 1:
                aux_logits = aux_logits[:, 0]
            loc_target = (target5 > 0).float()
            aux = F.binary_cross_entropy_with_logits(aux_logits, loc_target)

    loss = focal + dice + aux_loc_weight * aux
    return loss, {"focal": float(focal.detach().cpu()), "dice": float(dice.detach().cpu()), "aux": float(aux.detach().cpu())}


def masked_focal_bce_with_logits(logits: torch.Tensor, pseudo_targets: torch.Tensor, valid_mask: torch.Tensor, gamma: float, pos_weight: torch.Tensor) -> torch.Tensor:
    """Focal BCE only on confident target pixels. valid_mask is [B,H,W]."""
    if valid_mask.ndim == 3:
        valid = valid_mask.unsqueeze(1).float()
    else:
        valid = valid_mask.float()
    valid = valid.expand_as(logits)
    if valid.sum() < 1:
        return logits.sum() * 0.0

    bce = F.binary_cross_entropy_with_logits(
        logits,
        pseudo_targets,
        reduction="none",
        pos_weight=pos_weight.view(1, -1, 1, 1),
    )
    p = torch.sigmoid(logits)
    pt = torch.where(pseudo_targets > 0.5, p, 1.0 - p)
    focal = ((1.0 - pt).clamp_min(1e-6) ** gamma) * bce
    return (focal * valid).sum() / valid.sum().clamp_min(1.0)


def masked_dice_loss_with_logits(logits: torch.Tensor, pseudo_targets: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if valid_mask.ndim == 3:
        valid = valid_mask.unsqueeze(1).float()
    else:
        valid = valid_mask.float()
    valid = valid.expand_as(logits)
    if valid.sum() < 1:
        return logits.sum() * 0.0

    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * pseudo_targets * valid).sum(dim=dims)
    denom = ((probs * probs + pseudo_targets * pseudo_targets) * valid).sum(dim=dims)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)

    # Average only classes that have at least one confident positive pixel.
    class_has_pos = ((pseudo_targets * valid).sum(dim=dims) > 0).float()
    if class_has_pos.sum() > 0:
        return (dice * class_has_pos).sum() / class_has_pos.sum()
    return dice.mean()


def get_aux_logits(phase2_out) -> Optional[torch.Tensor]:
    if isinstance(phase2_out, (tuple, list)) and len(phase2_out) >= 2:
        aux = phase2_out[1]
    elif isinstance(phase2_out, dict):
        aux = phase2_out.get("aux_loc", None)
        if aux is None:
            aux = phase2_out.get("aux_loc_logits", None)
    else:
        aux = None
    if aux is not None and aux.ndim == 4 and aux.shape[1] == 1:
        aux = aux[:, 0]
    return aux


def target_pseudo_loss(
    v7,
    student_out,
    teacher_damage_logits: torch.Tensor,
    target_loc_mask: torch.Tensor,
    thresholds: torch.Tensor,
    focal_gamma: float,
    pos_weight: torch.Tensor,
    target_aux_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pseudo-label loss on confident target foreground pixels."""
    student_logits = get_damage_logits(v7, student_out)

    with torch.no_grad():
        probs = torch.sigmoid(teacher_damage_logits)
        conf, cls0 = probs.max(dim=1)  # cls0 0..3, actual class 1..4
        cls_id = cls0 + 1
        thr_map = thresholds[(cls_id - 1).clamp(0, 3)]
        valid = (target_loc_mask > 0.5) & (conf >= thr_map)
        pseudo_targets = torch.zeros_like(probs)
        pseudo_targets.scatter_(1, cls0.unsqueeze(1), 1.0)

    focal = masked_focal_bce_with_logits(student_logits, pseudo_targets, valid, gamma=focal_gamma, pos_weight=pos_weight)
    dice = masked_dice_loss_with_logits(student_logits, pseudo_targets, valid)

    aux = torch.tensor(0.0, device=student_logits.device)
    aux_logits = get_aux_logits(student_out)
    if aux_logits is not None:
        aux = F.binary_cross_entropy_with_logits(aux_logits, target_loc_mask.float())

    loss = focal + dice + target_aux_weight * aux

    valid_count = float(valid.sum().detach().cpu())
    loc_count = float((target_loc_mask > 0.5).sum().detach().cpu())
    coverage = valid_count / max(loc_count, 1.0)
    per_class = {}
    for c in [1, 2, 3, 4]:
        per_class[f"pseudo_c{c}"] = float(((cls_id == c) & valid).sum().detach().cpu())

    stats = {
        "target_focal": float(focal.detach().cpu()),
        "target_dice": float(dice.detach().cpu()),
        "target_aux": float(aux.detach().cpu()),
        "pseudo_pixels": valid_count,
        "pseudo_coverage": coverage,
        **per_class,
    }
    return loss, stats


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x.clamp(0.0, 1.0) - mean) / std


def strong_photometric_aug(x: torch.Tensor, noise_std: float = 0.03, drop_prob: float = 0.10) -> torch.Tensor:
    """Differentiable-ish photometric strong augmentation without spatial transforms."""
    y = unnormalize(x)
    b = y.shape[0]
    brightness = torch.empty(b, 1, 1, 1, device=y.device, dtype=y.dtype).uniform_(0.75, 1.25)
    contrast = torch.empty(b, 1, 1, 1, device=y.device, dtype=y.dtype).uniform_(0.75, 1.25)
    mean = y.mean(dim=(2, 3), keepdim=True)
    y = (y - mean) * contrast + mean
    y = y * brightness
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    if drop_prob > 0 and random.random() < drop_prob:
        ch = random.randrange(3)
        y[:, ch:ch + 1] = y[:, ch:ch + 1] * torch.empty(b, 1, 1, 1, device=y.device, dtype=y.dtype).uniform_(0.2, 0.6)
    return normalize(y)


def match_source_to_target_style(src: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Simple per-image channel mean/std matching in RGB space."""
    s = unnormalize(src)
    t = unnormalize(tgt)
    if t.shape[0] != s.shape[0]:
        # Repeat/crop target stats to match source batch size.
        reps = int(np.ceil(s.shape[0] / max(t.shape[0], 1)))
        t = t.repeat(reps, 1, 1, 1)[:s.shape[0]]
    s_mean = s.mean(dim=(2, 3), keepdim=True)
    s_std = s.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    t_mean = t.mean(dim=(2, 3), keepdim=True)
    t_std = t.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    styled = (s - s_mean) / s_std * t_std + t_mean
    return normalize(styled)


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, decay: float):
    sdict = student.state_dict()
    tdict = teacher.state_dict()
    for k in tdict.keys():
        if k in sdict and torch.is_floating_point(tdict[k]):
            tdict[k].mul_(decay).add_(sdict[k].detach(), alpha=1.0 - decay)
        elif k in sdict:
            tdict[k].copy_(sdict[k])


@torch.no_grad()
def evaluate_idabd(v7, phase1_model, phase2_model, loader, device, phase1_threshold, args):
    return idabd_direct.evaluate_idabd(
        v7=v7,
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=loader,
        device=device,
        phase1_threshold=phase1_threshold,
        dilation=args.postprocess_dilation,
        dilation_kernel=args.dilation_kernel,
    )


def save_checkpoint(path: Path, student: nn.Module, teacher: nn.Module, epoch: int, args: argparse.Namespace, metrics: Optional[Dict] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save teacher as model_state_dict because it is usually more stable at inference.
    payload = {
        "epoch": epoch,
        "model_state_dict": teacher.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "student_state_dict": student.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "experiment": "HRTBDA v7-MSDF xBD->IDA-BD foreground-aware Mean Teacher ST-UDA",
    }
    torch.save(payload, path)
    print(f"Saved: {path}", flush=True)


def write_summary(scores_dir: Path, metrics: Dict, args: argparse.Namespace, phase1_threshold: float, best_epoch: int):
    scores_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "HRTBDA v7-MSDF foreground-aware Mean Teacher ST-UDA xBD -> IDA-BD",
        "source_xbd_root": str(args.xbd_root),
        "target_idabd_root": str(args.idabd_root),
        "phase1_checkpoint": str(args.phase1_checkpoint),
        "source_phase2_checkpoint": str(args.phase2_checkpoint),
        "adapted_phase2_checkpoint": str(args.output_dir / "checkpoints" / "phase2_studa_teacher_latest.pt"),
        "phase1_threshold": float(phase1_threshold),
        "best_epoch_debug": int(best_epoch),
        "pseudo_thresholds": {
            "no_damage": args.conf_thresh_no,
            "minor": args.conf_thresh_minor,
            "major": args.conf_thresh_major,
            "destroyed": args.conf_thresh_destroyed,
        },
        "metrics": metrics,
    }
    json_path = scores_dir / "idabd_v7_msdf_studa_final_scores.json"
    txt_path = scores_dir / "summary_idabd_v7_msdf_studa_final.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = [
        "===== FINAL IDA-BD TEST EVALUATION =====",
        "Experiment: HRTBDA v7-MSDF foreground-aware Mean Teacher ST-UDA xBD -> IDA-BD",
        f"Source xBD root: {args.xbd_root}",
        f"Target IDA-BD root: {args.idabd_root}",
        f"Phase I checkpoint: {args.phase1_checkpoint}",
        f"Source Phase II checkpoint: {args.phase2_checkpoint}",
        f"Adapted Phase II checkpoint: {args.output_dir / 'checkpoints' / 'phase2_studa_teacher_latest.pt'}",
        f"Phase I threshold used for mask: {phase1_threshold:.2f}",
        f"Pseudo thresholds [no,minor,major,destroyed]: [{args.conf_thresh_no}, {args.conf_thresh_minor}, {args.conf_thresh_major}, {args.conf_thresh_destroyed}]",
        f"Target pseudo loss weight: {args.target_loss_weight}",
        f"EMA decay: {args.ema_decay}",
        f"Style match probability: {args.style_match_prob}",
        f"Best epoch selected on IDA-BD test-debug overall: {best_epoch}",
        f"Localization F1: {metrics['loc_f1']:.6f}",
        f"No Damage F1:    {metrics['no_damage_f1']:.6f}",
        f"Minor Damage F1: {metrics['minor_damage_f1']:.6f}",
        f"Major Damage F1: {metrics['major_damage_f1']:.6f}",
        f"Destroyed F1:    {metrics['destroyed_f1']:.6f}",
        f"Damage F1 harmonic: {metrics['damage_f1_hmean']:.6f}",
        f"Damage F1 macro:    {metrics['damage_f1_macro']:.6f}",
        f"Overall Score:      {metrics['overall_score']:.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v7-script", type=Path, default=Path("transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py"))
    p.add_argument("--phase1-checkpoint", type=Path, required=True)
    p.add_argument("--phase2-checkpoint", type=Path, required=True)
    p.add_argument("--xbd-root", type=Path, required=True)
    p.add_argument("--idabd-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, default=None)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--source-batch-size", type=int, default=2)
    p.add_argument("--target-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--phase2-crop-size", type=int, default=608)
    p.add_argument("--crop-candidate-count", type=int, default=8)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--aux-loc-weight", type=float, default=0.2)
    p.add_argument("--target-aux-weight", type=float, default=0.05)
    p.add_argument("--target-loss-weight", type=float, default=0.30)
    p.add_argument("--target-ramp-epochs", type=int, default=5)
    p.add_argument("--ema-decay", type=float, default=0.995)

    p.add_argument("--pos-weight-no", type=float, default=0.10)
    p.add_argument("--pos-weight-minor", type=float, default=1.80)
    p.add_argument("--pos-weight-major", type=float, default=1.80)
    p.add_argument("--pos-weight-destroyed", type=float, default=1.30)
    p.add_argument("--target-pos-weight-no", type=float, default=0.20)
    p.add_argument("--target-pos-weight-minor", type=float, default=2.00)
    p.add_argument("--target-pos-weight-major", type=float, default=2.00)
    p.add_argument("--target-pos-weight-destroyed", type=float, default=3.00)

    p.add_argument("--conf-thresh-no", type=float, default=0.88)
    p.add_argument("--conf-thresh-minor", type=float, default=0.70)
    p.add_argument("--conf-thresh-major", type=float, default=0.68)
    p.add_argument("--conf-thresh-destroyed", type=float, default=0.45)

    p.add_argument("--style-match-prob", type=float, default=0.50)
    p.add_argument("--strong-noise-std", type=float, default=0.03)
    p.add_argument("--phase1-threshold", type=float, default=0.50)
    p.add_argument("--postprocess-dilation", choices=["none", "minor", "minor_major"], default="minor")
    p.add_argument("--dilation-kernel", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--eval-every", type=int, default=1)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    scores_dir = args.output_dir / "scores"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    if args.split_json is None:
        args.split_json = args.output_dir / "idabd_splits_seed42_80_10_10.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    v7 = idabd_direct.load_module(args.v7_script)
    v7_args = make_v7_args(args)

    print("Loading xBD source loaders...", flush=True)
    source_train_loader, _, _, _ = v7.make_loaders(v7_args)

    print("Loading IDA-BD target loaders...", flush=True)
    idabd_samples = idabd_direct.discover_idabd_samples(args.idabd_root, require_mask=False)
    idabd_labeled_samples = idabd_direct.discover_idabd_samples(args.idabd_root, require_mask=True)
    split = idabd_direct.get_or_create_split(idabd_samples, args.split_json, args.seed)
    print("===== IDA-BD SPLIT SUMMARY =====", flush=True)
    print(f"Train target unlabeled: {len(split['train'])}", flush=True)
    print(f"Val: {len(split['val'])}", flush=True)
    print(f"Test: {len(split['test'])}", flush=True)
    print("=================================", flush=True)

    target_train_ds = idabd_direct.IDABDDataset(idabd_samples, split["train"], img_size=args.img_size, require_mask=False)
    target_test_ds = idabd_direct.IDABDDataset(idabd_labeled_samples, split["test"], img_size=args.img_size, require_mask=True)
    target_train_loader = DataLoader(target_train_ds, batch_size=args.target_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    target_test_loader = DataLoader(target_test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if len(target_train_ds) == 0:
        raise RuntimeError("No IDA-BD target train samples found.")
    if len(target_test_ds) == 0:
        print("WARNING: no labeled IDA-BD test samples found. Training will run, but final eval cannot run.", flush=True)

    print("Loading frozen Phase-I model...", flush=True)
    phase1_model, phase1_threshold, phase1_meta = v7.load_phase1_model_for_cascade(args=v7_args, device=device, phase1_ckpt=args.phase1_checkpoint)
    for p1 in phase1_model.parameters():
        p1.requires_grad_(False)
    phase1_model.eval()
    print(f"Phase-I threshold: {phase1_threshold}", flush=True)
    print(f"Phase-I meta: {phase1_meta}", flush=True)

    print("Loading student and teacher Phase-II v7-MSDF models from xBD checkpoint...", flush=True)
    student = v7.HRTBDAPhase2(base_channels=args.base_channels, decoder_channels=args.decoder_channels, window_size=args.window_size, num_classes=4).to(device)
    teacher = v7.HRTBDAPhase2(base_channels=args.base_channels, decoder_channels=args.decoder_channels, window_size=args.window_size, num_classes=4).to(device)
    phase2_ckpt = v7.load_model_weights(student, args.phase2_checkpoint, device)
    _ = v7.load_model_weights(teacher, args.phase2_checkpoint, device)
    for pt in teacher.parameters():
        pt.requires_grad_(False)
    teacher.eval()
    print(f"Loaded Phase-II source checkpoint epoch: {phase2_ckpt.get('epoch', 'unknown')}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    source_pos_weight = torch.tensor([args.pos_weight_no, args.pos_weight_minor, args.pos_weight_major, args.pos_weight_destroyed], dtype=torch.float32, device=device)
    target_pos_weight = torch.tensor([args.target_pos_weight_no, args.target_pos_weight_minor, args.target_pos_weight_major, args.target_pos_weight_destroyed], dtype=torch.float32, device=device)
    thresholds = torch.tensor([args.conf_thresh_no, args.conf_thresh_minor, args.conf_thresh_major, args.conf_thresh_destroyed], dtype=torch.float32, device=device)

    source_iter = cycle_loader(source_train_loader)
    target_iter = cycle_loader(target_train_loader)
    best_overall = -1.0
    best_epoch = -1
    best_metrics = None

    print("===== START FOREGROUND-AWARE MEAN TEACHER ST-UDA =====", flush=True)
    print(f"Source: xBD train+tier3 with labels: {args.xbd_root}", flush=True)
    print(f"Target: IDA-BD train without labels: {args.idabd_root}", flush=True)
    print(f"Epochs: {args.epochs} | steps/epoch: {args.steps_per_epoch}", flush=True)
    print(f"Target loss weight: {args.target_loss_weight}", flush=True)
    print(f"EMA decay: {args.ema_decay}", flush=True)
    print(f"Pseudo thresholds [no,minor,major,destroyed]: {thresholds.detach().cpu().tolist()}", flush=True)
    print(f"Style match probability: {args.style_match_prob}", flush=True)
    print("========================================================", flush=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        teacher.eval()
        ramp = min(1.0, epoch / max(float(args.target_ramp_epochs), 1.0))
        target_weight = args.target_loss_weight * ramp

        running = {
            "loss": 0.0,
            "src": 0.0,
            "tgt": 0.0,
            "src_focal": 0.0,
            "src_dice": 0.0,
            "src_aux": 0.0,
            "tgt_focal": 0.0,
            "tgt_dice": 0.0,
            "tgt_aux": 0.0,
            "pseudo_pixels": 0.0,
            "pseudo_coverage": 0.0,
            "pseudo_c1": 0.0,
            "pseudo_c2": 0.0,
            "pseudo_c3": 0.0,
            "pseudo_c4": 0.0,
        }

        for step in range(1, args.steps_per_epoch + 1):
            src = next(source_iter)
            tgt = next(target_iter)
            src_pre = src["pre"].to(device, non_blocking=True)
            src_post = src["post"].to(device, non_blocking=True)
            src_target5 = src["target5"].to(device, non_blocking=True).long()
            tgt_pre = tgt["pre"].to(device, non_blocking=True)
            tgt_post = tgt["post"].to(device, non_blocking=True)

            # Optional source-to-target style matching. Labels remain source labels.
            if args.style_match_prob > 0 and random.random() < args.style_match_prob:
                with torch.no_grad():
                    src_pre = match_source_to_target_style(src_pre, tgt_pre)
                    src_post = match_source_to_target_style(src_post, tgt_post)

            with torch.no_grad():
                tgt_loc_logits = phase1_forward_logits(phase1_model, tgt_pre)
                tgt_loc_mask = (torch.sigmoid(tgt_loc_logits) > phase1_threshold).float()
                teacher_out = teacher(tgt_pre, tgt_post)
                teacher_damage_logits = get_damage_logits(v7, teacher_out)

            tgt_pre_strong = strong_photometric_aug(tgt_pre, noise_std=args.strong_noise_std)
            tgt_post_strong = strong_photometric_aug(tgt_post, noise_std=args.strong_noise_std)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                src_out = student(src_pre, src_post)
                src_loss, src_parts = source_supervised_loss(v7, src_out, src_target5, args.focal_gamma, source_pos_weight, args.aux_loc_weight)

                tgt_out = student(tgt_pre_strong, tgt_post_strong)
                tgt_loss, tgt_stats = target_pseudo_loss(
                    v7=v7,
                    student_out=tgt_out,
                    teacher_damage_logits=teacher_damage_logits,
                    target_loc_mask=tgt_loc_mask,
                    thresholds=thresholds,
                    focal_gamma=args.focal_gamma,
                    pos_weight=target_pos_weight,
                    target_aux_weight=args.target_aux_weight,
                )
                loss = src_loss + target_weight * tgt_loss

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            ema_update(student, teacher, args.ema_decay)

            running["loss"] += float(loss.detach().cpu())
            running["src"] += float(src_loss.detach().cpu())
            running["tgt"] += float(tgt_loss.detach().cpu())
            running["src_focal"] += src_parts["focal"]
            running["src_dice"] += src_parts["dice"]
            running["src_aux"] += src_parts["aux"]
            for k in ["target_focal", "target_dice", "target_aux", "pseudo_pixels", "pseudo_coverage", "pseudo_c1", "pseudo_c2", "pseudo_c3", "pseudo_c4"]:
                short = k.replace("target_", "tgt_")
                running[short if short in running else k] += float(tgt_stats[k])

            if step % 50 == 0 or step == args.steps_per_epoch:
                d = float(step)
                print(
                    f"Epoch {epoch:03d}/{args.epochs} Step {step:04d}/{args.steps_per_epoch} "
                    f"loss={running['loss']/d:.4f} src={running['src']/d:.4f} tgt={running['tgt']/d:.4f} "
                    f"tgt_w={target_weight:.3f} src_focal={running['src_focal']/d:.4f} src_dice={running['src_dice']/d:.4f} "
                    f"tgt_focal={running['tgt_focal']/d:.4f} tgt_dice={running['tgt_dice']/d:.4f} "
                    f"pseudo_pix={running['pseudo_pixels']/d:.1f} cov={running['pseudo_coverage']/d:.4f} "
                    f"pseudo[c1,c2,c3,c4]=[{running['pseudo_c1']/d:.1f},{running['pseudo_c2']/d:.1f},{running['pseudo_c3']/d:.1f},{running['pseudo_c4']/d:.1f}]",
                    flush=True,
                )

        save_checkpoint(ckpt_dir / f"phase2_studa_teacher_epoch_{epoch:03d}.pt", student, teacher, epoch, args)
        save_checkpoint(ckpt_dir / "phase2_studa_teacher_latest.pt", student, teacher, epoch, args)

        if len(target_test_ds) > 0 and args.eval_every > 0 and epoch % args.eval_every == 0:
            teacher.eval()
            metrics = evaluate_idabd(v7, phase1_model, teacher, target_test_loader, device, phase1_threshold, args)
            print(
                f"IDA-BD TEST after epoch {epoch:03d} | "
                f"loc={metrics['loc_f1']:.6f} no={metrics['no_damage_f1']:.6f} "
                f"minor={metrics['minor_damage_f1']:.6f} major={metrics['major_damage_f1']:.6f} "
                f"destroyed={metrics['destroyed_f1']:.6f} damage_h={metrics['damage_f1_hmean']:.6f} "
                f"overall={metrics['overall_score']:.6f}",
                flush=True,
            )
            if metrics["overall_score"] > best_overall:
                best_overall = metrics["overall_score"]
                best_epoch = epoch
                best_metrics = metrics
                save_checkpoint(ckpt_dir / "phase2_studa_teacher_best_idabd_test_debug.pt", student, teacher, epoch, args, metrics)

    if len(target_test_ds) > 0:
        print("\n===== FINAL IDA-BD TEST EVALUATION =====", flush=True)
        # Evaluate latest teacher. Also save final summary.
        final_metrics = evaluate_idabd(v7, phase1_model, teacher, target_test_loader, device, phase1_threshold, args)
        write_summary(scores_dir, final_metrics, args, phase1_threshold, best_epoch)
        if best_metrics is not None:
            best_json = scores_dir / "idabd_v7_msdf_studa_best_debug_scores.json"
            with open(best_json, "w", encoding="utf-8") as f:
                json.dump({"best_epoch_debug": best_epoch, "best_metrics_debug": best_metrics}, f, indent=2)
            print(f"Wrote best-debug scores: {best_json}", flush=True)


if __name__ == "__main__":
    main()
