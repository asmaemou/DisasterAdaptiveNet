#!/usr/bin/env python3
"""
Unsupervised domain adaptation for HRTBDA v7-MSDF using foreground-aware DANN.

Source domain: xBD train+tier3 with labels.
Target domain: IDA-BD train split without labels.
Final eval: IDA-BD test split labels are used only for reporting.

This script starts from your xBD-trained v7-MSDF checkpoints and adapts Phase II features.
Phase I is kept frozen and is used to produce foreground masks for target IDA-BD images.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Import reusable IDA-BD loader/evaluator from the direct-transfer script.
_THIS_DIR = Path(__file__).resolve().parent
_DIRECT_SCRIPT = _THIS_DIR / "eval_idabd_v7_msdf_direct_transfer.py"
if not _DIRECT_SCRIPT.exists():
    # Allows running from project checkout after both scripts are copied to transformer/scripts.
    _DIRECT_SCRIPT = Path("transformer/scripts/eval_idabd_v7_msdf_direct_transfer.py")

import importlib.util as _importlib_util
_spec = _importlib_util.spec_from_file_location("idabd_direct", str(_DIRECT_SCRIPT))
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import direct-transfer helper script: {_DIRECT_SCRIPT}")
idabd_direct = _importlib_util.module_from_spec(_spec)
sys.modules["idabd_direct"] = idabd_direct
_spec.loader.exec_module(idabd_direct)


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverseFn.apply(x, lambd)


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


def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


class FeatureHook:
    def __init__(self, layer: nn.Module):
        self.feature = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.feature = output[0] if isinstance(output, (tuple, list)) else output

    def clear(self):
        self.feature = None

    def remove(self):
        self.handle.remove()


def masked_global_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average pool feature map using a foreground mask."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    mask = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")
    denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
    pooled = (feat * mask).sum(dim=(2, 3)) / denom
    return pooled


def phase1_forward_logits(phase1_model: torch.nn.Module, pre: torch.Tensor) -> torch.Tensor:
    return idabd_direct.phase1_forward_logits(phase1_model, pre)


def get_damage_logits(v7, out):
    return idabd_direct.get_damage_logits(v7, out)


def make_multilabel_targets(target5: torch.Tensor) -> torch.Tensor:
    """target5 [B,H,W] labels 0..4 -> [B,4,H,W] for classes 1..4."""
    chans = []
    for c in [1, 2, 3, 4]:
        chans.append((target5 == c).float())
    return torch.stack(chans, dim=1)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    denom = (probs * probs).sum(dim=dims) + (targets * targets).sum(dim=dims)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float, pos_weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight.view(1, -1, 1, 1))
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    focal = ((1.0 - pt).clamp_min(1e-6) ** gamma) * bce
    return focal.mean()


def source_supervised_loss(
    v7,
    phase2_out,
    target5: torch.Tensor,
    focal_gamma: float,
    pos_weight: torch.Tensor,
    aux_loc_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        aux_logits = phase2_out.get("aux_loc", None) or phase2_out.get("aux_loc_logits", None)
        if aux_logits is not None:
            if aux_logits.ndim == 4 and aux_logits.shape[1] == 1:
                aux_logits = aux_logits[:, 0]
            loc_target = (target5 > 0).float()
            aux = F.binary_cross_entropy_with_logits(aux_logits, loc_target)

    loss = focal + dice + aux_loc_weight * aux
    return loss, {"focal": float(focal.detach().cpu()), "dice": float(dice.detach().cpu()), "aux": float(aux.detach().cpu())}


def dann_lambda(step: int, total_steps: int, max_lambda: float) -> float:
    if total_steps <= 1:
        return max_lambda
    p = step / float(total_steps)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def quick_idabd_eval(v7, phase1_model, phase2_model, idabd_loader, device, phase1_threshold, args):
    return idabd_direct.evaluate_idabd(
        v7=v7,
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=idabd_loader,
        device=device,
        phase1_threshold=phase1_threshold,
        dilation=args.postprocess_dilation,
        dilation_kernel=args.dilation_kernel,
    )


def save_checkpoint(path: Path, phase2_model: nn.Module, discriminator: nn.Module, epoch: int, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": phase2_model.state_dict(),
        "domain_discriminator_state_dict": discriminator.state_dict(),
        "args": vars(args),
        "experiment": "HRTBDA v7-MSDF xBD->IDA-BD foreground-aware DANN",
    }, path)
    print(f"Saved: {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v7-script", type=Path, default=Path("transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py"))
    p.add_argument("--phase1-checkpoint", type=Path, required=True)
    p.add_argument("--phase2-checkpoint", type=Path, required=True)
    p.add_argument("--xbd-root", type=Path, required=True)
    p.add_argument("--idabd-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, default=None)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--source-batch-size", type=int, default=2)
    p.add_argument("--target-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--phase2-crop-size", type=int, default=608)
    p.add_argument("--crop-candidate-count", type=int, default=8)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--aux-loc-weight", type=float, default=0.2)
    p.add_argument("--domain-loss-weight", type=float, default=0.05)
    p.add_argument("--max-dann-lambda", type=float, default=1.0)
    p.add_argument("--target-layer", type=str, default="change_fusion.0")

    p.add_argument("--pos-weight-no", type=float, default=0.10)
    p.add_argument("--pos-weight-minor", type=float, default=1.80)
    p.add_argument("--pos-weight-major", type=float, default=1.80)
    p.add_argument("--pos-weight-destroyed", type=float, default=1.30)

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
    source_train_loader, source_val_loader, _, _ = v7.make_loaders(v7_args)

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
    phase1_model, phase1_threshold, phase1_meta = v7.load_phase1_model_for_cascade(
        args=v7_args, device=device, phase1_ckpt=args.phase1_checkpoint
    )
    for p1 in phase1_model.parameters():
        p1.requires_grad_(False)
    phase1_model.eval()
    print(f"Phase-I threshold: {phase1_threshold}", flush=True)
    print(f"Phase-I meta: {phase1_meta}", flush=True)

    print("Loading Phase-II v7-MSDF model from xBD checkpoint...", flush=True)
    phase2_model = v7.HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=4,
    ).to(device)
    phase2_ckpt = v7.load_model_weights(phase2_model, args.phase2_checkpoint, device)
    print(f"Loaded Phase-II source checkpoint epoch: {phase2_ckpt.get('epoch', 'unknown')}", flush=True)

    target_layer = get_module_by_path(phase2_model, args.target_layer)
    hook = FeatureHook(target_layer)
    print(f"DANN target layer: {args.target_layer}", flush=True)

    # Create discriminator lazily after first feature forward because channel dimension is model-dependent.
    discriminator: Optional[DomainDiscriminator] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    pos_weight = torch.tensor([
        args.pos_weight_no,
        args.pos_weight_minor,
        args.pos_weight_major,
        args.pos_weight_destroyed,
    ], dtype=torch.float32, device=device)

    source_iter = cycle_loader(source_train_loader)
    target_iter = cycle_loader(target_train_loader)
    total_steps = args.epochs * args.steps_per_epoch
    global_step = 0
    best_overall = -1.0
    best_epoch = -1

    print("===== START FOREGROUND-AWARE DANN ADAPTATION =====", flush=True)
    print(f"Source: xBD train+tier3 with labels: {args.xbd_root}", flush=True)
    print(f"Target: IDA-BD train without labels: {args.idabd_root}", flush=True)
    print(f"Epochs: {args.epochs} | steps/epoch: {args.steps_per_epoch}", flush=True)
    print(f"Domain loss weight: {args.domain_loss_weight}", flush=True)
    print("==================================================", flush=True)

    for epoch in range(1, args.epochs + 1):
        phase2_model.train()
        running = {"loss": 0.0, "sup": 0.0, "domain": 0.0, "focal": 0.0, "dice": 0.0, "aux": 0.0, "dom_acc": 0.0}

        for step in range(1, args.steps_per_epoch + 1):
            global_step += 1
            src = next(source_iter)
            tgt = next(target_iter)

            src_pre = src["pre"].to(device, non_blocking=True)
            src_post = src["post"].to(device, non_blocking=True)
            src_target5 = src["target5"].to(device, non_blocking=True).long()
            tgt_pre = tgt["pre"].to(device, non_blocking=True)
            tgt_post = tgt["post"].to(device, non_blocking=True)

            with torch.no_grad():
                src_mask = (src_target5 > 0).float()
                tgt_loc_logits = phase1_forward_logits(phase1_model, tgt_pre)
                tgt_mask = (torch.sigmoid(tgt_loc_logits) > phase1_threshold).float()

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                hook.clear()
                src_out = phase2_model(src_pre, src_post)
                src_feat = hook.feature
                if src_feat is None:
                    raise RuntimeError("DANN hook did not capture source features.")

                sup_loss, sup_parts = source_supervised_loss(
                    v7=v7,
                    phase2_out=src_out,
                    target5=src_target5,
                    focal_gamma=args.focal_gamma,
                    pos_weight=pos_weight,
                    aux_loc_weight=args.aux_loc_weight,
                )

                hook.clear()
                _ = phase2_model(tgt_pre, tgt_post)
                tgt_feat = hook.feature
                if tgt_feat is None:
                    raise RuntimeError("DANN hook did not capture target features.")

                src_vec = masked_global_pool(src_feat, src_mask)
                tgt_vec = masked_global_pool(tgt_feat, tgt_mask)

                if discriminator is None:
                    in_dim = int(src_vec.shape[1])
                    discriminator = DomainDiscriminator(in_dim=in_dim).to(device)
                    optimizer = torch.optim.AdamW(
                        [
                            {"params": phase2_model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
                            {"params": discriminator.parameters(), "lr": args.disc_lr, "weight_decay": args.weight_decay},
                        ]
                    )
                    optimizer.zero_grad(set_to_none=True)
                    print(f"Created domain discriminator with input dim={in_dim}", flush=True)

                lam = dann_lambda(global_step, total_steps, args.max_dann_lambda)
                dom_in = torch.cat([src_vec, tgt_vec], dim=0)
                dom_labels = torch.cat([
                    torch.zeros(src_vec.shape[0], device=device),
                    torch.ones(tgt_vec.shape[0], device=device),
                ], dim=0)
                dom_logits = discriminator(grad_reverse(dom_in, lam))
                domain_loss = F.binary_cross_entropy_with_logits(dom_logits, dom_labels)
                loss = sup_loss + args.domain_loss_weight * domain_loss

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(phase2_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                dom_pred = (torch.sigmoid(dom_logits) > 0.5).float()
                dom_acc = (dom_pred == dom_labels).float().mean().item()

            running["loss"] += float(loss.detach().cpu())
            running["sup"] += float(sup_loss.detach().cpu())
            running["domain"] += float(domain_loss.detach().cpu())
            running["focal"] += sup_parts["focal"]
            running["dice"] += sup_parts["dice"]
            running["aux"] += sup_parts["aux"]
            running["dom_acc"] += dom_acc

            if step % 50 == 0 or step == args.steps_per_epoch:
                denom = step
                print(
                    f"Epoch {epoch:03d}/{args.epochs} Step {step:04d}/{args.steps_per_epoch} "
                    f"loss={running['loss']/denom:.4f} sup={running['sup']/denom:.4f} "
                    f"domain={running['domain']/denom:.4f} dom_acc={running['dom_acc']/denom:.3f} "
                    f"focal={running['focal']/denom:.4f} dice={running['dice']/denom:.4f} aux={running['aux']/denom:.4f}",
                    flush=True,
                )

        save_checkpoint(ckpt_dir / f"phase2_dann_epoch_{epoch:03d}.pt", phase2_model, discriminator, epoch, args)
        save_checkpoint(ckpt_dir / "phase2_dann_latest.pt", phase2_model, discriminator, epoch, args)

        if len(target_test_ds) > 0 and args.eval_every > 0 and epoch % args.eval_every == 0:
            phase2_model.eval()
            metrics = quick_idabd_eval(v7, phase1_model, phase2_model, target_test_loader, device, phase1_threshold, args)
            print(
                f"IDA-BD TEST after epoch {epoch:03d} | "
                f"loc={metrics['loc_f1']:.6f} no={metrics['no_damage_f1']:.6f} "
                f"minor={metrics['minor_damage_f1']:.6f} major={metrics['major_damage_f1']:.6f} "
                f"destroyed={metrics['destroyed_f1']:.6f} damage_h={metrics['damage_f1_hmean']:.6f} "
                f"overall={metrics['overall_score']:.6f}",
                flush=True,
            )
            # NOTE: This best checkpoint uses test labels for convenience/debugging.
            # For a strict paper, report the final/latest checkpoint or choose using source hold only.
            if metrics["overall_score"] > best_overall:
                best_overall = metrics["overall_score"]
                best_epoch = epoch
                save_checkpoint(ckpt_dir / "phase2_dann_best_idabd_test_debug.pt", phase2_model, discriminator, epoch, args)

    hook.remove()

    print("===== FINAL IDA-BD TEST EVALUATION =====", flush=True)
    final_ckpt = ckpt_dir / "phase2_dann_latest.pt"
    metrics = None
    if len(target_test_ds) > 0:
        phase2_model.eval()
        metrics = quick_idabd_eval(v7, phase1_model, phase2_model, target_test_loader, device, phase1_threshold, args)
        result = {
            "experiment": "HRTBDA v7-MSDF foreground-aware DANN xBD -> IDA-BD",
            "source_xbd_root": str(args.xbd_root),
            "target_idabd_root": str(args.idabd_root),
            "phase1_checkpoint": str(args.phase1_checkpoint),
            "source_phase2_checkpoint": str(args.phase2_checkpoint),
            "adapted_phase2_checkpoint": str(final_ckpt),
            "split_json": str(args.split_json),
            "phase1_threshold": float(phase1_threshold),
            "domain_loss_weight": args.domain_loss_weight,
            "target_layer": args.target_layer,
            "best_epoch_by_idabd_test_debug": best_epoch,
            "best_overall_by_idabd_test_debug": best_overall,
            "metrics_final_latest": metrics,
        }
        json_path = scores_dir / "idabd_v7_msdf_dann_final_scores.json"
        txt_path = scores_dir / "summary_idabd_v7_msdf_dann_final.txt"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        lines = [
            "Experiment: HRTBDA v7-MSDF foreground-aware DANN xBD -> IDA-BD",
            f"Source xBD root: {args.xbd_root}",
            f"Target IDA-BD root: {args.idabd_root}",
            f"Phase I checkpoint: {args.phase1_checkpoint}",
            f"Source Phase II checkpoint: {args.phase2_checkpoint}",
            f"Adapted Phase II checkpoint: {final_ckpt}",
            f"Phase I threshold used for mask: {phase1_threshold:.2f}",
            f"DANN target layer: {args.target_layer}",
            f"Domain loss weight: {args.domain_loss_weight}",
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
    else:
        print("No IDA-BD test labels were found; skipped final evaluation.", flush=True)


if __name__ == "__main__":
    main()
