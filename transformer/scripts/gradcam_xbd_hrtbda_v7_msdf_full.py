#!/usr/bin/env python3
"""
Grad-CAM visualization for HRTBDA v7-MSDF full two-stage xBD experiment.

Outputs, for each selected xBD test sample:
  - pre-disaster image
  - post-disaster image
  - ground-truth damage map
  - predicted cascaded damage map
  - Grad-CAM overlays for selected damage classes

Default target layer:
  change_fusion.0

This is usually the best layer for segmentation-style Grad-CAM because it has
higher spatial resolution than the deepest branch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAME_TO_ID = {
    "no": 1,
    "no_damage": 1,
    "minor": 2,
    "minor_damage": 2,
    "major": 3,
    "major_damage": 3,
    "destroyed": 4,
}

CLASS_ID_TO_NAME = {
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}

# RGB colors for maps: background, no, minor, major, destroyed
PALETTE = np.array(
    [
        [0, 0, 90],        # 0 background: dark blue
        [0, 210, 0],       # 1 no damage: green
        [255, 255, 0],     # 2 minor: yellow
        [255, 150, 0],     # 3 major: orange
        [255, 0, 0],       # 4 destroyed: red
    ],
    dtype=np.uint8,
)


def load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("hrtbda_v7_module", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hrtbda_v7_module"] = module
    spec.loader.exec_module(module)
    return module


def make_v7_args(args: argparse.Namespace) -> SimpleNamespace:
    """
    Build a Namespace compatible with train_xbd_hrtbda_v7_msdf_full_two_stage.py.
    """
    return SimpleNamespace(
        phase="test",
        resume_phase1=None,
        phase1_checkpoint=str(args.phase1_checkpoint),
        phase2_checkpoint=str(args.phase2_checkpoint),

        xbd_root=str(args.xbd_root),
        train_split=["train", "tier3"],
        val_split="hold",
        test_split=args.test_split,
        output_dir=str(args.output_dir),

        phase1_epochs=150,
        phase2_epochs=60,
        phase1_batch_size=1,
        phase2_batch_size=2,
        batch_size=2,
        eval_batch_size=1,
        grad_accum_steps=4,
        num_workers=args.num_workers,

        img_size=args.img_size,
        phase2_crop_size=608,
        crop_candidate_count=8,

        lr=1e-4,
        weight_decay=1e-4,
        seed=42,
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        max_grad_norm=1.0,
        warmup_epochs=5,
        save_every=1,
        early_stopping_patience=999,

        focal_gamma=2.0,
        loc_loss_weight=1.0,
        cls_loss_weight=1.0,
        aux_loc_weight=0.2,

        minor_damage_boost=1.5,
        major_damage_boost=1.5,
        max_damage_class_weight=10.0,

        crop_weight_no_damage=1.0,
        crop_weight_minor=12.0,
        crop_weight_major=12.0,
        crop_weight_destroyed=4.0,

        finetune_epochs=3,
        finetune_lr=5e-5,

        postprocess_dilation=args.postprocess_dilation,
        dilation_kernel=args.dilation_kernel,

        phase1_threshold=0.50,
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        amp=False,
        extra_photometric_aug=False,
    )


def tensor_to_rgb_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor [3,H,W] back to uint8 RGB.
    Uses ImageNet mean/std because the training dataset does.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    arr = x.detach().cpu().float().numpy()
    arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return arr


def label_to_color(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(np.int64)
    mask = np.clip(mask, 0, 4)
    return PALETTE[mask]


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a CAM heatmap on RGB image.
    """
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - np.nanmin(cam)
    cam = cam / (np.nanmax(cam) + 1e-8)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Supports paths like:
      change_fusion.0
      change_fusion.0.out_fuse
      decoder
      backbone.stage4_b0
    """
    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        class_id: int,
        score_mask: torch.Tensor,
        v7,
    ) -> np.ndarray:
        """
        class_id: raw xBD class ID:
          1 no damage, 2 minor, 3 major, 4 destroyed
        score_mask: [1,H,W] bool/float mask used to average target score
        """
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        out = self.model(pre, post)
        damage_logits = v7.get_damage_logits(out)

        ch = class_id - 1
        logit_map = damage_logits[:, ch, :, :]  # [B,H,W]

        score_mask = score_mask.to(logit_map.device).float()
        if score_mask.ndim == 2:
            score_mask = score_mask.unsqueeze(0)
        if score_mask.sum() < 1:
            score = logit_map.mean()
        else:
            score = (logit_map * score_mask).sum() / (score_mask.sum() + 1e-6)

        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        acts = self.activations.detach()      # [B,C,h,w]
        grads = self.gradients.detach()       # [B,C,h,w]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))  # [B,1,h,w]

        cam = F.interpolate(
            cam,
            size=pre.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam[0, 0].detach().cpu().float().numpy()
        cam_np = cam_np - np.nanmin(cam_np)
        cam_np = cam_np / (np.nanmax(cam_np) + 1e-8)
        return cam_np


@torch.no_grad()
def get_cascade_prediction(
    phase1_model,
    phase2_model,
    pre: torch.Tensor,
    post: torch.Tensor,
    phase1_threshold: float,
    v7,
    postprocess_dilation: str,
    dilation_kernel: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      loc_pred:      [1,H,W]
      damage_pred:   [1,H,W], labels 1..4
      final_pred:    [1,H,W], labels 0..4
      damage_probs:  [1,4,H,W]
    """
    phase1_model.eval()
    phase2_model.eval()

    phase1_logits = phase1_model(pre)
    loc_pred = (torch.sigmoid(phase1_logits) > phase1_threshold).long()

    out = phase2_model(pre, post)
    damage_logits = v7.get_damage_logits(out)
    damage_probs = torch.sigmoid(damage_logits)
    damage_pred = v7.damage_logits_to_pred(damage_logits)
    damage_pred = v7.apply_damage_dilation(
        damage_pred,
        loc_pred,
        mode=postprocess_dilation,
        kernel_size=dilation_kernel,
    )

    final_pred = torch.zeros_like(damage_pred)
    final_pred[loc_pred.bool()] = damage_pred[loc_pred.bool()]
    return loc_pred, damage_pred, final_pred, damage_probs


def save_visual_grid(
    save_path: Path,
    pre_rgb: np.ndarray,
    post_rgb: np.ndarray,
    gt_color: np.ndarray,
    pred_color: np.ndarray,
    cam_overlays: Dict[int, np.ndarray],
    title: str,
):
    n_cam = len(cam_overlays)
    n_cols = 4 + n_cam

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    panels = [
        ("Pre", pre_rgb),
        ("Post", post_rgb),
        ("Ground truth", gt_color),
        ("Prediction", pred_color),
    ]

    for cls_id, overlay in cam_overlays.items():
        panels.append((f"Grad-CAM {CLASS_ID_TO_NAME[cls_id]}", overlay))

    for ax, (name, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v7-script",
        type=Path,
        default=Path("transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py"),
        help="Path to the v7-MSDF training script to import model/dataset helpers from.",
    )

    parser.add_argument(
        "--phase1-checkpoint",
        type=Path,
        default=Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/checkpoints/phase1_best.pt"),
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=Path,
        default=Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/checkpoints/phase2_best.pt"),
    )
    parser.add_argument(
        "--xbd-root",
        type=Path,
        default=Path("/homes/j244s673/documents/wsu/phd/xview2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/gradcam_visuals"),
    )

    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stems", nargs="*", default=None, help="Optional sample stems to visualize.")
    parser.add_argument("--cam-classes", nargs="+", default=["minor", "major", "destroyed"])
    parser.add_argument("--target-layer", type=str, default="change_fusion.0")
    parser.add_argument("--score-mask", type=str, default="pred_class", choices=["pred_class", "phase1_mask", "gt_class", "all"])

    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--postprocess-dilation", type=str, default="minor", choices=["none", "minor", "minor_major"])
    parser.add_argument("--dilation-kernel", type=int, default=3)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if not args.v7_script.exists():
        raise FileNotFoundError(f"v7 script not found: {args.v7_script}")
    if not args.phase1_checkpoint.exists():
        raise FileNotFoundError(f"Phase-I checkpoint not found: {args.phase1_checkpoint}")
    if not args.phase2_checkpoint.exists():
        raise FileNotFoundError(f"Phase-II checkpoint not found: {args.phase2_checkpoint}")

    v7 = load_module(args.v7_script)
    v7_args = make_v7_args(args)

    print("Loading test loader...", flush=True)
    _, _, test_loader, _ = v7.make_loaders(v7_args)

    print("Loading Phase-I model...", flush=True)
    phase1_model, phase1_threshold, phase1_meta = v7.load_phase1_model_for_cascade(
        args=v7_args,
        device=device,
        phase1_ckpt=args.phase1_checkpoint,
    )

    print(f"Phase-I threshold: {phase1_threshold}", flush=True)
    print(f"Phase-I meta: {phase1_meta}", flush=True)

    print("Loading Phase-II v7-MSDF model...", flush=True)
    phase2_model = v7.HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=4,
    ).to(device)

    ckpt = v7.load_model_weights(phase2_model, args.phase2_checkpoint, device)
    print(f"Loaded Phase-II checkpoint epoch: {ckpt.get('epoch', 'unknown')}", flush=True)

    phase1_model.eval()
    phase2_model.eval()

    try:
        target_layer = get_module_by_path(phase2_model, args.target_layer)
    except Exception as e:
        print(f"ERROR: Could not find target layer '{args.target_layer}'.", flush=True)
        print("Available modules containing 'change' or 'fusion':", flush=True)
        for name, _ in phase2_model.named_modules():
            if "change" in name or "fusion" in name:
                print(f"  {name}", flush=True)
        raise e

    cam_engine = GradCAM(phase2_model, target_layer)

    selected_class_ids: List[int] = []
    for name in args.cam_classes:
        key = name.lower()
        if key not in CLASS_NAME_TO_ID:
            raise ValueError(f"Unknown class '{name}'. Use one of: {sorted(CLASS_NAME_TO_ID)}")
        selected_class_ids.append(CLASS_NAME_TO_ID[key])

    processed = 0
    rows = []

    print("Generating Grad-CAM visualizations...", flush=True)

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx < args.start_index:
            continue

        stem = batch["stem"][0] if isinstance(batch["stem"], (list, tuple)) else str(batch["stem"])

        if args.stems is not None and len(args.stems) > 0 and stem not in set(args.stems):
            continue

        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        target5 = batch["target5"].to(device, non_blocking=True).long()

        pre_rgb = tensor_to_rgb_image(batch["pre"][0])
        post_rgb = tensor_to_rgb_image(batch["post"][0])

        loc_pred, damage_pred, final_pred, damage_probs = get_cascade_prediction(
            phase1_model=phase1_model,
            phase2_model=phase2_model,
            pre=pre,
            post=post,
            phase1_threshold=phase1_threshold,
            v7=v7,
            postprocess_dilation=args.postprocess_dilation,
            dilation_kernel=args.dilation_kernel,
        )

        gt_np = target5[0].detach().cpu().numpy()
        gt_np = np.where((gt_np >= 0) & (gt_np <= 4), gt_np, 0).astype(np.uint8)

        pred_np = final_pred[0].detach().cpu().numpy().astype(np.uint8)

        gt_color = label_to_color(gt_np)
        pred_color = label_to_color(pred_np)

        cam_overlays: Dict[int, np.ndarray] = {}

        for cls_id in selected_class_ids:
            if args.score_mask == "pred_class":
                mask = (final_pred == cls_id).float()
                if mask.sum() < 1:
                    mask = loc_pred.float()
            elif args.score_mask == "phase1_mask":
                mask = loc_pred.float()
            elif args.score_mask == "gt_class":
                mask = (target5 == cls_id).float()
                if mask.sum() < 1:
                    mask = loc_pred.float()
            else:
                mask = torch.ones_like(loc_pred).float()

            cam = cam_engine(
                pre=pre,
                post=post,
                class_id=cls_id,
                score_mask=mask[0],
                v7=v7,
            )
            cam_overlays[cls_id] = overlay_heatmap(post_rgb, cam, alpha=0.45)

            cam_path = args.output_dir / "heatmaps" / f"{stem}_gradcam_{CLASS_ID_TO_NAME[cls_id]}.png"
            cam_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cam_path), cv2.cvtColor((cam * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

        fig_path = args.output_dir / "figures" / f"{stem}_visual_comparison_gradcam.png"
        save_visual_grid(
            save_path=fig_path,
            pre_rgb=pre_rgb,
            post_rgb=post_rgb,
            gt_color=gt_color,
            pred_color=pred_color,
            cam_overlays=cam_overlays,
            title=f"{stem} | HRTBDA v7-MSDF Grad-CAM",
        )

        rows.append(
            {
                "stem": stem,
                "figure": str(fig_path),
                "phase1_threshold": float(phase1_threshold),
                "postprocess_dilation": args.postprocess_dilation,
                "dilation_kernel": int(args.dilation_kernel),
                "target_layer": args.target_layer,
                "cam_classes": ",".join(CLASS_ID_TO_NAME[c] for c in selected_class_ids),
            }
        )

        print(f"Wrote: {fig_path}", flush=True)

        processed += 1
        if processed >= args.sample_count:
            break

    cam_engine.remove()

    summary_path = args.output_dir / "gradcam_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase1_checkpoint": str(args.phase1_checkpoint),
                "phase2_checkpoint": str(args.phase2_checkpoint),
                "phase1_threshold": float(phase1_threshold),
                "phase1_meta": phase1_meta,
                "target_layer": args.target_layer,
                "score_mask": args.score_mask,
                "postprocess_dilation": args.postprocess_dilation,
                "dilation_kernel": args.dilation_kernel,
                "processed": rows,
            },
            f,
            indent=2,
        )

    print(f"Done. Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
PY