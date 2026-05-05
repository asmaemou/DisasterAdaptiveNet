#!/usr/bin/env python3
"""
Compact Grad-CAM visualization for HRTBDA v7-MSDF.

Output layout:
  Pre | Post | Ground truth | Prediction | Localization CAM | Predicted-class CAM

Localization CAM:
  - computed from Phase I localization model
  - overlaid on pre-disaster image

Predicted-class CAM:
  - computed from Phase II damage model
  - overlaid on post-disaster image
  - uses the dominant predicted non-background class in the final prediction map
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


CLASS_ID_TO_NAME = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}

PALETTE = np.array(
    [
        [0, 0, 90],        # background: dark blue
        [0, 210, 0],       # no damage: green
        [255, 255, 0],     # minor: yellow
        [255, 150, 0],     # major: orange
        [255, 0, 0],       # destroyed: red
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
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - np.nanmin(cam)
    cam = cam / (np.nanmax(cam) + 1e-8)

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    out = rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def find_last_conv2d(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_module = module

    if last_module is None:
        raise RuntimeError("No Conv2d layer found for automatic Grad-CAM target layer.")

    return last_name, last_module


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

    def compute_cam(self, output_size: Tuple[int, int]) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations/gradients.")

        acts = self.activations
        grads = self.gradients

        if isinstance(acts, (tuple, list)):
            acts = acts[0]
        if isinstance(grads, (tuple, list)):
            grads = grads[0]

        acts = acts.detach()
        grads = grads.detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))

        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam[0, 0].detach().cpu().float().numpy()
        cam_np = cam_np - np.nanmin(cam_np)
        cam_np = cam_np / (np.nanmax(cam_np) + 1e-8)
        return cam_np


def phase1_forward_logits(phase1_model: torch.nn.Module, pre: torch.Tensor) -> torch.Tensor:
    out = phase1_model(pre)

    if isinstance(out, dict):
        for key in ["loc_logits", "logits", "out", "mask_logits"]:
            if key in out:
                out = out[key]
                break
        else:
            raise RuntimeError(f"Could not find localization logits in Phase-I output keys: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    elif out.ndim == 4 and out.shape[1] > 1:
        out = out[:, 0]

    return out


def compute_phase1_localization_cam(
    phase1_model: torch.nn.Module,
    pre: torch.Tensor,
    phase1_cam_engine: GradCAM,
    target_mask: torch.Tensor | None,
    threshold: float,
) -> Tuple[np.ndarray, torch.Tensor]:
    phase1_model.zero_grad(set_to_none=True)
    phase1_cam_engine.activations = None
    phase1_cam_engine.gradients = None

    loc_logits = phase1_forward_logits(phase1_model, pre)
    loc_prob = torch.sigmoid(loc_logits)
    loc_pred = (loc_prob > threshold).long()

    if target_mask is None:
        mask = loc_pred.float()
    else:
        mask = target_mask.float()
        if mask.ndim == 4:
            mask = mask[:, 0]
        mask = (mask > 0).float()

    if mask.sum() < 1:
        score = loc_logits.mean()
    else:
        score = (loc_logits * mask).sum() / (mask.sum() + 1e-6)

    score.backward(retain_graph=False)

    cam = phase1_cam_engine.compute_cam(output_size=pre.shape[-2:])
    return cam, loc_pred


def choose_predicted_class(final_pred: torch.Tensor, damage_probs: torch.Tensor, mode: str) -> int:
    """
    final_pred: [1,H,W], labels 0..4
    damage_probs: [1,4,H,W]
    """
    pred_np = final_pred[0].detach().cpu().numpy()

    if mode == "rare_priority":
        for cls_id in [4, 3, 2, 1]:
            if np.any(pred_np == cls_id):
                return cls_id

    # dominant non-background class
    counts = []
    for cls_id in [1, 2, 3, 4]:
        counts.append((cls_id, int(np.sum(pred_np == cls_id))))

    counts_nonzero = [(c, n) for c, n in counts if n > 0]
    if len(counts_nonzero) > 0:
        counts_nonzero.sort(key=lambda x: x[1], reverse=True)
        return counts_nonzero[0][0]

    # fallback if no predicted buildings
    mean_probs = damage_probs.mean(dim=(0, 2, 3))
    return int(torch.argmax(mean_probs).item()) + 1


def compute_phase2_predicted_class_cam(
    phase2_model: torch.nn.Module,
    pre: torch.Tensor,
    post: torch.Tensor,
    phase2_cam_engine: GradCAM,
    v7,
    class_id: int,
    final_pred: torch.Tensor,
    loc_pred: torch.Tensor,
) -> np.ndarray:
    phase2_model.zero_grad(set_to_none=True)
    phase2_cam_engine.activations = None
    phase2_cam_engine.gradients = None

    out = phase2_model(pre, post)
    damage_logits = v7.get_damage_logits(out)

    ch = class_id - 1
    logit_map = damage_logits[:, ch, :, :]

    class_mask = (final_pred == class_id).float()
    if class_mask.sum() < 1:
        class_mask = loc_pred.float()
    if class_mask.sum() < 1:
        score = logit_map.mean()
    else:
        score = (logit_map * class_mask).sum() / (class_mask.sum() + 1e-6)

    score.backward(retain_graph=False)

    cam = phase2_cam_engine.compute_cam(output_size=pre.shape[-2:])
    return cam


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
):
    phase1_model.eval()
    phase2_model.eval()

    loc_logits = phase1_forward_logits(phase1_model, pre)
    loc_pred = (torch.sigmoid(loc_logits) > phase1_threshold).long()

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


def save_compact_grid(
    save_path: Path,
    pre_rgb: np.ndarray,
    post_rgb: np.ndarray,
    gt_color: np.ndarray,
    pred_color: np.ndarray,
    loc_overlay: np.ndarray,
    pred_class_overlay: np.ndarray,
    pred_class_name: str,
    title: str,
):
    panels = [
        ("Pre", pre_rgb),
        ("Post", post_rgb),
        ("Ground truth", gt_color),
        ("Prediction", pred_color),
        ("Grad-CAM localization", loc_overlay),
        (f"Grad-CAM predicted class\n{pred_class_name}", pred_class_overlay),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))

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
        default=Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/gradcam_compact_visuals"),
    )

    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stems", nargs="*", default=None)

    parser.add_argument("--phase1-target-layer", type=str, default="auto")
    parser.add_argument("--phase2-target-layer", type=str, default="change_fusion.0")
    parser.add_argument("--pred-class-mode", type=str, default="dominant", choices=["dominant", "rare_priority"])

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

    if args.phase1_target_layer == "auto":
        phase1_layer_name, phase1_target_layer = find_last_conv2d(phase1_model)
        print(f"Auto Phase-I target layer: {phase1_layer_name}", flush=True)
    else:
        phase1_layer_name = args.phase1_target_layer
        phase1_target_layer = get_module_by_path(phase1_model, args.phase1_target_layer)
        print(f"Phase-I target layer: {phase1_layer_name}", flush=True)

    try:
        phase2_target_layer = get_module_by_path(phase2_model, args.phase2_target_layer)
        print(f"Phase-II target layer: {args.phase2_target_layer}", flush=True)
    except Exception as e:
        print(f"ERROR: Could not find Phase-II target layer '{args.phase2_target_layer}'.", flush=True)
        print("Available modules containing 'change' or 'fusion':", flush=True)
        for name, _ in phase2_model.named_modules():
            if "change" in name or "fusion" in name:
                print(f"  {name}", flush=True)
        raise e

    phase1_cam_engine = GradCAM(phase1_model, phase1_target_layer)
    phase2_cam_engine = GradCAM(phase2_model, phase2_target_layer)

    processed = 0
    rows = []
    wanted_stems = set(args.stems) if args.stems else None

    print("Generating compact Grad-CAM figures...", flush=True)

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx < args.start_index:
            continue

        stem = batch["stem"][0] if isinstance(batch["stem"], (list, tuple)) else str(batch["stem"])

        if wanted_stems is not None and stem not in wanted_stems:
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

        # For localization CAM, use ground-truth building mask if available.
        # target5 > 0 means building in the xBD label.
        gt_building_mask = (target5 > 0).float()

        loc_cam, _ = compute_phase1_localization_cam(
            phase1_model=phase1_model,
            pre=pre,
            phase1_cam_engine=phase1_cam_engine,
            target_mask=gt_building_mask,
            threshold=phase1_threshold,
        )

        predicted_class_id = choose_predicted_class(
            final_pred=final_pred,
            damage_probs=damage_probs,
            mode=args.pred_class_mode,
        )
        predicted_class_name = CLASS_ID_TO_NAME[predicted_class_id]

        pred_class_cam = compute_phase2_predicted_class_cam(
            phase2_model=phase2_model,
            pre=pre,
            post=post,
            phase2_cam_engine=phase2_cam_engine,
            v7=v7,
            class_id=predicted_class_id,
            final_pred=final_pred,
            loc_pred=loc_pred,
        )

        loc_overlay = overlay_heatmap(pre_rgb, loc_cam, alpha=0.45)
        pred_class_overlay = overlay_heatmap(post_rgb, pred_class_cam, alpha=0.45)

        fig_path = args.output_dir / "figures" / f"{stem}_compact_gradcam.png"

        save_compact_grid(
            save_path=fig_path,
            pre_rgb=pre_rgb,
            post_rgb=post_rgb,
            gt_color=gt_color,
            pred_color=pred_color,
            loc_overlay=loc_overlay,
            pred_class_overlay=pred_class_overlay,
            pred_class_name=predicted_class_name,
            title=f"{stem} | HRTBDA v7-MSDF compact Grad-CAM",
        )

        rows.append(
            {
                "stem": stem,
                "figure": str(fig_path),
                "phase1_threshold": float(phase1_threshold),
                "phase1_target_layer": phase1_layer_name,
                "phase2_target_layer": args.phase2_target_layer,
                "predicted_class_cam": predicted_class_name,
                "postprocess_dilation": args.postprocess_dilation,
                "dilation_kernel": int(args.dilation_kernel),
            }
        )

        print(f"Wrote: {fig_path} | predicted-class CAM: {predicted_class_name}", flush=True)

        processed += 1
        if processed >= args.sample_count:
            break

    phase1_cam_engine.remove()
    phase2_cam_engine.remove()

    summary_path = args.output_dir / "compact_gradcam_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase1_checkpoint": str(args.phase1_checkpoint),
                "phase2_checkpoint": str(args.phase2_checkpoint),
                "phase1_threshold": float(phase1_threshold),
                "phase1_meta": phase1_meta,
                "phase1_target_layer": phase1_layer_name,
                "phase2_target_layer": args.phase2_target_layer,
                "pred_class_mode": args.pred_class_mode,
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