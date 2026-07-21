#!/usr/bin/env python3
"""Grad-CAM explanations for the fine-tuned HRTBDA-v5 Texas cascade.

This script does inference only. It loads the saved best Phase-I localization
and Phase-II damage checkpoints, runs the original test preprocessing, and
creates one compact explanation panel per pre/post image pair.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


CLASS_NAMES = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}

PALETTE = np.asarray(
    [
        [0, 0, 60],
        [0, 210, 0],
        [255, 255, 0],
        [255, 150, 0],
        [255, 0, 0],
    ],
    dtype=np.uint8,
)


def load_training_module(path: Path):
    name = "hrtbda_v5_xai_source"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import HRTBDA-v5 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint does not contain a 'model' state dict: {path}")
    return checkpoint


def checkpoint_int(checkpoint: Dict[str, Any], key: str, fallback: int) -> int:
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, dict) and key in saved_args:
        return int(saved_args[key])
    return int(fallback)


def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current = model
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


class GradCAM:
    def __init__(self, target_layer: torch.nn.Module):
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def clear(self) -> None:
        self.activations = None
        self.gradients = None

    def remove(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def heatmap(self, output_size: Tuple[int, int]) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        activations = self.activations[0] if isinstance(self.activations, (tuple, list)) else self.activations
        gradients = self.gradients[0] if isinstance(self.gradients, (tuple, list)) else self.gradients
        weights = gradients.detach().mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, output_size, mode="bilinear", align_corners=False)[0, 0]
        result = cam.float().cpu().numpy()
        result -= float(np.nanmin(result))
        maximum = float(np.nanmax(result))
        if maximum > 0:
            result /= maximum
        return result.astype(np.float32)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = tensor.detach().cpu().float().numpy() * std + mean
    image = np.clip(image.transpose(1, 2, 0), 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8)


def color_mask(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask.astype(np.int64), 0, 4)]


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = cv2.applyColorMap(np.rint(heatmap * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = image.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def concentration(heatmap: np.ndarray, mask: np.ndarray) -> float:
    if not np.asarray(mask, dtype=bool).any():
        return float("nan")
    total = float(np.asarray(heatmap, dtype=np.float64).sum())
    if total <= 1e-12:
        return float("nan")
    return float(np.asarray(heatmap, dtype=np.float64)[mask.astype(bool)].sum() / total)


def dominant_class(mask: torch.Tensor, probabilities: torch.Tensor) -> int:
    values = mask[0].detach().cpu().numpy()
    counts = [(class_id, int((values == class_id).sum())) for class_id in range(1, 5)]
    present = [item for item in counts if item[1] > 0]
    if present:
        return max(present, key=lambda item: item[1])[0]
    return int(probabilities.mean(dim=(0, 2, 3)).argmax().item()) + 1


@torch.no_grad()
def cascade_prediction(v5, phase1, phase2, pre, post, threshold: float, dilation: str, kernel: int):
    loc_logits = phase1(pre)
    loc_pred = (torch.sigmoid(loc_logits) > threshold).long()
    damage_logits = v5.get_damage_logits(phase2(pre, post))
    probabilities = torch.sigmoid(damage_logits)
    damage_pred = v5.damage_logits_to_pred(damage_logits)
    damage_pred = v5.apply_damage_dilation(damage_pred, loc_pred, mode=dilation, kernel_size=kernel)
    final_pred = torch.zeros_like(damage_pred)
    final_pred[loc_pred.bool()] = damage_pred[loc_pred.bool()]
    return loc_pred, final_pred, probabilities


def localization_cam(model, engine: GradCAM, pre: torch.Tensor, threshold: float):
    model.zero_grad(set_to_none=True)
    engine.clear()
    logits = model(pre)
    predicted_mask = (torch.sigmoid(logits) > threshold).float()
    score = (logits * predicted_mask).sum() / predicted_mask.sum().clamp_min(1.0)
    score.backward()
    heatmap = engine.heatmap(pre.shape[-2:])
    engine.clear()
    model.zero_grad(set_to_none=True)
    return heatmap


def damage_cam(model, v5, engine: GradCAM, pre, post, class_id: int, final_pred, loc_pred):
    model.zero_grad(set_to_none=True)
    engine.clear()
    logits = v5.get_damage_logits(model(pre, post))
    logit_map = logits[:, class_id - 1]
    target_mask = (final_pred == class_id).float()
    if target_mask.sum() < 1:
        target_mask = loc_pred.float()
    score = (logit_map * target_mask).sum() / target_mask.sum().clamp_min(1.0)
    score.backward()
    heatmap = engine.heatmap(pre.shape[-2:])
    engine.clear()
    model.zero_grad(set_to_none=True)
    return heatmap


def save_panel(path: Path, stem: str, class_name: str, pre, post, truth, prediction, loc_overlay, damage_overlay):
    panels = [
        ("Pre-disaster", pre),
        ("Post-disaster", post),
        ("Ground truth", color_mask(truth)),
        ("Cascade prediction", color_mask(prediction)),
        ("Phase I localization Grad-CAM", loc_overlay),
        (f"Phase II {class_name} Grad-CAM", damage_overlay),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"{stem} | HRTBDA-v5 Texas Tornadoes XAI", fontsize=14)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    root = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    experiment = root / "output/HRTBDA-v5-xBDInit-FullFineTune_TEXAS_TORNADOES_resetbest"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-script", type=Path, default=root / "transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py")
    parser.add_argument("--phase1-checkpoint", type=Path, default=experiment / "checkpoints/phase1_best.pt")
    parser.add_argument("--phase2-checkpoint", type=Path, default=experiment / "checkpoints/phase2_best.pt")
    parser.add_argument("--data-root", type=Path, default=Path("/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed"))
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=experiment / "xai_gradcam_test")
    parser.add_argument("--sample-count", type=int, default=0, help="0 processes the complete test set.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stems", nargs="*", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--phase1-threshold", type=float, default=None, help="Defaults to the checkpoint value (0.75 for this run).")
    parser.add_argument("--postprocess-dilation", choices=["none", "minor", "minor_major"], default="minor")
    parser.add_argument("--dilation-kernel", type=int, default=3)
    parser.add_argument("--phase1-target-layer", default="decoder.fuse.1.block.0")
    parser.add_argument("--phase2-target-layer", default="decoder.fuse.1.block.0")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model_script, args.phase1_checkpoint, args.phase2_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.data_root / args.test_split).is_dir():
        raise FileNotFoundError(args.data_root / args.test_split)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available.")

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "heatmaps").mkdir(exist_ok=True)

    print(f"Device: {device}", flush=True)
    v5 = load_training_module(args.model_script)
    phase1_ckpt = load_checkpoint(args.phase1_checkpoint, device)
    phase2_ckpt = load_checkpoint(args.phase2_checkpoint, device)

    base_channels = checkpoint_int(phase2_ckpt, "base_channels", 48)
    decoder_channels = checkpoint_int(phase2_ckpt, "decoder_channels", 128)
    window_size = checkpoint_int(phase2_ckpt, "window_size", 8)
    threshold = float(
        args.phase1_threshold
        if args.phase1_threshold is not None
        else phase1_ckpt.get("best_threshold", 0.5)
    )

    phase1 = v5.HRTBDAPhase1(base_channels, decoder_channels, window_size).to(device)
    phase2 = v5.HRTBDAPhase2(base_channels, decoder_channels, window_size, num_classes=4).to(device)
    phase1.load_state_dict(phase1_ckpt["model"], strict=True)
    phase2.load_state_dict(phase2_ckpt["model"], strict=True)
    phase1.eval()
    phase2.eval()

    dataset = v5.XBDHRTBDADataset(
        root=args.data_root,
        split=args.test_split,
        image_size=args.img_size,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    phase1_cam_engine = GradCAM(get_module_by_path(phase1, args.phase1_target_layer))
    phase2_cam_engine = GradCAM(get_module_by_path(phase2, args.phase2_target_layer))
    wanted = set(args.stems) if args.stems else None
    rows = []
    processed = 0

    print(f"Phase I checkpoint epoch: {phase1_ckpt.get('epoch')} | threshold: {threshold:.2f}", flush=True)
    print(f"Phase II checkpoint epoch: {phase2_ckpt.get('epoch')}", flush=True)
    print(f"Texas test samples: {len(dataset)}", flush=True)

    try:
        for index, batch in enumerate(loader):
            if index < args.start_index:
                continue
            stem = batch["stem"][0]
            if wanted is not None and stem not in wanted:
                continue

            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            target5 = batch["target5"].to(device, non_blocking=True)

            loc_pred, final_pred, probabilities = cascade_prediction(
                v5, phase1, phase2, pre, post, threshold, args.postprocess_dilation, args.dilation_kernel
            )
            class_id = dominant_class(final_pred, probabilities)

            loc_heatmap = localization_cam(phase1, phase1_cam_engine, pre, threshold)
            damage_heatmap = damage_cam(
                phase2, v5, phase2_cam_engine, pre, post, class_id, final_pred, loc_pred
            )

            pre_rgb = tensor_to_rgb(batch["pre"][0])
            post_rgb = tensor_to_rgb(batch["post"][0])
            truth = target5[0].detach().cpu().numpy()
            truth_valid = np.where((truth >= 0) & (truth <= 4), truth, 0).astype(np.uint8)
            prediction = final_pred[0].detach().cpu().numpy().astype(np.uint8)
            truth_building = (truth >= 1) & (truth <= 4)
            predicted_building = prediction > 0
            truth_class = truth == class_id
            predicted_class = prediction == class_id

            heatmap_dir = args.output_dir / "heatmaps" / stem
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            np.save(heatmap_dir / "phase1_localization.npy", loc_heatmap)
            np.save(heatmap_dir / f"phase2_{CLASS_NAMES[class_id]}.npy", damage_heatmap)
            cv2.imwrite(str(heatmap_dir / "phase1_localization.png"), np.rint(loc_heatmap * 255).astype(np.uint8))
            cv2.imwrite(str(heatmap_dir / f"phase2_{CLASS_NAMES[class_id]}.png"), np.rint(damage_heatmap * 255).astype(np.uint8))

            figure_path = args.output_dir / "figures" / f"{stem}_hrtbda_v5_gradcam.png"
            save_panel(
                figure_path,
                stem,
                CLASS_NAMES[class_id],
                pre_rgb,
                post_rgb,
                truth_valid,
                prediction,
                overlay(pre_rgb, loc_heatmap),
                overlay(post_rgb, damage_heatmap),
            )

            rows.append(
                {
                    "index": index,
                    "stem": stem,
                    "explained_class_id": class_id,
                    "explained_class": CLASS_NAMES[class_id],
                    "phase1_cam_in_true_buildings": concentration(loc_heatmap, truth_building),
                    "phase1_cam_in_predicted_buildings": concentration(loc_heatmap, predicted_building),
                    "phase2_cam_in_true_explained_class": concentration(damage_heatmap, truth_class),
                    "phase2_cam_in_predicted_explained_class": concentration(damage_heatmap, predicted_class),
                    "true_building_pixels": int(truth_building.sum()),
                    "predicted_building_pixels": int(predicted_building.sum()),
                    "true_explained_class_pixels": int(truth_class.sum()),
                    "predicted_explained_class_pixels": int(predicted_class.sum()),
                    "figure": str(figure_path),
                }
            )
            processed += 1
            print(f"[{processed}] {stem}: explained {CLASS_NAMES[class_id]}", flush=True)
            if args.sample_count > 0 and processed >= args.sample_count:
                break
    finally:
        phase1_cam_engine.remove()
        phase2_cam_engine.remove()

    if not rows:
        raise RuntimeError("No test samples were selected.")

    csv_path = args.output_dir / "xai_metrics_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "phase1_cam_in_true_buildings",
        "phase1_cam_in_predicted_buildings",
        "phase2_cam_in_true_explained_class",
        "phase2_cam_in_predicted_explained_class",
    ]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        aggregate[name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "valid_samples": int(np.isfinite(values).sum()),
        }

    summary = {
        "method": "Grad-CAM",
        "experiment": "HRTBDA-v5 xBD-initialized full fine-tune on Texas Tornadoes",
        "phase1_checkpoint": str(args.phase1_checkpoint),
        "phase1_epoch": int(phase1_ckpt.get("epoch", -1)),
        "phase1_threshold": threshold,
        "phase2_checkpoint": str(args.phase2_checkpoint),
        "phase2_epoch": int(phase2_ckpt.get("epoch", -1)),
        "data_root": str(args.data_root),
        "test_split": args.test_split,
        "processed_samples": processed,
        "phase1_target_layer": args.phase1_target_layer,
        "phase2_target_layer": args.phase2_target_layer,
        "postprocess_dilation": args.postprocess_dilation,
        "dilation_kernel": args.dilation_kernel,
        "aggregate_metrics": aggregate,
    }
    summary_path = args.output_dir / "xai_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote figures: {args.output_dir / 'figures'}", flush=True)
    print(f"Wrote raw heatmaps: {args.output_dir / 'heatmaps'}", flush=True)
    print(f"Wrote per-image metrics: {csv_path}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
