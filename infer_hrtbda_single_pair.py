#!/usr/bin/env python3
"""
Single-pair inference wrapper for DisasterAdaptiveNet HRTBDA cascade.

This version supports the correct two-stage setup:

  Phase I:
    pre image -> building/localization logits -> building mask

  Phase II:
    pre/post image -> 4 foreground damage logits:
      0 = no damage
      1 = minor
      2 = major
      3 = destroyed

  Final dashboard mask:
    0 outside Phase I building mask
    1 no damage
    2 minor
    3 major
    4 destroyed

The backend can still call this script using the existing standardized CLI:
  --checkpoint ...
  --pre_image ...
  --post_image ...
  --gt_mask ...
  --building_mask_output ...
  --damage_mask_output ...
  --damage_index_output ...
  --overlay_output ...
  --summary_json ...
  --summary_csv ...

Extra cascade paths are read from environment variables:
  HRTBDA_PHASE1_CHECKPOINT
  HRTBDA_PHASE2_CHECKPOINT
  HRTBDA_PHASE1_MODEL_MODULE
  HRTBDA_PHASE1_MODEL_CLASS
  HRTBDA_PHASE2_MODEL_MODULE
  HRTBDA_PHASE2_MODEL_CLASS
  HRTBDA_PHASE1_THRESHOLD
  HRTBDA_POSTPROCESS_DILATION
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch


CLASS_NAMES = {
    1: "no_damage",
    2: "minor",
    3: "major",
    4: "destroyed",
}

CLASS_RGB = {
    0: (0, 0, 0),          # background - black
    1: (34, 197, 94),     # no damage - green
    2: (234, 179, 8),     # minor - yellow
    3: (249, 115, 22),    # major - orange
    4: (220, 38, 38),     # destroyed - red
}

DEFAULT_PHASE1_MODEL_MODULE = "transformer.scripts.train_xbd_hrtbda_v2_cascaded_phase1mask"
DEFAULT_PHASE1_MODEL_CLASS = "HRTBDAPhase1"
DEFAULT_PHASE2_MODEL_MODULE = "transformer.scripts.train_xbd_hrtbda_v2_cascaded_phase1mask"
DEFAULT_PHASE2_MODEL_CLASS = "HRTBDAPhase2"


def add_project_to_path() -> None:
    current = Path(__file__).resolve()
    root = current.parent

    candidates = [
        root,
        root / "src",
        root / "models",
    ]

    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value if value is not None else default


def env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = os.getenv(name, "")
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def resize_if_needed(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> np.ndarray:
    if pre_bgr.shape[:2] == post_bgr.shape[:2]:
        return post_bgr

    print(
        "[WARNING] Pre and post images have different sizes. "
        "Resizing post image to match pre image.",
        flush=True,
    )

    return cv2.resize(
        post_bgr,
        (pre_bgr.shape[1], pre_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def pad_to_factor(image: np.ndarray, factor: int = 32) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    new_h = int(np.ceil(h / factor) * factor)
    new_w = int(np.ceil(w / factor) * factor)

    pad_h = new_h - h
    pad_w = new_w - w

    if pad_h == 0 and pad_w == 0:
        return image, (h, w)

    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded, (h, w)


def normalize_single_rgb(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Match the training dataset normalization:
      RGB
      /255
      ImageNet mean/std
      CHW tensor
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb = (rgb - mean) / std
    rgb = np.transpose(rgb, (2, 0, 1))

    return torch.from_numpy(rgb).unsqueeze(0).float()


def normalize_ground_truth_mask(mask_path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if gt_mask is None:
        raise FileNotFoundError(f"Could not read ground-truth mask: {mask_path}")

    if gt_mask.ndim == 3:
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

    gt_mask = gt_mask.astype(np.uint8)

    if gt_mask.shape[:2] != target_shape:
        print(
            f"[WARNING] Ground-truth mask shape {gt_mask.shape[:2]} does not match "
            f"prediction shape {target_shape}. Resizing with nearest-neighbor.",
            flush=True,
        )

        gt_mask = cv2.resize(
            gt_mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    unique_values = set(np.unique(gt_mask).tolist())
    allowed_values = {0, 1, 2, 3, 4, 255}

    if not unique_values.issubset(allowed_values):
        print(
            "[WARNING] Ground-truth mask has unexpected values:",
            sorted(unique_values),
            flush=True,
        )
        print(
            "[WARNING] Expected class IDs: 0, 1, 2, 3, 4, 255.",
            flush=True,
        )

    if unique_values.issubset({0, 255}):
        print(
            "[WARNING] Ground-truth mask appears binary. "
            "Per-class damage F1 needs class labels 0, 1, 2, 3, 4, 255.",
            flush=True,
        )

    return gt_mask


def harmonic_mean(values) -> float:
    values = [float(v) for v in values]
    return len(values) / sum((v + 1e-6) ** -1 for v in values)


def compute_f1_scores(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    class_metric_names = {
        1: "no_damage_f1",
        2: "minor_f1",
        3: "major_f1",
        4: "destroyed_f1",
    }

    valid = gt_mask != 255
    scores: Dict[str, float] = {}

    per_class = []

    for class_id, metric_name in class_metric_names.items():
        pred_class = (pred_mask == class_id) & valid
        gt_class = (gt_mask == class_id) & valid

        tp = np.logical_and(pred_class, gt_class).sum()
        fp = np.logical_and(pred_class, ~gt_class).sum()
        fn = np.logical_and(~pred_class, gt_class).sum()

        f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-9)
        f1 = float(f1)

        scores[metric_name] = round(f1, 4)
        per_class.append(f1)

    macro_f1 = float(np.mean(per_class))
    damage_f1 = harmonic_mean(per_class)

    scores["macro_f1"] = round(macro_f1, 4)
    scores["damage_f1"] = round(float(damage_f1), 4)

    return scores


def compute_localization_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    valid = gt_mask != 255

    pred_building = (pred_mask > 0) & valid
    gt_building = np.isin(gt_mask, [1, 2, 3, 4]) & valid

    tp = np.logical_and(pred_building, gt_building).sum()
    fp = np.logical_and(pred_building, ~gt_building).sum()
    fn = np.logical_and(~pred_building, gt_building).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-9)

    return {
        "localization_precision": round(float(precision), 4),
        "localization_recall": round(float(recall), 4),
        "localization_f1": round(float(f1), 4),
    }


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, rgb in CLASS_RGB.items():
        color[mask == class_id] = rgb

    return color


def make_overlay(
    post_bgr: np.ndarray,
    color_mask_rgb: np.ndarray,
    pred_mask: np.ndarray,
    overlay_output: str,
) -> None:
    post_rgb = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB)

    if color_mask_rgb.shape[:2] != post_rgb.shape[:2]:
        color_mask_rgb = cv2.resize(
            color_mask_rgb,
            (post_rgb.shape[1], post_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if pred_mask.shape[:2] != post_rgb.shape[:2]:
        pred_mask = cv2.resize(
            pred_mask,
            (post_rgb.shape[1], post_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_pixels = pred_mask > 0
    overlay = post_rgb.copy()

    if mask_pixels.any():
        overlay[mask_pixels] = (
            0.55 * post_rgb[mask_pixels] + 0.45 * color_mask_rgb[mask_pixels]
        ).astype(np.uint8)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_output), overlay_bgr)


def summarize_by_connected_components(pred_mask: np.ndarray, min_area: int = 20) -> Dict[str, Any]:
    building_binary = (pred_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        building_binary,
        connectivity=8,
    )

    counts = {
        "no_damage": 0,
        "minor": 0,
        "major": 0,
        "destroyed": 0,
    }

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        component_pixels = pred_mask[labels == label_id]
        component_pixels = component_pixels[component_pixels > 0]

        if component_pixels.size == 0:
            continue

        class_id = int(np.bincount(component_pixels, minlength=5).argmax())
        class_name = CLASS_NAMES.get(class_id)

        if class_name:
            counts[class_name] += 1

    total = sum(counts.values())

    if total == 0:
        percentages = {
            "no_damage": 0,
            "minor": 0,
            "major": 0,
            "destroyed": 0,
        }
    else:
        percentages = {
            key: round((value / total) * 100, 2)
            for key, value in counts.items()
        }

    return {
        "total_buildings": total,
        "damage_counts": counts,
        "damage_percentages": percentages,
        "counting_method": "connected_components_majority_class",
    }


def save_csv(csv_path: str, summary: Dict[str, Any]) -> None:
    counts = summary["damage_counts"]
    percentages = summary["damage_percentages"]
    metrics = summary.get("metrics", {})
    runtime = summary.get("runtime", {})
    model_info = summary.get("model_info", {})

    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["section", "name", "value"])

        for key, value in counts.items():
            writer.writerow(["damage_count", key, value])

        for key, value in percentages.items():
            writer.writerow(["damage_percentage", key, value])

        for key, value in metrics.items():
            writer.writerow(["metric", key, value])

        for key, value in runtime.items():
            writer.writerow(["runtime", key, value])

        for key, value in model_info.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            writer.writerow(["model_info", key, value])


def inspect_checkpoint(checkpoint_path: str) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    print("========== CHECKPOINT INSPECTION ==========")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print("Top-level keys:")
        for key in ckpt.keys():
            value = ckpt[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: tensor {tuple(value.shape)}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            else:
                print(f"  {key}: {type(value)}")

        state = extract_state_dict(ckpt)
        if state is not None:
            print(f"\nState dict keys: {len(state)}")
            print("First 40 state_dict keys:")
            for k in list(state.keys())[:40]:
                v = state[k]
                shape = tuple(v.shape) if hasattr(v, "shape") else ""
                print(f"  {k} {shape}")

            print("\nOutput-like keys:")
            for k, v in state.items():
                if any(s in k.lower() for s in ["out.weight", "out.bias", "decoder.out", "head", "classifier"]):
                    shape = tuple(v.shape) if hasattr(v, "shape") else ""
                    print(f"  {k} {shape}")

        if "args" in ckpt:
            print("\nCheckpoint args:")
            print(json.dumps(ckpt["args"], indent=2, default=str))

    elif isinstance(ckpt, torch.nn.Module):
        print("Checkpoint is a full torch.nn.Module.")
        print(ckpt)

    print("===========================================")


def extract_state_dict(ckpt: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ckpt, torch.nn.Module):
        return None

    if not isinstance(ckpt, dict):
        return None

    possible_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "module",
        "ema_state_dict",
    ]

    for key in possible_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    if all(isinstance(k, str) for k in ckpt.keys()):
        tensor_values = [v for v in ckpt.values() if isinstance(v, torch.Tensor)]
        if len(tensor_values) > 0:
            return ckpt

    return None


def clean_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ["module.", "model.", "net.", "network."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def find_model_class(module_name: str, class_name: str):
    if not module_name:
        raise ValueError("Missing model module name.")
    if not class_name:
        raise ValueError("Missing model class name.")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    return cls


def checkpoint_args(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        return dict(ckpt["args"])
    return {}


def instantiate_model(
    module_name: str,
    class_name: str,
    ckpt_args: Dict[str, Any],
    role: str,
):
    cls = find_model_class(module_name, class_name)

    signature = inspect.signature(cls.__init__)
    params = signature.parameters

    candidate_kwargs = {
        "base_channels": ckpt_args.get("base_channels", 48),
        "decoder_channels": ckpt_args.get("decoder_channels", 128),
        "window_size": ckpt_args.get("window_size", 8),
        "img_size": ckpt_args.get("img_size", 1024),
        "image_size": ckpt_args.get("img_size", 1024),
        "num_classes": 4 if role == "phase2" else 1,
        "out_channels": 4 if role == "phase2" else 1,
        "in_channels": 3,
        "in_chans": 3,
        "in_ch": 3,
    }

    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in params
    }

    print(
        f"[MODEL] Instantiating {role}: {module_name}.{class_name} "
        f"with kwargs={kwargs}",
        flush=True,
    )

    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            f"Could import {module_name}.{class_name}, but could not instantiate it.\n"
            f"Constructor signature: {signature}\n"
            f"Tried kwargs: {kwargs}\n"
            "Update instantiate_model() if this architecture needs different arguments."
        ) from exc


def load_model_from_checkpoint(
    checkpoint_path: str,
    module_name: str,
    class_name: str,
    device: torch.device,
    role: str,
    strict: bool = False,
):
    checkpoint_path = str(checkpoint_path)

    print(f"[MODEL] Loading {role} checkpoint: {checkpoint_path}", flush=True)

    if not checkpoint_path:
        raise ValueError(f"Missing checkpoint path for {role}.")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"{role} checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        print(f"[MODEL] {role} checkpoint contains a full torch.nn.Module.", flush=True)
        model = ckpt.to(device)
        model.eval()
        return model, ckpt

    state_dict = extract_state_dict(ckpt)

    if state_dict is None:
        raise RuntimeError(
            f"Could not find a state_dict in {role} checkpoint: {checkpoint_path}. "
            "Run --inspect_checkpoint to inspect it."
        )

    args_from_ckpt = checkpoint_args(ckpt)
    model = instantiate_model(
        module_name=module_name,
        class_name=class_name,
        ckpt_args=args_from_ckpt,
        role=role,
    )

    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    print(f"[MODEL] Loaded {role} state_dict with strict={strict}", flush=True)
    print(f"[MODEL] {role} missing keys: {len(missing)}", flush=True)
    print(f"[MODEL] {role} unexpected keys: {len(unexpected)}", flush=True)

    if len(missing) > 0:
        print(f"[MODEL] {role} sample missing keys: {list(missing)[:20]}", flush=True)

    if len(unexpected) > 0:
        print(f"[MODEL] {role} sample unexpected keys: {list(unexpected)[:20]}", flush=True)

    model = model.to(device)
    model.eval()

    return model, ckpt


def extract_tensor_from_model_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        if not tensors:
            raise RuntimeError("Model output tuple/list contained no tensors.")

        tensors = sorted(tensors, key=lambda t: t.ndim, reverse=True)
        return tensors[0]

    if isinstance(output, dict):
        preferred_keys = [
            "logits",
            "out",
            "output",
            "pred",
            "prediction",
            "damage",
            "damage_logits",
            "mask",
            "seg",
            "segmentation",
        ]

        for key in preferred_keys:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]

        tensors = [value for value in output.values() if isinstance(value, torch.Tensor)]

        if not tensors:
            raise RuntimeError("Model output dict contained no tensors.")

        tensors = sorted(tensors, key=lambda t: t.ndim, reverse=True)
        return tensors[0]

    raise RuntimeError(f"Unsupported model output type: {type(output)}")


def phase1_logits_to_mask(
    phase1_output: Any,
    original_shape: Tuple[int, int],
    threshold: float,
) -> Tuple[np.ndarray, torch.Tensor]:
    logits = extract_tensor_from_model_output(phase1_output)

    print(f"[PHASE1] Output tensor shape: {tuple(logits.shape)}", flush=True)

    if logits.ndim == 4:
        if logits.shape[1] == 1:
            logits_2d = logits[0, 0]
        else:
            logits_2d = logits[0, 0]
    elif logits.ndim == 3:
        if logits.shape[0] == 1:
            logits_2d = logits[0]
        else:
            logits_2d = logits[0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise RuntimeError(f"Unsupported Phase I output shape: {tuple(logits.shape)}")

    prob = torch.sigmoid(logits_2d)
    mask = (prob > float(threshold)).detach().cpu().numpy().astype(np.uint8)

    original_h, original_w = original_shape
    mask = mask[:original_h, :original_w]

    return mask, logits


def phase2_logits_to_damage(
    phase2_output: Any,
    original_shape: Tuple[int, int],
) -> Tuple[np.ndarray, torch.Tensor]:
    logits = extract_tensor_from_model_output(phase2_output)

    print(f"[PHASE2] Output tensor shape: {tuple(logits.shape)}", flush=True)

    if logits.ndim == 4:
        logits_3d = logits[0]
    elif logits.ndim == 3:
        logits_3d = logits
    else:
        raise RuntimeError(f"Unsupported Phase II output shape: {tuple(logits.shape)}")

    channels = logits_3d.shape[0]

    if channels == 4:
        damage = torch.argmax(logits_3d, dim=0).detach().cpu().numpy().astype(np.uint8) + 1
    elif channels == 5:
        damage = torch.argmax(logits_3d, dim=0).detach().cpu().numpy().astype(np.uint8)
    else:
        raise RuntimeError(
            f"Phase II should output 4 foreground damage classes or 5 full classes. "
            f"Got channels={channels}."
        )

    original_h, original_w = original_shape
    damage = damage[:original_h, :original_w]
    damage = damage.astype(np.uint8)

    damage[~np.isin(damage, [0, 1, 2, 3, 4])] = 0

    return damage, logits


def apply_damage_postprocess(
    pred_mask: np.ndarray,
    building_mask: np.ndarray,
    mode: str,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Optional light post-processing.

    mode:
      none  = no change
      minor = conservatively dilate minor-damage predictions inside Phase I building mask

    This is intentionally conservative for dashboard use.
    """
    mode = (mode or "none").lower().strip()

    if mode in {"", "none", "off", "false", "0"}:
        return pred_mask

    out = pred_mask.copy()
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    building = building_mask.astype(bool)

    if mode == "minor":
        minor = (out == 2).astype(np.uint8)
        dilated_minor = cv2.dilate(minor, kernel, iterations=1).astype(bool)

        # Only spread minor into Phase I building pixels currently predicted no-damage.
        update = dilated_minor & building & (out == 1)
        out[update] = 2

        return out

    print(f"[WARNING] Unknown postprocess mode '{mode}'. Skipping.", flush=True)
    return pred_mask


def run_cascade(
    args: argparse.Namespace,
    device: torch.device,
    pre_tensor: torch.Tensor,
    post_tensor: torch.Tensor,
    original_shape: Tuple[int, int],
):
    phase1_checkpoint = args.phase1_checkpoint or env_str("HRTBDA_PHASE1_CHECKPOINT", "")
    phase2_checkpoint = args.phase2_checkpoint or env_str("HRTBDA_PHASE2_CHECKPOINT", "") or args.checkpoint

    phase1_module = args.phase1_model_module
    phase1_class = args.phase1_model_class
    phase2_module = args.phase2_model_module
    phase2_class = args.phase2_model_class

    if not phase1_checkpoint:
        raise ValueError(
            "Missing Phase I checkpoint. Set HRTBDA_PHASE1_CHECKPOINT or pass --phase1_checkpoint."
        )

    if not phase2_checkpoint:
        raise ValueError(
            "Missing Phase II checkpoint. Set HRTBDA_PHASE2_CHECKPOINT or pass --phase2_checkpoint/--checkpoint."
        )

    phase1_model, phase1_ckpt = load_model_from_checkpoint(
        checkpoint_path=phase1_checkpoint,
        module_name=phase1_module,
        class_name=phase1_class,
        device=device,
        role="phase1",
        strict=False,
    )

    phase2_model, phase2_ckpt = load_model_from_checkpoint(
        checkpoint_path=phase2_checkpoint,
        module_name=phase2_module,
        class_name=phase2_class,
        device=device,
        role="phase2",
        strict=False,
    )

    phase1_threshold = args.phase1_threshold
    if phase1_threshold is None:
        phase1_threshold = env_float("HRTBDA_PHASE1_THRESHOLD", None)

    if phase1_threshold is None and isinstance(phase1_ckpt, dict):
        phase1_threshold = float(phase1_ckpt.get("best_threshold", args.threshold))

    if phase1_threshold is None:
        phase1_threshold = float(args.threshold)

    print(f"[CASCADE] Phase I threshold: {phase1_threshold}", flush=True)

    autocast_ctx = (
        torch.cuda.amp.autocast()
        if args.amp and device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            phase1_output = phase1_model(pre_tensor)
            phase2_output = phase2_model(pre_tensor, post_tensor)

    loc_mask, phase1_logits = phase1_logits_to_mask(
        phase1_output=phase1_output,
        original_shape=original_shape,
        threshold=float(phase1_threshold),
    )

    damage_pred, phase2_logits = phase2_logits_to_damage(
        phase2_output=phase2_output,
        original_shape=original_shape,
    )

    final_pred = np.zeros_like(damage_pred, dtype=np.uint8)
    final_pred[loc_mask.astype(bool)] = damage_pred[loc_mask.astype(bool)]

    final_pred = apply_damage_postprocess(
        pred_mask=final_pred,
        building_mask=loc_mask,
        mode=args.postprocess_dilation,
        kernel_size=args.dilation_kernel,
    )

    final_pred[~np.isin(final_pred, [0, 1, 2, 3, 4])] = 0

    cascade_info = {
        "phase1_checkpoint": str(phase1_checkpoint),
        "phase2_checkpoint": str(phase2_checkpoint),
        "phase1_threshold": float(phase1_threshold),
        "phase1_model_module": phase1_module,
        "phase1_model_class": phase1_class,
        "phase2_model_module": phase2_module,
        "phase2_model_class": phase2_class,
        "phase1_epoch": int(phase1_ckpt.get("epoch", -1)) if isinstance(phase1_ckpt, dict) else None,
        "phase2_epoch": int(phase2_ckpt.get("epoch", -1)) if isinstance(phase2_ckpt, dict) else None,
        "phase1_best_metric": float(phase1_ckpt.get("best_metric", -1.0)) if isinstance(phase1_ckpt, dict) else None,
        "phase2_best_metric": float(phase2_ckpt.get("best_metric", -1.0)) if isinstance(phase2_ckpt, dict) else None,
        "postprocess_dilation": args.postprocess_dilation,
        "phase1_output_tensor_shape": list(phase1_logits.shape),
        "phase2_output_tensor_shape": list(phase2_logits.shape),
    }

    return final_pred, loc_mask, cascade_info


def save_outputs(
    args: argparse.Namespace,
    pred_np: np.ndarray,
    loc_mask: np.ndarray,
    post_bgr: np.ndarray,
) -> None:
    output_parent = Path(args.building_mask_output).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    building_mask = (loc_mask.astype(np.uint8)) * 255
    color_mask_rgb = colorize_mask(pred_np)
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(args.building_mask_output), building_mask)
    cv2.imwrite(str(args.damage_mask_output), color_mask_bgr)
    cv2.imwrite(str(args.damage_index_output), pred_np)

    make_overlay(
        post_bgr=post_bgr,
        color_mask_rgb=color_mask_rgb,
        pred_mask=pred_np,
        overlay_output=args.overlay_output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-pair inference wrapper for DisasterAdaptiveNet HRTBDA cascade."
    )

    parser.add_argument("--pre_image")
    parser.add_argument("--post_image")
    parser.add_argument("--gt_mask")

    # Existing backend-compatible args.
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.40)

    parser.add_argument("--building_mask_output")
    parser.add_argument("--damage_mask_output")
    parser.add_argument("--damage_index_output")
    parser.add_argument("--overlay_output")
    parser.add_argument("--summary_json")
    parser.add_argument("--summary_csv")

    parser.add_argument("--model_name", default="High Resolution Transformer Building Damage Detection")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--min_area", type=int, default=20)

    # New cascade args. These default to environment variables.
    parser.add_argument(
        "--phase1_checkpoint",
        default=env_str("HRTBDA_PHASE1_CHECKPOINT", ""),
    )
    parser.add_argument(
        "--phase2_checkpoint",
        default=env_str("HRTBDA_PHASE2_CHECKPOINT", ""),
    )

    parser.add_argument(
        "--phase1_model_module",
        default=env_str(
            "HRTBDA_PHASE1_MODEL_MODULE",
            env_str("HRTBDA_MODEL_MODULE", DEFAULT_PHASE1_MODEL_MODULE),
        ),
    )
    parser.add_argument(
        "--phase1_model_class",
        default=env_str(
            "HRTBDA_PHASE1_MODEL_CLASS",
            env_str("HRTBDA_MODEL_CLASS", DEFAULT_PHASE1_MODEL_CLASS),
        ),
    )
    parser.add_argument(
        "--phase2_model_module",
        default=env_str(
            "HRTBDA_PHASE2_MODEL_MODULE",
            env_str("HRTBDA_MODEL_MODULE", DEFAULT_PHASE2_MODEL_MODULE),
        ),
    )
    parser.add_argument(
        "--phase2_model_class",
        default=env_str(
            "HRTBDA_PHASE2_MODEL_CLASS",
            env_str("HRTBDA_MODEL_CLASS", DEFAULT_PHASE2_MODEL_CLASS),
        ),
    )

    parser.add_argument(
        "--phase1_threshold",
        type=float,
        default=env_float("HRTBDA_PHASE1_THRESHOLD", None),
    )

    parser.add_argument(
        "--postprocess_dilation",
        default=env_str("HRTBDA_POSTPROCESS_DILATION", "none"),
        choices=["none", "minor"],
    )
    parser.add_argument(
        "--dilation_kernel",
        type=int,
        default=int(env_str("HRTBDA_DILATION_KERNEL", "3")),
    )

    parser.add_argument("--inspect_checkpoint", action="store_true")
    parser.add_argument("--inspect_cascade", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    add_project_to_path()

    if args.inspect_checkpoint:
        inspect_checkpoint(args.checkpoint)
        return

    if args.inspect_cascade:
        print("========== CASCADE CONFIG ==========")
        print(f"checkpoint:              {args.checkpoint}")
        print(f"phase1_checkpoint:       {args.phase1_checkpoint}")
        print(f"phase2_checkpoint:       {args.phase2_checkpoint or args.checkpoint}")
        print(f"phase1_model_module:     {args.phase1_model_module}")
        print(f"phase1_model_class:      {args.phase1_model_class}")
        print(f"phase2_model_module:     {args.phase2_model_module}")
        print(f"phase2_model_class:      {args.phase2_model_class}")
        print(f"phase1_threshold:        {args.phase1_threshold}")
        print(f"postprocess_dilation:    {args.postprocess_dilation}")
        print("====================================")
        return

    required = [
        "pre_image",
        "post_image",
        "gt_mask",
        "building_mask_output",
        "damage_mask_output",
        "damage_index_output",
        "overlay_output",
        "summary_json",
        "summary_csv",
    ]

    missing = [name for name in required if not getattr(args, name)]

    if missing:
        raise ValueError(f"Missing required arguments for inference: {missing}")

    start_time = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)

    pre_bgr = read_image(args.pre_image)
    post_bgr = read_image(args.post_image)
    post_bgr = resize_if_needed(pre_bgr, post_bgr)

    original_h, original_w = pre_bgr.shape[:2]
    original_shape = (original_h, original_w)

    pre_padded, _ = pad_to_factor(pre_bgr, 32)
    post_padded, _ = pad_to_factor(post_bgr, 32)

    pre_tensor = normalize_single_rgb(pre_padded).to(device)
    post_tensor = normalize_single_rgb(post_padded).to(device)

    print(f"[INPUT] Original shape: {original_shape}", flush=True)
    print(f"[INPUT] Padded pre tensor: {tuple(pre_tensor.shape)}", flush=True)
    print(f"[INPUT] Padded post tensor: {tuple(post_tensor.shape)}", flush=True)

    pred_np, loc_mask, cascade_info = run_cascade(
        args=args,
        device=device,
        pre_tensor=pre_tensor,
        post_tensor=post_tensor,
        original_shape=original_shape,
    )

    gt_mask = normalize_ground_truth_mask(
        mask_path=args.gt_mask,
        target_shape=pred_np.shape[:2],
    )

    localization_metrics = compute_localization_metrics(pred_np, gt_mask)
    damage_metrics = compute_f1_scores(pred_np, gt_mask)

    damage_f1 = float(damage_metrics.get("damage_f1", 0.0))
    localization_f1 = float(localization_metrics.get("localization_f1", 0.0))
    overall_score = 0.3 * localization_f1 + 0.7 * damage_f1

    metrics = {
        **localization_metrics,
        **damage_metrics,
        "overall_score": round(float(overall_score), 4),
        "metrics_note": (
            "Image-specific F1 calculated using uploaded ground-truth post-disaster mask. "
            "Localization comes from Phase I mask. Damage classes come from Phase II foreground classifier."
        ),
    }

    save_outputs(
        args=args,
        pred_np=pred_np,
        loc_mask=loc_mask,
        post_bgr=post_bgr,
    )

    summary = summarize_by_connected_components(
        pred_mask=pred_np,
        min_area=args.min_area,
    )

    total_seconds = time.perf_counter() - start_time

    summary["metrics"] = metrics

    summary["runtime"] = {
        "model_name": args.model_name,
        "model_family": "hrtbda_cascade",
        "device": str(device),
        "model_total_seconds": round(float(total_seconds), 3),
    }

    summary["model_info"] = {
        "model_name": args.model_name,
        "model_family": "hrtbda_cascade",
        "backend_checkpoint_arg": str(args.checkpoint),
        "checkpoint_dir": str(args.checkpoint_dir),
        "threshold_arg": args.threshold,
        "cascade": cascade_info,
        "prediction_unique_values": sorted(np.unique(pred_np).astype(int).tolist()),
        "phase1_mask_unique_values": sorted(np.unique(loc_mask).astype(int).tolist()),
        "gt_mask_unique_values": sorted(np.unique(gt_mask).astype(int).tolist()),
        "class_map": {
            "0": "background",
            "1": "no_damage",
            "2": "minor",
            "3": "major",
            "4": "destroyed",
            "255": "ignore/unknown in ground truth only",
        },
        "note": (
            "This script runs the HRTBDA two-phase cascade: Phase I localization mask "
            "plus Phase II 4-class foreground damage prediction."
        ),
    }

    with open(str(args.summary_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_csv(args.summary_csv, summary)

    print("[SUMMARY]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()