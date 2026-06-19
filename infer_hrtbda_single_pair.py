#!/usr/bin/env python3

import argparse
import csv
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

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


def add_project_to_path():
    current = Path(__file__).resolve()
    root = current.parent

    candidates = [
        root,
        root / "src",
        root / "models",
    ]

    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)

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


def pad_to_factor(image: np.ndarray, factor: int = 32):
    h, w = image.shape[:2]
    new_h = int(np.ceil(h / factor) * factor)
    new_w = int(np.ceil(w / factor) * factor)

    pad_h = new_h - h
    pad_w = new_w - w

    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded, (h, w)


def normalize_6ch(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> torch.Tensor:
    """
    Returns tensor shape [1, 6, H, W].

    Uses ImageNet-style normalization for RGB channels. If your
    DisasterAdaptiveNet training used a different normalization, update here.
    """
    pre_rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    post_rgb = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    x = np.concatenate([pre_rgb, post_rgb], axis=2)

    mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32)

    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))

    return torch.from_numpy(x).unsqueeze(0).float()


def normalize_single_rgb(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb = (rgb - mean) / std
    rgb = np.transpose(rgb, (2, 0, 1))

    return torch.from_numpy(rgb).unsqueeze(0).float()


def normalize_ground_truth_mask(mask_path: str, target_shape) -> np.ndarray:
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

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
            "Per-class F1 needs class labels 0, 1, 2, 3, 4, 255.",
            flush=True,
        )

    return gt_mask


def compute_f1_scores(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    class_names = {
        1: "no_damage_f1",
        2: "minor_f1",
        3: "major_f1",
        4: "destroyed_f1",
    }

    valid = gt_mask != 255
    scores = {}

    for class_id, metric_name in class_names.items():
        pred_class = (pred_mask == class_id) & valid
        gt_class = (gt_mask == class_id) & valid

        tp = np.logical_and(pred_class, gt_class).sum()
        fp = np.logical_and(pred_class, ~gt_class).sum()
        fn = np.logical_and(~pred_class, gt_class).sum()

        f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-9)
        scores[metric_name] = round(float(f1), 4)

    scores["macro_f1"] = round(
        float(
            np.mean(
                [
                    scores["no_damage_f1"],
                    scores["minor_f1"],
                    scores["major_f1"],
                    scores["destroyed_f1"],
                ]
            )
        ),
        4,
    )

    return scores


def compute_localization_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
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
):
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


def summarize_by_connected_components(pred_mask: np.ndarray, min_area: int = 20) -> dict:
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


def save_csv(csv_path: str, summary: dict):
    counts = summary["damage_counts"]
    percentages = summary["damage_percentages"]
    metrics = summary.get("metrics", {})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["section", "name", "value"])

        for key, value in counts.items():
            writer.writerow(["damage_count", key, value])

        for key, value in percentages.items():
            writer.writerow(["damage_percentage", key, value])

        for key, value in metrics.items():
            writer.writerow(["metric", key, value])


def inspect_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    print("========== CHECKPOINT INSPECTION ==========")
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

        for possible_key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
            "ema_state_dict",
        ]:
            if possible_key in ckpt:
                print(f"\nFound possible model key: {possible_key}")
                value = ckpt[possible_key]
                print(f"Type: {type(value)}")

                if isinstance(value, dict):
                    sample_keys = list(value.keys())[:30]
                    print("Sample state_dict keys:")
                    for k in sample_keys:
                        print(f"  {k}")

    elif isinstance(ckpt, torch.nn.Module):
        print("Checkpoint is a full torch.nn.Module.")
        print(ckpt)

    print("===========================================")


def extract_state_dict(ckpt):
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

    # Sometimes checkpoint itself is a state_dict.
    if all(isinstance(k, str) for k in ckpt.keys()):
        tensor_values = [v for v in ckpt.values() if isinstance(v, torch.Tensor)]
        if len(tensor_values) > 0:
            return ckpt

    return None


def clean_state_dict_keys(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ["module.", "model.", "net.", "network."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def find_model_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def instantiate_model_from_args(args):
    """
    This is the only model-specific part.

    If your checkpoint is a state_dict, the script needs to know the exact
    DisasterAdaptiveNet model class. You can provide it using:
      --model_module some.python.module
      --model_class SomeClass

    Example:
      --model_module models.hrtbda
      --model_class HRTBDA
    """
    if not args.model_module or not args.model_class:
        return None

    cls = find_model_class(args.model_module, args.model_class)

    try:
        return cls()
    except TypeError as exc:
        raise RuntimeError(
            f"Could import {args.model_module}.{args.model_class}, "
            "but could not instantiate it with no arguments. "
            "Edit instantiate_model_from_args() to pass the same constructor "
            "arguments used during training."
        ) from exc


def load_model(args, device):
    checkpoint_path = str(args.checkpoint)

    print(f"[MODEL] Loading checkpoint: {checkpoint_path}", flush=True)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        print("[MODEL] Checkpoint contains a full torch.nn.Module.", flush=True)
        model = ckpt.to(device)
        model.eval()
        return model

    # Sometimes the full module is stored under a key.
    if isinstance(ckpt, dict):
        for key in ["model_object", "model_module", "full_model"]:
            if key in ckpt and isinstance(ckpt[key], torch.nn.Module):
                print(f"[MODEL] Found full model under checkpoint key: {key}", flush=True)
                model = ckpt[key].to(device)
                model.eval()
                return model

    state_dict = extract_state_dict(ckpt)

    if state_dict is None:
        raise RuntimeError(
            "Could not find a model or state_dict inside the checkpoint. "
            "Run this script with --inspect_checkpoint to see checkpoint contents."
        )

    model = instantiate_model_from_args(args)

    if model is None:
        raise RuntimeError(
            "\nThe checkpoint appears to contain only a state_dict, not the full model object.\n"
            "That means this wrapper needs the DisasterAdaptiveNet architecture class.\n\n"
            "Run this first:\n"
            f"  python {Path(__file__).name} --inspect_checkpoint --checkpoint {checkpoint_path}\n\n"
            "Then find the model class used by your training script and rerun with:\n"
            "  --model_module MODULE_NAME --model_class CLASS_NAME\n\n"
            "Example if your code has DisasterAdaptiveNet/models/hrtbda.py with class HRTBDA:\n"
            "  --model_module models.hrtbda --model_class HRTBDA\n\n"
            "If the class constructor needs arguments, edit instantiate_model_from_args()."
        )

    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[MODEL] Loaded state_dict with strict=False", flush=True)
    print(f"[MODEL] Missing keys: {len(missing)}", flush=True)
    print(f"[MODEL] Unexpected keys: {len(unexpected)}", flush=True)

    if len(unexpected) > 0:
        print("[MODEL] Sample unexpected keys:", unexpected[:20], flush=True)

    if len(missing) > 0:
        print("[MODEL] Sample missing keys:", missing[:20], flush=True)

    model = model.to(device)
    model.eval()

    return model


def extract_tensor_from_model_output(output):
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


def run_model_forward(model, x6, pre_tensor, post_tensor, args):
    """
    Tries common forward signatures:
      model(x6)
      model(pre, post)
      model({"image": x6, "pre": pre, "post": post})
    """

    errors = []

    try:
        return model(x6)
    except Exception as exc:
        errors.append(f"model(x6) failed: {repr(exc)}")

    try:
        return model(pre_tensor, post_tensor)
    except Exception as exc:
        errors.append(f"model(pre, post) failed: {repr(exc)}")

    try:
        return model(
            {
                "image": x6,
                "x": x6,
                "pre": pre_tensor,
                "post": post_tensor,
                "pre_image": pre_tensor,
                "post_image": post_tensor,
            }
        )
    except Exception as exc:
        errors.append(f"model(dict) failed: {repr(exc)}")

    raise RuntimeError(
        "Could not run model forward with common signatures.\n"
        + "\n".join(errors)
        + "\n\nEdit run_model_forward() to match the forward() method of your DisasterAdaptiveNet model."
    )


def tensor_to_prediction_mask(output_tensor: torch.Tensor, original_shape, threshold: float) -> np.ndarray:
    """
    Converts model output tensor to class mask with values:
      0 = background
      1 = no damage
      2 = minor
      3 = major
      4 = destroyed
    """

    if output_tensor.ndim == 4:
        output_tensor = output_tensor[0]

    if output_tensor.ndim == 3:
        channels, h, w = output_tensor.shape

        if channels >= 5:
            pred = torch.argmax(output_tensor, dim=0).detach().cpu().numpy().astype(np.uint8)

        elif channels == 4:
            pred = torch.argmax(output_tensor, dim=0).detach().cpu().numpy().astype(np.uint8)
            pred = pred + 1

        elif channels == 1:
            prob = torch.sigmoid(output_tensor[0])
            pred = (prob > float(threshold)).detach().cpu().numpy().astype(np.uint8)

        else:
            raise RuntimeError(f"Unsupported output channel count: {channels}")

    elif output_tensor.ndim == 2:
        values = output_tensor.detach().cpu().numpy()

        if np.issubdtype(values.dtype, np.integer):
            pred = values.astype(np.uint8)
        else:
            pred = (values > float(threshold)).astype(np.uint8)

    else:
        raise RuntimeError(f"Unsupported output tensor shape: {tuple(output_tensor.shape)}")

    original_h, original_w = original_shape
    pred = pred[:original_h, :original_w]

    pred = pred.astype(np.uint8)

    # Keep only expected labels.
    pred[~np.isin(pred, [0, 1, 2, 3, 4])] = 0

    return pred


def main():
    parser = argparse.ArgumentParser(
        description="Single-pair inference wrapper for DisasterAdaptiveNet HRTBDA."
    )

    parser.add_argument("--pre_image")
    parser.add_argument("--post_image")
    parser.add_argument("--gt_mask")

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

    # Optional: needed if checkpoint is state_dict only.
    parser.add_argument("--model_module", default=os.getenv("HRTBDA_MODEL_MODULE", ""))
    parser.add_argument("--model_class", default=os.getenv("HRTBDA_MODEL_CLASS", ""))

    parser.add_argument("--inspect_checkpoint", action="store_true")

    args = parser.parse_args()

    add_project_to_path()

    if args.inspect_checkpoint:
        inspect_checkpoint(args.checkpoint)
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

    pre_padded, _ = pad_to_factor(pre_bgr, 32)
    post_padded, _ = pad_to_factor(post_bgr, 32)

    x6 = normalize_6ch(pre_padded, post_padded).to(device)
    pre_tensor = normalize_single_rgb(pre_padded).to(device)
    post_tensor = normalize_single_rgb(post_padded).to(device)

    model = load_model(args, device)

    with torch.no_grad():
        if args.amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                output = run_model_forward(model, x6, pre_tensor, post_tensor, args)
        else:
            output = run_model_forward(model, x6, pre_tensor, post_tensor, args)

    output_tensor = extract_tensor_from_model_output(output)

    print(f"[MODEL] Output tensor shape: {tuple(output_tensor.shape)}", flush=True)

    pred_np = tensor_to_prediction_mask(
        output_tensor=output_tensor,
        original_shape=(original_h, original_w),
        threshold=args.threshold,
    )

    gt_mask = normalize_ground_truth_mask(
        mask_path=args.gt_mask,
        target_shape=pred_np.shape[:2],
    )

    localization_metrics = compute_localization_metrics(pred_np, gt_mask)
    damage_metrics = compute_f1_scores(pred_np, gt_mask)

    metrics = {
        **localization_metrics,
        **damage_metrics,
        "metrics_note": "Image-specific F1 calculated using uploaded ground-truth post-disaster mask.",
    }

    output_parent = Path(args.building_mask_output).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    building_mask = ((pred_np > 0).astype(np.uint8)) * 255
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

    summary = summarize_by_connected_components(
        pred_mask=pred_np,
        min_area=args.min_area,
    )

    total_seconds = time.perf_counter() - start_time

    summary["metrics"] = metrics

    summary["runtime"] = {
        "model_name": args.model_name,
        "model_family": "hrtbda",
        "device": str(device),
        "model_total_seconds": round(float(total_seconds), 3),
    }

    summary["model_info"] = {
        "model_name": args.model_name,
        "model_family": "hrtbda",
        "checkpoint": str(args.checkpoint),
        "checkpoint_dir": str(args.checkpoint_dir),
        "threshold": args.threshold,
        "model_module": args.model_module,
        "model_class": args.model_class,
        "output_tensor_shape": list(output_tensor.shape),
        "prediction_unique_values": sorted(np.unique(pred_np).astype(int).tolist()),
        "gt_mask_unique_values": sorted(np.unique(gt_mask).astype(int).tolist()),
        "note": (
            "This wrapper assumes the model outputs either 5-class logits, "
            "4-class damage logits, binary logits, or a class-index mask."
        ),
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_csv(args.summary_csv, summary)

    print("[SUMMARY]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()