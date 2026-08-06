#!/usr/bin/env python3
"""Export probabilities from the Texas-fine-tuned xView2 third-place ensemble."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


MODEL_WEIGHTS = {
    "Dec30_15_34_resnet34_unet_v2_512_fold0_fp16_pseudo_crops.pth": [0.51244243, 1.42747062, 1.23648384, 0.90290896, 0.88912514],
    "Dec30_15_34_resnet101_fpncatv2_256_512_fold0_fp16_pseudo_crops.pth": [0.50847073, 1.15392272, 1.2059733, 1.1340391, 1.03196719],
    "Dec22_22_24_seresnext50_unet_v2_512_fold1_fp16_crops.pth": [0.54324459, 1.76890163, 1.20782899, 0.85128004, 0.83100698],
    "Dec31_02_09_resnet34_unet_v2_512_fold1_fp16_pseudo_crops.pth": [0.48269921, 1.22874469, 1.38328066, 0.96695393, 0.91348539],
    "Dec31_03_55_densenet201_fpncatv2_256_512_fold1_fp16_pseudo_crops.pth": [0.48804137, 1.14809462, 1.24851827, 1.11798428, 1.00790482],
    "Dec17_19_12_inceptionv4_fpncatv2_256_512_fold2_fp16_crops.pth": [0.34641084, 1.63486251, 1.14186036, 0.86668715, 1.12193125],
    "Dec27_14_08_densenet169_unet_v2_512_fold2_fp16_crops.pth": [0.55429115, 1.34944309, 1.1087044, 0.89542089, 1.17257541],
    "Dec31_12_45_resnet34_unet_v2_512_fold2_fp16_pseudo_crops.pth": [0.65977938, 1.50252452, 0.97098732, 0.74048182, 1.08712367],
    "Dec15_23_24_resnet34_unet_v2_512_fold3_crops.pth": [0.84090623, 1.02953555, 1.2526516, 0.9298182, 0.94053529],
    "Dec21_11_50_seresnext50_unet_v2_512_fold3_fp16_crops.pth": [0.43108046, 1.30222898, 1.09660616, 0.94958969, 1.07063753],
    "Dec31_18_17_efficientb4_fpncatv2_256_512_fold3_fp16_pseudo_crops.pth": [0.59338243, 1.17347438, 1.186104, 1.06860638, 1.03041829],
    "Dec19_06_18_resnet34_unet_v2_512_fold4_fp16_crops.pth": [0.83915734, 1.02560309, 0.77639015, 1.17487775, 1.05632771],
    "Dec27_14_37_resnet101_unet_v2_512_fold4_fp16_crops.pth": [0.57414314, 1.19599486, 1.05561912, 0.98815567, 1.2274592],
}


def arguments():
    project = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-root", type=Path, default=project / "baselines/xview2_winners/xview2_third_place")
    parser.add_argument("--checkpoint-root", type=Path, default=project / "output/xview2_baselines/third_place_texas_tornadoes_FULL_SOLUTION_finetune_official_split/models")
    parser.add_argument("--data-root", type=Path, default=project / "../texas_tornadoes_preprocessed")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--expected-val-samples", type=int, default=45)
    parser.add_argument("--expected-test-samples", type=int, default=46)
    return parser.parse_args()


def read_rgb(path: Path, size: int):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def read_mask(path: Path, size: int):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def input_tensor(pre: Path, post: Path, size: int):
    image = np.dstack([read_rgb(pre, size), read_rgb(post, size)]).astype(np.float32) / 255.0
    mean = np.asarray([.485, .456, .406, .485, .456, .406], np.float32)
    std = np.asarray([.229, .224, .225, .229, .224, .225], np.float32)
    image = (image - mean) / std
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 13-model third-place exporter")
    sys.path.insert(0, str(args.winner_root))
    from xview.dataset import OUTPUT_MASK_KEY  # pylint: disable=import-outside-toplevel
    from xview.inference import model_from_checkpoint  # pylint: disable=import-outside-toplevel
    from xview.postprocessing import make_predictions_naive  # pylint: disable=import-outside-toplevel

    checkpoints = [(args.checkpoint_root / name, weights) for name, weights in MODEL_WEIGHTS.items()]
    missing = [str(path) for path, _ in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fine-tuned third-place checkpoints:\n" + "\n".join(missing))
    samples = []
    expected = {"val": args.expected_val_samples, "test": args.expected_test_samples}
    for split in args.splits:
        image_dir = args.data_root / split / "images"
        ids = sorted(path.name.removesuffix("_pre_disaster.png") for path in image_dir.glob("*_pre_disaster.png"))
        if split in expected and len(ids) != expected[split]:
            raise RuntimeError(f"Expected {expected[split]} {split} samples, found {len(ids)}")
        for tile_id in ids:
            samples.append((split, tile_id))
    print("Third-place export samples:", {s: sum(x[0] == s for x in samples) for s in args.splits}, flush=True)

    sums = {tile_id: np.zeros((5, args.output_size, args.output_size), np.float32) for _, tile_id in samples}
    device = torch.device("cuda")
    for index, (checkpoint, class_weights) in enumerate(checkpoints, 1):
        print(f"[{index}/13] Loading {checkpoint.name}", flush=True)
        model, _ = model_from_checkpoint(str(checkpoint), activation_after="model", report=False)
        model = model.to(device).eval()
        weights = np.asarray(class_weights, np.float32)[:, None, None]
        with torch.inference_mode():
            for split, tile_id in samples:
                base = args.data_root / split / "images"
                tensor = input_tensor(base / f"{tile_id}_pre_disaster.png", base / f"{tile_id}_post_disaster.png", args.input_size).to(device)
                with torch.amp.autocast("cuda", enabled=True):
                    probability = model(tensor)[OUTPUT_MASK_KEY]
                if probability.shape[-2:] != (args.output_size, args.output_size):
                    probability = F.interpolate(probability, (args.output_size, args.output_size), mode="bilinear", align_corners=False)
                sums[tile_id] += probability[0].float().cpu().numpy() * weights
        del model
        torch.cuda.empty_cache()

    for split, tile_id in samples:
        probability5 = sums[tile_id] / len(checkpoints)
        loc_prediction, damage_prediction = make_predictions_naive(probability5)
        foreground = np.clip(probability5[1:5], 0, None)
        foreground /= np.maximum(foreground.sum(axis=0, keepdims=True), 1e-7)
        building = np.clip(probability5[1:].sum(axis=0), 0, None)
        background = np.clip(probability5[0], 0, None)
        loc_probability = building / np.maximum(building + background, 1e-7)
        mask_dir = args.data_root / split / "masks"
        loc_true = (read_mask(mask_dir / f"{tile_id}_pre_disaster.png", args.output_size) > 0).astype(np.uint8)
        damage_true = read_mask(mask_dir / f"{tile_id}_post_disaster.png", args.output_size)
        destination = args.output_root / split / f"{tile_id}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination,
            loc_probability=loc_probability.astype(np.float16),
            damage_probability=foreground.astype(np.float16),
            damage_probability5=probability5.astype(np.float16),
            loc_prediction=loc_prediction.astype(np.uint8),
            damage_prediction=damage_prediction.astype(np.uint8),
            loc_true=loc_true, damage_true=damage_true)
    for split in args.splits:
        (args.output_root / split / "_SUCCESS").touch()
    print(f"Wrote third-place probability maps under: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
