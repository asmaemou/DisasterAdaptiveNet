#!/usr/bin/env python3
"""Validation-calibrated TTA ensemble for SwinDamageFormer.

This evaluator deliberately selects every calibration/post-processing choice on
the validation split, freezes that choice, and evaluates the test split once.
That keeps the ensemble scientifically usable rather than tuning on test labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_texas_swin_damageformer as base
import train_xbd_hrtbda_v5_swin_pretrained_cascade as legacy


TTA_NAMES = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270", "transpose", "anti_transpose")


def transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity": return x
    if name == "hflip": return x.flip(-1)
    if name == "vflip": return x.flip(-2)
    if name == "rot90": return torch.rot90(x, 1, (-2, -1))
    if name == "rot180": return torch.rot90(x, 2, (-2, -1))
    if name == "rot270": return torch.rot90(x, 3, (-2, -1))
    if name == "transpose": return x.transpose(-2, -1)
    if name == "anti_transpose": return torch.rot90(x.transpose(-2, -1), 2, (-2, -1))
    raise ValueError(f"Unknown TTA transform: {name}")


def inverse_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name in {"identity", "hflip", "vflip", "transpose", "anti_transpose"}:
        return transform(x, name)
    if name == "rot90": return torch.rot90(x, 3, (-2, -1))
    if name == "rot180": return torch.rot90(x, 2, (-2, -1))
    if name == "rot270": return torch.rot90(x, 1, (-2, -1))
    raise ValueError(f"Unknown TTA transform: {name}")


def architecture_args(saved: Dict, image_size: int) -> Namespace:
    values = dict(saved.get("args", {}))
    defaults = {
        "swin_variant": "swin_tiny_patch4_window7_224", "img_size": image_size,
        "decoder_channels": 192, "temporal_heads": 6, "temporal_window": 7,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    if int(values["img_size"]) != image_size:
        raise ValueError(
            f"Checkpoint image size {values['img_size']} does not match evaluator --img-size {image_size}"
        )
    return Namespace(**values)


def load_models(paths: Sequence[Path], device: torch.device, image_size: int):
    models = []
    identities = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = base.SwinDamageFormer(architecture_args(checkpoint, image_size)).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)
        identities.append({
            "path": str(path), "epoch": int(checkpoint.get("epoch", -1)),
            "best_metric": float(checkpoint.get("best_metric", -1.0)),
        })
        print(f"Loaded ensemble member: {identities[-1]}", flush=True)
    return models, identities


def ordinal_distribution(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits.float())
    # Enforce monotonic cumulative probabilities P(y>0)>=P(y>1)>=P(y>2).
    q1 = cumulative[:, 0]
    q2 = torch.minimum(q1, cumulative[:, 1])
    q3 = torch.minimum(q2, cumulative[:, 2])
    probability = torch.stack((1.0 - q1, q1 - q2, q2 - q3, q3), dim=1)
    return probability.clamp_min(1e-7) / probability.sum(1, keepdim=True).clamp_min(1e-7)


@torch.no_grad()
def predict(models, pre: torch.Tensor, post: torch.Tensor, tta: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loc_sum = damage_sum = ordinal_sum = None
    count = 0
    for model in models:
        for name in tta:
            output = model(transform(pre, name), transform(post, name))
            loc = inverse_transform(torch.sigmoid(output["loc"].float()).unsqueeze(1), name).squeeze(1)
            damage = inverse_transform(torch.softmax(output["damage"].float(), 1), name)
            ordinal = inverse_transform(ordinal_distribution(output["ordinal"]), name)
            loc_sum = loc if loc_sum is None else loc_sum + loc
            damage_sum = damage if damage_sum is None else damage_sum + damage
            ordinal_sum = ordinal if ordinal_sum is None else ordinal_sum + ordinal
            count += 1
    return loc_sum / count, damage_sum / count, ordinal_sum / count


@torch.no_grad()
def cache_split(models, loader: DataLoader, device: torch.device, tta: Sequence[str]):
    cached = []
    for index, batch in enumerate(loader, 1):
        loc, damage, ordinal = predict(models, batch["pre"].to(device), batch["post"].to(device), tta)
        for bi, stem in enumerate(batch["stem"]):
            cached.append({
                "stem": stem, "loc_prob": loc[bi].cpu(), "damage_prob": damage[bi].cpu(),
                "ordinal_prob": ordinal[bi].cpu(), "loc_true": batch["loc"][bi].long(),
                "target5": batch["target5"][bi].long(),
            })
        print(f"Cached ensemble predictions: batch {index}/{len(loader)}", flush=True)
    return cached


def morph(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, element)


def remove_small(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum:
            keep[labels == component] = 1
    return keep


def postprocess(item: Dict, config: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    loc_np = (item["loc_prob"].numpy() >= config["loc_threshold"]).astype(np.uint8)
    loc_np = remove_small(morph(loc_np, config["morph_kernel"]), config["min_building_pixels"])
    loc = torch.from_numpy(loc_np).long()
    probability = (
        (1.0 - config["ordinal_blend"]) * item["damage_prob"]
        + config["ordinal_blend"] * item["ordinal_prob"]
    )
    boosts = torch.tensor(config["class_boosts"], dtype=probability.dtype).view(4, 1, 1)
    probability = probability * boosts
    probability = probability / probability.sum(0, keepdim=True).clamp_min(1e-7)
    if config["object_vote"]:
        prediction = base.object_vote(probability.unsqueeze(0), loc.unsqueeze(0), 1)[0]
    else:
        prediction = (probability.argmax(0) + 1) * loc
    return prediction, loc


def score(cached: Sequence[Dict], config: Dict) -> Dict[str, float]:
    counts = {"loc_tp": 0, "loc_fp": 0, "loc_fn": 0, **{c: {"tp": 0, "fp": 0, "fn": 0} for c in range(1, 5)}}
    for item in cached:
        prediction, loc = postprocess(item, config)
        base.update_counts(prediction, loc, item["loc_true"], item["target5"], counts)
    result = base.finalize_counts(counts, config["loc_threshold"])
    result["damage_macro_f1"] = float(np.mean([
        result["damage_f1_no_damage"], result["damage_f1_minor_damage"],
        result["damage_f1_major_damage"], result["damage_f1_destroyed"],
    ]))
    return result


def candidate_configs(args: argparse.Namespace) -> Iterable[Dict]:
    boost_sets = [tuple(x) for x in args.class_boosts]
    for threshold, blend, kernel, minimum, vote, boosts in product(
        args.thresholds, args.ordinal_blends, args.morph_kernels,
        args.min_building_pixels, args.object_vote, boost_sets,
    ):
        yield {
            "loc_threshold": threshold, "ordinal_blend": blend, "morph_kernel": kernel,
            "min_building_pixels": minimum, "object_vote": bool(vote), "class_boosts": boosts,
        }


def make_loader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = legacy.XBDHRTBDADataset(args.data_root, split, args.img_size, training=False)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calibrated SwinDamageFormer ensemble")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--img-size", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta", nargs="+", choices=TTA_NAMES, default=list(TTA_NAMES))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65, 0.75])
    parser.add_argument("--ordinal-blends", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--morph-kernels", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--min-building-pixels", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--object-vote", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument(
        "--class-boosts", type=float, nargs=4, action="append",
        default=None, metavar=("NO", "MINOR", "MAJOR", "DESTROYED"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.class_boosts is None:
        args.class_boosts = [[1, 1, 1, 1], [1, 1.25, 1.25, 1], [1, 1.5, 1.5, 1.1]]
    paths = [Path(path) for pattern in args.checkpoints for path in sorted(Path().glob(pattern))] if any("*" in x for x in args.checkpoints) else [Path(x) for x in args.checkpoints]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"One or more ensemble checkpoints do not exist: {paths}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device} | members={len(paths)} | TTA={args.tta}", flush=True)
    models, identities = load_models(paths, device, args.img_size)
    validation = cache_split(models, make_loader(args, args.val_split), device, args.tta)
    trials = []
    best_config = best_result = None
    for config in candidate_configs(args):
        result = score(validation, config)
        trials.append({"config": config, "result": result})
        if best_result is None or result["score"] > best_result["score"]:
            best_config, best_result = config, result
    print(f"Selected only on validation: {best_config}", flush=True)
    base.print_result("BEST VALIDATION ENSEMBLE", best_result)
    test_cache = cache_split(models, make_loader(args, args.test_split), device, args.tta)
    test_result = score(test_cache, best_config)
    base.print_result("FINAL TEST ENSEMBLE", test_result)
    report = {
        "architecture": "SwinDamageFormer calibrated probability ensemble",
        "members": identities, "tta": args.tta, "selected_on": args.val_split,
        "best_postprocess": best_config, "validation": best_result,
        "test": test_result, "calibration_trials": len(trials),
    }
    (output / "ensemble_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "calibration_trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
