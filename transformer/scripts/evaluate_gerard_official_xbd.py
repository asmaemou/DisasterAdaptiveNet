#!/usr/bin/env python3
"""Evaluate frozen Gerard Strong Baseline on the official xBD Test partition.

Evaluation follows the RS-BDNet pixel-level scoring conventions:
  - localization: sigmoid(channel 0) > 0.5;
  - damage: argmax(channels 1..4) + 1, gated by localization;
  - valid GT damage: GT pre-building mask > 0 AND post target in 1..4;
  - four-class arithmetic damage Macro-F1 (NOT harmonic xView2 damage score).

Inputs use the baseline's original BGR + (value/127 - 1) normalization.
Image and label maps are resized to 896 x 896 by default to match the
spatial evaluation grid used for the published RS-BDNet experiment.

No train/val/test split shuffling, checkpoint writes, training, or test tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

CLASSES = ["no_damage", "minor_damage", "major_damage", "destroyed"]
DEFAULT_CHECKPOINT = PROJECT / "output/reviewer3_gerard_xbd_nocudnn_32359/networks/strong_baseline_xview2.pt"
DEFAULT_DATASET = Path("/homes/j244s673/documents/wsu/phd/xview2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_samples(root: Path):
    meta_path = root / "metadata.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    # Train / tier3 metadata counts may vary between xBD distributions. Only
    # enforce the published combined split and official Hold / Test counts.
    assert len(metadata["train"]["patches"]) + len(metadata["tier3"]["patches"]) == 9168, "Train+Tier3 count mismatch"
    assert len(metadata["hold"]["patches"]) == 933, "Hold count mismatch"
    raw = metadata["test"]["patches"]
    assert len(raw) == 933, "Test count mismatch"
    keys = {(str(s["event"]), str(s["patch_id"])) for s in raw}
    assert len(keys) == 933, "Duplicate test pair IDs"
    train_keys = {(str(s["event"]), str(s["patch_id"])) for ss in ("train", "tier3") for s in metadata[ss]["patches"]}
    val_keys = {(str(s["event"]), str(s["patch_id"])) for s in metadata["hold"]["patches"]}
    assert keys.isdisjoint(train_keys) and keys.isdisjoint(val_keys) and train_keys.isdisjoint(val_keys), "Partition overlap"
    samples = []
    for sample in sorted(raw, key=lambda s: (str(s["event"]), str(s["patch_id"]))):
        name = f'{sample["event"]}_{sample["patch_id"]}'
        # Same pre/post target files used by the RS-BDNet supervised evaluator.
        paths = {
            "pre": root / "test/images" / f"{name}_pre_disaster.png",
            "post": root / "test/images" / f"{name}_post_disaster.png",
            "loc": root / "test/targets" / f"{name}_pre_disaster_target.png",
            "dmg": root / "test/targets" / f"{name}_post_disaster_target.png",
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label} for {name}: {path}")
        samples.append((name, paths))
    return samples


def read_pair(paths, size: int):
    images = []
    for kind in ("pre", "post"):
        img = cv2.imread(str(paths[kind]), cv2.IMREAD_COLOR)  # BGR, as original Strong Baseline
        if img is None:
            raise RuntimeError(f"Cannot read {paths[kind]}")
        if img.shape[:2] != (size, size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        images.append(img)
    loc = cv2.imread(str(paths["loc"]), cv2.IMREAD_UNCHANGED)
    dmg = cv2.imread(str(paths["dmg"]), cv2.IMREAD_UNCHANGED)
    if loc is None or dmg is None:
        raise RuntimeError(f"Cannot read GT targets: {paths['loc']}, {paths['dmg']}")
    if loc.ndim == 3:
        loc = loc[:, :, 0]
    if dmg.ndim == 3:
        dmg = dmg[:, :, 0]
    if loc.shape != (size, size):
        loc = cv2.resize(loc, (size, size), interpolation=cv2.INTER_NEAREST)
    if dmg.shape != (size, size):
        dmg = cv2.resize(dmg, (size, size), interpolation=cv2.INTER_NEAREST)
    # Original baseline preprocessing: BGR concat; value / 127.0 - 1.0.
    arr = np.concatenate(images, axis=2).astype(np.float32)
    arr /= 127.0
    arr -= 1.0
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return torch.from_numpy(arr), loc > 0, dmg


def f1_record(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p and r) else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r, "f1": f}


def evaluate(samples, checkpoint_path: Path, size: int, max_pairs: int):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EVAL: device={device}, cudnn_enabled={torch.backends.cudnn.enabled}", flush=True)
    if device.type != "cuda":
        raise RuntimeError("GPU required for this scheduled evaluation; CPU evaluation not intended")

    from utils import models
    # Avoid any download/reinitialization of ImageNet weights. The exact trained
    # weights are loaded strictly from our local frozen checkpoint below.
    original = models.torchvision.models.resnet34
    def resnet34_no_download(*args, **kwargs):
        kwargs["weights"] = None
        kwargs.pop("pretrained", None)
        return original(*args, **kwargs)
    models.torchvision.models.resnet34 = resnet34_no_download
    try:
        cfg = SimpleNamespace(MODEL=SimpleNamespace(TYPE="strongbaseline", OUT_CHANNELS=5))
        net = models.create_network(cfg)  # DataParallel, same 'module.' key prefix as training
    finally:
        models.torchvision.models.resnet34 = original

    print("EVAL: loading frozen checkpoint", checkpoint_path, flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "network" not in checkpoint or "epoch" not in checkpoint:
        raise RuntimeError("Unexpected checkpoint format; expected network and epoch")
    net.load_state_dict(checkpoint["network"], strict=True)
    epoch_selected = float(checkpoint["epoch"])
    del checkpoint
    net.to(device)
    net.eval()
    net.requires_grad_(False)
    print(f"EVAL: selected validation checkpoint epoch={epoch_selected:g}", flush=True)

    loc_tp = loc_fp = loc_fn = 0
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(1, 5)}
    limit = len(samples) if max_pairs == 0 else min(max_pairs, len(samples))

    with torch.inference_mode():
        for i, (name, paths) in enumerate(samples[:limit], 1):
            img, true_loc, true_dmg = read_pair(paths, size)
            logits = net(img.unsqueeze(0).to(device, non_blocking=False))
            if logits.shape != (1, 5, size, size):
                raise RuntimeError(f"Unexpected logits shape for {name}: {tuple(logits.shape)}")
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"Non-finite logits for {name}")
            # Threshold and gated 4-class prediction identical to RS-BDNet F1 runner.
            pred_loc = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy()
            pred_dmg = (logits[0, 1:5].argmax(dim=0) + 1).cpu().numpy().astype(np.uint8)
            pred_dmg[~pred_loc] = 0

            loc_tp += int(np.count_nonzero(pred_loc & true_loc))
            loc_fp += int(np.count_nonzero(pred_loc & ~true_loc))
            loc_fn += int(np.count_nonzero(~pred_loc & true_loc))

            valid = true_loc & (true_dmg >= 1) & (true_dmg <= 4)
            target = true_dmg[valid]
            predicted = pred_dmg[valid]
            for c in range(1, 5):
                counts[c]["tp"] += int(np.count_nonzero((predicted == c) & (target == c)))
                counts[c]["fp"] += int(np.count_nonzero((predicted == c) & (target != c)))
                counts[c]["fn"] += int(np.count_nonzero((predicted != c) & (target == c)))

            if i == 1 or i % 25 == 0 or i == limit:
                print(f"EVAL: {i}/{limit} images processed; latest={name}", flush=True)

    loc = f1_record(loc_tp, loc_fp, loc_fn)
    damage = {CLASSES[c - 1]: f1_record(**counts[c]) for c in range(1, 5)}
    macro = sum(v["f1"] for v in damage.values()) / 4.0
    harmonic = 4.0 / sum(1.0 / (v["f1"] + 1e-6) for v in damage.values())
    results = {
        "model": "Gerard-style Strong Baseline (Hafner implementation)",
        "checkpoint": str(checkpoint_path),
        "selected_val_epoch": epoch_selected,
        "parameter_count": sum(p.numel() for p in net.module.parameters()),
        "evaluated_split": "official_xbd_test",
        "num_test_pairs": limit,
        "expected_full_test_pairs": len(samples),
        "image_size": size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_preprocessing": "BGR, concatenate pre/post, float32 / 127 - 1",
        "loc_threshold": 0.5,
        "damage_gated_by_localization": True,
        "damage_valid_gt": "pre-localization-positive and damage target in 1..4",
        "localization_f1": loc["f1"],
        "damage_macro_f1": macro,
        "damage_harmonic_f1_for_reference_only": harmonic,
        "classwise_damage_f1": {k: v["f1"] for k, v in damage.items()},
        "counts": {"localization": loc, **damage},
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0, help="diagnostic only; NEVER publish partial-set scores")
    args = parser.parse_args()
    if args.image_size <= 0 or args.image_size % 32:
        parser.error("--image-size must be a positive multiple of 32")
    if args.max_pairs < 0:
        parser.error("--max-pairs must be nonnegative")
    args.data_dir = args.data_dir.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    print("EVAL: checkpoint found", args.checkpoint, flush=True)
    samples = discover_samples(args.data_dir)
    print(f"EVAL PREFLIGHT PASSED: {len(samples)} official xBD test pairs, complete RGB/GT paths, disjoint split IDs", flush=True)
    if args.preflight:
        print("EVAL: preflight only; no checkpoint load or GPU inference", flush=True)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight")
    args.output_dir = args.output_dir.absolute()
    if args.output_dir.is_symlink() or args.output_dir.exists():
        raise FileExistsError(f"Output path already exists (refusing overwrite): {args.output_dir}")
    if not args.output_dir.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {args.output_dir.parent}")
    # Guarantee evaluation writes cannot overlap the trained checkpoint.
    if args.checkpoint == args.output_dir or args.output_dir in args.checkpoint.parents or args.checkpoint in args.output_dir.parents:
        raise RuntimeError("Evaluation output overlaps the frozen training checkpoint")
    args.output_dir.mkdir(exist_ok=False)
    results = evaluate(samples, args.checkpoint, args.image_size, args.max_pairs)
    if args.max_pairs:
        results["warning"] = "PARTIAL DIAGNOSTIC ONLY - DO NOT PUBLISH"
    with (args.output_dir / "scores.json").open("x", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    with (args.output_dir / "scores.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "checkpoint_epoch", "num_test_pairs", "parameters_M", "localization_f1", "damage_macro_f1", *CLASSES])
        writer.writerow(["gerard_strong_baseline", results["selected_val_epoch"], results["num_test_pairs"], results["parameter_count"] / 1e6,
                         results["localization_f1"], results["damage_macro_f1"], *[results["classwise_damage_f1"][k] for k in CLASSES]])
    print("EVAL: RESULTS", json.dumps({k: results[k] for k in ("num_test_pairs", "selected_val_epoch", "localization_f1", "damage_macro_f1", "classwise_damage_f1")}, indent=2), flush=True)
    print("EVAL: saved to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
