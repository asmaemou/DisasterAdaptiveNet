#!/usr/bin/env python3
"""Export Swin-HRTBDA DirectTurkey probability maps for late fusion."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_module(path: Path):
    name = "swin_hrtbda_v5_turkey_ensemble_source"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return checkpoint


def saved(checkpoint: dict, key: str, fallback):
    saved_args = checkpoint.get("args", {})
    return saved_args.get(key, fallback) if isinstance(saved_args, dict) else fallback


def native_truth(dataset, index: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    sample = dataset.samples[index]
    loc_raw = dataset._read_mask(sample.pre_target_path)
    damage_raw = dataset._read_mask(sample.post_target_path)
    target5 = dataset._target5_from_masks(loc_raw, damage_raw)
    loc = (loc_raw > 0).astype(np.uint8)
    if loc.shape != (output_size, output_size):
        loc = cv2.resize(loc, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        target5 = cv2.resize(
            target5, (output_size, output_size), interpolation=cv2.INTER_NEAREST
        )
    return loc, target5.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    project = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    experiment = project / "output/HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-script",
        type=Path,
        default=project / "transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py",
    )
    parser.add_argument("--phase1-checkpoint", type=Path, default=experiment / "checkpoints/phase1_best.pt")
    parser.add_argument("--phase2-checkpoint", type=Path, default=experiment / "checkpoints/phase2_best.pt")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project
        / "output/hybrid_swin_hrtbda_first_place_xbd_zero_shot_turkey/probabilities/swin_hrtbda",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=project
        / "output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT/official_split_manifest.csv",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model_script, args.phase1_checkpoint, args.phase2_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    device = torch.device(args.device)
    model_code = load_module(args.model_script)
    phase1_checkpoint = load_checkpoint(args.phase1_checkpoint, device)
    phase2_checkpoint = load_checkpoint(args.phase2_checkpoint, device)

    decoder_channels = int(saved(phase2_checkpoint, "decoder_channels", 128))
    swin_variant = str(saved(phase2_checkpoint, "swin_variant", "swin_tiny_patch4_window7_224"))
    img_size = int(saved(phase2_checkpoint, "img_size", 896))
    patch_size = int(saved(phase2_checkpoint, "swin_patch_size", 4))
    window_size = int(saved(phase2_checkpoint, "swin_window_size", 7))
    phase1_threshold = float(phase1_checkpoint.get("best_threshold", 0.5))

    # The complete trained state is loaded immediately, so downloading ImageNet
    # initialization again is unnecessary and could fail on an offline node.
    phase1 = model_code.HRTBDAPhase1(
        decoder_channels=decoder_channels,
        swin_variant=swin_variant,
        swin_pretrained=False,
        img_size=img_size,
        swin_patch_size=patch_size,
        swin_window_size=window_size,
    ).to(device)
    phase2 = model_code.HRTBDAPhase2(
        decoder_channels=decoder_channels,
        swin_variant=swin_variant,
        swin_pretrained=False,
        img_size=img_size,
        swin_patch_size=patch_size,
        swin_window_size=window_size,
        num_classes=4,
    ).to(device)
    phase1.load_state_dict(phase1_checkpoint["model"], strict=True)
    phase2.load_state_dict(phase2_checkpoint["model"], strict=True)
    phase1.eval()
    phase2.eval()

    allowed_by_split: dict[str, set[str]] = {}
    if not args.split_manifest.is_file():
        raise FileNotFoundError(args.split_manifest)
    with args.split_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            allowed_by_split.setdefault(row["split"], set()).add(row["id"])

    print(f"Device: {device}", flush=True)
    print(f"Swin variant: {swin_variant}", flush=True)
    print(f"Model input size: {img_size}; exported map size: {args.output_size}", flush=True)
    print(
        f"Phase I epoch={phase1_checkpoint.get('epoch')} threshold={phase1_threshold:.2f}",
        flush=True,
    )
    print(f"Phase II epoch={phase2_checkpoint.get('epoch')}", flush=True)
    print(f"Common split manifest: {args.split_manifest}", flush=True)

    with torch.no_grad():
        for split in args.splits:
            dataset = model_code.XBDHRTBDADataset(
                root=args.data_root,
                split=split,
                image_size=img_size,
                training=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            allowed = allowed_by_split.get(split)
            if not allowed:
                raise RuntimeError(f"No official manifest IDs found for split: {split}")
            split_output = args.output_root / split
            split_output.mkdir(parents=True, exist_ok=True)
            for stale in split_output.glob("*.npz"):
                if stale.stem not in allowed:
                    stale.unlink()

            print(
                f"{split}: loader sees {len(dataset)}; common manifest selects {len(allowed)}",
                flush=True,
            )
            written: set[str] = set()
            for zero_index, batch in enumerate(loader):
                stem = batch["stem"][0]
                if stem not in allowed:
                    print(f"[{split} {zero_index + 1}/{len(dataset)}] skip {stem}", flush=True)
                    continue
                written.add(stem)
                destination = split_output / f"{stem}.npz"
                if destination.exists() and not args.overwrite:
                    print(f"[{split} {zero_index + 1}/{len(dataset)}] reuse {stem}", flush=True)
                    continue

                pre = batch["pre"].to(device, non_blocking=True)
                post = batch["post"].to(device, non_blocking=True)
                loc_probability = torch.sigmoid(phase1(pre)).unsqueeze(1)
                damage_logits = model_code.get_damage_logits(phase2(pre, post))
                damage_probability = torch.sigmoid(damage_logits)
                damage_probability /= damage_probability.sum(dim=1, keepdim=True).clamp_min(1e-7)

                if img_size != args.output_size:
                    loc_probability = F.interpolate(
                        loc_probability,
                        size=(args.output_size, args.output_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    damage_probability = F.interpolate(
                        damage_probability,
                        size=(args.output_size, args.output_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    damage_probability /= damage_probability.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-7)

                loc_true, damage_true = native_truth(dataset, zero_index, args.output_size)
                np.savez_compressed(
                    destination,
                    loc_probability=loc_probability[0, 0].float().cpu().numpy().astype(np.float16),
                    damage_probability=damage_probability[0].float().cpu().numpy().astype(np.float16),
                    loc_true=loc_true,
                    damage_true=damage_true,
                    phase1_threshold=np.asarray(phase1_threshold, dtype=np.float32),
                )
                print(f"[{split} {zero_index + 1}/{len(dataset)}] wrote {stem}", flush=True)

            missing = sorted(allowed - written)
            if missing:
                raise RuntimeError(
                    f"{split}: {len(missing)} official IDs absent from the Swin dataset; "
                    f"first IDs: {missing[:10]}"
                )

    print(f"Wrote Swin-HRTBDA probability maps under: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
