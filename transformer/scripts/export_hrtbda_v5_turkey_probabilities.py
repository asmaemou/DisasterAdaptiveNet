#!/usr/bin/env python3
"""Export HRTBDA-v5 Turkey probabilities for validation/test fusion."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_module(path: Path):
    name = "hrtbda_v5_turkey_ensemble_source"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint: {path}")
    return checkpoint


def saved_int(checkpoint, key: str, fallback: int) -> int:
    saved_args = checkpoint.get("args", {})
    return int(saved_args.get(key, fallback)) if isinstance(saved_args, dict) else int(fallback)


def parse_args() -> argparse.Namespace:
    project = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    experiment = project / "output/HRTBDA-v5-xBDInit-FullFineTune_EARTHQUAKE_TURKEY_resetbest"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-script",
        type=Path,
        default=project / "transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py",
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
        default=project / "output/hybrid_hrtbda_first_place_turkey/probabilities/hrtbda",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=project
        / "output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT/official_split_manifest.csv",
        help="Official common split IDs used to keep both model families identical.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--img-size", type=int, default=1024)
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

    base_channels = saved_int(phase2_checkpoint, "base_channels", 48)
    decoder_channels = saved_int(phase2_checkpoint, "decoder_channels", 128)
    window_size = saved_int(phase2_checkpoint, "window_size", 8)
    phase1_threshold = float(phase1_checkpoint.get("best_threshold", 0.5))

    phase1 = model_code.HRTBDAPhase1(base_channels, decoder_channels, window_size).to(device)
    phase2 = model_code.HRTBDAPhase2(
        base_channels, decoder_channels, window_size, num_classes=4
    ).to(device)
    phase1.load_state_dict(phase1_checkpoint["model"], strict=True)
    phase2.load_state_dict(phase2_checkpoint["model"], strict=True)
    phase1.eval()
    phase2.eval()

    allowed_by_split = {}
    if args.split_manifest:
        if not args.split_manifest.is_file():
            raise FileNotFoundError(args.split_manifest)
        with args.split_manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                allowed_by_split.setdefault(row["split"], set()).add(row["id"])
        print(f"Using common split manifest: {args.split_manifest}", flush=True)

    print(f"Device: {device}", flush=True)
    print(
        f"Phase I epoch={phase1_checkpoint.get('epoch')} threshold={phase1_threshold:.2f}",
        flush=True,
    )
    print(f"Phase II epoch={phase2_checkpoint.get('epoch')}", flush=True)

    with torch.no_grad():
        for split in args.splits:
            dataset = model_code.XBDHRTBDADataset(
                root=args.data_root,
                split=split,
                image_size=args.img_size,
                training=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            split_output = args.output_root / split
            split_output.mkdir(parents=True, exist_ok=True)
            allowed = allowed_by_split.get(split)
            if allowed is not None:
                for stale in split_output.glob("*.npz"):
                    if stale.stem not in allowed:
                        stale.unlink()
                print(
                    f"{split}: loader sees {len(dataset)}; common manifest selects {len(allowed)}",
                    flush=True,
                )
            else:
                print(f"{split}: {len(dataset)} samples", flush=True)

            written_stems = set()
            for index, batch in enumerate(loader, start=1):
                stem = batch["stem"][0]
                if allowed is not None and stem not in allowed:
                    print(f"[{split} {index}/{len(dataset)}] skip non-common ID {stem}", flush=True)
                    continue
                written_stems.add(stem)
                destination = split_output / f"{stem}.npz"
                if destination.exists() and not args.overwrite:
                    print(f"[{split} {index}/{len(dataset)}] reuse {stem}", flush=True)
                    continue

                pre = batch["pre"].to(device, non_blocking=True)
                post = batch["post"].to(device, non_blocking=True)
                loc_logits = phase1(pre)
                damage_logits = model_code.get_damage_logits(phase2(pre, post))

                loc_probability = torch.sigmoid(loc_logits)[0].float().cpu().numpy()
                damage_probability = torch.sigmoid(damage_logits)[0].float().cpu().numpy()
                damage_probability /= np.maximum(
                    damage_probability.sum(axis=0, keepdims=True), 1e-7
                )

                target5 = batch["target5"][0].cpu().numpy().astype(np.uint8)
                loc_true = batch["loc"][0].cpu().numpy().astype(np.uint8)

                np.savez_compressed(
                    destination,
                    loc_probability=loc_probability.astype(np.float16),
                    damage_probability=damage_probability.astype(np.float16),
                    loc_true=loc_true,
                    damage_true=target5,
                    phase1_threshold=np.asarray(phase1_threshold, dtype=np.float32),
                )
                print(f"[{split} {index}/{len(dataset)}] wrote {stem}", flush=True)

            if allowed is not None:
                missing = sorted(allowed - written_stems)
                if missing:
                    raise RuntimeError(
                        f"{split}: {len(missing)} manifest IDs were absent from the HRTBDA dataset; "
                        f"first IDs: {missing[:10]}"
                    )

    print(f"Wrote HRTBDA probabilities under: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
