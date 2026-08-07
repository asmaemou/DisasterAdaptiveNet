#!/usr/bin/env python3
"""Validate IDs and required arrays in a completed Swin probability cache."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def source_truth(data_root: Path, split: str, stem: str, output_size: int):
    targets = data_root / split / "targets"
    loc = cv2.imread(str(targets / f"{stem}_pre_disaster_target.png"), cv2.IMREAD_UNCHANGED)
    damage = cv2.imread(str(targets / f"{stem}_post_disaster_target.png"), cv2.IMREAD_UNCHANGED)
    if loc is None or damage is None:
        raise RuntimeError(f"{stem}: failed to read current source targets under {targets}")
    if loc.ndim == 3:
        loc = loc[..., 0]
    if damage.ndim == 3:
        damage = damage[..., 0]
    loc_binary = loc > 0
    target5 = np.zeros(loc.shape, dtype=np.uint8)
    for class_id in (1, 2, 3, 4):
        target5[(damage == class_id) & loc_binary] = class_id
    target5[loc_binary & ~np.isin(damage, [1, 2, 3, 4])] = 255
    if loc.shape[0] <= output_size and loc.shape[1] <= output_size:
        loc_canvas = np.zeros((output_size, output_size), dtype=np.uint8)
        damage_canvas = np.zeros((output_size, output_size), dtype=np.uint8)
        loc_canvas[: loc.shape[0], : loc.shape[1]] = loc_binary.astype(np.uint8)
        damage_canvas[: loc.shape[0], : loc.shape[1]] = target5
        return loc_canvas, damage_canvas
    return (
        cv2.resize(loc_binary.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_NEAREST),
        cv2.resize(target5, (output_size, output_size), interpolation=cv2.INTER_NEAREST),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-size", type=int, default=1024)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        expected = {row["id"] for row in csv.DictReader(handle) if row["split"].strip() == args.split}
    files = {path.stem: path for path in args.root.glob("*.npz")}
    if not expected:
        raise RuntimeError(f"Manifest has no IDs for split={args.split}")
    if set(files) != expected:
        missing = sorted(expected - set(files)); extra = sorted(set(files) - expected)
        raise RuntimeError(f"Cache ID mismatch: missing={missing[:10]}, extra={extra[:10]}")
    required = {"loc_probability", "damage_probability", "loc_true", "damage_true", "phase1_threshold"}
    for stem, path in files.items():
        with np.load(path) as data:
            absent = required - set(data.files)
            if absent:
                raise RuntimeError(f"{stem}: missing arrays {sorted(absent)}")
            if data["loc_probability"].shape != data["loc_true"].shape:
                raise RuntimeError(f"{stem}: localization probability/truth shape mismatch")
            if data["damage_probability"].shape[1:] != data["damage_true"].shape:
                raise RuntimeError(f"{stem}: damage probability/truth shape mismatch")
            if args.data_root is not None:
                current_loc, current_damage = source_truth(
                    args.data_root, args.split, stem, args.output_size
                )
                if not np.array_equal(data["loc_true"], current_loc):
                    differing = int(np.count_nonzero(data["loc_true"] != current_loc))
                    raise RuntimeError(
                        f"{stem}: cached localization truth is stale ({differing} pixels differ)"
                    )
                if not np.array_equal(data["damage_true"], current_damage):
                    differing = int(np.count_nonzero(data["damage_true"] != current_damage))
                    raise RuntimeError(
                        f"{stem}: cached damage truth is stale ({differing} pixels differ)"
                    )
    print(f"VALID Swin cache: split={args.split}, samples={len(files)}", flush=True)


if __name__ == "__main__":
    main()
