#!/usr/bin/env python3
"""Prepare a leak-free Texas Tornadoes split for second-place xView2."""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SRC = Path("/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed")
OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/second_place_texas_tornadoes"
)

SPLITS = ["train", "val", "test"]
FOLD_MAP = {"train": 1, "val": 0, "test": 2}


def symlink_force(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.symlink(source.resolve(), destination)


def find_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def main():
    if OUT.exists():
        print("Removing old prepared folder:", OUT)
        shutil.rmtree(OUT)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for split in SPLITS:
        image_dir = SRC / split / "images"
        mask_dir = SRC / split / "masks"
        target_dir = SRC / split / "targets"

        if not image_dir.exists():
            missing.append((split, "missing images folder", str(image_dir)))
            continue

        pre_images = sorted(image_dir.glob("*_pre_disaster.png"))
        if not pre_images:
            missing.append((split, "no pre-disaster images", str(image_dir)))
            continue

        for pre_image in pre_images:
            tile_id = pre_image.name.replace("_pre_disaster.png", "")
            post_image = image_dir / f"{tile_id}_post_disaster.png"
            pre_mask = find_existing(
                [
                    target_dir / f"{tile_id}_pre_disaster_target.png",
                    target_dir / f"{tile_id}_pre_disaster.png",
                    mask_dir / f"{tile_id}_pre_disaster.png",
                ]
            )
            post_mask = find_existing(
                [
                    target_dir / f"{tile_id}_post_disaster_target.png",
                    target_dir / f"{tile_id}_post_disaster.png",
                    mask_dir / f"{tile_id}_post_disaster.png",
                ]
            )

            if not post_image.exists():
                missing.append((split, tile_id, "missing post image"))
                continue
            if pre_mask is None:
                missing.append((split, tile_id, "missing pre mask/target"))
                continue
            if post_mask is None:
                missing.append((split, tile_id, "missing post mask/target"))
                continue

            damage = read_mask(post_mask)
            unexpected = sorted(set(np.unique(damage).astype(int)) - {0, 1, 2, 3, 4})
            if unexpected:
                raise ValueError(
                    f"Unexpected Texas Tornadoes damage-mask values for {tile_id}: "
                    f"{unexpected}"
                )

            symlink_force(pre_image, OUT / "images" / pre_image.name)
            symlink_force(post_image, OUT / "images" / post_image.name)
            symlink_force(pre_mask, OUT / "masks" / f"{tile_id}_pre_disaster.png")
            symlink_force(post_mask, OUT / "masks" / f"{tile_id}_post_disaster.png")

            present = set(np.unique(damage).astype(int))
            rows.append(
                {
                    "id": tile_id,
                    "fold": FOLD_MAP[split],
                    "split": split,
                    "nondamage": 1 in present,
                    "minor": 2 in present,
                    "major": 3 in present,
                    "destroyed": 4 in present,
                    "empty": not bool(present & {1, 2, 3, 4}),
                }
            )

    if missing:
        print("ERROR: missing required files.")
        for item in missing[:50]:
            print(item)
        print("Total missing:", len(missing))
        raise SystemExit(2)

    columns = [
        "id", "fold", "split", "nondamage", "minor", "major", "destroyed", "empty"
    ]
    folds = pd.DataFrame(rows)[columns]
    folds.to_csv(OUT / "folds.csv", index=False)

    train_val = folds[folds["split"].isin(["train", "val"])].copy()
    train_val.to_csv(OUT / "folds_train_val.csv", index=False)

    train_ids = set(folds[folds["split"] == "train"]["id"].astype(str))
    val_ids = set(folds[folds["split"] == "val"]["id"].astype(str))
    test_ids = set(folds[folds["split"] == "test"]["id"].astype(str))

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Texas Tornadoes train/val/test IDs are not disjoint")
    if set(train_val[train_val["fold"] != 0]["split"]) != {"train"}:
        raise RuntimeError("Training folds contain non-training rows outside fold 0")
    if set(train_val[train_val["fold"] == 0]["split"]) != {"val"}:
        raise RuntimeError("Training folds contain non-validation rows in fold 0")

    print("Prepared Texas Tornadoes for second-place xView2 fine-tuning")
    print("SRC:", SRC)
    print("OUT:", OUT)
    print("Split counts:")
    print(folds["split"].value_counts())
    print("Training rows:", len(train_ids))
    print("Validation rows:", len(val_ids))
    print("Held-out test rows:", len(test_ids))
    print("Test leakage: none")
    print("folds.csv:", OUT / "folds.csv")
    print("Training folds:", OUT / "folds_train_val.csv")


if __name__ == "__main__":
    main()
