#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

RAW = Path("/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_earthquake_turkey_OFFICIAL_SPLIT")

FOLD_MAP = {
    "train": 1,
    "val": 0,
}


def symlink_force(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def find_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def validate_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        size = path.stat().st_size if path.exists() else -1
        with path.open("rb") as stream:
            first_bytes = stream.read(32)
        raise RuntimeError(
            f"OpenCV could not read image: {path}\n"
            f"File size: {size}\n"
            f"First bytes: {first_bytes}"
        )


def validate_mask(path):
    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise RuntimeError(f"PIL could not read mask: {path}: {error}") from error


def source_files(split, tile_id):
    image_dir = RAW / split / "images"
    mask_dir = RAW / split / "masks"
    target_dir = RAW / split / "targets"

    pre_image = image_dir / f"{tile_id}_pre_disaster.png"
    post_image = image_dir / f"{tile_id}_post_disaster.png"

    pre_mask = find_existing([
        target_dir / f"{tile_id}_pre_disaster_target.png",
        target_dir / f"{tile_id}_pre_disaster.png",
        mask_dir / f"{tile_id}_pre_disaster.png",
    ])
    post_mask = find_existing([
        target_dir / f"{tile_id}_post_disaster_target.png",
        target_dir / f"{tile_id}_post_disaster.png",
        mask_dir / f"{tile_id}_post_disaster.png",
    ])

    if not pre_image.exists():
        raise FileNotFoundError(pre_image)
    if not post_image.exists():
        raise FileNotFoundError(post_image)
    if pre_mask is None:
        raise FileNotFoundError(f"Missing pre-disaster mask for {split}/{tile_id}")
    if post_mask is None:
        raise FileNotFoundError(f"Missing post-disaster mask for {split}/{tile_id}")

    return pre_image, post_image, pre_mask, post_mask


def training_row(tile_id, event_type, image_path, mask_path, fold, sample_id):
    return {
        "event_name": "earthquake-turkey",
        "event_type": event_type,
        "folder": "train",
        "image_fname": image_path.as_posix(),
        "image_id": f"{tile_id}_{event_type}_disaster",
        "mask_fname": mask_path.as_posix(),
        "sample_id": sample_id,
        "fold": fold,
    }


def main():
    if OUT.exists():
        print("Removing old prepared dataset:", OUT)
        shutil.rmtree(OUT)

    for split in ["train", "test"]:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "masks").mkdir(parents=True, exist_ok=True)

    training_rows = []
    test_rows = []
    split_ids = {}
    sample_id = 0

    for split in ["train", "val", "test"]:
        image_dir = RAW / split / "images"
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)

        tile_ids = sorted(
            path.name.replace("_pre_disaster.png", "")
            for path in image_dir.glob("*_pre_disaster.png")
        )
        split_ids[split] = set(tile_ids)
        print(f"{split} samples:", len(tile_ids))

        for tile_id in tile_ids:
            pre_image, post_image, pre_mask, post_mask = source_files(split, tile_id)
            validate_image(pre_image)
            validate_image(post_image)
            validate_mask(pre_mask)
            validate_mask(post_mask)

            destination_split = "test" if split == "test" else "train"
            out_pre_image = Path(destination_split) / "images" / f"{tile_id}_pre_disaster.png"
            out_post_image = Path(destination_split) / "images" / f"{tile_id}_post_disaster.png"
            out_pre_mask = Path(destination_split) / "masks" / f"{tile_id}_pre_disaster.png"
            out_post_mask = Path(destination_split) / "masks" / f"{tile_id}_post_disaster.png"

            symlink_force(pre_image, OUT / out_pre_image)
            symlink_force(post_image, OUT / out_post_image)
            symlink_force(pre_mask, OUT / out_pre_mask)
            symlink_force(post_mask, OUT / out_post_mask)

            if split == "test":
                test_rows.append({"id": tile_id, "fold": 0})
            else:
                fold = FOLD_MAP[split]
                training_rows.append(
                    training_row(tile_id, "pre", out_pre_image, out_pre_mask, fold, sample_id)
                )
                training_rows.append(
                    training_row(tile_id, "post", out_post_image, out_post_mask, fold, sample_id)
                )
                sample_id += 1

    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Earthquake Turkey train/val/test IDs are not disjoint")

    train_folds = pd.DataFrame(training_rows)
    train_folds.to_csv(OUT / "train_folds.csv", index=False)

    folds = pd.DataFrame(test_rows)
    folds.to_csv(OUT / "folds.csv", index=False)

    for name in ["images", "masks"]:
        symlink_force(OUT / "test" / name, OUT / name)

    print("Prepared third-place Earthquake Turkey official split")
    print("OUT:", OUT)
    print("Training samples:", len(train_ids))
    print("Validation samples:", len(val_ids))
    print("Held-out test samples:", len(test_ids))
    print("Training CSV rows:", len(train_folds))
    print("Test leakage: none")
    print("train_folds.csv:", OUT / "train_folds.csv")
    print("folds.csv:", OUT / "folds.csv")


if __name__ == "__main__":
    main()
