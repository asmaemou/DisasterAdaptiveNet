#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

RAW = Path("/homes/j244s673/documents/wsu/phd/mount_semeru_eruption_preprocessed")
OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/first_place_mount_semeru_FINE_TUNE_OFFICIAL_SPLIT"
)
TARGET_SIZE = 1024


def tile_ids_from_images(image_dir):
    return sorted(
        path.name.replace("_pre_disaster.png", "")
        for path in image_dir.glob("*_pre_disaster.png")
    )


def find_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def source_files(split, tile_id):
    image_dir = RAW / split / "images"
    mask_dir = RAW / split / "masks"
    target_dir = RAW / split / "targets"
    pre_image = image_dir / f"{tile_id}_pre_disaster.png"
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

    missing = []
    if not pre_image.exists():
        missing.append(str(pre_image))
    if not post_image.exists():
        missing.append(str(post_image))
    if pre_mask is None:
        missing.append(f"pre mask for {split}/{tile_id}")
    if post_mask is None:
        missing.append(f"post mask for {split}/{tile_id}")
    if missing:
        raise FileNotFoundError("Missing required Semeru files:\n" + "\n".join(missing))
    return pre_image, post_image, pre_mask, post_mask


def symlink_force(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.symlink(source.resolve(), destination)


def fit_1024(image, is_mask=False):
    width, height = image.size
    if width == TARGET_SIZE and height == TARGET_SIZE:
        return image
    if width <= TARGET_SIZE and height <= TARGET_SIZE:
        mode = "L" if is_mask else "RGB"
        fill = 0 if is_mask else (0, 0, 0)
        canvas = Image.new(mode, (TARGET_SIZE, TARGET_SIZE), fill)
        canvas.paste(image.convert(mode), (0, 0))
        return canvas
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return image.resize((TARGET_SIZE, TARGET_SIZE), resample=resample)


def load_rgb_image(path):
    path = Path(path)
    try:
        return Image.open(path).convert("RGB")
    except Exception as pil_error:
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if array is None:
            size = path.stat().st_size if path.exists() else -1
            with path.open("rb") as stream:
                first_bytes = stream.read(32)
            raise RuntimeError(
                f"Could not read image with PIL or OpenCV: {path}\n"
                f"File size: {size}\n"
                f"First bytes: {first_bytes}\n"
                f"PIL error: {pil_error}"
            )
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def read_mask(path):
    path = Path(path)
    try:
        array = np.array(Image.open(path))
    except Exception as pil_error:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            size = path.stat().st_size if path.exists() else -1
            with path.open("rb") as stream:
                first_bytes = stream.read(32)
            raise RuntimeError(
                f"Could not read mask with PIL or OpenCV: {path}\n"
                f"File size: {size}\n"
                f"First bytes: {first_bytes}\n"
                f"PIL error: {pil_error}"
            )
    if array.ndim == 3:
        array = array[:, :, 0]
    return array.astype(np.uint8)


def normalize_masks(pre_mask, post_mask):
    localization = (pre_mask > 0).astype(np.uint8)
    if localization.shape != post_mask.shape:
        localization = cv2.resize(
            localization,
            (post_mask.shape[1], post_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    values = set(np.unique(post_mask).astype(int).tolist())
    unexpected = sorted(values - {0, 1, 2, 3, 4})
    if unexpected:
        raise ValueError(f"Unexpected Mount Semeru damage-mask values: {unexpected}")
    damage = post_mask.astype(np.uint8)
    damage[localization == 0] = 0
    return localization * 255, damage


def save_rgb_1024(source, destination):
    fit_1024(load_rgb_image(source), is_mask=False).save(destination)


def save_mask_1024(array, destination):
    image = Image.fromarray(array.astype(np.uint8), mode="L")
    fit_1024(image, is_mask=True).save(destination)


def main():
    if OUT.exists():
        print("Removing old prepared dataset:", OUT)
        shutil.rmtree(OUT)

    for split in ["train", "val", "test"]:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "masks").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    test_rows = []
    split_ids = {}

    for split in ["train", "val", "test"]:
        image_dir = RAW / split / "images"
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)
        tile_ids = tile_ids_from_images(image_dir)
        if not tile_ids:
            raise RuntimeError(f"No pre-disaster images found under {image_dir}")
        split_ids[split] = set(tile_ids)
        print(f"{split} samples: {len(tile_ids)}")

        for tile_id in tile_ids:
            pre_image, post_image, pre_mask_path, post_mask_path = source_files(split, tile_id)
            localization, damage = normalize_masks(
                read_mask(pre_mask_path), read_mask(post_mask_path)
            )
            out_pre_image = OUT / split / "images" / f"{tile_id}_pre_disaster.png"
            out_post_image = OUT / split / "images" / f"{tile_id}_post_disaster.png"
            out_pre_mask = OUT / split / "masks" / f"{tile_id}_pre_disaster.png"
            out_post_mask = OUT / split / "masks" / f"{tile_id}_post_disaster.png"

            save_rgb_1024(pre_image, out_pre_image)
            save_rgb_1024(post_image, out_post_image)
            save_mask_1024(localization, out_pre_mask)
            save_mask_1024(damage, out_post_mask)

            symlink_force(out_pre_mask, OUT / "masks" / out_pre_mask.name)
            symlink_force(out_post_mask, OUT / "masks" / out_post_mask.name)
            symlink_force(out_pre_mask, OUT / "masks" / f"test_localization_{tile_id}_target.png")
            symlink_force(out_post_mask, OUT / "masks" / f"test_damage_{tile_id}_target.png")

            present = set(np.unique(damage).astype(int).tolist())
            row = {
                "id": tile_id,
                "split": split,
                "nondamage": 1 in present,
                "minor": 2 in present,
                "major": 3 in present,
                "destroyed": 4 in present,
                "empty": not any(value in present for value in [1, 2, 3, 4]),
            }
            manifest_rows.append(row)
            if split == "test":
                test_rows.append(
                    {
                        "id": tile_id,
                        "fold": 0,
                        "nondamage": row["nondamage"],
                        "minor": row["minor"],
                        "major": row["major"],
                        "destroyed": row["destroyed"],
                        "empty": row["empty"],
                    }
                )

    train_ids, val_ids, test_ids = (
        split_ids["train"], split_ids["val"], split_ids["test"]
    )
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Mount Semeru train/val/test IDs are not disjoint")

    pd.DataFrame(manifest_rows).to_csv(OUT / "official_split_manifest.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUT / "folds.csv", index=False)

    print("Prepared Mount Semeru official train/val/test split for 1st-place xView2")
    print("All images and masks saved as 1024x1024 PNG")
    print("Damage labels validated as xView2 classes 1-4 with background 0")
    print("Test leakage: none")
    print("OUT:", OUT)
    print("Test rows in folds.csv:", len(test_rows))


if __name__ == "__main__":
    main()
