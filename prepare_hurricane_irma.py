from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2

random.seed(42)

RAW_ROOT = Path("/homes/j244s673/documents/wsu/phd/HURRICANE-IRMA")
IMG_DIR = RAW_ROOT / "images" / "images"
MSK_DIR = RAW_ROOT / "images" / "masks"

OUT_ROOT = Path("/homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet")

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


for split in ["train", "val", "test"]:
    for sub in ["images", "masks", "targets"]:
        (OUT_ROOT / split / sub).mkdir(parents=True, exist_ok=True)


def find_existing_file(directory: Path, stem_base: str) -> Path | None:
    for ext in IMG_EXTS:
        p = directory / f"{stem_base}{ext}"
        if p.exists():
            return p
    return None


def collect_pairs():
    pairs = []

    pre_files = []
    for ext in IMG_EXTS:
        pre_files.extend(IMG_DIR.glob(f"*_pre_disaster{ext}"))
    pre_files = sorted(pre_files)

    for pre_path in pre_files:
        prefix = pre_path.stem.replace("_pre_disaster", "")

        post_img = find_existing_file(IMG_DIR, f"{prefix}_post_disaster")
        pre_msk = find_existing_file(MSK_DIR, f"{prefix}_pre_disaster")
        post_msk = find_existing_file(MSK_DIR, f"{prefix}_post_disaster")

        if post_img is not None and pre_msk is not None and post_msk is not None:
            pairs.append(prefix)

    return pairs


def summarize_sample(prefix: str, subset: str):
    post_mask_path = OUT_ROOT / subset / "masks" / f"{prefix}_post_disaster.png"
    m = cv2.imread(str(post_mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read {post_mask_path}")

    return {
        "prefix": prefix,
        "subset": subset,
        "loc": int((m > 0).sum() > 0),
        "cls_1": int((m == 1).sum() > 0),
        "cls_2": int((m == 2).sum() > 0),
        "cls_3": int((m == 3).sum() > 0),
        "cls_4": int((m == 4).sum() > 0),
    }


def copy_as_png(src_path: Path, dst_path: Path):
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read source file: {src_path}")
    ok = cv2.imwrite(str(dst_path), img)
    if not ok:
        raise RuntimeError(f"Could not write destination file: {dst_path}")


pairs = collect_pairs()
print(f"Found {len(pairs)} valid pre/post pairs.")

random.shuffle(pairs)

n = len(pairs)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

split_map = {
    "train": pairs[:n_train],
    "val": pairs[n_train:n_train + n_val],
    "test": pairs[n_train + n_val:],
}

metadata = {}

for split, split_prefixes in split_map.items():
    patch_entries = []

    for prefix in split_prefixes:
        pre_img = find_existing_file(IMG_DIR, f"{prefix}_pre_disaster")
        post_img = find_existing_file(IMG_DIR, f"{prefix}_post_disaster")
        pre_msk = find_existing_file(MSK_DIR, f"{prefix}_pre_disaster")
        post_msk = find_existing_file(MSK_DIR, f"{prefix}_post_disaster")

        if pre_img is None or post_img is None or pre_msk is None or post_msk is None:
            print(f"Skipping incomplete sample: {prefix}")
            continue

        copy_as_png(pre_img, OUT_ROOT / split / "images" / f"{prefix}_pre_disaster.png")
        copy_as_png(post_img, OUT_ROOT / split / "images" / f"{prefix}_post_disaster.png")

        copy_as_png(pre_msk, OUT_ROOT / split / "masks" / f"{prefix}_pre_disaster.png")
        copy_as_png(post_msk, OUT_ROOT / split / "masks" / f"{prefix}_post_disaster.png")

        copy_as_png(pre_msk, OUT_ROOT / split / "targets" / f"{prefix}_pre_disaster_target.png")
        copy_as_png(post_msk, OUT_ROOT / split / "targets" / f"{prefix}_post_disaster_target.png")

        patch_entries.append(summarize_sample(prefix, split))

    metadata[split] = {"patches": patch_entries}
    print(f"{split}: {len(patch_entries)} samples")

with open(OUT_ROOT / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Prepared IRMA dataset at: {OUT_ROOT}")