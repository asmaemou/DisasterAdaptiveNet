#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2
import numpy as np

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def read_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return m

def check_split(root: Path, split: str):
    split_root = root / split
    images_dir = split_root / "images"
    targets_dir = split_root / "targets"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not targets_dir.exists():
        raise FileNotFoundError(f"Missing targets dir: {targets_dir}")

    post_files = []
    for ext in IMAGE_EXTS:
        post_files.extend(images_dir.glob(f"*_post_disaster{ext}"))

    post_files = sorted(post_files)

    missing = []
    class_counts = np.zeros(5, dtype=np.int64)

    for post_path in post_files:
        stem = post_path.stem.replace("_post_disaster", "")
        ext = post_path.suffix

        pre_path = images_dir / f"{stem}_pre_disaster{ext}"
        pre_target = targets_dir / f"{stem}_pre_disaster_target.png"
        post_target = targets_dir / f"{stem}_post_disaster_target.png"

        if not pre_path.exists() or not pre_target.exists() or not post_target.exists():
            missing.append((stem, str(pre_path.exists()), str(pre_target.exists()), str(post_target.exists())))
            continue

        dmg = read_mask(post_target)
        valid = (dmg >= 0) & (dmg <= 4)
        vals, freqs = np.unique(dmg[valid], return_counts=True)
        for v, f in zip(vals, freqs):
            class_counts[int(v)] += int(f)

    print(f"\n===== SPLIT: {split} =====")
    print(f"Images dir:  {images_dir}")
    print(f"Targets dir: {targets_dir}")
    print(f"Paired post images found: {len(post_files)}")
    print(f"Missing paired files: {len(missing)}")
    print(f"Target pixel counts [background,no,minor,major,destroyed]: {class_counts.tolist()}")

    if missing[:10]:
        print("First missing examples:")
        for x in missing[:10]:
            print(x)

    if len(post_files) == 0:
        raise RuntimeError(f"No *_post_disaster images found in {images_dir}")

    if len(missing) > 0:
        raise RuntimeError(f"{split}: found missing paired files. Fix filenames before training.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    print("Checking IDA-BD xBD-style dataset format")
    print(f"Root: {args.root}")

    for split in args.splits:
        check_split(args.root, split)

    print("\nDataset format looks OK.")

if __name__ == "__main__":
    main()