#!/usr/bin/env python3
"""Convert raster Texas splits into the official xView2 baseline inputs."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    return mask[..., 0] if mask.ndim == 3 else mask


def collect(root: Path, split: str):
    images = root / split / "images"
    targets = root / split / "targets"
    for post in sorted(images.glob("*_post_disaster.*")):
        stem = post.stem.replace("_post_disaster", "")
        pre = images / f"{stem}_pre_disaster{post.suffix}"
        loc = targets / f"{stem}_pre_disaster_target.png"
        damage = targets / f"{stem}_post_disaster_target.png"
        if not (pre.exists() and loc.exists() and damage.exists()):
            raise FileNotFoundError(f"Incomplete pair for {stem} in {split}")
        yield stem, pre, post, loc, damage


def write_localization(root: Path, output: Path) -> None:
    base = output / "localization"
    image_dir, label_dir, dataset_dir = base / "images", base / "labels", base / "dataSet"
    for directory in (image_dir, label_dir, dataset_dir):
        directory.mkdir(parents=True, exist_ok=True)
    lists = {"train": [], "val": [], "test": []}
    means = []
    for split in lists:
        for stem, pre, _, loc, _ in collect(root, split):
            name = f"{split}_{stem}.png"
            image = cv2.imread(str(pre), cv2.IMREAD_COLOR)
            mask = (read_mask(loc) > 0).astype(np.uint8) * 255
            cv2.imwrite(str(image_dir / name), image)
            cv2.imwrite(str(label_dir / name), mask)
            lists[split].append(name)
            if split == "train":
                means.append(image.astype(np.float64).mean((0, 1)))
    for split, names in lists.items():
        (dataset_dir / f"{split}.txt").write_text("\n".join(names) + "\n")
    mean = np.mean(means, axis=0).astype(np.float32) if means else np.zeros(3, np.float32)
    np.save(dataset_dir / "mean.npy", mean)


def component_chips(image: np.ndarray, target: np.ndarray, stem: str, out: Path, rows: list) -> None:
    height, width = target.shape
    for damage_class in range(1, 5):
        count, labels = cv2.connectedComponents((target == damage_class).astype(np.uint8), connectivity=8)
        for component in range(1, count):
            ys, xs = np.where(labels == component)
            if len(xs) < 4:
                continue
            pad = max(4, int(round(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.15)))
            x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad + 1)
            y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad + 1)
            chip = image[y0:y1, x0:x1]
            filename = f"{stem}_{damage_class}_{component}.png"
            cv2.imwrite(str(out / filename), chip)
            rows.append({"uuid": filename, "labels": damage_class - 1})


def write_classification(root: Path, output: Path) -> None:
    base = output / "classification"
    for split in ("train", "val", "test"):
        chip_dir = base / split
        chip_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for stem, _, post, _, damage_path in collect(root, split):
            image = cv2.imread(str(post), cv2.IMREAD_COLOR)
            component_chips(image, read_mask(damage_path), stem, chip_dir, rows)
        with (base / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["uuid", "labels"])
            writer.writeheader()
            writer.writerows(rows)
        counts = {label: sum(int(row["labels"]) == label for row in rows) for label in range(4)}
        print(f"{split}: {len(rows)} building chips; class counts={counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root, output = Path(args.data_root), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    write_localization(root, output)
    write_classification(root, output)
    print(f"Prepared official xView2 inputs under {output}")


if __name__ == "__main__":
    main()
