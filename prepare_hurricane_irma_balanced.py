from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SEED = 42
random.seed(SEED)

RAW_ROOT = Path("/homes/j244s673/documents/wsu/phd/HURRICANE-IRMA")
IMG_DIR = RAW_ROOT / "images"
MSK_DIR = RAW_ROOT / "masks"

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


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return m


def copy_as_png(src_path: Path, dst_path: Path) -> None:
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read source file: {src_path}")
    ok = cv2.imwrite(str(dst_path), img)
    if not ok:
        raise RuntimeError(f"Could not write destination file: {dst_path}")


def collect_pairs():
    samples = []

    pre_files = []
    for ext in IMG_EXTS:
        pre_files.extend(IMG_DIR.glob(f"*_pre_disaster{ext}"))
    pre_files = sorted(pre_files)

    for pre_path in pre_files:
        prefix = pre_path.stem.replace("_pre_disaster", "")

        post_img = find_existing_file(IMG_DIR, f"{prefix}_post_disaster")
        pre_msk = find_existing_file(MSK_DIR, f"{prefix}_pre_disaster")
        post_msk = find_existing_file(MSK_DIR, f"{prefix}_post_disaster")

        if post_img is None or pre_msk is None or post_msk is None:
            continue

        post_mask = read_mask(post_msk)
        unique_vals = np.unique(post_mask).tolist()

        sample = {
            "prefix": prefix,
            "pre_img": str(pre_path),
            "post_img": str(post_img),
            "pre_msk": str(pre_msk),
            "post_msk": str(post_msk),
            "is_empty": int(len(unique_vals) == 1 and unique_vals[0] == 0),
            "has_loc": int((post_mask > 0).any()),
            "has_cls_1": int((post_mask == 1).any()),
            "has_cls_2": int((post_mask == 2).any()),
            "has_cls_3": int((post_mask == 3).any()),
            "has_cls_4": int((post_mask == 4).any()),
            "unique_values": unique_vals,
        }
        samples.append(sample)

    return samples


def split_bucket(items: list[dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    items = items[:]
    random.shuffle(items)

    n = len(items)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:n_train + n_val + n_test]

    return train_items, val_items, test_items


def summarize_subset(subset_name: str, subset_samples: list[dict]) -> dict:
    summary = Counter()
    for s in subset_samples:
        summary["num_samples"] += 1
        summary["empty_tiles"] += int(s["is_empty"])
        summary["non_empty_tiles"] += int(not s["is_empty"])
        summary["has_loc"] += int(s["has_loc"])
        summary["has_cls_1"] += int(s["has_cls_1"])
        summary["has_cls_2"] += int(s["has_cls_2"])
        summary["has_cls_3"] += int(s["has_cls_3"])
        summary["has_cls_4"] += int(s["has_cls_4"])

    return {
        "subset": subset_name,
        "num_samples": int(summary["num_samples"]),
        "empty_tiles": int(summary["empty_tiles"]),
        "non_empty_tiles": int(summary["non_empty_tiles"]),
        "has_loc": int(summary["has_loc"]),
        "has_cls_1": int(summary["has_cls_1"]),
        "has_cls_2": int(summary["has_cls_2"]),
        "has_cls_3": int(summary["has_cls_3"]),
        "has_cls_4": int(summary["has_cls_4"]),
    }


def export_subset(subset_name: str, subset_samples: list[dict]) -> list[dict]:
    exported_entries = []

    for s in subset_samples:
        prefix = s["prefix"]

        pre_img = Path(s["pre_img"])
        post_img = Path(s["post_img"])
        pre_msk = Path(s["pre_msk"])
        post_msk = Path(s["post_msk"])

        copy_as_png(pre_img, OUT_ROOT / subset_name / "images" / f"{prefix}_pre_disaster.png")
        copy_as_png(post_img, OUT_ROOT / subset_name / "images" / f"{prefix}_post_disaster.png")

        copy_as_png(pre_msk, OUT_ROOT / subset_name / "masks" / f"{prefix}_pre_disaster.png")
        copy_as_png(post_msk, OUT_ROOT / subset_name / "masks" / f"{prefix}_post_disaster.png")

        copy_as_png(pre_msk, OUT_ROOT / subset_name / "targets" / f"{prefix}_pre_disaster_target.png")
        copy_as_png(post_msk, OUT_ROOT / subset_name / "targets" / f"{prefix}_post_disaster_target.png")

        exported_entries.append(
            {
                "prefix": prefix,
                "subset": subset_name,
                "is_empty": s["is_empty"],
                "has_loc": s["has_loc"],
                "has_cls_1": s["has_cls_1"],
                "has_cls_2": s["has_cls_2"],
                "has_cls_3": s["has_cls_3"],
                "has_cls_4": s["has_cls_4"],
                "unique_values": s["unique_values"],
            }
        )

    return exported_entries


samples = collect_pairs()
print(f"Found {len(samples)} valid pre/post pairs.")

empty_samples = [s for s in samples if s["is_empty"] == 1]
non_empty_samples = [s for s in samples if s["is_empty"] == 0]

print(f"Empty tiles: {len(empty_samples)}")
print(f"Non-empty tiles: {len(non_empty_samples)}")

train_empty, val_empty, test_empty = split_bucket(empty_samples)
train_non_empty, val_non_empty, test_non_empty = split_bucket(non_empty_samples)

split_map = {
    "train": train_empty + train_non_empty,
    "val": val_empty + val_non_empty,
    "test": test_empty + test_non_empty,
}

for split in split_map:
    random.shuffle(split_map[split])

metadata = {
    "seed": SEED,
    "raw_root": str(RAW_ROOT),
    "img_dir": str(IMG_DIR),
    "msk_dir": str(MSK_DIR),
    "out_root": str(OUT_ROOT),
    "global_summary": {
        "num_samples": len(samples),
        "empty_tiles": len(empty_samples),
        "non_empty_tiles": len(non_empty_samples),
    },
    "splits": {},
}

for split_name, split_samples in split_map.items():
    exported_entries = export_subset(split_name, split_samples)
    split_summary = summarize_subset(split_name, split_samples)
    metadata["splits"][split_name] = {
        "summary": split_summary,
        "patches": exported_entries,
    }
    print(
        f"{split_name}: {split_summary['num_samples']} samples | "
        f"empty={split_summary['empty_tiles']} | "
        f"non_empty={split_summary['non_empty_tiles']}"
    )

with open(OUT_ROOT / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Prepared IRMA dataset at: {OUT_ROOT}")
print(f"Metadata written to: {OUT_ROOT / 'metadata.json'}")