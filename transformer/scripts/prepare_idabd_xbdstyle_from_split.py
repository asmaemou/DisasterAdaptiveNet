#!/usr/bin/env python3
"""
Prepare IDA-BD as an xBD-style directory so existing xBD/HRTBDA scripts can
run zero-shot evaluation without changing model code.

Input IDA-BD structure expected:
  IDABD_ROOT/images/*_pre_disaster.*
  IDABD_ROOT/images/*_post_disaster.*
  IDABD_ROOT/masks/*

Output xBD-style structure:
  OUT_ROOT/train/images, OUT_ROOT/train/targets
  OUT_ROOT/hold/images,  OUT_ROOT/hold/targets
  OUT_ROOT/test/images,  OUT_ROOT/test/targets

For each sample:
  images/<stem>_pre_disaster.png
  images/<stem>_post_disaster.png
  targets/<stem>_pre_disaster_target.png   # binary building mask from IDA-BD mask > 0
  targets/<stem>_post_disaster_target.png  # original 0..4 damage mask

The split JSON is expected to contain keys: train, val, test.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def tile_id_from_name(path_or_name: str | Path) -> str:
    base = Path(path_or_name).stem
    for suffix in [
        "_pre_disaster_target",
        "_post_disaster_target",
        "_pre_disaster_mask",
        "_post_disaster_mask",
        "_pre_disaster",
        "_post_disaster",
        "_target",
        "_mask",
    ]:
        base = base.replace(suffix, "")
    return base


def list_images_by_split(images_dir: Path, split: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in IMG_EXTS:
        for p in images_dir.glob(f"*_{split}_disaster{ext}"):
            out[tile_id_from_name(p)] = p
    return dict(sorted(out.items()))


def find_mask(masks_dir: Path, stem: str, split: str = "post") -> Optional[Path]:
    candidate_bases = [
        f"{stem}_{split}_disaster_target",
        f"{stem}_{split}_disaster_mask",
        f"{stem}_{split}_disaster",
    ]
    if split == "post":
        candidate_bases += [f"{stem}_target", f"{stem}_mask", stem]

    for base in candidate_bases:
        for ext in IMG_EXTS:
            p = masks_dir / f"{base}{ext}"
            if p.exists():
                return p
    return None


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.int64)
    legal = (m == 0) | (m == 1) | (m == 2) | (m == 3) | (m == 4) | (m == 255)
    m = np.where(legal, m, 255).astype(np.uint8)
    return m


def collect_samples(idabd_root: Path) -> Dict[str, Dict[str, Path]]:
    images_dir = idabd_root / "images"
    masks_dir = idabd_root / "masks"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing masks dir: {masks_dir}")

    pre_map = list_images_by_split(images_dir, "pre")
    post_map = list_images_by_split(images_dir, "post")
    stems = sorted(set(pre_map) & set(post_map))
    out: Dict[str, Dict[str, Path]] = {}
    missing = []
    for stem in stems:
        post_mask = find_mask(masks_dir, stem, "post")
        if post_mask is None:
            missing.append(stem)
            continue
        pre_mask = find_mask(masks_dir, stem, "pre")
        out[stem] = {
            "pre": pre_map[stem],
            "post": post_map[stem],
            "post_mask": post_mask,
        }
        if pre_mask is not None:
            out[stem]["pre_mask"] = pre_mask
    if not out:
        raise RuntimeError(f"No valid IDA-BD samples found under {idabd_root}")
    if missing:
        print(f"WARNING: skipped {len(missing)} samples with missing post masks")
    return out


def make_split(samples: Dict[str, Dict[str, Path]], seed: int) -> Dict[str, List[str]]:
    stems = list(samples.keys())
    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(round(0.80 * n))
    n_val = int(round(0.10 * n))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    return {
        "train": sorted(stems[:n_train]),
        "val": sorted(stems[n_train:n_train + n_val]),
        "test": sorted(stems[n_train + n_val:]),
    }


def load_or_make_split(split_file: Path, samples: Dict[str, Dict[str, Path]], seed: int, force_resplit: bool) -> Dict[str, List[str]]:
    if split_file.exists() and not force_resplit:
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        print(f"Loaded split file: {split_file}")
    else:
        splits = make_split(samples, seed)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"Wrote split file: {split_file}")

    all_stems = set(samples)
    clean = {}
    for key in ["train", "val", "test"]:
        clean[key] = [s for s in splits.get(key, []) if s in all_stems]
        if not clean[key]:
            raise RuntimeError(f"Split {key!r} is empty after filtering to discovered samples")
    return clean


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def prepare(args: argparse.Namespace) -> None:
    idabd_root = Path(args.idabd_root)
    out_root = Path(args.out_root)
    split_file = Path(args.split_file) if args.split_file else out_root / f"idabd_splits_seed{args.seed}_80_10_10.json"

    samples = collect_samples(idabd_root)
    splits = load_or_make_split(split_file, samples, args.seed, args.force_resplit)

    if args.clean and out_root.exists():
        print(f"Cleaning output root: {out_root}")
        shutil.rmtree(out_root)

    split_map = {"train": "train", "val": "hold", "test": "test"}

    for src_split, xbd_split in split_map.items():
        for sub in ["images", "targets"]:
            (out_root / xbd_split / sub).mkdir(parents=True, exist_ok=True)

        for stem in splits[src_split]:
            item = samples[stem]
            pre_dst = out_root / xbd_split / "images" / f"{stem}_pre_disaster{item['pre'].suffix.lower()}"
            post_dst = out_root / xbd_split / "images" / f"{stem}_post_disaster{item['post'].suffix.lower()}"
            copy_or_link(item["pre"], pre_dst, args.image_mode)
            copy_or_link(item["post"], post_dst, args.image_mode)

            post_mask = read_mask(item["post_mask"])
            if "pre_mask" in item:
                loc = (read_mask(item["pre_mask"]) > 0).astype(np.uint8)
            else:
                loc = np.isin(post_mask, [1, 2, 3, 4]).astype(np.uint8)

            pre_tgt = out_root / xbd_split / "targets" / f"{stem}_pre_disaster_target.png"
            post_tgt = out_root / xbd_split / "targets" / f"{stem}_post_disaster_target.png"
            write_png(pre_tgt, loc)
            write_png(post_tgt, post_mask)

    print("===== IDA-BD xBD-style split prepared =====")
    print(f"IDA-BD root: {idabd_root}")
    print(f"Output root: {out_root}")
    print(f"Split file:  {split_file}")
    print(f"Train: {len(splits['train'])}")
    print(f"Hold:  {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")
    print("Expected test image count is 8 if using your existing seed-42 80/10/10 split.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare IDA-BD into xBD-style folders")
    p.add_argument("--idabd-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--split-file", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-resplit", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--image-mode", choices=["copy", "symlink"], default="symlink")
    return p.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
