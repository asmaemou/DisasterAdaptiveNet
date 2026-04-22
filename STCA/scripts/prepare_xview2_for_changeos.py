#!/usr/bin/env python3
"""Prepare raw xView2 splits for torchange ChangeOS training.

Creates a torchange-compatible directory layout:
  <out_root>/<split>/images   (symlink to raw images dir)
  <out_root>/<split>/targets  (*.png masks)

Expected raw layout (common xView2 layout):
  <raw_root>/<split>/images/*.png
  <raw_root>/<split>/labels/*.json

Output masks:
  *_pre_disaster_target.png   values: 0 background, 1 building
  *_post_disaster_target.png  values: 0 bg, 1 no-damage, 2 minor, 3 major,
                               4 destroyed, 255 un-classified/ignore
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw

DAMAGE_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 255,
    "unclassified": 255,
}


def parse_polygon_wkt(wkt: str) -> List[Tuple[float, float]]:
    wkt = wkt.strip()
    if not wkt.startswith("POLYGON"):
        return []
    m = re.search(r"POLYGON\s*\(\((.*)\)\)", wkt)
    if not m:
        return []
    ring = m.group(1).split(",")
    pts: List[Tuple[float, float]] = []
    for item in ring:
        item = item.strip()
        if not item:
            continue
        parts = item.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        pts.append((x, y))
    return pts


def draw_polygon(mask: Image.Image, pts: Iterable[Tuple[float, float]], value: int) -> None:
    pts = list(pts)
    if len(pts) < 3:
        return
    draw = ImageDraw.Draw(mask)
    draw.polygon(pts, fill=int(value))


def build_masks_from_json(label_fp: Path, image_size=(1024, 1024)) -> Tuple[Image.Image, Image.Image]:
    pre_mask = Image.new("L", image_size, 0)
    post_mask = Image.new("L", image_size, 0)
    with open(label_fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    feats = data.get("features", {}).get("xy", [])
    for feat in feats:
        props = feat.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        subtype = str(props.get("subtype", "") or "").strip().lower()
        wkt = feat.get("wkt", "")
        pts = parse_polygon_wkt(wkt)
        if not pts:
            continue
        draw_polygon(pre_mask, pts, 1)
        draw_polygon(post_mask, pts, DAMAGE_MAP.get(subtype, 255 if subtype else 255))
    return pre_mask, post_mask


def process_split(raw_root: Path, out_root: Path, split: str, force: bool = False) -> None:
    split_dir = raw_root / split
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir} to exist")

    out_split = out_root / split
    out_images = out_split / "images"
    out_targets = out_split / "targets"
    out_split.mkdir(parents=True, exist_ok=True)
    out_targets.mkdir(parents=True, exist_ok=True)

    if out_images.exists() or out_images.is_symlink():
        if force:
            if out_images.is_symlink() or out_images.is_file():
                out_images.unlink()
        else:
            pass
    if not out_images.exists():
        os.symlink(image_dir, out_images)

    json_files = sorted(label_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No label json files found in {label_dir}")

    for fp in json_files:
        stem = fp.stem
        pre_name = f"{stem}_pre_disaster_target.png"
        post_name = f"{stem}_post_disaster_target.png"
        pre_out = out_targets / pre_name
        post_out = out_targets / post_name
        if pre_out.exists() and post_out.exists() and not force:
            continue
        pre_mask, post_mask = build_masks_from_json(fp)
        pre_mask.save(pre_out)
        post_mask.save(post_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["tier3", "train", "test", "hold"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"Preparing split: {split}")
        process_split(raw_root, out_root, split, force=args.force)
    print(f"Done. Prepared dataset at: {out_root}")


if __name__ == "__main__":
    main()
