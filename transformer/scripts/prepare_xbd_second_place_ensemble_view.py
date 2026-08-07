#!/usr/bin/env python3
"""Create an isolated flattened xBD hold/test view for second-place inference."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Non-matching existing link: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    os.symlink(source.resolve(), destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    images = args.output_root / "images"; masks = args.output_root / "masks"
    images.mkdir(parents=True, exist_ok=True); masks.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, source_split in (("val", "hold"), ("test", "test")):
        source = args.xbd_root / source_split
        ids = sorted(path.name.removesuffix("_pre_disaster.png") for path in (source / "images").glob("*_pre_disaster.png"))
        if not ids:
            raise RuntimeError(f"No images found for xBD split={source_split}")
        for tile_id in ids:
            for timepoint in ("pre", "post"):
                image = source / "images" / f"{tile_id}_{timepoint}_disaster.png"
                target = source / "targets" / f"{tile_id}_{timepoint}_disaster_target.png"
                if not image.is_file() or not target.is_file():
                    raise FileNotFoundError(image if not image.is_file() else target)
                link(image, images / image.name)
                link(target, masks / f"{tile_id}_{timepoint}_disaster.png")
            rows.append({"id": tile_id, "fold": 0 if split == "val" else 2, "split": split})
        print(f"{split} <- {source_split}: {len(ids)} samples", flush=True)
    manifest = args.output_root / "folds.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "fold", "split"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote second-place xBD manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
