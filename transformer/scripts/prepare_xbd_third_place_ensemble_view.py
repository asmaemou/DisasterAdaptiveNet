#!/usr/bin/env python3
"""Create an isolated xBD hold/test view for third-place ensemble inference."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Non-matching existing link: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    os.symlink(source.resolve(), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for split, source_split in (("val", "hold"), ("test", "test")):
        source = args.xbd_root / source_split
        image_source = source / "images"
        target_source = source / "targets"
        ids = sorted(
            path.name.removesuffix("_pre_disaster.png")
            for path in image_source.glob("*_pre_disaster.png")
        )
        if not ids:
            raise RuntimeError(f"No images found for xBD split={source_split}")
        for tile_id in ids:
            for timepoint in ("pre", "post"):
                image = image_source / f"{tile_id}_{timepoint}_disaster.png"
                target = target_source / f"{tile_id}_{timepoint}_disaster_target.png"
                if not image.is_file() or not target.is_file():
                    raise FileNotFoundError(image if not image.is_file() else target)
                link(image, args.output_root / split / "images" / image.name)
                link(
                    target,
                    args.output_root / split / "masks" / f"{tile_id}_{timepoint}_disaster.png",
                )
            rows.append({"id": tile_id, "split": split})
        print(f"{split} <- {source_split}: {len(ids)} samples", flush=True)
    manifest = args.output_root / "official_split_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote third-place xBD manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
