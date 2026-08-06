#!/usr/bin/env python3
"""Create a non-destructive common hold/test view for xBD ensemble inference."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    mapping = {"val": "hold", "test": "test"}
    rows = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for destination_split, source_split in mapping.items():
        source = (args.xbd_root / source_split).resolve()
        for required in (source / "images", source / "targets"):
            if not required.is_dir():
                raise FileNotFoundError(required)
        destination = args.output_root / destination_split
        if destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"Refusing to replace non-matching link: {destination}")
        elif destination.exists():
            raise RuntimeError(f"Refusing to replace existing path: {destination}")
        else:
            os.symlink(source, destination, target_is_directory=True)
        ids = sorted(path.name.removesuffix("_pre_disaster.png") for path in (source / "images").glob("*_pre_disaster.png"))
        if not ids:
            raise RuntimeError(f"No pre-disaster images found: {source / 'images'}")
        for tile_id in ids:
            for suffix in ("pre", "post"):
                image = source / "images" / f"{tile_id}_{suffix}_disaster.png"
                target = source / "targets" / f"{tile_id}_{suffix}_disaster_target.png"
                if not image.is_file() or not target.is_file():
                    raise FileNotFoundError(image if not image.is_file() else target)
            rows.append({"id": tile_id, "split": destination_split})
        print(f"{destination_split} <- {source_split}: {len(ids)} samples", flush=True)
    manifest = args.output_root / "official_split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "split"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote common manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
