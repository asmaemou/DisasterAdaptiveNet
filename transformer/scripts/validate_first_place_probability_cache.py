#!/usr/bin/env python3
"""Validate every expected first-place xView2 probability file for a split."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


LOCALIZATION_FOLDERS = (
    "pred50_loc_tuned", "pred92_loc_tuned", "pred34_loc", "pred154_loc",
)
DAMAGE_FOLDERS = (
    "dpn92cls_cce_0_tuned", "dpn92cls_cce_1_tuned", "dpn92cls_cce_2_tuned",
    "res34cls2_0_tuned", "res34cls2_1_tuned", "res34cls2_2_tuned",
    "res50cls_cce_0_tuned", "res50cls_cce_1_tuned", "res50cls_cce_2_tuned",
    "se154cls_0_tuned", "se154cls_1_tuned", "se154cls_2_tuned",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--require-success-marker", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.require_success_marker and not (args.root / "_SUCCESS").is_file():
        raise RuntimeError(f"Missing success marker: {args.root / '_SUCCESS'}")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        stems = [row["id"] for row in csv.DictReader(handle) if row["split"] == args.split]
    if not stems:
        raise RuntimeError(f"Manifest contains no IDs for split={args.split}: {args.manifest}")
    missing = []
    for stem in stems:
        filename = f"{stem}_pre_disaster_part1.png"
        for folder in LOCALIZATION_FOLDERS:
            path = args.root / folder / filename
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(path)
        for folder in DAMAGE_FOLDERS:
            part1 = args.root / folder / filename
            part2 = args.root / folder / filename.replace("_part1.png", "_part2.png")
            for path in (part1, part2):
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(path)
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        raise RuntimeError(
            f"Incomplete first-place cache for split={args.split}: "
            f"{len(missing)} missing/empty files. First entries:\n{preview}"
        )
    print(
        f"VALID first-place cache: split={args.split}, samples={len(stems)}, "
        f"localization_folders={len(LOCALIZATION_FOLDERS)}, damage_folders={len(DAMAGE_FOLDERS)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"INVALID first-place cache: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
