#!/usr/bin/env python3
"""Validate IDs and required arrays in a completed Swin probability cache."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        expected = {row["id"] for row in csv.DictReader(handle) if row["split"].strip() == args.split}
    files = {path.stem: path for path in args.root.glob("*.npz")}
    if not expected:
        raise RuntimeError(f"Manifest has no IDs for split={args.split}")
    if set(files) != expected:
        missing = sorted(expected - set(files)); extra = sorted(set(files) - expected)
        raise RuntimeError(f"Cache ID mismatch: missing={missing[:10]}, extra={extra[:10]}")
    required = {"loc_probability", "damage_probability", "loc_true", "damage_true", "phase1_threshold"}
    for stem, path in files.items():
        with np.load(path) as data:
            absent = required - set(data.files)
            if absent:
                raise RuntimeError(f"{stem}: missing arrays {sorted(absent)}")
            if data["loc_probability"].shape != data["loc_true"].shape:
                raise RuntimeError(f"{stem}: localization probability/truth shape mismatch")
            if data["damage_probability"].shape[1:] != data["damage_true"].shape:
                raise RuntimeError(f"{stem}: damage probability/truth shape mismatch")
    print(f"VALID Swin cache: split={args.split}, samples={len(files)}", flush=True)


if __name__ == "__main__":
    main()
