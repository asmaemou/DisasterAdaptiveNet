#!/usr/bin/env python3
"""Create the held-out Texas Tornadoes test view for evaluation."""

import os
import shutil
from pathlib import Path

import pandas as pd


FULL = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/second_place_texas_tornadoes"
)
TEST = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/second_place_texas_tornadoes_TEST_ONLY"
)
DATASET_NAME = "Texas Tornadoes"


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.symlink(source.resolve(), destination)


def main():
    folds = pd.read_csv(FULL / "folds.csv")
    test_folds = folds[folds["split"].astype(str).str.lower() == "test"].copy()
    if test_folds.empty:
        raise RuntimeError(f"No {DATASET_NAME} test rows found")

    if TEST.exists():
        shutil.rmtree(TEST)
    (TEST / "images").mkdir(parents=True, exist_ok=True)
    (TEST / "masks").mkdir(parents=True, exist_ok=True)

    missing = []
    for tile_id in test_folds["id"].astype(str):
        for folder, suffix in [
            ("images", "pre_disaster"),
            ("images", "post_disaster"),
            ("masks", "pre_disaster"),
            ("masks", "post_disaster"),
        ]:
            source = FULL / folder / f"{tile_id}_{suffix}.png"
            if not source.exists():
                missing.append(str(source))
            else:
                link(source, TEST / folder / source.name)

    if missing:
        raise FileNotFoundError(
            f"Missing {DATASET_NAME} test files:\n" + "\n".join(missing[:50])
        )

    test_folds["fold"] = 0
    columns = ["id", "fold", "nondamage", "minor", "major", "destroyed", "empty"]
    test_folds[columns].to_csv(TEST / "folds.csv", index=False)

    print(f"Created {DATASET_NAME} TEST_ONLY folder")
    print("TEST:", TEST)
    print("Test samples:", len(test_folds))
    print("Image links:", len(list((TEST / "images").iterdir())))
    print("Mask links:", len(list((TEST / "masks").iterdir())))


if __name__ == "__main__":
    main()
