#!/usr/bin/env python3
"""Prepare the Pakistan Flooding official split for first-place fine-tuning."""

from pathlib import Path

import prepare_semeru_official_split_1024 as preparation


preparation.RAW = Path(
    "/homes/j244s673/documents/wsu/phd/pakistan_flooding_preprocessed"
)
preparation.OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/"
    "first_place_pakistan_flooding_FINE_TUNE_OFFICIAL_SPLIT"
)
preparation.DATASET_NAME = "Pakistan Flooding"


if __name__ == "__main__":
    preparation.main()
