#!/usr/bin/env python3
"""Prepare the Pakistan Flooding official split for third-place fine-tuning."""

from pathlib import Path

import prepare_third_place_turkey_official_split as preparation


preparation.RAW = Path(
    "/homes/j244s673/documents/wsu/phd/pakistan_flooding_preprocessed"
)
preparation.OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/third_place_pakistan_flooding_OFFICIAL_SPLIT"
)
preparation.EVENT_NAME = "pakistan-flooding"
preparation.DATASET_LABEL = "Pakistan Flooding"


if __name__ == "__main__":
    preparation.main()
