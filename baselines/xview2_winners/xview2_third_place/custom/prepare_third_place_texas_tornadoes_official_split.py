#!/usr/bin/env python3
"""Prepare the Texas Tornadoes official split for third-place fine-tuning."""

from pathlib import Path

import prepare_third_place_turkey_official_split as preparation


preparation.RAW = Path(
    "/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed"
)
preparation.OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/third_place_texas_tornadoes_OFFICIAL_SPLIT"
)
preparation.EVENT_NAME = "texas-tornadoes"
preparation.DATASET_LABEL = "Texas Tornadoes"


if __name__ == "__main__":
    preparation.main()
