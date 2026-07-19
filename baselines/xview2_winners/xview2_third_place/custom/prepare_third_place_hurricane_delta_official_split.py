#!/usr/bin/env python3
"""Prepare the Hurricane Delta official split for third-place fine-tuning."""

from pathlib import Path

import prepare_third_place_turkey_official_split as preparation


preparation.RAW = Path(
    "/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed"
)
preparation.OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/third_place_hurricane_delta_OFFICIAL_SPLIT"
)
preparation.EVENT_NAME = "hurricane-delta"
preparation.DATASET_LABEL = "Hurricane Delta"


if __name__ == "__main__":
    preparation.main()
