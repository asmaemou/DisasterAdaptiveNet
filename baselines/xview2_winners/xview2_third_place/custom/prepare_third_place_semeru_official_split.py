#!/usr/bin/env python3
"""Prepare the Mount Semeru official split for third-place fine-tuning."""

from pathlib import Path

import prepare_third_place_turkey_official_split as preparation


preparation.RAW = Path(
    "/homes/j244s673/documents/wsu/phd/mount_semeru_eruption_preprocessed"
)
preparation.OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/third_place_mount_semeru_OFFICIAL_SPLIT"
)
preparation.EVENT_NAME = "mount-semeru-eruption"
preparation.DATASET_LABEL = "Mount Semeru"


if __name__ == "__main__":
    preparation.main()
