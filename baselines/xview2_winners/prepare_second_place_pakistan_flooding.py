#!/usr/bin/env python3
"""Prepare a leak-free Pakistan Flooding split for second-place xView2."""

from pathlib import Path

import prepare_second_place_texas_tornadoes as preparation


preparation.SRC = Path(
    "/homes/j244s673/documents/wsu/phd/pakistan_flooding_preprocessed"
)
preparation.OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/second_place_pakistan_flooding"
)
preparation.DATASET_NAME = "Pakistan Flooding"


if __name__ == "__main__":
    preparation.main()
