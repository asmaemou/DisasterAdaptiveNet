#!/usr/bin/env python3
"""Prepare a leak-free Hurricane Delta split for second-place xView2."""

from pathlib import Path

import prepare_second_place_texas_tornadoes as preparation


preparation.SRC = Path(
    "/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed"
)
preparation.OUT = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/second_place_hurricane_delta"
)
preparation.DATASET_NAME = "Hurricane Delta"


if __name__ == "__main__":
    preparation.main()
