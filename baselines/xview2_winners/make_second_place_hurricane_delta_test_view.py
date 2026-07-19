#!/usr/bin/env python3
"""Create the held-out Hurricane Delta test view for evaluation."""

from pathlib import Path

import make_second_place_texas_tornadoes_test_view as test_view


test_view.FULL = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/second_place_hurricane_delta"
)
test_view.TEST = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/"
    "xview2_baseline_datasets/second_place_hurricane_delta_TEST_ONLY"
)
test_view.DATASET_NAME = "Hurricane Delta"


if __name__ == "__main__":
    test_view.main()
