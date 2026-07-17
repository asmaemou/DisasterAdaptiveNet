#!/usr/bin/env python3
"""Set up first-place checkpoints for Hurricane Delta fine-tuning."""

import setup_semeru_ft_weights as setup


setup.FT = setup.BASE / "weights" / "hurricane_delta_finetuned_weights_official_split"
setup.DATASET_NAME = "Hurricane Delta"


if __name__ == "__main__":
    setup.copy_initial_weights()
    setup.create_aliases()
