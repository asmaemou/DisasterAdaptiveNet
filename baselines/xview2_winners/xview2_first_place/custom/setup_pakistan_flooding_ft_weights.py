#!/usr/bin/env python3
"""Set up first-place checkpoints for Pakistan Flooding fine-tuning."""

import setup_semeru_ft_weights as setup


setup.FT = setup.BASE / "weights" / "pakistan_flooding_finetuned_weights_official_split"
setup.DATASET_NAME = "Pakistan Flooding"


if __name__ == "__main__":
    setup.copy_initial_weights()
    setup.create_aliases()
