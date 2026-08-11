#!/usr/bin/env python3
"""Report the exact parameter count of the implemented HBG-CAON model."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal import (
    BuildingGuidedCrossAttentionOrdinalNet,
)


def count_parameters(module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def format_count(value: int) -> str:
    return f"{value:,} ({value / 1_000_000:.3f} M)"


def main() -> None:
    # Parameter count is independent of the dataset and trained values. The
    # constructor loads ImageNet initialization exactly as the experiments do.
    model = BuildingGuidedCrossAttentionOrdinalNet(image_size=896, width=96)
    total, trainable = count_parameters(model)

    print("=" * 84)
    print("Hybrid Building-Guided Cross-Attention Ordinal Network (HBG-CAON)")
    print("Siamese ResNet34 U-Net + Siamese Swin-Tiny; shared weights per time point")
    print("=" * 84)
    print(f"Total parameters:     {format_count(total)}")
    print(f"Trainable parameters: {format_count(trainable)}")
    print(f"Frozen parameters:    {format_count(total - trainable)}")
    print("\nTop-level component breakdown (non-overlapping):")

    component_sum = 0
    for name, child in model.named_children():
        child_total, child_trainable = count_parameters(child)
        component_sum += child_total
        print(
            f"  {name:28s} total={format_count(child_total):>24s}  "
            f"trainable={format_count(child_trainable)}"
        )

    if component_sum != total:
        raise RuntimeError(
            f"Component sum {component_sum:,} does not match model total {total:,}"
        )

    print("\nImportant: the Siamese pre/post branches share encoder weights.")
    print("Their parameters are therefore counted once, not twice.")


if __name__ == "__main__":
    main()
