#!/usr/bin/env python3
"""Expose the original first-place xBD checkpoints under inference aliases.

The locally patched first-place prediction scripts use ``*_tuned_best`` names.
For this zero-shot branch those names must resolve to the original released xBD
checkpoints, not to any Turkey-fine-tuned checkpoint.  This script creates a
small directory of symbolic links so the source files remain unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def checkpoint_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for seed in range(3):
        mapping.update(
            {
                f"res50_loc_{seed}_tuned_best": f"res50_loc_{seed}_0_best",
                f"dpn92_loc_{seed}_tuned_best": f"dpn92_loc_{seed}_0_best",
                f"res34_loc_{seed}_1_best": f"res34_loc_{seed}_1_best",
                f"se154_loc_{seed}_1_best": f"se154_loc_{seed}_1_best",
                f"res34_cls2_{seed}_tuned_best": f"res34_cls2_{seed}_0_best",
                f"res50_cls_cce_{seed}_tuned_best": f"res50_cls_cce_{seed}_0_best",
                f"dpn92_cls_cce_{seed}_tuned_best": f"dpn92_cls_cce_{seed}_1_best",
                f"se154_cls_cce_{seed}_tuned_best": f"se154_cls_cce_{seed}_1_best",
            }
        )
    return mapping


def parse_args() -> argparse.Namespace:
    project = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    first_place = project / "baselines/xview2_winners/xview2_first_place"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=first_place / "weights/extracted_weights/weights",
        help="Directory containing the original released first-place xBD weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project
        / "output/hybrid_swin_hrtbda_first_place_xbd_zero_shot_turkey/first_place_xbd_inference_weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"Original first-place xBD weight directory is missing: {args.source}")

    mapping = checkpoint_mapping()
    missing = sorted({source for source in mapping.values() if not (args.source / source).is_file()})
    if missing:
        raise FileNotFoundError(
            "Missing original first-place xBD checkpoints:\n"
            + "\n".join(f"  - {args.source / name}" for name in missing)
        )

    args.output.mkdir(parents=True, exist_ok=True)
    expected = set(mapping)
    for stale in args.output.iterdir():
        if stale.name not in expected:
            raise RuntimeError(
                f"Unexpected file in the dedicated alias directory: {stale}. "
                "Remove it manually before rerunning."
            )

    for alias, source_name in sorted(mapping.items()):
        source = (args.source / source_name).resolve()
        destination = args.output / alias
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() and destination.resolve() == source:
                continue
            raise RuntimeError(
                f"Refusing to replace an existing non-matching checkpoint alias: {destination}"
            )
        os.symlink(source, destination)
        print(f"{alias} -> {source}", flush=True)

    print(f"Validated {len(mapping)} inference aliases.", flush=True)
    print("All aliases point to original released first-place xBD checkpoints.", flush=True)
    print(f"Alias directory: {args.output}", flush=True)


if __name__ == "__main__":
    main()
