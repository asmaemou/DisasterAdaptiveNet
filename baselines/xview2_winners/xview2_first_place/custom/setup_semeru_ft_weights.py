#!/usr/bin/env python3

import shutil
from pathlib import Path

BASE = Path(
    "/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/"
    "baselines/xview2_winners/xview2_first_place"
)
SRC = BASE / "weights" / "extracted_weights" / "weights"
FT = BASE / "weights" / "mount_semeru_finetuned_weights_official_split"
DATASET_NAME = "Mount Semeru"


def copy_initial_weights():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing original xView2 weights: {SRC}")
    FT.mkdir(parents=True, exist_ok=True)
    for source in sorted(SRC.iterdir()):
        if source.is_file():
            destination = FT / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
    print(f"{DATASET_NAME} fine-tuning weight folder:", FT)
    print("Weight files before aliases:", len(list(FT.iterdir())))


def create_aliases():
    aliases = {}
    for seed in [0, 1, 2]:
        aliases[f"res50_loc_{seed}_0_best"] = f"res50_loc_{seed}_tuned_best"
        aliases[f"res50_loc_{seed}_best"] = f"res50_loc_{seed}_tuned_best"
        aliases[f"dpn92_loc_{seed}_0_best"] = f"dpn92_loc_{seed}_tuned_best"
        aliases[f"dpn92_loc_{seed}_best"] = f"dpn92_loc_{seed}_tuned_best"
        aliases[f"se154_loc_{seed}_0_best"] = f"se154_loc_{seed}_1_best"
        aliases[f"se154_loc_{seed}_best"] = f"se154_loc_{seed}_1_best"
        aliases[f"res34_loc_{seed}_0_best"] = f"res34_loc_{seed}_1_best"
        aliases[f"res34_loc_{seed}_best"] = f"res34_loc_{seed}_1_best"
        aliases[f"res34_cls2_{seed}_0_best"] = f"res34_cls2_{seed}_tuned_best"
        aliases[f"res34_cls2_{seed}_best"] = f"res34_cls2_{seed}_tuned_best"
        aliases[f"res50_cls_cce_{seed}_0_best"] = f"res50_cls_cce_{seed}_tuned_best"
        aliases[f"res50_cls_cce_{seed}_best"] = f"res50_cls_cce_{seed}_tuned_best"
        aliases[f"dpn92_cls_cce_{seed}_0_best"] = f"dpn92_cls_cce_{seed}_tuned_best"
        aliases[f"dpn92_cls_cce_{seed}_1_best"] = f"dpn92_cls_cce_{seed}_tuned_best"
        aliases[f"dpn92_cls_cce_{seed}_best"] = f"dpn92_cls_cce_{seed}_tuned_best"
        aliases[f"se154_cls_cce_{seed}_0_best"] = f"se154_cls_cce_{seed}_tuned_best"
        aliases[f"se154_cls_cce_{seed}_1_best"] = f"se154_cls_cce_{seed}_tuned_best"
        aliases[f"se154_cls_cce_{seed}_best"] = f"se154_cls_cce_{seed}_tuned_best"

    for alias, original in aliases.items():
        source = FT / original
        destination = FT / alias
        if source.exists():
            shutil.copy2(source, destination)
            print(f"created/updated: {alias} -> {original}")
        else:
            print(f"WARNING missing source: {original}")

    required = []
    for seed in [0, 1, 2]:
        required.extend(
            [
                f"res50_loc_{seed}_0_best",
                f"dpn92_loc_{seed}_0_best",
                f"res34_cls2_{seed}_0_best",
                f"res50_cls_cce_{seed}_0_best",
                f"dpn92_cls_cce_{seed}_1_best",
                f"se154_cls_cce_{seed}_1_best",
            ]
        )

    missing = [name for name in required if not (FT / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints required by the {DATASET_NAME} fine-tuning scripts:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )
    print("Validated required fine-tuning checkpoints:", len(required))
    print("Total files now:", len(list(FT.iterdir())))


if __name__ == "__main__":
    copy_initial_weights()
    create_aliases()
