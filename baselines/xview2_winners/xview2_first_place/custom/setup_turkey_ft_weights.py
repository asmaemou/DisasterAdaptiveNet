#!/usr/bin/env python3

from pathlib import Path
import shutil

BASE = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place")
SRC = BASE / "weights" / "extracted_weights" / "weights"
FT = BASE / "weights" / "earthquake_turkey_finetuned_weights_official_split"


def copy_initial_weights():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing original xView2 weights: {SRC}")

    FT.mkdir(parents=True, exist_ok=True)

    for src in sorted(SRC.iterdir()):
        if src.is_file():
            dst = FT / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    print("Fine-tuning weight folder:", FT)
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
        aliases[f"dpn92_cls_cce_{seed}_best"] = f"dpn92_cls_cce_{seed}_tuned_best"

        aliases[f"se154_cls_cce_{seed}_0_best"] = f"se154_cls_cce_{seed}_tuned_best"
        aliases[f"se154_cls_cce_{seed}_best"] = f"se154_cls_cce_{seed}_tuned_best"

    for alias, original in aliases.items():
        src = FT / original
        dst = FT / alias

        if src.exists():
            shutil.copy2(src, dst)
            print(f"created/updated: {alias} -> {original}")
        else:
            print(f"WARNING missing source: {original}")

    print("Total files now:", len(list(FT.iterdir())))


if __name__ == "__main__":
    copy_initial_weights()
    create_aliases()
