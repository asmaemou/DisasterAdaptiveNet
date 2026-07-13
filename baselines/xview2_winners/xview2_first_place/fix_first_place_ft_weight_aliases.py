from pathlib import Path
import shutil

FT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/weights/earthquake_turkey_finetuned_weights_official_split")

if not FT.exists():
    raise FileNotFoundError(f"Missing fine-tuning weight folder: {FT}")

aliases = {}

for seed in [0, 1, 2]:
    aliases[f"res50_loc_{seed}_0_best"] = f"res50_loc_{seed}_tuned_best"
    aliases[f"res50_loc_{seed}_best"] = f"res50_loc_{seed}_tuned_best"

    aliases[f"dpn92_loc_{seed}_0_best"] = f"dpn92_loc_{seed}_tuned_best"
    aliases[f"dpn92_loc_{seed}_best"] = f"dpn92_loc_{seed}_tuned_best"

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
        print(f"missing source: {original}")

print("Total files now:", len(list(FT.iterdir())))
