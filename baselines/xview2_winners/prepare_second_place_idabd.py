import os
import pandas as pd
from pathlib import Path

SRC = Path("/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_idabd")

image_out = OUT / "images"
mask_out = OUT / "masks"

image_out.mkdir(parents=True, exist_ok=True)
mask_out.mkdir(parents=True, exist_ok=True)

rows = []
debug_rows = []

def safe_symlink(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def get_mask_dir(split_root):
    if (split_root / "masks").exists():
        return split_root / "masks"
    if (split_root / "targets").exists():
        return split_root / "targets"
    raise FileNotFoundError(f"No masks/targets folder found in {split_root}")

def process_split(split, fold_value=None):
    split_root = SRC / split
    img_dir = split_root / "images"
    mask_dir = get_mask_dir(split_root)

    pre_images = sorted(img_dir.glob("*_pre_disaster.png"))

    for pre_img in pre_images:
        sample_id = pre_img.name.replace("_pre_disaster.png", "")
        post_img = img_dir / f"{sample_id}_post_disaster.png"

        pre_mask = mask_dir / f"{sample_id}_pre_disaster.png"
        post_mask = mask_dir / f"{sample_id}_post_disaster.png"

        if not post_img.exists():
            print(f"WARNING: missing post image: {post_img}")
            continue
        if not pre_mask.exists():
            print(f"WARNING: missing pre mask: {pre_mask}")
            continue
        if not post_mask.exists():
            print(f"WARNING: missing post mask: {post_mask}")
            continue

        safe_symlink(pre_img, image_out / pre_img.name)
        safe_symlink(post_img, image_out / post_img.name)
        safe_symlink(pre_mask, mask_out / pre_mask.name)
        safe_symlink(post_mask, mask_out / post_mask.name)

        debug_rows.append({
            "id": sample_id,
            "split": split,
            "fold": fold_value,
        })

        # Only train/val go into folds.csv.
        # fold 0 = validation, fold != 0 = training.
        if fold_value is not None:
            rows.append({
                "id": sample_id,
                "fold": fold_value,
                "nondamage": False,
                "minor": False,
                "major": False,
                "destroyed": False,
                "empty": False,
            })

process_split("train", 1)
process_split("val", 0)
process_split("test", None)

df = pd.DataFrame(rows)
df.to_csv(OUT / "folds.csv", index=False)

debug = pd.DataFrame(debug_rows)
debug.to_csv(OUT / "splits_debug.csv", index=False)

print("Prepared second-place IDA-BD dataset:")
print("SRC:", SRC)
print("OUT:", OUT)
print(debug["split"].value_counts())
print("Train/val rows in folds.csv:", len(df))
print("Images:", len(list(image_out.glob('*.png'))))
print("Masks:", len(list(mask_out.glob('*.png'))))
print("folds.csv:", OUT / "folds.csv")
print("folds.csv columns:", list(df.columns))
print(df.head())
