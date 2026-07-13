import os
import pandas as pd
from pathlib import Path

SRC = Path("/homes/j244s673/documents/wsu/phd/hurricane_dorian_preprocessed")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_hurricane_dorian")

image_out = OUT / "images"
mask_out = OUT / "masks"

image_out.mkdir(parents=True, exist_ok=True)
mask_out.mkdir(parents=True, exist_ok=True)

rows = []

def safe_symlink(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def process_split(split, fold_value):
    split_root = SRC / split
    img_dir = split_root / "images"
    mask_dir = split_root / "masks"

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

        rows.append({
            "id": sample_id,
            "fold": fold_value,
            "nondamage": False,
            "minor": False,
            "major": False,
            "destroyed": False,
            "empty": False,
            "split": split,
        })

# fold 0 = validation, fold != 0 = training for this repo
process_split("train", 1)
process_split("val", 0)

df = pd.DataFrame(rows)

# Keep the exact expected columns for the second-place xView2 loader.
df_for_repo = df[["id", "fold", "nondamage", "minor", "major", "destroyed", "empty"]]
df_for_repo.to_csv(OUT / "folds.csv", index=False)

# Save a more informative copy for our own checking.
df.to_csv(OUT / "folds_with_split_debug.csv", index=False)

print("Prepared second-place Dorian dataset:")
print("OUT:", OUT)
print(df["split"].value_counts())
print("Total image pairs:", len(df))
print("Images:", len(list(image_out.glob('*.png'))))
print("Masks:", len(list(mask_out.glob('*.png'))))
print("folds.csv:", OUT / "folds.csv")
print("folds.csv columns:", list(df_for_repo.columns))
print(df_for_repo.head())