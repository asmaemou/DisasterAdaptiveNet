import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAW = Path("/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE")

if not RAW.exists():
    raise FileNotFoundError(f"Missing raw Earthquake Turkey dataset: {RAW}")

if OUT.exists():
    print(f"Removing old folder: {OUT}")
    shutil.rmtree(OUT)

(OUT / "train" / "images").mkdir(parents=True, exist_ok=True)
(OUT / "test" / "images").mkdir(parents=True, exist_ok=True)
(OUT / "masks").mkdir(parents=True, exist_ok=True)

def symlink_force(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def collect_files(root):
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        files.extend(root.rglob(ext))
    return files

def find_file(files, tile_id, kind):
    tile_id = str(tile_id)
    tile_num = tile_id.split("_")[-1]

    hits = []
    for f in files:
        name = f.name.lower()
        if tile_id.lower() not in name and tile_num not in name:
            continue

        if kind == "pre_img":
            if "pre" in name and "post" not in name:
                hits.append(f)
        elif kind == "post_img":
            if "post" in name:
                hits.append(f)
        elif kind == "loc_mask":
            if ("pre" in name or "localization" in name or "building" in name or "loc" in name) and "damage" not in name:
                hits.append(f)
        elif kind == "damage_mask":
            if "post" in name or "damage" in name or "target" in name:
                hits.append(f)

    if not hits:
        raise FileNotFoundError(f"Could not find {kind} for {tile_id}")
    return sorted(hits)[0]

def tile_ids_from_images(image_dir):
    ids = set()
    for f in collect_files(image_dir):
        name = f.name
        if "_pre_disaster" in name:
            ids.add(name.split("_pre_disaster")[0])
        elif "_post_disaster" in name:
            ids.add(name.split("_post_disaster")[0])
    return sorted(ids)

def read_mask(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr

rows = []

for split in ["train", "test"]:
    img_dir = RAW / split / "images"
    mask_dir = RAW / split / "masks"
    target_dir = RAW / split / "targets"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {img_dir}")

    image_files = collect_files(img_dir)

    mask_files = []
    if mask_dir.exists():
        mask_files.extend(collect_files(mask_dir))
    if target_dir.exists():
        mask_files.extend(collect_files(target_dir))

    if not mask_files:
        raise FileNotFoundError(f"Missing masks/targets for split: {split}")

    ids = tile_ids_from_images(img_dir)
    print(split, "samples:", len(ids))

    for i, tile_id in enumerate(ids):
        pre_img = find_file(image_files, tile_id, "pre_img")
        post_img = find_file(image_files, tile_id, "post_img")

        # For localization, prefer pre mask if available; otherwise target is okay because building pixels are >0.
        try:
            loc_mask = find_file(mask_files, tile_id, "loc_mask")
        except Exception:
            loc_mask = find_file(mask_files, tile_id, "damage_mask")

        # For damage, prefer post/target.
        damage_mask = find_file(mask_files, tile_id, "damage_mask")

        out_img_dir = OUT / split / "images"
        symlink_force(pre_img, out_img_dir / f"{tile_id}_pre_disaster.png")
        symlink_force(post_img, out_img_dir / f"{tile_id}_post_disaster.png")

        symlink_force(loc_mask, OUT / "masks" / f"{tile_id}_pre_disaster.png")
        symlink_force(damage_mask, OUT / "masks" / f"{tile_id}_post_disaster.png")

        symlink_force(loc_mask, OUT / "masks" / f"test_localization_{tile_id}_target.png")
        symlink_force(damage_mask, OUT / "masks" / f"test_damage_{tile_id}_target.png")

        if split == "train":
            arr = read_mask(damage_mask)
            vals = set(np.unique(arr).astype(int).tolist())

            rows.append({
                "id": tile_id,
                "fold": i % 3,
                "nondamage": 1 in vals,
                "minor": 2 in vals,
                "major": 3 in vals,
                "destroyed": 4 in vals,
                "empty": not any(v in vals for v in [1, 2, 3, 4]),
            })

df = pd.DataFrame(rows)
df.to_csv(OUT / "folds.csv", index=False)

print("Prepared Earthquake Turkey fine-tuning dataset for 1st-place xView2")
print("RAW:", RAW)
print("OUT:", OUT)
print("Train samples:", len(df))
print("Test images:", len(list((OUT / "test" / "images").iterdir())))
print("Train images:", len(list((OUT / "train" / "images").iterdir())))
print("Masks:", len(list((OUT / "masks").iterdir())))
print(df.head())
