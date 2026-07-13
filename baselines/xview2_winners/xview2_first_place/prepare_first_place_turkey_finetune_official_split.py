import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAW = Path("/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT")

TARGET_SIZE = 1024

if not RAW.exists():
    raise FileNotFoundError(f"Missing raw Earthquake Turkey dataset: {RAW}")

for split in ["train", "val", "test"]:
    if not (RAW / split / "images").exists():
        raise FileNotFoundError(f"Missing {split}/images: {RAW / split / 'images'}")
    if not ((RAW / split / "targets").exists() or (RAW / split / "masks").exists()):
        raise FileNotFoundError(f"Missing {split}/targets or {split}/masks under: {RAW / split}")

if OUT.exists():
    print(f"Removing old folder: {OUT}")
    shutil.rmtree(OUT)

for split in ["train", "val", "test"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "masks").mkdir(parents=True, exist_ok=True)

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
    return sorted(files)

def tile_ids_from_images(image_dir):
    ids = set()
    for f in collect_files(image_dir):
        name = f.name
        if "_pre_disaster" in name:
            ids.add(name.split("_pre_disaster")[0])
        elif "_post_disaster" in name:
            ids.add(name.split("_post_disaster")[0])
    return sorted(ids)

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

        elif kind == "damage_mask":
            if "post" in name or "damage" in name or "target" in name:
                hits.append(f)

    if not hits:
        raise FileNotFoundError(f"Could not find {kind} for {tile_id}")

    return sorted(hits)[0]

def fit_1024_image(im, is_mask=False):
    w, h = im.size

    if w == TARGET_SIZE and h == TARGET_SIZE:
        return im

    if w <= TARGET_SIZE and h <= TARGET_SIZE:
        if is_mask:
            canvas = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 0)
            canvas.paste(im.convert("L"), (0, 0))
        else:
            canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
            canvas.paste(im.convert("RGB"), (0, 0))
        return canvas

    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return im.resize((TARGET_SIZE, TARGET_SIZE), resample=resample)

def save_rgb_1024(src, dst):
    im = Image.open(src).convert("RGB")
    im = fit_1024_image(im, is_mask=False)
    im.save(dst)

def read_mask(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)

def save_mask_1024(arr, dst):
    im = Image.fromarray(arr.astype(np.uint8), mode="L")
    im = fit_1024_image(im, is_mask=True)
    im.save(dst)

def write_loc_mask_from_damage(damage_mask_path, out_path):
    arr = read_mask(damage_mask_path)
    loc = (arr > 0).astype(np.uint8) * 255
    save_mask_1024(loc, out_path)

def copy_damage_mask(damage_mask_path, out_path):
    arr = read_mask(damage_mask_path)
    save_mask_1024(arr, out_path)

manifest_rows = []
test_rows = []

for split in ["train", "val", "test"]:
    img_dir = RAW / split / "images"

    mask_files = []
    if (RAW / split / "targets").exists():
        mask_files.extend(collect_files(RAW / split / "targets"))
    if (RAW / split / "masks").exists():
        mask_files.extend(collect_files(RAW / split / "masks"))

    image_files = collect_files(img_dir)
    ids = tile_ids_from_images(img_dir)

    print(f"{split} samples:", len(ids))

    missing = []

    for tile_id in ids:
        try:
            pre_img = find_file(image_files, tile_id, "pre_img")
            post_img = find_file(image_files, tile_id, "post_img")
            damage_mask = find_file(mask_files, tile_id, "damage_mask")

            out_pre_img = OUT / split / "images" / f"{tile_id}_pre_disaster.png"
            out_post_img = OUT / split / "images" / f"{tile_id}_post_disaster.png"
            out_loc_mask = OUT / split / "masks" / f"{tile_id}_pre_disaster.png"
            out_damage_mask = OUT / split / "masks" / f"{tile_id}_post_disaster.png"

            save_rgb_1024(pre_img, out_pre_img)
            save_rgb_1024(post_img, out_post_img)

            write_loc_mask_from_damage(damage_mask, out_loc_mask)
            copy_damage_mask(damage_mask, out_damage_mask)

            symlink_force(out_loc_mask, OUT / "masks" / f"{tile_id}_pre_disaster.png")
            symlink_force(out_damage_mask, OUT / "masks" / f"{tile_id}_post_disaster.png")

            symlink_force(out_loc_mask, OUT / "masks" / f"test_localization_{tile_id}_target.png")
            symlink_force(out_damage_mask, OUT / "masks" / f"test_damage_{tile_id}_target.png")

            arr = read_mask(out_damage_mask)
            vals = set(np.unique(arr).astype(int).tolist())

            row = {
                "id": tile_id,
                "split": split,
                "nondamage": 1 in vals,
                "minor": 2 in vals,
                "major": 3 in vals,
                "destroyed": 4 in vals,
                "empty": not any(v in vals for v in [1, 2, 3, 4]),
            }

            manifest_rows.append(row)

            if split == "test":
                test_rows.append({
                    "id": tile_id,
                    "fold": 0,
                    "nondamage": row["nondamage"],
                    "minor": row["minor"],
                    "major": row["major"],
                    "destroyed": row["destroyed"],
                    "empty": row["empty"],
                })

        except Exception as e:
            missing.append((tile_id, str(e)))

    if missing:
        print("Missing files in split:", split)
        for x in missing[:40]:
            print(x)
        print("Total missing:", len(missing))
        raise SystemExit(1)

pd.DataFrame(manifest_rows).to_csv(OUT / "official_split_manifest.csv", index=False)
pd.DataFrame(test_rows).to_csv(OUT / "folds.csv", index=False)

print("Prepared official train/val/test split for 1st-place xView2 Turkey fine-tuning")
print("All images and masks were saved as 1024x1024 PNG files.")
print("RAW:", RAW)
print("OUT:", OUT)

for split in ["train", "val", "test"]:
    print(split, "images:", len(list((OUT / split / "images").iterdir())))
    print(split, "masks:", len(list((OUT / split / "masks").iterdir())))

print("Root masks:", len(list((OUT / "masks").iterdir())))
print("Test rows in folds.csv:", len(test_rows))
print(pd.DataFrame(test_rows).head())
