
import os

import shutil

from pathlib import Path

import pandas as pd



SRC = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_texas_tornadoes_TEST_ONLY")

OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_texas_tornadoes_TEST_ONLY")



if not SRC.exists():

    raise FileNotFoundError(f"Missing source TEST_ONLY dataset: {SRC}")



src_images = SRC / "images"

src_masks = SRC / "masks"

src_folds = SRC / "folds.csv"



if not src_images.exists():

    raise FileNotFoundError(f"Missing images folder: {src_images}")

if not src_masks.exists():

    raise FileNotFoundError(f"Missing masks folder: {src_masks}")

if not src_folds.exists():

    raise FileNotFoundError(f"Missing folds.csv: {src_folds}")



if OUT.exists():

    print(f"Removing old folder: {OUT}")

    shutil.rmtree(OUT)



out_images = OUT / "images"

out_masks = OUT / "masks"

out_images.mkdir(parents=True, exist_ok=True)

out_masks.mkdir(parents=True, exist_ok=True)



df = pd.read_csv(src_folds)



if "id" not in df.columns:

    raise ValueError(f"folds.csv must contain an id column. Columns found: {list(df.columns)}")



ids = df["id"].astype(str).tolist()



def symlink_force(src, dst):

    dst = Path(dst)

    if dst.exists() or dst.is_symlink():

        dst.unlink()

    os.symlink(src, dst)



def find_one(root, tile_id, kind):

    tile_id = str(tile_id)



    if kind == "pre":

        keywords = ["pre_disaster", "_pre_", "pre"]

        reject = ["post", "damage", "localization", "target", "mask"]

    elif kind == "post":

        keywords = ["post_disaster", "_post_", "post"]

        reject = ["pre", "damage", "localization", "target", "mask"]

    elif kind == "localization":

        keywords = ["localization", "localisation", "building", "loc"]

        reject = ["damage"]

    elif kind == "damage":

        keywords = ["damage", "dmg", "target"]

        reject = []

    else:

        raise ValueError(kind)



    files = []

    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:

        files.extend(root.rglob(ext))



    candidates = []

    for f in files:

        name = f.name.lower()

        full = str(f).lower()

        if tile_id.lower() not in name and tile_id.split("_")[-1].lower() not in name:

            continue

        if any(r in name for r in reject):

            continue

        if any(k in name or k in full for k in keywords):

            candidates.append(f)



    if candidates:

        return sorted(candidates)[0]



    # fallback: for masks, sometimes only id + target naming exists

    if kind in ["localization", "damage"]:

        fallback = [f for f in files if tile_id.lower() in f.name.lower()]

        if fallback:

            return sorted(fallback)[0]



    raise FileNotFoundError(f"Could not find {kind} file for {tile_id} under {root}")



rows = []

missing = []



for tile_id in ids:

    try:

        pre = find_one(src_images, tile_id, "pre")

        post = find_one(src_images, tile_id, "post")

        loc = find_one(src_masks, tile_id, "localization")

        dmg = find_one(src_masks, tile_id, "damage")



        symlink_force(pre, out_images / f"test_pre_{tile_id}.png")

        symlink_force(post, out_images / f"test_post_{tile_id}.png")

        symlink_force(loc, out_masks / f"test_localization_{tile_id}_target.png")

        symlink_force(dmg, out_masks / f"test_damage_{tile_id}_target.png")



        rows.append({

            "id": tile_id,

            "fold": 0,

            "nondamage": False,

            "minor": False,

            "major": False,

            "destroyed": False,

            "empty": False,

        })

    except Exception as e:

        missing.append((tile_id, str(e)))



if missing:

    print("ERROR: missing files")

    for item in missing[:30]:

        print(item)

    print("Total missing:", len(missing))

    raise SystemExit(1)



out_df = pd.DataFrame(rows)

out_df.to_csv(OUT / "folds.csv", index=False)



print("Prepared Texas Tornadoes TEST_ONLY for 3rd-place xView2 code")

print("SRC:", SRC)

print("OUT:", OUT)

print("Test samples:", len(out_df))

print("Image links:", len(list(out_images.iterdir())))

print("Mask links:", len(list(out_masks.iterdir())))

print(out_df.head())


# Also create xView2 third-place expected layout:

# predict_37_weighted.py expects DATA/test/images

test_dir = OUT / "test"

test_dir.mkdir(parents=True, exist_ok=True)



test_images_link = test_dir / "images"

test_masks_link = test_dir / "masks"



for link, target in [(test_images_link, Path("../images")), (test_masks_link, Path("../masks"))]:

    if link.exists() or link.is_symlink():

        link.unlink()

    os.symlink(target, link)



print("Created expected test/images and test/masks links")



