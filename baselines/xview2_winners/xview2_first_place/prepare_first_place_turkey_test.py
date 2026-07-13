import os
import shutil
from pathlib import Path
import pandas as pd

SRC = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_earthquake_turkey_TEST_ONLY")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/first_place_earthquake_turkey_TEST_ONLY")

if not SRC.exists():
    raise FileNotFoundError(f"Missing source dataset: {SRC}")

src_images_candidates = [SRC / "test" / "images", SRC / "images"]
src_images = next((p for p in src_images_candidates if p.exists()), None)
src_masks = SRC / "masks"
src_folds = SRC / "folds.csv"

if src_images is None:
    raise FileNotFoundError(f"Could not find images under {SRC}")
if not src_masks.exists():
    raise FileNotFoundError(f"Missing masks folder: {src_masks}")
if not src_folds.exists():
    raise FileNotFoundError(f"Missing folds.csv: {src_folds}")

if OUT.exists():
    print(f"Removing old folder: {OUT}")
    shutil.rmtree(OUT)

out_images = OUT / "test" / "images"
out_masks = OUT / "masks"
out_images.mkdir(parents=True, exist_ok=True)
out_masks.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src_folds)
if "id" not in df.columns:
    raise ValueError(f"folds.csv must contain id column. Found: {list(df.columns)}")

def symlink_force(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def find_file(root, tile_id, kind):
    tile_id = str(tile_id)
    tile_num = tile_id.split("_")[-1]

    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        files.extend(root.rglob(ext))

    hits = []
    for f in files:
        name = f.name.lower()
        if tile_id.lower() not in name and tile_num.lower() not in name:
            continue

        if kind == "pre":
            if "pre" in name and "post" not in name:
                hits.append(f)
        elif kind == "post":
            if "post" in name:
                hits.append(f)
        elif kind == "loc":
            if ("localization" in name or "localisation" in name or "building" in name or "loc" in name or "pre_disaster" in name) and "damage" not in name:
                hits.append(f)
        elif kind == "damage":
            if "damage" in name or "post_disaster" in name or "dmg" in name:
                hits.append(f)

    if not hits:
        raise FileNotFoundError(f"Could not find {kind} for {tile_id} under {root}")

    return sorted(hits)[0]

missing = []

for tile_id in df["id"].astype(str):
    try:
        pre = find_file(src_images, tile_id, "pre")
        post = find_file(src_images, tile_id, "post")
        loc = find_file(src_masks, tile_id, "loc")
        dmg = find_file(src_masks, tile_id, "damage")

        # 1st-place code expects original xView2-style names
        symlink_force(pre, out_images / f"{tile_id}_pre_disaster.png")
        symlink_force(post, out_images / f"{tile_id}_post_disaster.png")

        # evaluator-friendly ground truth names
        symlink_force(loc, out_masks / f"{tile_id}_pre_disaster.png")
        symlink_force(dmg, out_masks / f"{tile_id}_post_disaster.png")

        # also keep third-place style target names
        symlink_force(loc, out_masks / f"test_localization_{tile_id}_target.png")
        symlink_force(dmg, out_masks / f"test_damage_{tile_id}_target.png")

    except Exception as e:
        missing.append((tile_id, str(e)))

if missing:
    print("ERROR: missing files")
    for x in missing[:40]:
        print(x)
    print("Total missing:", len(missing))
    raise SystemExit(1)

df.to_csv(OUT / "folds.csv", index=False)

print("Prepared Earthquake Turkey TEST_ONLY for 1st-place xView2 code")
print("SRC:", SRC)
print("OUT:", OUT)
print("Test samples:", len(df))
print("Image links:", len(list(out_images.iterdir())))
print("Mask links:", len(list(out_masks.iterdir())))
print(df.head())
