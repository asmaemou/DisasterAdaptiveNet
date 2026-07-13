from pathlib import Path
import os
import shutil
import pandas as pd

SRC = Path("/homes/j244s673/documents/wsu/phd/pakistan_flooding_preprocessed")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_pakistan_flooding_TEST_ONLY")

SPLIT = "test"

def symlink_force(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)

def find_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

if OUT.exists():
    print("Removing old folder:", OUT)
    shutil.rmtree(OUT)

# 3rd-place code expects test/images layout.
(OUT / "test" / "images").mkdir(parents=True, exist_ok=True)
(OUT / "test" / "masks").mkdir(parents=True, exist_ok=True)

img_dir = SRC / SPLIT / "images"
mask_dir = SRC / SPLIT / "masks"
target_dir = SRC / SPLIT / "targets"

if not img_dir.exists():
    raise SystemExit(f"ERROR: missing images folder: {img_dir}")

rows = []
missing = []

pre_images = sorted(img_dir.glob("*_pre_disaster.png"))

for pre_img in pre_images:
    tile_id = pre_img.name.replace("_pre_disaster.png", "")
    post_img = img_dir / f"{tile_id}_post_disaster.png"

    pre_mask = find_existing([
        target_dir / f"{tile_id}_pre_disaster_target.png",
        target_dir / f"{tile_id}_pre_disaster.png",
        mask_dir / f"{tile_id}_pre_disaster.png",
    ])

    post_mask = find_existing([
        target_dir / f"{tile_id}_post_disaster_target.png",
        target_dir / f"{tile_id}_post_disaster.png",
        mask_dir / f"{tile_id}_post_disaster.png",
    ])

    if not post_img.exists():
        missing.append((tile_id, "missing post image"))
        continue
    if pre_mask is None:
        missing.append((tile_id, "missing pre mask/target"))
        continue
    if post_mask is None:
        missing.append((tile_id, "missing post mask/target"))
        continue

    symlink_force(pre_img, OUT / "test" / "images" / f"{tile_id}_pre_disaster.png")
    symlink_force(post_img, OUT / "test" / "images" / f"{tile_id}_post_disaster.png")
    symlink_force(pre_mask, OUT / "test" / "masks" / f"{tile_id}_pre_disaster.png")
    symlink_force(post_mask, OUT / "test" / "masks" / f"{tile_id}_post_disaster.png")

    rows.append({
        "id": tile_id,
        "fold": 0,
        "nondamage": False,
        "minor": False,
        "major": False,
        "destroyed": False,
        "empty": False,
    })

if missing:
    print("ERROR: missing required files.")
    for item in missing[:50]:
        print(item)
    print("Total missing:", len(missing))
    raise SystemExit(2)

df = pd.DataFrame(rows)
df = df[["id", "fold", "nondamage", "minor", "major", "destroyed", "empty"]]
df.to_csv(OUT / "folds.csv", index=False)

# Also create root-level images/masks symlinks for metric evaluation and compatibility.
if (OUT / "images").exists() or (OUT / "images").is_symlink():
    (OUT / "images").unlink()
if (OUT / "masks").exists() or (OUT / "masks").is_symlink():
    (OUT / "masks").unlink()

os.symlink((OUT / "test" / "images").resolve(), OUT / "images")
os.symlink((OUT / "test" / "masks").resolve(), OUT / "masks")

print("Prepared Pakistan Flooding TEST_ONLY for 3rd-place xView2 code")
print("SRC:", SRC)
print("OUT:", OUT)
print("Test samples:", len(df))
print("Image links:", len(list((OUT / "test" / "images").iterdir())))
print("Mask links:", len(list((OUT / "test" / "masks").iterdir())))
print("Broken links:", len(list(OUT.glob('**/*'))))
print(df.head())
