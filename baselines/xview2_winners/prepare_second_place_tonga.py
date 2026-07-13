from pathlib import Path
import os
import shutil
import pandas as pd

SRC = Path("/homes/j244s673/documents/wsu/phd/tonga_volcano_preprocessed")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_tonga_volcano")

SPLITS = ["train", "val", "test"]
FOLD_MAP = {"train": 1, "val": 0, "test": 2}

def symlink_force(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def find_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

if OUT.exists():
    print(f"Removing old prepared folder: {OUT}")
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "masks").mkdir(parents=True, exist_ok=True)

rows = []
missing = []

for split in SPLITS:
    img_dir = SRC / split / "images"
    mask_dir = SRC / split / "masks"
    target_dir = SRC / split / "targets"

    if not img_dir.exists():
        missing.append((split, "missing images folder", str(img_dir)))
        continue

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
            missing.append((split, tile_id, "missing post image"))
            continue
        if pre_mask is None:
            missing.append((split, tile_id, "missing pre mask/target"))
            continue
        if post_mask is None:
            missing.append((split, tile_id, "missing post mask/target"))
            continue

        symlink_force(pre_img, OUT / "images" / f"{tile_id}_pre_disaster.png")
        symlink_force(post_img, OUT / "images" / f"{tile_id}_post_disaster.png")
        symlink_force(pre_mask, OUT / "masks" / f"{tile_id}_pre_disaster.png")
        symlink_force(post_mask, OUT / "masks" / f"{tile_id}_post_disaster.png")

        rows.append({
            "id": tile_id,
            "fold": FOLD_MAP[split],
            "split": split,
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
df = df[["id", "fold", "split", "nondamage", "minor", "major", "destroyed", "empty"]]
df.to_csv(OUT / "folds.csv", index=False)

print("Prepared Tonga Volcano for 2nd-place xView2 code")
print("SRC:", SRC)
print("OUT:", OUT)
print()
print("Split counts:")
print(df["split"].value_counts())
print()
print("Total image pairs:", len(df))
print("Image symlinks:", len(list((OUT / "images").iterdir())))
print("Mask symlinks:", len(list((OUT / "masks").iterdir())))
print("folds.csv:", OUT / "folds.csv")
print(df.head())
