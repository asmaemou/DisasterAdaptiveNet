from pathlib import Path
import os
import shutil
import pandas as pd

SRC = Path("/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed")
OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_earthquake_turkey")

SPLITS = ["train", "val", "test"]

# fold convention:
# train = 1
# val   = 0
# test  = 2
# We also save a "split" column so later evaluation can explicitly select test.
FOLD_MAP = {
    "train": 1,
    "val": 0,
    "test": 2,
}

def symlink_force(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def find_file(candidates):
    for c in candidates:
        if c.exists():
            return c
    return None

def get_tile_id(pre_path):
    return pre_path.name.replace("_pre_disaster.png", "")

if OUT.exists():
    print(f"Removing old prepared folder: {OUT}")
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "masks").mkdir(parents=True, exist_ok=True)

rows = []
missing = []
split_counts = {}

for split in SPLITS:
    img_dir = SRC / split / "images"
    mask_dir = SRC / split / "masks"
    target_dir = SRC / split / "targets"

    if not img_dir.exists():
        missing.append((split, "missing image folder", str(img_dir)))
        continue

    pre_files = sorted(img_dir.glob("*_pre_disaster.png"))
    split_counts[split] = 0

    for pre_path in pre_files:
        tile_id = get_tile_id(pre_path)
        post_path = img_dir / f"{tile_id}_post_disaster.png"

        if not post_path.exists():
            missing.append((split, tile_id, "missing post image"))
            continue

        # For masks, prefer targets first because targets usually contain class labels.
        pre_mask = find_file([
            target_dir / f"{tile_id}_pre_disaster.png",
            mask_dir / f"{tile_id}_pre_disaster.png",
        ])

        post_mask = find_file([
            target_dir / f"{tile_id}_post_disaster.png",
            mask_dir / f"{tile_id}_post_disaster.png",
            target_dir / f"{tile_id}.png",
            mask_dir / f"{tile_id}.png",
        ])

        if pre_mask is None:
            missing.append((split, tile_id, "missing pre mask/target"))
            continue

        if post_mask is None:
            missing.append((split, tile_id, "missing post mask/target"))
            continue

        symlink_force(pre_path, OUT / "images" / pre_path.name)
        symlink_force(post_path, OUT / "images" / post_path.name)

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

        split_counts[split] += 1

if missing:
    print("ERROR: missing required files.")
    print("First 40 missing items:")
    for item in missing[:40]:
        print(item)
    print("Total missing:", len(missing))
    raise SystemExit(2)

df = pd.DataFrame(rows)
df = df[["id", "fold", "split", "nondamage", "minor", "major", "destroyed", "empty"]]
df.to_csv(OUT / "folds.csv", index=False)

# The second-place loader defines training rows as every row whose fold differs
# from the validation fold. Keep test rows out of the training CSV entirely so
# fold 2 can never leak into fine-tuning when validation uses fold 0.
train_val_df = df[df["split"].isin(["train", "val"])].copy()
train_val_df.to_csv(OUT / "folds_train_val.csv", index=False)

train_ids = set(train_val_df[train_val_df["split"] == "train"]["id"].astype(str))
val_ids = set(train_val_df[train_val_df["split"] == "val"]["id"].astype(str))
test_ids = set(df[df["split"] == "test"]["id"].astype(str))

if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
    raise RuntimeError("Earthquake Turkey train/val/test IDs are not disjoint")

if set(train_val_df[train_val_df["fold"] != 0]["split"]) != {"train"}:
    raise RuntimeError("folds_train_val.csv contains non-training rows outside fold 0")

if set(train_val_df[train_val_df["fold"] == 0]["split"]) != {"val"}:
    raise RuntimeError("folds_train_val.csv fold 0 is not exclusively validation")

print("Prepared second-place Earthquake Turkey dataset from scratch")
print("SRC:", SRC)
print("OUT:", OUT)
print()
print("Split counts:")
for k, v in split_counts.items():
    print(f"{k}: {v}")
print()
print("Total image pairs:", len(df))
print("Image symlinks:", len(list((OUT / "images").iterdir())))
print("Mask symlinks:", len(list((OUT / "masks").iterdir())))
print("folds.csv:", OUT / "folds.csv")
print("Training folds:", OUT / "folds_train_val.csv")
print("Training rows:", len(train_ids))
print("Validation rows:", len(val_ids))
print("Held-out test rows:", len(test_ids))
print("folds.csv columns:", list(df.columns))
print(df.head())
