from pathlib import Path
import os
import shutil
import pandas as pd

FULL = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_mount_semeru")
TEST = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_mount_semeru_TEST_ONLY")

df = pd.read_csv(FULL / "folds.csv")
test_df = df[df["split"].astype(str).str.lower() == "test"].copy()

if len(test_df) == 0:
    raise SystemExit("ERROR: no test rows found.")

if TEST.exists():
    shutil.rmtree(TEST)

(TEST / "images").mkdir(parents=True, exist_ok=True)
(TEST / "masks").mkdir(parents=True, exist_ok=True)

def link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)

missing = []

for tile_id in test_df["id"]:
    files = [
        FULL / "images" / f"{tile_id}_pre_disaster.png",
        FULL / "images" / f"{tile_id}_post_disaster.png",
        FULL / "masks" / f"{tile_id}_pre_disaster.png",
        FULL / "masks" / f"{tile_id}_post_disaster.png",
    ]

    for src in files:
        if not src.exists() and not src.is_symlink():
            missing.append(str(src))
        else:
            link(src, TEST / src.parent.name / src.name)

if missing:
    print("Missing files:")
    for m in missing[:50]:
        print(m)
    print("Total missing:", len(missing))
    raise SystemExit(2)

test_df["fold"] = 0

cols = ["id", "fold", "nondamage", "minor", "major", "destroyed", "empty"]
test_df[cols].to_csv(TEST / "folds.csv", index=False)

print("Created Mount Semeru TEST_ONLY folder")
print("TEST:", TEST)
print("Test samples:", len(test_df))
print("Image links:", len(list((TEST / "images").iterdir())))
print("Mask links:", len(list((TEST / "masks").iterdir())))
print(TEST / "folds.csv")
print(test_df[cols].head())
