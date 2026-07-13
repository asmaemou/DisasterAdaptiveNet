from pathlib import Path
import argparse
import os
import shutil
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-id", required=True)
parser.add_argument("--source-test-only", required=True)
args = parser.parse_args()

SRC = Path(args.source_test_only)
OUT = Path(f"/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_{args.dataset_id}_TEST_ONLY")

if not (SRC / "images").exists():
    raise SystemExit(f"ERROR: missing source images folder: {SRC / 'images'}")

if not (SRC / "masks").exists():
    raise SystemExit(f"ERROR: missing source masks folder: {SRC / 'masks'}")

if not (SRC / "folds.csv").exists():
    raise SystemExit(f"ERROR: missing folds.csv: {SRC / 'folds.csv'}")

if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "test" / "images").mkdir(parents=True, exist_ok=True)
(OUT / "test" / "masks").mkdir(parents=True, exist_ok=True)

def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)

for p in sorted((SRC / "images").glob("*.png")):
    link(p, OUT / "test" / "images" / p.name)

for p in sorted((SRC / "masks").glob("*.png")):
    link(p, OUT / "test" / "masks" / p.name)

shutil.copy2(SRC / "folds.csv", OUT / "folds.csv")

# Add root-level images/masks symlinks too, so the code works with either expected layout.
os.symlink((OUT / "test" / "images").resolve(), OUT / "images")
os.symlink((OUT / "test" / "masks").resolve(), OUT / "masks")

df = pd.read_csv(OUT / "folds.csv")

print("Created third-place TEST_ONLY dataset")
print("DATASET_ID:", args.dataset_id)
print("SRC:", SRC)
print("OUT:", OUT)
print("Samples:", len(df))
print("Image links:", len(list((OUT / "test" / "images").iterdir())))
print("Mask links:", len(list((OUT / "test" / "masks").iterdir())))
print("Broken links:", len(list(OUT.glob('**/*'))))
print(df.head())
