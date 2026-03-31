#!/bin/bash
#SBATCH --job-name=check255
#SBATCH --output=check255_%j.out
#SBATCH --error=check255_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# go to the folder you submit from
cd "$SLURM_SUBMIT_DIR"

# uncomment/edit these only if your cluster needs them
# module load python
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

python <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def is_img(p):
    return p.is_file() and p.suffix.lower() in IMG_EXTS

for split in ["train", "val", "test"]:
    root = Path(split)
    loc_dir = root / "masks" / "localization"
    dmg_dir = root / "masks" / "damage"

    total_255 = 0
    inside_building_255 = 0
    outside_building_255 = 0
    imgs_with_255 = 0

    for dmg_path in dmg_dir.rglob("*"):
        if not is_img(dmg_path):
            continue

        rel = dmg_path.relative_to(dmg_dir)
        loc_path = loc_dir / rel
        if not loc_path.exists():
            continue

        dmg = np.array(Image.open(dmg_path))
        loc = np.array(Image.open(loc_path))
        loc_bin = (loc > 0)

        mask255 = (dmg == 255)
        n255 = int(mask255.sum())

        if n255 > 0:
            imgs_with_255 += 1

        total_255 += n255
        inside_building_255 += int((mask255 & loc_bin).sum())
        outside_building_255 += int((mask255 & ~loc_bin).sum())

    print(f"\n=== {split} ===")
    print("images with 255:", imgs_with_255)
    print("total 255 pixels:", total_255)
    print("255 inside buildings:", inside_building_255)
    print("255 outside buildings:", outside_building_255)
PY