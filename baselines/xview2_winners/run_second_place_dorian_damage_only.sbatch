#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodelist=compute202517
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=10:00:00
#SBATCH --job-name=xv2_2nd_dorian_dmg
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_dorian_dmg_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_dorian_dmg_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners
REPO=$BASE/xview2_second_place
DATA_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_hurricane_dorian
EXP_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_hurricane_dorian_finetune

mkdir -p "$BASE/logs"
mkdir -p "$EXP_OUT"
mkdir -p "$EXP_OUT/weights_localization"
mkdir -p "$EXP_OUT/weights_damage"
mkdir -p "$EXP_OUT/logs_localization"
mkdir -p "$EXP_OUT/logs_damage"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found."
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "================================================"
echo "Prepare Dorian dataset for xView2 2nd-place baseline"
echo "================================================"

cd "$BASE"
python -u prepare_second_place_dorian.py

echo "================================================"
echo "Patch folds.csv for second-place dataset loader"
echo "================================================"

python - <<PY2
import pandas as pd
from pathlib import Path

folds = Path("$DATA_OUT") / "folds.csv"
df = pd.read_csv(folds)

for col in ["nondamage", "minor", "major", "destroyed", "empty"]:
    if col not in df.columns:
        df[col] = False

df = df[["id", "fold", "nondamage", "minor", "major", "destroyed", "empty"]]
df.to_csv(folds, index=False)

print("Updated folds.csv columns:", list(df.columns))
print(df.head())
PY2

echo "================================================"
echo "Check second-place configs"
echo "================================================"

cd "$REPO"
ls -lh configs

LOC_CONFIG="configs/d92_loc_512.json"
CLS_CONFIG="configs/d92_softmax_512.json"

echo "LOC_CONFIG=$LOC_CONFIG"
echo "CLS_CONFIG=$CLS_CONFIG"

if [ -z "$LOC_CONFIG" ]; then
  echo "ERROR: could not find localization DPN92 config."
  exit 1
fi

if [ -z "$CLS_CONFIG" ]; then
  echo "ERROR: could not find classification Siamese DPN92 config."
  exit 1
fi

LOC_SRC=weights/localization_dpn_unet_dpn92_0_best_dice
CLS_SRC=weights/softmax_dpn_seamese_unet_shared_dpn92_0_best_xview

LOC_INIT=$EXP_OUT/localization_dpn_unet_dpn92_0_epoch0_init
CLS_INIT=$EXP_OUT/softmax_dpn_seamese_unet_shared_dpn92_0_epoch0_init

echo "================================================"
echo "Reset checkpoint epochs so pretrained weights act as initialization"
echo "================================================"

python - <<PY
import torch
from pathlib import Path

pairs = [
    ("$LOC_SRC", "$LOC_INIT"),
    ("$CLS_SRC", "$CLS_INIT"),
]

for src, dst in pairs:
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(src)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    ckpt["epoch"] = 0
    ckpt["dice_best"] = 0
    ckpt["xview_best"] = 0

    for k in ["optimizer", "scheduler", "amp"]:
        if k in ckpt:
            ckpt[k] = None

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)

    print(f"Saved init checkpoint: {dst}")
PY

echo "================================================"
echo "Skip localization fine-tuning because it already completed"
echo "================================================"
ls -lh "$EXP_OUT/weights_localization" || true

echo "================================================"
echo "Fine-tune damage classification model"
echo "================================================"

python -u train.py \
  --config "$CLS_CONFIG" \
  --workers 8 \
  --gpu 0 \
  --output-dir "$EXP_OUT/weights_damage" \
  --resume "$CLS_INIT" \
  --fold 0 \
  --prefix dorian_damage_ \
  --data-dir "$DATA_OUT" \
  --folds-csv "$DATA_OUT/folds.csv" \
  --logdir "$EXP_OUT/logs_damage" \
  --opt-level O0 \
  --test_every 1

echo "================================================"
echo "DONE: second-place xView2 baseline fine-tuned on Hurricane Dorian"
echo "Output directory:"
echo "$EXP_OUT"
echo "================================================"

find "$EXP_OUT" -maxdepth 4 -type f | sort