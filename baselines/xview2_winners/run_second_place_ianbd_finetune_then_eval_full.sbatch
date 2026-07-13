#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=11:00:00
#SBATCH --job-name=xv2_2nd_ianbd_ft
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_ianbd_ft_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_ianbd_ft_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners
REPO=$BASE/xview2_second_place
MANIFEST=$BASE/second_place_full_solution_manifest.tsv

DATA_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_ianbd
EXP_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_ianbd_FULL_SOLUTION_finetune

mkdir -p "$BASE/logs"
mkdir -p "$EXP_OUT"
mkdir -p "$EXP_OUT/init"
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
echo "Prepare IAN-BD dataset"
echo "================================================"

cd "$BASE"
python -u prepare_second_place_ianbd.py

echo "================================================"
echo "Run all second-place localization models on IAN-BD"
echo "================================================"

cd "$REPO"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r TASK FAMILY TAG CONFIG WEIGHT; do
  if [ "$TASK" != "loc" ]; then
    continue
  fi

  echo "================================================"
  echo "Localization model: $TAG"
  echo "Config: $CONFIG"
  echo "Weight: $WEIGHT"
  echo "================================================"

  INIT="$EXP_OUT/init/${TAG}_epoch0_init"

  python - <<PY
import torch
from pathlib import Path

src = Path("$WEIGHT")
dst = Path("$INIT")
ckpt = torch.load(src, map_location="cpu", weights_only=False)

ckpt["epoch"] = 0
ckpt["dice_best"] = 0
ckpt["xview_best"] = 0

for k in ["optimizer", "scheduler", "amp"]:
    if k in ckpt:
        ckpt[k] = None

dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, dst)
print("Saved init:", dst)
PY

  mkdir -p "$EXP_OUT/weights_localization/$TAG"
  mkdir -p "$EXP_OUT/logs_localization/$TAG"

  python -u train_localization.py \
    --config "$CONFIG" \
    --workers 8 \
    --gpu 0 \
    --output-dir "$EXP_OUT/weights_localization/$TAG" \
    --resume "$INIT" \
    --fold 0 \
    --prefix "ianbd_full_${TAG}_" \
    --data-dir "$DATA_OUT" \
    --folds-csv "$DATA_OUT/folds.csv" \
    --logdir "$EXP_OUT/logs_localization/$TAG" \
    --opt-level O0 \
    --test_every 1
done

echo "================================================"
echo "Run all second-place damage classification models on IAN-BD"
echo "================================================"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r TASK FAMILY TAG CONFIG WEIGHT; do
  if [ "$TASK" != "damage" ]; then
    continue
  fi

  echo "================================================"
  echo "Damage model: $TAG"
  echo "Config: $CONFIG"
  echo "Weight: $WEIGHT"
  echo "================================================"

  INIT="$EXP_OUT/init/${TAG}_epoch0_init"

  python - <<PY
import torch
from pathlib import Path

src = Path("$WEIGHT")
dst = Path("$INIT")
ckpt = torch.load(src, map_location="cpu", weights_only=False)

ckpt["epoch"] = 0
ckpt["dice_best"] = 0
ckpt["xview_best"] = 0

for k in ["optimizer", "scheduler", "amp"]:
    if k in ckpt:
        ckpt[k] = None

dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, dst)
print("Saved init:", dst)
PY

  mkdir -p "$EXP_OUT/weights_damage/$TAG"
  mkdir -p "$EXP_OUT/logs_damage/$TAG"

  python -u train.py \
    --config "$CONFIG" \
    --workers 8 \
    --gpu 0 \
    --output-dir "$EXP_OUT/weights_damage/$TAG" \
    --resume "$INIT" \
    --fold 0 \
    --prefix "ianbd_full_${TAG}_" \
    --data-dir "$DATA_OUT" \
    --folds-csv "$DATA_OUT/folds.csv" \
    --logdir "$EXP_OUT/logs_damage/$TAG" \
    --opt-level O0 \
    --test_every 1
done

echo "================================================"
echo "Training completed. Now run final IAN-BD test evaluation."
echo "================================================"

cd "$BASE"
python -u eval_second_place_ianbd_full.py --mode finetuned

echo "================================================"
echo "DONE: full second-place xView2 solution fine-tuned and tested on IAN-BD"
echo "$EXP_OUT"
echo "================================================"

find "$EXP_OUT" -maxdepth 5 -type f | sort
