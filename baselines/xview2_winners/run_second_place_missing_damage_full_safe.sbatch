#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=12:00:00
#SBATCH --job-name=xv2_2nd_missing_dmg_safe
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_missing_dmg_safe_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_missing_dmg_safe_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export CUDA_LAUNCH_BLOCKING=1

echo "================================================"
echo "ENV / GPU CHECK INSIDE SLURM JOB"
echo "================================================"
which python
python -V
hostname
nvidia-smi || true
python - <<'PY2'
import sys, torch
print("Python executable:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY2
echo "================================================"


DATASET="${1:-}"

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners
REPO=$BASE/xview2_second_place
MANIFEST=$BASE/second_place_full_solution_manifest.tsv

if [ "$DATASET" = "dorian" ]; then
  PREP_SCRIPT=prepare_second_place_dorian.py
  DATA_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_hurricane_dorian
  EXP_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_hurricane_dorian_FULL_SOLUTION_finetune
  PREFIX=dorian_full
elif [ "$DATASET" = "idabd" ]; then
  PREP_SCRIPT=prepare_second_place_idabd.py
  DATA_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_idabd
  EXP_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_idabd_FULL_SOLUTION_finetune
  PREFIX=idabd_full
elif [ "$DATASET" = "ianbd" ]; then
  PREP_SCRIPT=prepare_second_place_ianbd.py
  DATA_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_ianbd
  EXP_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_ianbd_FULL_SOLUTION_finetune
  PREFIX=ianbd_full
else
  echo "ERROR: pass dataset name: dorian, idabd, or ianbd"
  exit 1
fi

mkdir -p "$BASE/logs"
mkdir -p "$EXP_OUT/init"
mkdir -p "$EXP_OUT/weights_damage"
mkdir -p "$EXP_OUT/logs_damage"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

echo "================================================"
echo "Prepare dataset: $DATASET"
echo "================================================"
cd "$BASE"
python -u "$PREP_SCRIPT"

echo "================================================"
echo "Continue missing damage models only: $DATASET"
echo "================================================"

cd "$REPO"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r TASK FAMILY TAG CONFIG WEIGHT; do
  if [ "$TASK" != "damage" ]; then
    continue
  fi

  OUT_DIR="$EXP_OUT/weights_damage/$TAG"
  LOG_DIR="$EXP_OUT/logs_damage/$TAG"

  if find "$OUT_DIR" -type f 2>/dev/null | grep -q .; then
    echo "SKIP existing damage checkpoint: $TAG"
    find "$OUT_DIR" -type f | sort
    continue
  fi

  echo "================================================"
  echo "Train missing damage model: $TAG"
  echo "Config: $CONFIG"
  echo "Weight: $WEIGHT"
  echo "================================================"

  mkdir -p "$OUT_DIR"
  mkdir -p "$LOG_DIR"

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

  python -u train.py \
    --config "$CONFIG" \
    --workers 0 \
    --gpu 0 \
    --output-dir "$OUT_DIR" \
    --resume "$INIT" \
    --fold 0 \
    --prefix "${PREFIX}_${TAG}_" \
    --data-dir "$DATA_OUT" \
    --folds-csv "$DATA_OUT/folds.csv" \
    --logdir "$LOG_DIR" \
    --opt-level O0 \
    --test_every 1
done

echo "================================================"
echo "DONE missing damage continuation for $DATASET"
echo "================================================"

find "$EXP_OUT/weights_damage" -mindepth 1 -maxdepth 1 -type d | sort
