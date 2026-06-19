#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=hrtbda_E0030_xbdTL
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATASET_BASE=/homes/j244s673/documents/wsu/phd

cd "$PROJECT_ROOT"
mkdir -p cross_DS/sbatch/logs

EXPERIMENT_ID=E0030
EXPERIMENT_NAME=HRTBDA-v5_E0030_xBDInit_FullFineTune_Train_EARTHQUAKE-TURKEY_PLUS_HURRICANE-LAURA_Test_HURRICANE-DELTA

TRAIN_DATASETS=EARTHQUAKE-TURKEY,HURRICANE-LAURA
TEST_DATASET=HURRICANE-DELTA

COMBINED_ROOT=$PROJECT_ROOT/output/HRTBDA_cross_dataset_combined/$EXPERIMENT_ID
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

PREP_SCRIPT=cross_DS/scripts/prepare_hrtbda_E0019_E0030_cross_DS.py
HRTBDA_SCRIPT=transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py
PHASE2_WRAPPER=transformer/scripts/train_pakistan_flooding_hrtbda_v5_phase2_finetune.py

XBD_PHASE1_CKPT=$PROJECT_ROOT/output/train_plus_tier3_test_xbd_hrtbda_v5_FINAL_TEST_th040_dilminor/threshold_checkpoints/phase1_best_threshold_0.40.pt
XBD_PHASE2_CKPT=$PROJECT_ROOT/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

TRAIN_SPLITS=(train)
VAL_SPLIT=val
TEST_SPLIT=test

PHASE1_EPOCHS=150
PHASE2_EPOCHS=30

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8

IMG_SIZE=1024
PHASE2_CROP_SIZE=608
CROP_CANDIDATES=8

PHASE1_LR=5e-5
PHASE2_LR=5e-5
WEIGHT_DECAY=1e-4
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0
WARMUP_EPOCHS=3

FOCAL_GAMMA=2.0
AUX_LOC_WEIGHT=0.2

POSTPROCESS_DILATION=minor
DILATION_KERNEL=3

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

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

echo "===== HRTBDA-v5 CROSS-DATASET E0030 xBD TRANSFER LEARNING ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "EXPERIMENT_ID: $EXPERIMENT_ID"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "TRAIN_DATASETS: $TRAIN_DATASETS"
echo "TEST_DATASET: $TEST_DATASET"
echo "DATASET_BASE: $DATASET_BASE"
echo "COMBINED_ROOT: $COMBINED_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "==============================================================="

if [ ! -f "$XBD_PHASE1_CKPT" ]; then
  echo "ERROR: missing xBD Phase-I checkpoint:"
  echo "$XBD_PHASE1_CKPT"
  exit 1
fi

if [ ! -f "$XBD_PHASE2_CKPT" ]; then
  echo "ERROR: missing xBD Phase-II checkpoint:"
  echo "$XBD_PHASE2_CKPT"
  exit 1
fi

python -u "$PREP_SCRIPT" \
  --dataset-base "$DATASET_BASE" \
  --train-datasets "$TRAIN_DATASETS" \
  --test-dataset "$TEST_DATASET" \
  --output-root "$COMBINED_ROOT" \
  --mode symlink

echo "===== CHECK COMBINED DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$COMBINED_ROOT" \
  --splits train val test

echo "===== CREATE RESET-BEST PHASE-I CHECKPOINT ====="

XBD_PHASE1_RESET_CKPT=$OUTPUT_DIR/checkpoints/phase1_xbdinit_resetbest_for_E0030.pt

python - <<PY
import torch
from pathlib import Path

src = Path("$XBD_PHASE1_CKPT")
dst = Path("$XBD_PHASE1_RESET_CKPT")
dst.parent.mkdir(parents=True, exist_ok=True)

try:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(src, map_location="cpu")

ckpt["epoch"] = 0
ckpt["best_metric"] = -1.0
ckpt["best_score"] = -1.0
ckpt["best_f1"] = -1.0
ckpt["best_loc_f1"] = -1.0
ckpt["best_threshold"] = 0.40

for k in [
    "optimizer",
    "optimizer_state_dict",
    "scheduler",
    "scheduler_state_dict",
    "scaler",
    "amp_scaler",
]:
    ckpt.pop(k, None)

torch.save(ckpt, dst)
print(f"Wrote reset-best Phase-I checkpoint: {dst}")
PY

echo "===== STEP 1: xBD Phase-I weights -> fine-tune Phase I ====="

python -u "$HRTBDA_SCRIPT" \
  --phase phase1 \
  --resume-phase1 "$XBD_PHASE1_RESET_CKPT" \
  --xbd-root "$COMBINED_ROOT" \
  --train-split "${TRAIN_SPLITS[@]}" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --phase1-epochs "$PHASE1_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" \
  --img-size "$IMG_SIZE" \
  --lr "$PHASE1_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --base-channels "$BASE_CHANNELS" \
  --decoder-channels "$DECODER_CHANNELS" \
  --window-size "$WINDOW_SIZE" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --focal-gamma "$FOCAL_GAMMA" \
  --amp

PHASE1_CKPT=$OUTPUT_DIR/checkpoints/phase1_best.pt

if [ ! -f "$PHASE1_CKPT" ]; then
  echo "ERROR: expected fine-tuned Phase-I checkpoint not found:"
  echo "$PHASE1_CKPT"
  exit 1
fi

echo "===== STEP 2: xBD Phase-II weights -> fine-tune Phase II and test on HURRICANE-DELTA ====="

python -u "$PHASE2_WRAPPER" \
  --phase phase2_test \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --init-phase2-from "$XBD_PHASE2_CKPT" \
  --xbd-root "$COMBINED_ROOT" \
  --train-split "${TRAIN_SPLITS[@]}" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --phase2-epochs "$PHASE2_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" \
  --img-size "$IMG_SIZE" \
  --phase2-crop-size "$PHASE2_CROP_SIZE" \
  --crop-candidate-count "$CROP_CANDIDATES" \
  --lr "$PHASE2_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --base-channels "$BASE_CHANNELS" \
  --decoder-channels "$DECODER_CHANNELS" \
  --window-size "$WINDOW_SIZE" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --focal-gamma "$FOCAL_GAMMA" \
  --aux-loc-weight "$AUX_LOC_WEIGHT" \
  --postprocess-dilation "$POSTPROCESS_DILATION" \
  --dilation-kernel "$DILATION_KERNEL" \
  --amp

echo "Done E0030."
