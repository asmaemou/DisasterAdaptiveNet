#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=E0058_xBD_TurkeySemeru_to_PakFlood
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATASET_BASE=/homes/j244s673/documents/wsu/phd

EXPERIMENT_ID=E0058
EXPERIMENT_LABEL=Planned_E4_xBDTL_ETurkey_MSemeru_to_PakFlood
MODE=xbdtl

cd "$PROJECT_ROOT"
mkdir -p cross_DS/sbatch/logs

EXPERIMENT_NAME=HRTBDA-v5_${EXPERIMENT_ID}_${EXPERIMENT_LABEL}
COMBINED_ROOT=$PROJECT_ROOT/output/HRTBDA_planned_combined/$EXPERIMENT_ID
STAGING_ROOT=$PROJECT_ROOT/output/HRTBDA_planned_staging/$EXPERIMENT_ID
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

PREP_SCRIPT=cross_DS/scripts/prepare_hrtbda_planned_E0055_E0060.py
HRTBDA_SCRIPT=transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py

XBD_PHASE1_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_FINAL_TEST_th040_dilminor/threshold_checkpoints/phase1_best_threshold_0.40.pt
XBD_PHASE2_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

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
echo "HRTBDA-v5 Planned Experiment E0058"
echo "xBD-pretrained fine-tuning: Earthquake Turkey + Mount Semeru"
echo "Testing on unseen Pakistan Flooding"
echo "================================================"
date
hostname
pwd
which python
python --version
nvidia-smi || true

echo "EXPERIMENT_ID: $EXPERIMENT_ID"
echo "EXPERIMENT_LABEL: $EXPERIMENT_LABEL"
echo "MODE: $MODE"
echo "COMBINED_ROOT: $COMBINED_ROOT"
echo "STAGING_ROOT: $STAGING_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "================================================"

echo "===== CLEAN OLD OUTPUT FOR E0058 ====="
rm -rf "$COMBINED_ROOT"
rm -rf "$STAGING_ROOT"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

echo "===== PREPARE COMBINED DATASET ====="
python -u "$PREP_SCRIPT" \
  --experiment-id "$EXPERIMENT_ID" \
  --project-root "$PROJECT_ROOT" \
  --dataset-base "$DATASET_BASE" \
  --output-root "$COMBINED_ROOT" \
  --staging-root "$STAGING_ROOT" \
  --mode symlink \
  --clean

echo "===== CHECK COMBINED DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$COMBINED_ROOT" \
  --splits train val test

PHASE1_INIT_ARGS=()
PHASE2_INIT_ARGS=()

echo "===== xBD TRANSFER MODE ENABLED ====="

if [ ! -f "$XBD_PHASE1_CKPT" ]; then
  echo "ERROR: xBD Phase-I checkpoint not found:"
  echo "$XBD_PHASE1_CKPT"
  exit 1
fi

# The Python script does NOT support --phase1-init-checkpoint.
# It supports --resume-phase1-from.
# To use the xBD checkpoint as initialization instead of a true resume,
# we copy it and reset epoch/optimizer/scheduler/scaler.
XBD_PHASE1_INIT_CKPT=$OUTPUT_DIR/checkpoints/xbd_phase1_init_epoch0.pt

echo "===== CREATE xBD PHASE-I INITIALIZATION CHECKPOINT ====="
python - <<PY
import torch
from pathlib import Path

src = Path("$XBD_PHASE1_CKPT")
dst = Path("$XBD_PHASE1_INIT_CKPT")

ckpt = torch.load(src, map_location="cpu")

ckpt["epoch"] = 0
ckpt["best_metric"] = -1.0
ckpt["best_threshold"] = 0.5
ckpt["optimizer"] = None
ckpt["scheduler"] = None
ckpt["scaler"] = None

dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, dst)

print(f"Created xBD initialization checkpoint: {dst}", flush=True)
PY

PHASE1_INIT_ARGS=(--resume-phase1-from "$XBD_PHASE1_INIT_CKPT")

echo "PHASE1_INIT_ARGS: ${PHASE1_INIT_ARGS[@]}"
echo "PHASE2_INIT_ARGS: ${PHASE2_INIT_ARGS[@]}"

echo "===== STEP 1: TRAIN PHASE I WITH xBD INITIALIZATION ====="

python -u "$HRTBDA_SCRIPT" \
  --phase phase1 \
  "${PHASE1_INIT_ARGS[@]}" \
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
  echo "ERROR: expected Phase-I checkpoint not found:"
  echo "$PHASE1_CKPT"
  exit 1
fi

echo "Found Phase-I checkpoint:"
ls -lh "$PHASE1_CKPT"

echo "===== STEP 2: TRAIN PHASE II AND TEST ON PAKISTAN FLOODING ====="

python -u "$HRTBDA_SCRIPT" \
  --phase phase2_test \
  "${PHASE2_INIT_ARGS[@]}" \
  --phase1-checkpoint "$PHASE1_CKPT" \
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

echo "===== DONE E0058: xBD-pretrained Turkey + Semeru -> Pakistan Flooding ====="