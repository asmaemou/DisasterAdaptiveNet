#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=hrtbda_v2_cleanvit_s4b3
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v2_cascaded_phase1mask_cleanvit_s4b3
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2

TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

PHASE1_EPOCHS=150
PHASE2_EPOCHS=30
WARMUP_EPOCHS=10

BATCH_SIZE=1
NUM_WORKERS=8
IMG_SIZE=1024

LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8

# Clean compatible ViT setup:
# Phase I and Phase II use the SAME ViT replacement stages.
# This makes Phase I -> Phase II backbone loading strict with 0 missing and 0 unexpected keys.
VIT_STAGES=stage4_b3
VIT_MLP_RATIO=4.0
VIT_DROPOUT=0.0

MAX_GRAD_NORM=1.0
SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999
FOCAL_GAMMA=2.0
LOC_LOSS_WEIGHT=1.0
CLS_LOSS_WEIGHT=1.0

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

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "MODEL: HRTBDA v2 clean compatible ViT: Phase I + Phase II both use $VIT_STAGES"
echo "MODE: train Phase I from scratch, train Phase II from the compatible Phase I checkpoint, then test cascade"
echo "STRICT_BACKBONE_LOAD: enabled by default"
echo "DOMAIN_ADAPTATION: none"
echo "XBD_ROOT: $XBD_ROOT"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "WARMUP_EPOCHS: $WARMUP_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "IMG_SIZE: $IMG_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "WEIGHT_DECAY: $WEIGHT_DECAY"
echo "BASE_CHANNELS: $BASE_CHANNELS"
echo "DECODER_CHANNELS: $DECODER_CHANNELS"
echo "WINDOW_SIZE: $WINDOW_SIZE"
echo "PHASE1_VIT_STAGES: $VIT_STAGES"
echo "PHASE2_VIT_STAGES: $VIT_STAGES"
echo "VIT_MLP_RATIO: $VIT_MLP_RATIO"
echo "VIT_DROPOUT: $VIT_DROPOUT"
echo "MAX_GRAD_NORM: $MAX_GRAD_NORM"
echo "SAVE_EVERY: $SAVE_EVERY"
echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
echo "FOCAL_GAMMA: $FOCAL_GAMMA"
echo "AMP: enabled"
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask_vit_clean.py
  --phase both
  --xbd-root "$XBD_ROOT"
  --train-split "${TRAIN_SPLITS[@]}"
  --val-split "$VAL_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --phase1-epochs "$PHASE1_EPOCHS"
  --phase2-epochs "$PHASE2_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --lr "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --base-channels "$BASE_CHANNELS"
  --decoder-channels "$DECODER_CHANNELS"
  --window-size "$WINDOW_SIZE"
  --phase1-vit-stages "$VIT_STAGES"
  --phase2-vit-stages "$VIT_STAGES"
  --vit-mlp-ratio "$VIT_MLP_RATIO"
  --vit-dropout "$VIT_DROPOUT"
  --max-grad-norm "$MAX_GRAD_NORM"
  --warmup-epochs "$WARMUP_EPOCHS"
  --save-every "$SAVE_EVERY"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --focal-gamma "$FOCAL_GAMMA"
  --loc-loss-weight "$LOC_LOSS_WEIGHT"
  --cls-loss-weight "$CLS_LOSS_WEIGHT"
  --amp
)

echo "===== TRAIN CLEAN COMPATIBLE PHASE I + PHASE II + TEST COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "===================================================================="
"${CMD[@]}"
