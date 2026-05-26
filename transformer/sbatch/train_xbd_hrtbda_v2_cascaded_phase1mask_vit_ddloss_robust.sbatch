#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=cleanvit_s4b3_ddloss_robust
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v2_cascaded_phase1mask_cleanvit_s4b3_ddloss_robust
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2

TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

PHASE1_EPOCHS=150
PHASE2_EPOCHS=30
WARMUP_EPOCHS=15

BATCH_SIZE=1
NUM_WORKERS=8
IMG_SIZE=1024

LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8

# Clean compatible ViT setup:
# Phase I and Phase II use the SAME ViT replacement stages.
VIT_STAGES=stage4_b3
VIT_MLP_RATIO=4.0
VIT_DROPOUT=0.0

MAX_GRAD_NORM=0.5
SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999

# Existing fallback focal/dice params remain available, but this job uses DDNet-style losses.
FOCAL_GAMMA=2.0
LOC_LOSS_WEIGHT=1.0
CLS_LOSS_WEIGHT=1.0

# =========================
# DDNet-style loss settings
# =========================
PHASE1_LOSS=unified_focal
PHASE2_LOSS=combo_seesaw

# Phase I Offset Unified Focal Loss / O-UFL defaults inspired by DDNet.
OUFL_LAMBDA=0.5
OUFL_BETA=0.9
OUFL_GAMMA=0.3

# Phase II Combo + Seesaw defaults.
# Combo:Seesaw = 1:2, alpha=2.0, class weights=[no, minor, major, destroyed]
COMBO_ALPHA=2.0
COMBO_LOSS_WEIGHT=1.0
SEESAW_LOSS_WEIGHT=2.0
SEESAW_P=0.8
SEESAW_Q=2.0
COMBO_CLASS_WEIGHTS=(0.1 0.4 0.3 0.1)

SCRIPT=transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask_vit_ddloss_robust.py

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

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: Python script not found: $SCRIPT"
  echo "Copy train_xbd_hrtbda_v2_cascaded_phase1mask_vit_ddloss.py into transformer/scripts/ first."
  exit 1
fi

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "MODEL: HRTBDA-CleanViT-S4B3 + DDNet-style losses (robust AMP-safe)"
echo "MODE: train Phase I from scratch, train Phase II from compatible Phase I checkpoint, then test cascade"
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
echo "PHASE1_LOSS: $PHASE1_LOSS"
echo "PHASE2_LOSS: $PHASE2_LOSS"
echo "OUFL_LAMBDA: $OUFL_LAMBDA"
echo "OUFL_BETA: $OUFL_BETA"
echo "OUFL_GAMMA: $OUFL_GAMMA"
echo "COMBO_ALPHA: $COMBO_ALPHA"
echo "COMBO_LOSS_WEIGHT: $COMBO_LOSS_WEIGHT"
echo "SEESAW_LOSS_WEIGHT: $SEESAW_LOSS_WEIGHT"
echo "SEESAW_P: $SEESAW_P"
echo "SEESAW_Q: $SEESAW_Q"
echo "COMBO_CLASS_WEIGHTS: ${COMBO_CLASS_WEIGHTS[*]}"
echo "MAX_GRAD_NORM: $MAX_GRAD_NORM"
echo "SAVE_EVERY: $SAVE_EVERY"
echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
echo "AMP: enabled"
nvidia-smi || true
echo "===================="

CMD=(
  python -u "$SCRIPT"
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
  --phase1-loss "$PHASE1_LOSS"
  --phase2-loss "$PHASE2_LOSS"
  --oufl-lambda "$OUFL_LAMBDA"
  --oufl-beta "$OUFL_BETA"
  --oufl-gamma "$OUFL_GAMMA"
  --combo-alpha "$COMBO_ALPHA"
  --combo-loss-weight "$COMBO_LOSS_WEIGHT"
  --seesaw-loss-weight "$SEESAW_LOSS_WEIGHT"
  --seesaw-p "$SEESAW_P"
  --seesaw-q "$SEESAW_Q"
  --combo-class-weights "${COMBO_CLASS_WEIGHTS[@]}"
  --amp
)

echo "===== TRAIN CLEANVIT-S4B3 + DDNET-STYLE LOSSES ROBUST COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "============================================================="
"${CMD[@]}"
