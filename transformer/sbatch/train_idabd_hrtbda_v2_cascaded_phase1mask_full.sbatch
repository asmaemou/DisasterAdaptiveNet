#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=idabd_v2_cascade
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10

IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd
OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/$EXPERIMENT_NAME

# This is the faithful v2-cascade version:
#   Phase I is trained from scratch on IDA-BD.
#   Phase II remains foreground-only 4-class softmax damage classification.
#   Final inference uses Phase-I mask + Phase-II damage.
PHASE1_EPOCHS=150
PHASE2_EPOCHS=30
WARMUP_EPOCHS=10

# Keep the v2 cascade training setup.
BATCH_SIZE=1
NUM_WORKERS=8
IMG_SIZE=1024

LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0

SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999

FOCAL_GAMMA=2.0
LOC_LOSS_WEIGHT=1.0
CLS_LOSS_WEIGHT=1.0

TRAIN_RATIO=0.80
VAL_RATIO=0.10
TEST_RATIO=0.10

THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"

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
echo "MODEL: IDA-BD HRTBDA v2 cascaded Phase-I mask + foreground-only Phase-II damage"
echo "MODE: train Phase I from scratch, train Phase II, test with Phase-I mask"
echo "IDABD_ROOT: $IDABD_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SPLIT: 80/10/10"
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
echo "MAX_GRAD_NORM: $MAX_GRAD_NORM"
echo "FOCAL_GAMMA: $FOCAL_GAMMA"
echo "THRESHOLDS: $THRESHOLDS"
echo "AMP: enabled"
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_idabd_hrtbda_v2_cascaded_phase1mask_full.py
  --phase both
  --idabd-root "$IDABD_ROOT"
  --output-dir "$OUTPUT_DIR"
  --train-ratio "$TRAIN_RATIO"
  --val-ratio "$VAL_RATIO"
  --test-ratio "$TEST_RATIO"
  --train-split train80
  --val-split val10
  --test-split test10
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
  --max-grad-norm "$MAX_GRAD_NORM"
  --warmup-epochs "$WARMUP_EPOCHS"
  --save-every "$SAVE_EVERY"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --focal-gamma "$FOCAL_GAMMA"
  --loc-loss-weight "$LOC_LOSS_WEIGHT"
  --cls-loss-weight "$CLS_LOSS_WEIGHT"
  --thresholds $THRESHOLDS
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"
