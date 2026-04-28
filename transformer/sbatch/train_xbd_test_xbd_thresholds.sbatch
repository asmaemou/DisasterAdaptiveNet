#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=train_xbd_test_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

# =========================
# Experiment settings
# =========================
EXPERIMENT_NAME=train_xbd_test_xbd

EPOCHS=40
BATCH_SIZE=16
LEARNING_RATE=2e-4
NUM_WORKERS=8
LOG_FREQ=10
SEED=42

# Thresholds tested on HOLD validation.
# Best HOLD threshold is then used once on TEST.
THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70"

# ComboLoss weights.
# Keep these as 1.0 1.0 unless you know your old config used different values.
LOSS_WEIGHTS="1.0 1.0"

# Leave empty to let the script try your project's default class-weight logic.
# Example if needed later:
# CLASS_WEIGHTS="1.0 1.0 1.0 1.0 1.0"
CLASS_WEIGHTS=""

# =========================
# Dataset split paths
# =========================
TRAIN_ROOT=/homes/j244s673/documents/wsu/phd/xview2/tier3
VAL_ROOT=/homes/j244s673/documents/wsu/phd/xview2/hold
TEST_ROOT=/homes/j244s673/documents/wsu/phd/xview2/test

# Exact output folder requested.
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_test_xbd
SPLIT_ROOT=${RUN_DIR}/xview2_explicit_splits

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/scores"
mkdir -p "$SPLIT_ROOT"

ln -sfn "$TRAIN_ROOT" "$SPLIT_ROOT/train"
ln -sfn "$TRAIN_ROOT" "$SPLIT_ROOT/tier3"

ln -sfn "$VAL_ROOT" "$SPLIT_ROOT/val"
ln -sfn "$VAL_ROOT" "$SPLIT_ROOT/hold"

ln -sfn "$TEST_ROOT" "$SPLIT_ROOT/test"

# =========================
# Conda setup
# =========================
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

export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_DISABLED=true

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "MODEL: DisasterAdaptiveNet"
echo "MODEL_IMPORT: from utils.models import DisasterAdaptiveNet"
echo "CONDITIONING_INFORMATION: True"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "LOG_FREQ: $LOG_FREQ"
echo "SEED: $SEED"
echo "THRESHOLDS: $THRESHOLDS"
echo "LOSS_WEIGHTS: $LOSS_WEIGHTS"
echo "CLASS_WEIGHTS: ${CLASS_WEIGHTS:-none}"
echo "TRAIN_ROOT: $TRAIN_ROOT"
echo "VAL_ROOT: $VAL_ROOT"
echo "TEST_ROOT: $TEST_ROOT"
echo "SPLIT_ROOT: $SPLIT_ROOT"
echo "RUN_DIR: $RUN_DIR"
echo "===== SPLIT LINKS ====="
ls -lah "$SPLIT_ROOT"
echo "======================="
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_tier3_hold_test_disasteradaptivenet_f1_thresholds.py
  --experiment-name "$EXPERIMENT_NAME"
  --data-root "$SPLIT_ROOT"
  --output-dir "$RUN_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LEARNING_RATE"
  --num-workers "$NUM_WORKERS"
  --log-freq "$LOG_FREQ"
  --seed "$SEED"
  --thresholds $THRESHOLDS
  --loss-weights $LOSS_WEIGHTS
)

if [ -n "$CLASS_WEIGHTS" ]; then
  CMD+=(--class-weights $CLASS_WEIGHTS)
fi

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"