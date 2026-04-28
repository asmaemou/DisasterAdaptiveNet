#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=changeformer_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_xbd_test_xbd_changeformer

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLIT=tier3
VAL_SPLIT=hold
TEST_SPLIT=test

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_test_xbd_changeformer

EPOCHS=40
BATCH_SIZE=8
NUM_WORKERS=8

# ChangeFormer repo examples use 256 image size.
# Start with 256. Use 512 only if GPU memory allows.
IMG_SIZE=256

LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-4
SEED=42
EMBED_DIM=256

SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=10

LOC_BCE_WEIGHT=1.0
LOC_DICE_WEIGHT=1.0
DMG_CE_WEIGHT=1.0

THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/scores"

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
echo "MODEL: ChangeFormerV6"
echo "DOMAIN_ADAPTATION: none"
echo "XBD_ROOT: $XBD_ROOT"
echo "TRAIN_SPLIT: $TRAIN_SPLIT"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "IMG_SIZE: $IMG_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "WEIGHT_DECAY: $WEIGHT_DECAY"
echo "EMBED_DIM: $EMBED_DIM"
echo "THRESHOLDS: $THRESHOLDS"
echo "CHANGEFORMER_ROOT: /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/external/ChangeFormer"
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_xbd_changeformer_damage.py
  --xbd-root "$XBD_ROOT"
  --train-split "$TRAIN_SPLIT"
  --val-split "$VAL_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --lr "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --embed-dim "$EMBED_DIM"
  --save-every "$SAVE_EVERY"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --loc-bce-weight "$LOC_BCE_WEIGHT"
  --loc-dice-weight "$LOC_DICE_WEIGHT"
  --dmg-ce-weight "$DMG_CE_WEIGHT"
  --thresholds $THRESHOLDS
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"