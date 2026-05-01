#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --job-name=hrtbda_v2_cascade
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet

# IMPORTANT:
# This directory must exist before sbatch opens the log file.
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v2_cascaded_phase1mask

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2

# Train on BOTH xBD train and tier3. Keep hold/test clean.
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

# Old run that already contains your good Phase I localization checkpoint.
PHASE1_SOURCE_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp
PHASE1_BEST=$PHASE1_SOURCE_DIR/checkpoints/phase1_best.pt

# New output directory for the corrected cascaded two-phase experiment.
OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/$EXPERIMENT_NAME

# We do NOT retrain Phase I here.
# We only inspect phase1_best.pt, train Phase II, then test with Phase I mask + Phase II damage.
PHASE2_EPOCHS=30
WARMUP_EPOCHS=10

# Full xBD resolution. Batch size 1 is safest for 1024 and 4-branch model.
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
echo "MODEL: HRTBDA v2 cascaded Phase I localization mask + foreground-only Phase II damage classification"
echo "MODE: use existing phase1_best.pt, train Phase II, then test with Phase I mask"
echo "DOMAIN_ADAPTATION: none"
echo "XBD_ROOT: $XBD_ROOT"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "PHASE1_BEST: $PHASE1_BEST"
echo "OUTPUT_DIR: $OUTPUT_DIR"
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
echo "SAVE_EVERY: $SAVE_EVERY"
echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
echo "FOCAL_GAMMA: $FOCAL_GAMMA"
echo "AMP: enabled"
nvidia-smi || true
echo "===================="

if [ ! -f "$PHASE1_BEST" ]; then
  echo "ERROR: Phase I best checkpoint not found:"
  echo "$PHASE1_BEST"
  exit 1
fi

echo "Found Phase I best checkpoint:"
ls -lh "$PHASE1_BEST"

# First, inspect what phase1_best.pt contains.
INSPECT_CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask.py
  --phase inspect_phase1
  --phase1-checkpoint "$PHASE1_BEST"
  --xbd-root "$XBD_ROOT"
  --train-split "${TRAIN_SPLITS[@]}"
  --val-split "$VAL_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --seed "$SEED"
  --base-channels "$BASE_CHANNELS"
  --decoder-channels "$DECODER_CHANNELS"
  --window-size "$WINDOW_SIZE"
)

echo "===== INSPECT PHASE I CHECKPOINT COMMAND ====="
printf '%q ' "${INSPECT_CMD[@]}"
echo
echo "=============================================="
"${INSPECT_CMD[@]}"

# Then train foreground-only Phase II and test using the Phase I building mask.
CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask.py
  --phase phase2_test
  --phase1-checkpoint "$PHASE1_BEST"
  --xbd-root "$XBD_ROOT"
  --train-split "${TRAIN_SPLITS[@]}"
  --val-split "$VAL_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
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
  --cls-loss-weight "$CLS_LOSS_WEIGHT"
  --amp
)

echo "===== TRAIN PHASE II + CASCADED TEST COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "=================================================="
"${CMD[@]}"
