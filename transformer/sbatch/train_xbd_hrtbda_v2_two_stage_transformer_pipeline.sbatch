#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=hrtbda_2stage
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet

# Slurm opens these paths before the script body on some systems, so create once before sbatch too:
# mkdir -p /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v2_two_stage_transformer_pipeline

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

# Existing strong Phase-I localization checkpoint.
PHASE1_SOURCE_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/hrtbda_2stage
PHASE1_BEST=$PHASE1_SOURCE_DIR/checkpoints/phase1_best.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/$EXPERIMENT_NAME

# We do NOT retrain Phase I here.
# We train transformer Phase II damage only, then run strict pipeline inference:
# Phase I mask -> Phase II severity inside mask.
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
MAX_GRAD_NORM=1.0

SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999

FOCAL_GAMMA=2.0
CLS_LOSS_WEIGHT=1.0

# Borrowed from your older two-stage pipeline:
# 1) rare-damage sampling for Phase II
# 2) foreground-only damage class reweighting
# 3) vector-scaling damage calibration on hold
# 4) Phase-I threshold sweep on hold for full pipeline score
RARE_MINOR_WEIGHT=4.0
RARE_MAJOR_WEIGHT=2.5
RARE_DESTROYED_WEIGHT=1.5

MINOR_DAMAGE_BOOST=2.0
MAJOR_DAMAGE_BOOST=1.5
DESTROYED_DAMAGE_BOOST=1.0

CALIB_PIXELS_PER_CLASS=50000
CALIB_EPOCHS=10
CALIB_LR=5e-2
CALIB_BATCH=8192

MASK_THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"

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
echo "MODEL: HRTBDA v2 two-stage transformer pipeline"
echo "MODE: existing Phase-I mask + foreground-only transformer Phase-II damage + calibration + threshold sweep"
echo "XBD_ROOT: $XBD_ROOT"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "PHASE1_BEST: $PHASE1_BEST"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "WARMUP_EPOCHS: $WARMUP_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "IMG_SIZE: $IMG_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "RARE SAMPLER: minor=$RARE_MINOR_WEIGHT major=$RARE_MAJOR_WEIGHT destroyed=$RARE_DESTROYED_WEIGHT"
echo "CLASS BOOSTS: minor=$MINOR_DAMAGE_BOOST major=$MAJOR_DAMAGE_BOOST destroyed=$DESTROYED_DAMAGE_BOOST"
echo "CALIBRATION: pixels/class=$CALIB_PIXELS_PER_CLASS epochs=$CALIB_EPOCHS lr=$CALIB_LR"
echo "MASK_THRESHOLDS: $MASK_THRESHOLDS"
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

INSPECT_CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v2_two_stage_transformer_pipeline.py
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

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v2_two_stage_transformer_pipeline.py
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
  --rare-sampler
  --rare-minor-weight "$RARE_MINOR_WEIGHT"
  --rare-major-weight "$RARE_MAJOR_WEIGHT"
  --rare-destroyed-weight "$RARE_DESTROYED_WEIGHT"
  --minor-damage-boost "$MINOR_DAMAGE_BOOST"
  --major-damage-boost "$MAJOR_DAMAGE_BOOST"
  --destroyed-damage-boost "$DESTROYED_DAMAGE_BOOST"
  --calibrate-damage
  --calib-pixels-per-class "$CALIB_PIXELS_PER_CLASS"
  --calib-epochs "$CALIB_EPOCHS"
  --calib-lr "$CALIB_LR"
  --calib-batch "$CALIB_BATCH"
  --amp
  --mask-thresholds $MASK_THRESHOLDS
)

echo "===== TRAIN PHASE II + FINAL PIPELINE COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "=================================================="
"${CMD[@]}"
