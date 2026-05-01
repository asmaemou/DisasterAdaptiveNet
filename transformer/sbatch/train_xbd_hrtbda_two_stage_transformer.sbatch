#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --job-name=hrtbda_2stage_full
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v2_two_stage_transformer_full

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/$EXPERIMENT_NAME

# Full run: train Phase I localization, train Phase II damage, then test the strict two-stage pipeline.
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
MAX_GRAD_NORM=1.0

SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999

FOCAL_GAMMA=2.0
LOC_LOSS_WEIGHT=1.0
CLS_LOSS_WEIGHT=1.0

# Ideas borrowed from your older two-stage pipeline:
# 1) foreground-only Phase-II damage training
# 2) rare-damage sampling for Phase II
# 3) vector-scaling damage calibration on hold
# 4) Phase-I mask threshold sweep on hold for full pipeline score
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

THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"
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
echo "MODEL: HRTBDA v2 full two-stage transformer pipeline"
echo "MODE: train Phase I -> train foreground-only Phase II -> cascaded test"
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
echo "MAX_GRAD_NORM: $MAX_GRAD_NORM"
echo "SAVE_EVERY: $SAVE_EVERY"
echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
echo "FOCAL_GAMMA: $FOCAL_GAMMA"
echo "RARE SAMPLER: minor=$RARE_MINOR_WEIGHT major=$RARE_MAJOR_WEIGHT destroyed=$RARE_DESTROYED_WEIGHT"
echo "CLASS BOOSTS: minor=$MINOR_DAMAGE_BOOST major=$MAJOR_DAMAGE_BOOST destroyed=$DESTROYED_DAMAGE_BOOST"
echo "CALIBRATION: pixels/class=$CALIB_PIXELS_PER_CLASS epochs=$CALIB_EPOCHS lr=$CALIB_LR"
echo "THRESHOLDS: $THRESHOLDS"
echo "MASK_THRESHOLDS: $MASK_THRESHOLDS"
echo "AMP: enabled"
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_two_stage_transformer.py
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
  --max-grad-norm "$MAX_GRAD_NORM"
  --warmup-epochs "$WARMUP_EPOCHS"
  --save-every "$SAVE_EVERY"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --focal-gamma "$FOCAL_GAMMA"
  --loc-loss-weight "$LOC_LOSS_WEIGHT"
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
  --thresholds $THRESHOLDS
  --mask-thresholds $MASK_THRESHOLDS
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="
"${CMD[@]}"
