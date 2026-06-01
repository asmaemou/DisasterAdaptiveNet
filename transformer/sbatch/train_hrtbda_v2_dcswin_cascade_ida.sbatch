#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=hrtbda_v2_ida
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=HRTBDA-v2-DCSwin-Cascade_IDA

IDABD_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet

PHASE1_BEST=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_FINAL_TEST_th040_dilminor/threshold_checkpoints/phase1_best_threshold_0.40.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/$EXPERIMENT_NAME

TRAIN_SPLITS=(train)
VAL_SPLIT=val
TEST_SPLIT=test

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
echo "EXPERIMENT: $EXPERIMENT_NAME"
echo "MODEL: HRTBDA v2 DCSwin Cascade"
echo "MODE: fixed xBD Phase I, train Phase II on IDA-BD, test on IDA-BD"
echo "DOMAIN_ADAPTATION: none"
echo "IDABD_ROOT: $IDABD_ROOT"
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
echo "FOCAL_GAMMA: $FOCAL_GAMMA"
echo "AMP: enabled"
nvidia-smi || true
echo "===================="

if [ ! -f "$PHASE1_BEST" ]; then
  echo "ERROR: Phase I checkpoint not found:"
  echo "$PHASE1_BEST"
  exit 1
fi

if [ ! -f transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask.py ]; then
  echo "ERROR: missing training script:"
  echo "transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask.py"
  exit 1
fi

if [ ! -d "$IDABD_ROOT/train/images" ]; then
  echo "ERROR: missing dataset directory:"
  echo "$IDABD_ROOT/train/images"
  exit 1
fi

if [ ! -d "$IDABD_ROOT/train/targets" ]; then
  echo "ERROR: missing dataset directory:"
  echo "$IDABD_ROOT/train/targets"
  exit 1
fi

echo "===== CHECK IDA-BD DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$IDABD_ROOT" \
  --splits train val test

echo "===== INSPECT PHASE I CHECKPOINT ====="
python -u transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask.py \
  --phase inspect_phase1 \
  --phase1-checkpoint "$PHASE1_BEST" \
  --xbd-root "$IDABD_ROOT" \
  --train-split "${TRAIN_SPLITS[@]}" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --img-size "$IMG_SIZE" \
  --seed "$SEED" \
  --base-channels "$BASE_CHANNELS" \
  --decoder-channels "$DECODER_CHANNELS" \
  --window-size "$WINDOW_SIZE"

echo "===== TRAIN PHASE II + TEST CASCADE ON IDA-BD ====="
python -u transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask.py \
  --phase phase2_test \
  --phase1-checkpoint "$PHASE1_BEST" \
  --xbd-root "$IDABD_ROOT" \
  --train-split "${TRAIN_SPLITS[@]}" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --phase2-epochs "$PHASE2_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --img-size "$IMG_SIZE" \
  --lr "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --base-channels "$BASE_CHANNELS" \
  --decoder-channels "$DECODER_CHANNELS" \
  --window-size "$WINDOW_SIZE" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --save-every "$SAVE_EVERY" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --focal-gamma "$FOCAL_GAMMA" \
  --cls-loss-weight "$CLS_LOSS_WEIGHT" \
  --amp

echo "Done."