#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=idabd_v5_cascade
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_idabd_hrtbda_v5_multilabel_rare_crop_cascade_80_10_10

IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd
OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/$EXPERIMENT_NAME

# Automatic preprocessing split: 80% train, 10% validation, 10% test.
TRAIN_RATIO=0.80
VAL_RATIO=0.10
TEST_RATIO=0.10
SEED=42

# Phase I trains localization from scratch on IDA-BD.
PHASE1_EPOCHS=150
PHASE1_BATCH_SIZE=1

# Phase II uses the same v5 settings that gave the strong xBD result.
PHASE2_EPOCHS=60
PHASE2_BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
EFFECTIVE_BATCH_SIZE=$((PHASE2_BATCH_SIZE * GRAD_ACCUM_STEPS))

WARMUP_EPOCHS=5
FINETUNE_EPOCHS=3
FINETUNE_LR=5e-5

NUM_WORKERS=8
IMG_SIZE=1024
PHASE2_CROP_SIZE=608
CROP_CANDIDATES=8

LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0

SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999
FOCAL_GAMMA=2.0
LOC_LOSS_WEIGHT=1.0
CLS_LOSS_WEIGHT=1.0
AUX_LOC_WEIGHT=0.2

MINOR_DAMAGE_BOOST=1.5
MAJOR_DAMAGE_BOOST=1.5
MAX_DAMAGE_CLASS_WEIGHT=10.0

# Rare-crop selection weights: same idea that helped xBD v5.
CROP_WEIGHT_NO_DAMAGE=1.0
CROP_WEIGHT_MINOR=12.0
CROP_WEIGHT_MAJOR=12.0
CROP_WEIGHT_DESTROYED=4.0

# During final validation ablation and test, evaluate the same threshold/dilation settings.
ABLATION_THRESHOLDS="0.40 0.45 0.50 0.55"
ABLATION_DILATIONS="none minor"
DILATION_KERNEL=3

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
nvidia-smi || true
echo "MODEL: IDA-BD HRTBDA v5 multi-label damage heads + rare-damage crop training + cascaded Phase-I mask"
echo "IDABD_ROOT: $IDABD_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SPLIT: $TRAIN_RATIO / $VAL_RATIO / $TEST_RATIO"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "PHASE1_BATCH_SIZE: $PHASE1_BATCH_SIZE"
echo "PHASE2_BATCH_SIZE: $PHASE2_BATCH_SIZE"
echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
echo "EFFECTIVE_BATCH_SIZE: $EFFECTIVE_BATCH_SIZE"
echo "PHASE2_CROP_SIZE: $PHASE2_CROP_SIZE"
echo "CROP_CANDIDATES: $CROP_CANDIDATES"
echo "ABLATION_THRESHOLDS: $ABLATION_THRESHOLDS"
echo "ABLATION_DILATIONS: $ABLATION_DILATIONS"
echo "AMP: enabled"
echo "===================="

if [ ! -d "$IDABD_ROOT/images" ]; then
  echo "ERROR: IDA-BD images folder not found: $IDABD_ROOT/images"
  exit 1
fi

if [ ! -d "$IDABD_ROOT/masks" ]; then
  echo "ERROR: IDA-BD masks folder not found: $IDABD_ROOT/masks"
  exit 1
fi

CMD=(
  python -u transformer/scripts/train_idabd_hrtbda_v5_multilabel_split_cascade.py
  --phase both
  --idabd-root "$IDABD_ROOT"
  --output-dir "$OUTPUT_DIR"
  --train-ratio "$TRAIN_RATIO"
  --val-ratio "$VAL_RATIO"
  --test-ratio "$TEST_RATIO"
  --seed "$SEED"
  --phase1-epochs "$PHASE1_EPOCHS"
  --phase2-epochs "$PHASE2_EPOCHS"
  --phase1-batch-size "$PHASE1_BATCH_SIZE"
  --phase2-batch-size "$PHASE2_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --phase2-crop-size "$PHASE2_CROP_SIZE"
  --crop-candidate-count "$CROP_CANDIDATES"
  --crop-weight-no-damage "$CROP_WEIGHT_NO_DAMAGE"
  --crop-weight-minor "$CROP_WEIGHT_MINOR"
  --crop-weight-major "$CROP_WEIGHT_MAJOR"
  --crop-weight-destroyed "$CROP_WEIGHT_DESTROYED"
  --extra-photometric-aug
  --lr "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
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
  --aux-loc-weight "$AUX_LOC_WEIGHT"
  --minor-damage-boost "$MINOR_DAMAGE_BOOST"
  --major-damage-boost "$MAJOR_DAMAGE_BOOST"
  --max-damage-class-weight "$MAX_DAMAGE_CLASS_WEIGHT"
  --finetune-epochs "$FINETUNE_EPOCHS"
  --finetune-lr "$FINETUNE_LR"
  --ablation-thresholds $ABLATION_THRESHOLDS
  --ablation-dilations $ABLATION_DILATIONS
  --dilation-kernel "$DILATION_KERNEL"
  --thresholds $THRESHOLDS
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"
