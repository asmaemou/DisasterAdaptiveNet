#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=hrtbda_v5_pmkf
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/pmkf/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/pmkf/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"

mkdir -p cross_DS/sbatch/logs/pmkf

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v5_PMKF_multilabel_crop_cascade

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

# Reuse your good Phase-I localization checkpoint.
PHASE1_SOURCE_DIR=$PROJECT_ROOT/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp
PHASE1_BEST=$PHASE1_SOURCE_DIR/checkpoints/phase1_best.pt

OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

PMKF_SCRIPT=$PROJECT_ROOT/cross_DS/scripts/pmkf/train_xbd_hrtbda_v5_pmkf_multilabel_crop_cascade.py

PHASE2_EPOCHS=60
WARMUP_EPOCHS=5
FINETUNE_EPOCHS=3
FINETUNE_LR=5e-5

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8

IMG_SIZE=1024
PHASE2_CROP_SIZE=608
CROP_CANDIDATES=8

LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0

SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999

FOCAL_GAMMA=2.0
CLS_LOSS_WEIGHT=1.0
AUX_LOC_WEIGHT=0.2

MINOR_DAMAGE_BOOST=1.5
MAJOR_DAMAGE_BOOST=1.5
MAX_DAMAGE_CLASS_WEIGHT=10.0

CROP_WEIGHT_NO_DAMAGE=1.0
CROP_WEIGHT_MINOR=12.0
CROP_WEIGHT_MAJOR=12.0
CROP_WEIGHT_DESTROYED=4.0

POSTPROCESS_DILATION=none
DILATION_KERNEL=3

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

echo "===== HRTBDA V5 + PMKF JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "MODEL: HRTBDA v5 + PMKF multi-label damage heads + rare-damage crop training + cascaded Phase-I mask"
echo "SCRIPT: $PMKF_SCRIPT"
echo "XBD_ROOT: $XBD_ROOT"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "PHASE1_BEST: $PHASE1_BEST"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "WARMUP_EPOCHS: $WARMUP_EPOCHS"
echo "FINETUNE_EPOCHS: $FINETUNE_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
echo "EFFECTIVE_BATCH_SIZE: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "IMG_SIZE: $IMG_SIZE"
echo "PHASE2_CROP_SIZE: $PHASE2_CROP_SIZE"
echo "CROP_CANDIDATES: $CROP_CANDIDATES"
echo "POSTPROCESS_DILATION: $POSTPROCESS_DILATION"
echo "AMP: enabled"
nvidia-smi || true
echo "====================================="

if [ ! -f "$PHASE1_BEST" ]; then
  echo "ERROR: Phase I best checkpoint not found:"
  echo "$PHASE1_BEST"
  exit 1
fi

if [ ! -f "$PMKF_SCRIPT" ]; then
  echo "ERROR: PMKF training script not found:"
  echo "$PMKF_SCRIPT"
  exit 1
fi

python -m py_compile "$PMKF_SCRIPT"

ls -lh "$PHASE1_BEST"
ls -lh "$PMKF_SCRIPT"

CMD=(
  python -u "$PMKF_SCRIPT"
  --phase phase2_test
  --phase1-checkpoint "$PHASE1_BEST"
  --xbd-root "$XBD_ROOT"
  --train-split "${TRAIN_SPLITS[@]}"
  --val-split "$VAL_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --phase2-epochs "$PHASE2_EPOCHS"
  --batch-size "$BATCH_SIZE"
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
  --aux-loc-weight "$AUX_LOC_WEIGHT"
  --minor-damage-boost "$MINOR_DAMAGE_BOOST"
  --major-damage-boost "$MAJOR_DAMAGE_BOOST"
  --max-damage-class-weight "$MAX_DAMAGE_CLASS_WEIGHT"
  --finetune-epochs "$FINETUNE_EPOCHS"
  --finetune-lr "$FINETUNE_LR"
  --postprocess-dilation "$POSTPROCESS_DILATION"
  --dilation-kernel "$DILATION_KERNEL"
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"

echo "Done HRTBDA v5 + PMKF."