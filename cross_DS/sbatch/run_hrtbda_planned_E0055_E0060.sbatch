#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=hrtbda_planned_E0055_E0060
#SBATCH --array=0-5%1
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%A_%a.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%A_%a.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATASET_BASE=/homes/j244s673/documents/wsu/phd

cd "$PROJECT_ROOT"
mkdir -p cross_DS/sbatch/logs

EXP_IDS=(
  E0055
  E0056
  E0057
  E0058
  E0059
  E0060
)

EXP_LABELS=(
  Planned_E1_Scratch_HLaura_HDelta_to_HIrma
  Planned_E2_xBDTL_HLaura_HDelta_to_HIrma
  Planned_E3_Scratch_ETurkey_MSemeru_to_PakFlood
  Planned_E4_xBDTL_ETurkey_MSemeru_to_PakFlood
  Planned_E5_Scratch_LOO_AllExceptPakFlood_to_PakFlood
  Planned_E6_xBDTL_LOO_AllExceptPakFlood_to_PakFlood
)

MODES=(
  scratch
  xbdtl
  scratch
  xbdtl
  scratch
  xbdtl
)

IDX=${SLURM_ARRAY_TASK_ID}
EXPERIMENT_ID=${EXP_IDS[$IDX]}
EXPERIMENT_LABEL=${EXP_LABELS[$IDX]}
MODE=${MODES[$IDX]}

EXPERIMENT_NAME=HRTBDA-v5_${EXPERIMENT_ID}_${EXPERIMENT_LABEL}
COMBINED_ROOT=$PROJECT_ROOT/output/HRTBDA_planned_combined/$EXPERIMENT_ID
STAGING_ROOT=$PROJECT_ROOT/output/HRTBDA_planned_staging
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

PREP_SCRIPT=cross_DS/scripts/prepare_hrtbda_planned_E0055_E0060.py
HRTBDA_SCRIPT=transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py

XBD_PHASE1_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_FINAL_TEST_th040_dilminor/threshold_checkpoints/phase1_best_threshold_0.40.pt
XBD_PHASE2_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

TRAIN_SPLITS=(train)
VAL_SPLIT=val
TEST_SPLIT=test

PHASE1_EPOCHS=150
PHASE2_EPOCHS=30

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8

IMG_SIZE=1024
PHASE2_CROP_SIZE=608
CROP_CANDIDATES=8

PHASE1_LR=5e-5
PHASE2_LR=5e-5
WEIGHT_DECAY=1e-4
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0
WARMUP_EPOCHS=3

FOCAL_GAMMA=2.0
AUX_LOC_WEIGHT=0.2

POSTPROCESS_DILATION=minor
DILATION_KERNEL=3

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

echo "===== HRTBDA-v5 PLANNED CROSS-DISASTER EXPERIMENT ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true

echo "EXPERIMENT_ID: $EXPERIMENT_ID"
echo "EXPERIMENT_LABEL: $EXPERIMENT_LABEL"
echo "MODE: $MODE"
echo "COMBINED_ROOT: $COMBINED_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "================================================"

echo "===== CLEAN OLD OUTPUT FOR THIS EXPERIMENT ====="
rm -rf "$COMBINED_ROOT"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

echo "===== PREPARE PLANNED COMBINED DATASET ====="
python -u "$PREP_SCRIPT" \
  --experiment-id "$EXPERIMENT_ID" \
  --project-root "$PROJECT_ROOT" \
  --dataset-base "$DATASET_BASE" \
  --output-root "$COMBINED_ROOT" \
  --staging-root "$STAGING_ROOT" \
  --mode symlink \
  --clean

echo "===== CHECK COMBINED DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$COMBINED_ROOT" \
  --splits train val test

PHASE1_INIT_ARGS=()
PHASE2_INIT_ARGS=()

if [ "$MODE" = "xbdtl" ]; then
  echo "===== xBD TRANSFER MODE ENABLED ====="

  if [ ! -f "$XBD_PHASE1_CKPT" ]; then
    echo "ERROR: xBD Phase-I checkpoint not found:"
    echo "$XBD_PHASE1_CKPT"
    exit 1
  fi

  if [ ! -f "$XBD_PHASE2_CKPT" ]; then
    echo "WARNING: xBD Phase-II checkpoint not found:"
    echo "$XBD_PHASE2_CKPT"
    echo "Phase II will run without Phase-II initialization if unsupported/missing."
  fi

  HELP_TEXT=$(python -u "$HRTBDA_SCRIPT" --help 2>&1 || true)

  if echo "$HELP_TEXT" | grep -q -- "--phase1-init-checkpoint"; then
    PHASE1_INIT_ARGS=(--phase1-init-checkpoint "$XBD_PHASE1_CKPT")
  elif echo "$HELP_TEXT" | grep -q -- "--pretrained-checkpoint"; then
    PHASE1_INIT_ARGS=(--pretrained-checkpoint "$XBD_PHASE1_CKPT")
  elif echo "$HELP_TEXT" | grep -q -- "--init-checkpoint"; then
    PHASE1_INIT_ARGS=(--init-checkpoint "$XBD_PHASE1_CKPT")
  else
    echo "ERROR: Could not find a supported Phase-I initialization argument in $HRTBDA_SCRIPT."
    echo "Check the script help manually:"
    echo "python $HRTBDA_SCRIPT --help | grep checkpoint"
    exit 1
  fi

  if [ -f "$XBD_PHASE2_CKPT" ]; then
    if echo "$HELP_TEXT" | grep -q -- "--phase2-init-checkpoint"; then
      PHASE2_INIT_ARGS=(--phase2-init-checkpoint "$XBD_PHASE2_CKPT")
    elif echo "$HELP_TEXT" | grep -q -- "--init-phase2-checkpoint"; then
      PHASE2_INIT_ARGS=(--init-phase2-checkpoint "$XBD_PHASE2_CKPT")
    elif echo "$HELP_TEXT" | grep -q -- "--phase2-pretrained-checkpoint"; then
      PHASE2_INIT_ARGS=(--phase2-pretrained-checkpoint "$XBD_PHASE2_CKPT")
    else
      echo "WARNING: No supported Phase-II initialization argument found."
      echo "Continuing with Phase II trained on the selected source datasets."
    fi
  fi

  echo "PHASE1_INIT_ARGS: ${PHASE1_INIT_ARGS[@]}"
  echo "PHASE2_INIT_ARGS: ${PHASE2_INIT_ARGS[@]}"
fi

echo "===== STEP 1: Train HRTBDA-v5 Phase I ====="

python -u "$HRTBDA_SCRIPT" \
  --phase phase1 \
  "${PHASE1_INIT_ARGS[@]}" \
  --xbd-root "$COMBINED_ROOT" \
  --train-split "${TRAIN_SPLITS[@]}" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --phase1-epochs "$PHASE1_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" \
  --img-size "$IMG_SIZE" \
  --lr "$PHASE1_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --base-channels "$BASE_CHANNELS" \
  --decoder-channels "$DECODER_CHANNELS" \
  --window-size "$WINDOW_SIZE" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --focal-gamma "$FOCAL_GAMMA" \
  --amp

PHASE1_CKPT=$OUTPUT_DIR/checkpoints/phase1_best.pt

if [ ! -f "$PHASE1_CKPT" ]; then
  echo "ERROR: expected Phase-I checkpoint not found:"
  echo "$PHASE1_CKPT"
  exit 1
fi

echo "Found Phase-I checkpoint:"
ls -lh "$PHASE1_CKPT"

echo "===== STEP 2: Train HRTBDA-v5 Phase II and test on unseen target dataset ====="

python -u "$HRTBDA_SCRIPT" \
  --phase phase2_test \
  "${PHASE2_INIT_ARGS[@]}" \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --xbd-root "$COMBINED_ROOT" \
  --train-split "${TRAIN_SPLITS[@]}" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --phase2-epochs "$PHASE2_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" \
  --img-size "$IMG_SIZE" \
  --phase2-crop-size "$PHASE2_CROP_SIZE" \
  --crop-candidate-count "$CROP_CANDIDATES" \
  --lr "$PHASE2_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --base-channels "$BASE_CHANNELS" \
  --decoder-channels "$DECODER_CHANNELS" \
  --window-size "$WINDOW_SIZE" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --focal-gamma "$FOCAL_GAMMA" \
  --aux-loc-weight "$AUX_LOC_WEIGHT" \
  --postprocess-dilation "$POSTPROCESS_DILATION" \
  --dilation-kernel "$DILATION_KERNEL" \
  --amp

echo "===== DONE $EXPERIMENT_ID / $EXPERIMENT_LABEL ====="

