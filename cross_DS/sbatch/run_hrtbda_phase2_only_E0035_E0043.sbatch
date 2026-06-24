#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=hrtbda_phase2_only
#SBATCH --array=0-3
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%A_%a.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/cross_DS/sbatch/logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATASET_BASE=/homes/j244s673/documents/wsu/phd

cd "$PROJECT_ROOT"
mkdir -p cross_DS/sbatch/logs

EXP_IDS=(E0035 E0037 E0039 E0043)
DATASETS=(EARTHQUAKE-TURKEY HURRICANE-LAURA MOUNT-SEMERU-ERUPTION STVINCENT-VOLCANO)

IDX=${SLURM_ARRAY_TASK_ID}
EXPERIMENT_ID=${EXP_IDS[$IDX]}
DATASET=${DATASETS[$IDX]}

EXPERIMENT_NAME=HRTBDA-v5_${EXPERIMENT_ID}_Scratch_Train_${DATASET}_Test_${DATASET}
COMBINED_ROOT=$PROJECT_ROOT/output/HRTBDA_single_dataset_combined/$EXPERIMENT_ID
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

PREP_SCRIPT=cross_DS/scripts/prepare_hrtbda_single_dataset_finetune.py
HRTBDA_SCRIPT=transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py

TRAIN_SPLITS=(train)
VAL_SPLIT=val
TEST_SPLIT=test

PHASE2_EPOCHS=30

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8

IMG_SIZE=1024
PHASE2_CROP_SIZE=608
CROP_CANDIDATES=8

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

echo "===== HRTBDA-v5 PHASE-II-ONLY SINGLE-DATASET EXPERIMENT ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true

echo "EXPERIMENT_ID: $EXPERIMENT_ID"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "DATASET: $DATASET"
echo "COMBINED_ROOT: $COMBINED_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "================================================"

PHASE1_CKPT=$OUTPUT_DIR/checkpoints/phase1_best.pt

if [ ! -f "$PHASE1_CKPT" ]; then
  echo "ERROR: missing Phase-I checkpoint:"
  echo "$PHASE1_CKPT"
  exit 1
fi

echo "Found Phase-I checkpoint:"
ls -lh "$PHASE1_CKPT"

echo "===== PREPARE COMBINED ROOT AGAIN, SAFE FOR PHASE II ====="
python -u "$PREP_SCRIPT" \
  --dataset-base "$DATASET_BASE" \
  --dataset "$DATASET" \
  --output-root "$COMBINED_ROOT" \
  --mode symlink

echo "===== CHECK COMBINED DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$COMBINED_ROOT" \
  --splits train val test

echo "===== STEP 2 ONLY: Train HRTBDA-v5 Phase II and test on $DATASET ====="

python -u "$HRTBDA_SCRIPT" \
  --phase phase2_test \
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

echo "Done $EXPERIMENT_ID Phase II + test."