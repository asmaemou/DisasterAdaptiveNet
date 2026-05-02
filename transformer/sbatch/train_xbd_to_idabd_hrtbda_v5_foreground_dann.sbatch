#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=idabd_dann_v5
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

# -------------------------
# Source and target datasets
# -------------------------
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd
IDABD_SPLIT_FILE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/idabd_splits_seed42_80_10_10.json

# -------------------------
# xBD-trained initialization
# -------------------------
XBD_PHASE1=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp/checkpoints/phase1_best.pt
XBD_PHASE2=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_to_idabd_hrtbda_v5_foreground_dann

# -------------------------
# Model/training parameters
# -------------------------
EPOCHS=20
STEPS_PER_EPOCH=500
WARMUP_EPOCHS=3
SOURCE_BATCH_SIZE=1
TARGET_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8
IMG_SIZE=1024
PHASE2_CROP_SIZE=608
CROP_CANDIDATE_COUNT=8

LR=1e-5
WEIGHT_DECAY=1e-4
MAX_GRAD_NORM=0.5
EARLY_STOPPING_PATIENCE=8
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8

FOCAL_GAMMA=2.0
CLS_LOSS_WEIGHT=1.0
AUX_LOC_WEIGHT=0.2
MINOR_DAMAGE_BOOST=1.5
MAJOR_DAMAGE_BOOST=1.5
DESTROYED_DAMAGE_BOOST=1.0
MAX_DAMAGE_CLASS_WEIGHT=10.0

# Source rare-crop weights. Target crops are random and label-free.
CROP_WEIGHT_NO=1.0
CROP_WEIGHT_MINOR=12.0
CROP_WEIGHT_MAJOR=12.0
CROP_WEIGHT_DESTROYED=4.0

# Domain-adaptation settings. Keep conservative at first.
LAMBDA_DOMAIN=0.03
LAMBDA_DOMAIN_LOSS=0.10
DOMAIN_WARMUP_EPOCHS=5
LAMBDA_ENTROPY=0.0
DOMAIN_HIDDEN_DIM=256
DOMAIN_DROPOUT=0.2

THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"
DILATION_OPTIONS="none minor"
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

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "Experiment: xBD -> IDA-BD foreground-aware adversarial domain adaptation"
echo "XBD_ROOT: $XBD_ROOT"
echo "IDABD_ROOT: $IDABD_ROOT"
echo "IDABD_SPLIT_FILE: $IDABD_SPLIT_FILE"
echo "XBD_PHASE1: $XBD_PHASE1"
echo "XBD_PHASE2: $XBD_PHASE2"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "STEPS_PER_EPOCH: $STEPS_PER_EPOCH"
echo "SOURCE_BATCH_SIZE: $SOURCE_BATCH_SIZE"
echo "TARGET_BATCH_SIZE: $TARGET_BATCH_SIZE"
echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
echo "LR: $LR"
echo "LAMBDA_DOMAIN: $LAMBDA_DOMAIN"
echo "LAMBDA_DOMAIN_LOSS: $LAMBDA_DOMAIN_LOSS"
echo "LAMBDA_ENTROPY: $LAMBDA_ENTROPY"
echo "===================="

if [ ! -f "$XBD_PHASE1" ]; then
  echo "ERROR: missing xBD Phase-I checkpoint: $XBD_PHASE1"
  exit 1
fi
if [ ! -f "$XBD_PHASE2" ]; then
  echo "ERROR: missing xBD Phase-II checkpoint: $XBD_PHASE2"
  exit 1
fi
if [ ! -f "$IDABD_SPLIT_FILE" ]; then
  echo "ERROR: missing IDA-BD split file: $IDABD_SPLIT_FILE"
  exit 1
fi

CMD=(
  python -u transformer/scripts/train_xbd_to_idabd_hrtbda_v5_foreground_dann.py
  --phase adapt
  --xbd-root "$XBD_ROOT"
  --source-train-split train tier3
  --idabd-root "$IDABD_ROOT"
  --split-file "$IDABD_SPLIT_FILE"
  --phase1-checkpoint "$XBD_PHASE1"
  --phase2-checkpoint "$XBD_PHASE2"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --steps-per-epoch "$STEPS_PER_EPOCH"
  --warmup-epochs "$WARMUP_EPOCHS"
  --source-batch-size "$SOURCE_BATCH_SIZE"
  --target-batch-size "$TARGET_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --phase2-crop-size "$PHASE2_CROP_SIZE"
  --crop-candidate-count "$CROP_CANDIDATE_COUNT"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --seed "$SEED"
  --base-channels "$BASE_CHANNELS"
  --decoder-channels "$DECODER_CHANNELS"
  --window-size "$WINDOW_SIZE"
  --focal-gamma "$FOCAL_GAMMA"
  --cls-loss-weight "$CLS_LOSS_WEIGHT"
  --aux-loc-weight "$AUX_LOC_WEIGHT"
  --minor-damage-boost "$MINOR_DAMAGE_BOOST"
  --major-damage-boost "$MAJOR_DAMAGE_BOOST"
  --destroyed-damage-boost "$DESTROYED_DAMAGE_BOOST"
  --max-damage-class-weight "$MAX_DAMAGE_CLASS_WEIGHT"
  --crop-weight-no-damage "$CROP_WEIGHT_NO"
  --crop-weight-minor "$CROP_WEIGHT_MINOR"
  --crop-weight-major "$CROP_WEIGHT_MAJOR"
  --crop-weight-destroyed "$CROP_WEIGHT_DESTROYED"
  --lambda-domain "$LAMBDA_DOMAIN"
  --lambda-domain-loss "$LAMBDA_DOMAIN_LOSS"
  --domain-warmup-epochs "$DOMAIN_WARMUP_EPOCHS"
  --lambda-entropy "$LAMBDA_ENTROPY"
  --domain-hidden-dim "$DOMAIN_HIDDEN_DIM"
  --domain-dropout "$DOMAIN_DROPOUT"
  --dilation-kernel "$DILATION_KERNEL"
  --dilation-options $DILATION_OPTIONS
  --thresholds $THRESHOLDS
  --extra-photometric-aug
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"
