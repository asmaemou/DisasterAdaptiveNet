#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=turkey_v5_swin
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

# ---------------------------------------------------------------------------
# Trains HRTBDA v5 (Swin ImageNet-pretrained backbone variant) DIRECTLY on
# Earthquake Turkey -- Phase I localization then Phase II damage severity
# (no/minor/major/destroyed), both starting from ImageNet weights, no xBD
# pretraining stage in between. Uses --phase both, which trains Phase I,
# then automatically initializes Phase II's backbone from the freshly
# Turkey-trained Phase I checkpoint (see load_phase1_backbone_into_phase2
# inside train_xbd_hrtbda_v5_swin_pretrained_cascade.py), then tests --
# all in one job, on this small (755 train image) dataset, so it finishes
# in hours rather than the multi-day xBD-pretraining route
# (train_xbd_hrtbda_v5_swin_pretrained_cascade.sbatch).
#
# This is a single-variable comparison against the very first, clean
# Turkey baseline run (experiment HRTBDA-v5-xBDInit-FullFineTune_EARTHQUAKE_
# TURKEY_resetbest, overall score 0.509, test Loc F1 0.932, Minor F1 0.270,
# Major F1 0.179, Destroyed F1 0.547): every hyperparameter below (LR,
# warmup, epoch budget, class weights, finetune tail) matches that original
# run's values as closely as the Swin resolution constraint allows. The
# only thing that changed is the backbone: custom from-scratch HRNet+DCSwin
# -> ImageNet-pretrained Swin Transformer. Any difference in the result is
# attributable to that swap alone, not confounded with protocol tuning like
# the two follow-up experiments earlier in this series were.
#
# Does NOT touch any existing run's output directory or files.
# ---------------------------------------------------------------------------

EXPERIMENT_NAME=HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY

TURKEY_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed

OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

TRAIN_SPLITS=(train)
VAL_SPLIT=val
TEST_SPLIT=test

# Matches the original clean baseline run's epoch budget.
PHASE1_EPOCHS=150
PHASE2_EPOCHS=30
WARMUP_EPOCHS=3
FINETUNE_EPOCHS=3
FINETUNE_LR=5e-5

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8

# Must each be a multiple of 224 (swin-patch-size * swin-window-size * 8 for
# the default variant). Closest clean multiples to the original run's
# 1024 / 608.
IMG_SIZE=896
PHASE2_CROP_SIZE=672
CROP_CANDIDATES=8

# Matches the original clean baseline run's LR/weight-decay exactly.
LEARNING_RATE=5e-5
WEIGHT_DECAY=1e-4
SEED=42

SWIN_VARIANT=swin_tiny_patch4_window7_224
SWIN_PATCH_SIZE=4
SWIN_WINDOW_SIZE=7
DECODER_CHANNELS=128
MAX_GRAD_NORM=1.0

FOCAL_GAMMA=2.0
AUX_LOC_WEIGHT=0.2

# Unchanged from the original clean baseline (script defaults): do not
# re-tune these alongside the backbone swap -- keep this a single-variable
# comparison.
MINOR_DAMAGE_BOOST=1.5
MAJOR_DAMAGE_BOOST=1.5
MAX_DAMAGE_CLASS_WEIGHT=10.0

POSTPROCESS_DILATION=minor
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

echo "===== EARTHQUAKE TURKEY -- SWIN IMAGENET-PRETRAINED DIRECT TRAIN JOB ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true

echo "===== CHECK / INSTALL timm ====="
if ! python -c "import timm" >/dev/null 2>&1; then
  echo "timm not found in disasteradaptivenet_cu128 -- installing..."
  pip install --no-input "timm>=0.9,<2"
fi
python -c "import timm; print('timm version:', timm.__version__)"

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "TURKEY_ROOT: $TURKEY_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SWIN_VARIANT: $SWIN_VARIANT"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "IMG_SIZE: $IMG_SIZE"
echo "PHASE2_CROP_SIZE: $PHASE2_CROP_SIZE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
echo "EFFECTIVE_BATCH_SIZE: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "==========================================================================="

if [ ! -d "$TURKEY_ROOT/train/images" ]; then
  echo "ERROR: missing Turkey train/images:"
  echo "$TURKEY_ROOT/train/images"
  exit 1
fi

echo "===== CHECK TURKEY DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$TURKEY_ROOT" \
  --splits train val test

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py
  --phase both
  --xbd-root "$TURKEY_ROOT"
  --train-split "${TRAIN_SPLITS[@]}"
  --val-split "$VAL_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --phase1-epochs "$PHASE1_EPOCHS"
  --phase2-epochs "$PHASE2_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --phase2-crop-size "$PHASE2_CROP_SIZE"
  --crop-candidate-count "$CROP_CANDIDATES"
  --lr "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --swin-variant "$SWIN_VARIANT"
  --swin-patch-size "$SWIN_PATCH_SIZE"
  --swin-window-size "$SWIN_WINDOW_SIZE"
  --decoder-channels "$DECODER_CHANNELS"
  --max-grad-norm "$MAX_GRAD_NORM"
  --warmup-epochs "$WARMUP_EPOCHS"
  --focal-gamma "$FOCAL_GAMMA"
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

echo "Done."
