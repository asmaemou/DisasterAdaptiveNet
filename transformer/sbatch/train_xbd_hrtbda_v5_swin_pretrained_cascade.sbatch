#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=hrtbda_v5_swin
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

# ---------------------------------------------------------------------------
# Trains HRTBDA v5 on xBD (train + tier3, val on hold, test on test) using
# the ImageNet-pretrained Swin Transformer backbone variant of the model
# (transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py), a
# standalone new script that does not modify the original
# train_xbd_hrtbda_v5_multilabel_crop_cascade.py at all -- only the backbone
# differs (SwinPretrainedBackbone instead of HighResolutionTransformerBackbone);
# everything else (CSFModule pre/post fusion, MultiScaleDecoder, two-phase
# cascade, multilabel damage head, rare-crop sampling, losses) is identical.
#
# This is an xBD-level PRETRAINING run (Phase I localization + Phase II
# damage, both from scratch on xBD, on top of the Swin backbone's ImageNet
# weights) -- the Swin-backbone analogue of what originally produced
# XBD_PHASE1_CKPT / XBD_PHASE2_CKPT for the custom-backbone model. Its
# output checkpoints (phase1_best.pt, phase2_best.pt) are the ones a later,
# separate Turkey fine-tuning sbatch would start from -- this run does not
# touch Turkey data at all.
#
# Resolution note: img/crop sizes must be multiples of
# swin-patch-size * swin-window-size * 8 = 224 for the default
# swin_tiny_patch4_window7_224 variant. IMG_SIZE=896 and PHASE2_CROP_SIZE=672
# below are the closest clean multiples of 224 to the original backbone's
# 1024 / 608 -- the script itself will raise a clear error at startup if
# these are changed to a non-multiple of 224.
#
# Memory note: Swin-Tiny (~28M backbone params, up to 768 channels at its
# deepest stage vs. the original backbone's 384) is heavier than the
# original from-scratch backbone at base_channels=48. BATCH_SIZE/GRAD_ACCUM
# below start at the same values the original pipeline used; if you hit
# CUDA OOM, first try BATCH_SIZE=1 with GRAD_ACCUM_STEPS doubled to keep the
# same effective batch size, before reducing IMG_SIZE.
# ---------------------------------------------------------------------------

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v5_swin_pretrained_cascade

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

PHASE1_EPOCHS=60
PHASE2_EPOCHS=60
WARMUP_EPOCHS=5
FINETUNE_EPOCHS=3
FINETUNE_LR=5e-5

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8

# Must each be a multiple of 224 (see resolution note above).
IMG_SIZE=896
PHASE2_CROP_SIZE=672
CROP_CANDIDATES=8

# Conservative LR for fine-tuning a pretrained transformer backbone end-to-end
# (higher than this risks destroying the ImageNet features early in training,
# the same failure mode diagnosed on the un-pretrained backbone's Phase II
# runs, just for a different underlying reason). Treat as a starting point.
LEARNING_RATE=5e-5
WEIGHT_DECAY=1e-4
SEED=42

SWIN_VARIANT=swin_tiny_patch4_window7_224
SWIN_PATCH_SIZE=4
SWIN_WINDOW_SIZE=7
DECODER_CHANNELS=128
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

echo "===== HRTBDA V5 SWIN-PRETRAINED BACKBONE -- xBD TRAIN JOB ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true

echo "===== CHECK / INSTALL timm ====="
# timm is not in requirements_cu128.txt yet -- install it into this conda env
# if missing, rather than failing deep inside model construction after the
# job has already been running. If the compute node has no outbound internet
# access, pre-install timm from the login node first:
#   conda activate disasteradaptivenet_cu128 && pip install "timm>=0.9,<2"
if ! python -c "import timm" >/dev/null 2>&1; then
  echo "timm not found in disasteradaptivenet_cu128 -- installing..."
  pip install --no-input "timm>=0.9,<2"
fi
python -c "import timm; print('timm version:', timm.__version__)"

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "XBD_ROOT: $XBD_ROOT"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SWIN_VARIANT: $SWIN_VARIANT"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "IMG_SIZE: $IMG_SIZE"
echo "PHASE2_CROP_SIZE: $PHASE2_CROP_SIZE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
echo "EFFECTIVE_BATCH_SIZE: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "==============================================================="

if [ ! -d "$XBD_ROOT" ]; then
  echo "ERROR: missing xBD root: $XBD_ROOT"
  exit 1
fi

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py
  --phase both
  --xbd-root "$XBD_ROOT"
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
  --crop-weight-no-damage "$CROP_WEIGHT_NO_DAMAGE"
  --crop-weight-minor "$CROP_WEIGHT_MINOR"
  --crop-weight-major "$CROP_WEIGHT_MAJOR"
  --crop-weight-destroyed "$CROP_WEIGHT_DESTROYED"
  --extra-photometric-aug
  --lr "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --swin-variant "$SWIN_VARIANT"
  --swin-patch-size "$SWIN_PATCH_SIZE"
  --swin-window-size "$SWIN_WINDOW_SIZE"
  --decoder-channels "$DECODER_CHANNELS"
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

echo "Done."
