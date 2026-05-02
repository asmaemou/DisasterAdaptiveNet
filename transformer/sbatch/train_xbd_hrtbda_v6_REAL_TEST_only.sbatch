#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=hrtbda_v6_cp_test
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=train_plus_tier3_test_xbd_hrtbda_v6_damage_copypaste_aug_REAL_TEST

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

# Reuse your verified xBD Phase-I localization checkpoint, but test with threshold 0.40.
PHASE1_SOURCE_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp
PHASE1_ORIG=$PHASE1_SOURCE_DIR/checkpoints/phase1_best.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/$EXPERIMENT_NAME
THRESH_CKPT_DIR=$OUTPUT_DIR/threshold_checkpoints
PHASE1_TEST_CKPT=$THRESH_CKPT_DIR/phase1_best_threshold_0.40.pt

# Phase II only: Phase I remains the cascaded localization mask.
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

LEARNING_RATE=8e-5
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

# Rare-crop selection weights.
CROP_WEIGHT_NO_DAMAGE=1.0
CROP_WEIGHT_MINOR=12.0
CROP_WEIGHT_MAJOR=12.0
CROP_WEIGHT_DESTROYED=4.0

# v6 augmentation: paired building/damage copy-paste.
DAMAGE_COPY_PASTE_PROB=0.25
DAMAGE_COPY_PASTE_CLASSES=minor,major,destroyed
DAMAGE_COPY_PASTE_MAX_OBJECTS=2
DAMAGE_COPY_PASTE_MIN_AREA=80
DAMAGE_COPY_PASTE_MAX_AREA=25000

# Final inference setting for the REAL xBD test split.
# This uses the same setting that was best on your v5 hold ablation.
PHASE1_TEST_THRESHOLD=0.40
POSTPROCESS_DILATION=minor
DILATION_KERNEL=3

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores" "$THRESH_CKPT_DIR"

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

if [ ! -f "$PHASE1_ORIG" ]; then
  echo "ERROR: original Phase I checkpoint not found:"
  echo "$PHASE1_ORIG"
  exit 1
fi

# Make a safe copied checkpoint with threshold 0.40.
# The original phase1_best.pt is NOT modified.
python - <<PY
import torch
from pathlib import Path
src = Path("$PHASE1_ORIG")
dst = Path("$PHASE1_TEST_CKPT")
dst.parent.mkdir(parents=True, exist_ok=True)
ckpt = torch.load(src, map_location="cpu")
ckpt["best_threshold"] = float("$PHASE1_TEST_THRESHOLD")
torch.save(ckpt, dst)
print(f"Wrote threshold checkpoint: {dst}")
print(f"Stored best_threshold: {ckpt.get('best_threshold')}")
PY

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "MODEL: HRTBDA v6 = v5 multi-label rare-crop cascade + paired Damage Object Copy-Paste"
echo "MODE: train Phase II and run FINAL REAL xBD TEST"
echo "XBD_ROOT: $XBD_ROOT"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "PHASE1_ORIG: $PHASE1_ORIG"
echo "PHASE1_TEST_CKPT: $PHASE1_TEST_CKPT"
echo "PHASE1_TEST_THRESHOLD: $PHASE1_TEST_THRESHOLD"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
echo "EFFECTIVE_BATCH_SIZE: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "PHASE2_CROP_SIZE: $PHASE2_CROP_SIZE"
echo "CROP_CANDIDATES: $CROP_CANDIDATES"
echo "DAMAGE_COPY_PASTE_PROB: $DAMAGE_COPY_PASTE_PROB"
echo "DAMAGE_COPY_PASTE_CLASSES: $DAMAGE_COPY_PASTE_CLASSES"
echo "DAMAGE_COPY_PASTE_MAX_OBJECTS: $DAMAGE_COPY_PASTE_MAX_OBJECTS"
echo "POSTPROCESS_DILATION: $POSTPROCESS_DILATION"
echo "DILATION_KERNEL: $DILATION_KERNEL"
echo "AMP: enabled"
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v6_damage_copypaste_aug.py
  --phase phase2_test
  --phase1-checkpoint "$PHASE1_TEST_CKPT"
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
  --damage-copy-paste-prob "$DAMAGE_COPY_PASTE_PROB"
  --damage-copy-paste-classes "$DAMAGE_COPY_PASTE_CLASSES"
  --damage-copy-paste-max-objects "$DAMAGE_COPY_PASTE_MAX_OBJECTS"
  --damage-copy-paste-min-area "$DAMAGE_COPY_PASTE_MIN_AREA"
  --damage-copy-paste-max-area "$DAMAGE_COPY_PASTE_MAX_AREA"
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
