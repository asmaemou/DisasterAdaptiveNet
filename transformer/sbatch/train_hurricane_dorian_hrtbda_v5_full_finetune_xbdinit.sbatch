#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=dorian_v5_fullft
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

EXPERIMENT_NAME=HRTBDA-v5-xBDInit-FullFineTune_HURRICANE_DORIAN_resetbest

DORIAN_ROOT=/homes/j244s673/documents/wsu/phd/hurricane_dorian_preprocessed

XBD_PHASE1_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_FINAL_TEST_th040_dilminor/threshold_checkpoints/phase1_best_threshold_0.40.pt

XBD_PHASE2_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

PHASE2_WRAPPER=transformer/scripts/train_hurricane_dorian_hrtbda_v5_phase2_finetune.py

OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

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

echo "===== HURRICANE DORIAN FULL xBD-INITIALIZED FINE-TUNING JOB ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "DORIAN_ROOT: $DORIAN_ROOT"
echo "XBD_PHASE1_CKPT: $XBD_PHASE1_CKPT"
echo "XBD_PHASE2_CKPT: $XBD_PHASE2_CKPT"
echo "PHASE2_WRAPPER: $PHASE2_WRAPPER"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "======================================================="

if [ ! -f "$XBD_PHASE1_CKPT" ]; then
  echo "ERROR: missing xBD Phase-I checkpoint:"
  echo "$XBD_PHASE1_CKPT"
  exit 1
fi

if [ ! -f "$XBD_PHASE2_CKPT" ]; then
  echo "ERROR: missing xBD Phase-II checkpoint:"
  echo "$XBD_PHASE2_CKPT"
  exit 1
fi

if [ ! -f "$PHASE2_WRAPPER" ]; then
  echo "ERROR: missing Phase-II wrapper:"
  echo "$PHASE2_WRAPPER"
  exit 1
fi

if [ ! -d "$DORIAN_ROOT/train/images" ]; then
  echo "ERROR: missing Dorian train/images:"
  echo "$DORIAN_ROOT/train/images"
  exit 1
fi

if [ ! -d "$DORIAN_ROOT/train/targets" ]; then
  echo "ERROR: missing Dorian train/targets:"
  echo "$DORIAN_ROOT/train/targets"
  exit 1
fi

echo "===== CHECK DORIAN DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$DORIAN_ROOT" \
  --splits train val test

echo "===== CREATE RESET-BEST PHASE-I CHECKPOINT ====="
XBD_PHASE1_RESET_CKPT=$OUTPUT_DIR/checkpoints/phase1_xbdinit_resetbest_for_dorian.pt

python - <<PY
import torch
from pathlib import Path

src = Path("$XBD_PHASE1_CKPT")
dst = Path("$XBD_PHASE1_RESET_CKPT")
dst.parent.mkdir(parents=True, exist_ok=True)

try:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(src, map_location="cpu")

# Keep xBD pretrained weights, but reset old xBD validation bookkeeping.
ckpt["epoch"] = 0
ckpt["best_metric"] = -1.0
ckpt["best_score"] = -1.0
ckpt["best_f1"] = -1.0
ckpt["best_loc_f1"] = -1.0
ckpt["best_threshold"] = 0.40

for k in [
    "optimizer",
    "optimizer_state_dict",
    "scheduler",
    "scheduler_state_dict",
    "scaler",
    "amp_scaler",
]:
    ckpt.pop(k, None)

torch.save(ckpt, dst)
print(f"Wrote reset-best Phase-I checkpoint: {dst}")
print("best_metric:", ckpt.get("best_metric"))
print("best_threshold:", ckpt.get("best_threshold"))
PY

echo
echo "======================================================="
echo "STEP 1: xBD Phase-I weights -> fine-tune Phase I on Hurricane Dorian"
echo "======================================================="

python -u transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py \
  --phase phase1 \
  --resume-phase1 "$XBD_PHASE1_RESET_CKPT" \
  --xbd-root "$DORIAN_ROOT" \
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

DORIAN_PHASE1_CKPT=$OUTPUT_DIR/checkpoints/phase1_best.pt

if [ ! -f "$DORIAN_PHASE1_CKPT" ]; then
  echo "ERROR: expected Dorian fine-tuned Phase-I checkpoint not found:"
  echo "$DORIAN_PHASE1_CKPT"
  exit 1
fi

echo "Fine-tuned Dorian Phase-I checkpoint:"
ls -lh "$DORIAN_PHASE1_CKPT"

echo
echo "======================================================="
echo "STEP 2: xBD Phase-II weights -> fine-tune Phase II on Hurricane Dorian"
echo "Using the newly fine-tuned Dorian Phase-I checkpoint"
echo "======================================================="

python -u "$PHASE2_WRAPPER" \
  --phase phase2_test \
  --phase1-checkpoint "$DORIAN_PHASE1_CKPT" \
  --init-phase2-from "$XBD_PHASE2_CKPT" \
  --xbd-root "$DORIAN_ROOT" \
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

echo "Done."