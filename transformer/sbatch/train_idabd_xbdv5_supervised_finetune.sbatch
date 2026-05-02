#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=idabd_xbdv5_ft
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

# ---------------------------
# Data
# ---------------------------
IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd

# Reuse the exact same 80/10/10 split you already used for IDA-BD.
SPLIT_FILE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/idabd_splits_seed42_80_10_10.json

# ---------------------------
# xBD-trained initialization
# ---------------------------
XBD_PHASE1_INIT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp/checkpoints/phase1_best.pt
XBD_PHASE2_INIT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

# ---------------------------
# Output
# ---------------------------
OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_xbdv5_supervised_finetune_80_10_10
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

# ---------------------------
# Fine-tuning schedule
# ---------------------------
PHASE1_EPOCHS=40
PHASE2_EPOCHS=50
PHASE1_LR=1e-5
PHASE2_LR=2e-5
WARMUP_EPOCHS=5
EARLY_STOPPING_PATIENCE=10
PHASE1_EARLY_STOPPING_PATIENCE=10

# ---------------------------
# Batch / crop
# ---------------------------
PHASE1_BATCH_SIZE=1
PHASE2_BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8
IMG_SIZE=1024
PHASE2_CROP_SIZE=608
CROP_CANDIDATE_COUNT=8

# Keep crop weighting moderate. Do not overforce destroyed on this tiny dataset.
CROP_WEIGHT_NO_DAMAGE=1.0
CROP_WEIGHT_MINOR=8.0
CROP_WEIGHT_MAJOR=8.0
CROP_WEIGHT_DESTROYED=6.0

# ---------------------------
# Model / loss
# ---------------------------
BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
WEIGHT_DECAY=1e-4
FOCAL_GAMMA=2.0
LOC_LOSS_WEIGHT=1.0
CLS_LOSS_WEIGHT=1.0
AUX_LOC_WEIGHT=0.2
MINOR_DAMAGE_BOOST=1.5
MAJOR_DAMAGE_BOOST=1.5
MAX_DAMAGE_CLASS_WEIGHT=10.0
MAX_GRAD_NORM=0.5
SEED=42
SAVE_EVERY=1

# Evaluate several Phase-I thresholds and postprocessing options on IDA-BD validation.
ABLATION_THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"
ABLATION_DILATIONS="none minor"

# ---------------------------
# Conda
# ---------------------------
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
echo "IDABD_ROOT: $IDABD_ROOT"
echo "SPLIT_FILE: $SPLIT_FILE"
echo "XBD_PHASE1_INIT: $XBD_PHASE1_INIT"
echo "XBD_PHASE2_INIT: $XBD_PHASE2_INIT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PHASE1_LR: $PHASE1_LR"
echo "PHASE2_LR: $PHASE2_LR"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "EARLY_STOPPING_PATIENCE: $EARLY_STOPPING_PATIENCE"
echo "ABLATION_THRESHOLDS: $ABLATION_THRESHOLDS"
echo "ABLATION_DILATIONS: $ABLATION_DILATIONS"
echo "===================="

for f in "$XBD_PHASE1_INIT" "$XBD_PHASE2_INIT" "$SPLIT_FILE"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required file not found: $f"
    exit 1
  fi
  ls -lh "$f"
done

CMD=(
  python -u transformer/scripts/train_idabd_xbdv5_supervised_finetune.py
  --phase both
  --idabd-root "$IDABD_ROOT"
  --split-file "$SPLIT_FILE"
  --output-dir "$OUTPUT_DIR"
  --init-phase1-from "$XBD_PHASE1_INIT"
  --init-phase2-from "$XBD_PHASE2_INIT"
  --phase1-epochs "$PHASE1_EPOCHS"
  --phase2-epochs "$PHASE2_EPOCHS"
  --phase1-lr "$PHASE1_LR"
  --phase2-lr "$PHASE2_LR"
  --warmup-epochs "$WARMUP_EPOCHS"
  --phase1-early-stopping-patience "$PHASE1_EARLY_STOPPING_PATIENCE"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --phase1-batch-size "$PHASE1_BATCH_SIZE"
  --phase2-batch-size "$PHASE2_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
  --phase2-crop-size "$PHASE2_CROP_SIZE"
  --crop-candidate-count "$CROP_CANDIDATE_COUNT"
  --crop-weight-no-damage "$CROP_WEIGHT_NO_DAMAGE"
  --crop-weight-minor "$CROP_WEIGHT_MINOR"
  --crop-weight-major "$CROP_WEIGHT_MAJOR"
  --crop-weight-destroyed "$CROP_WEIGHT_DESTROYED"
  --base-channels "$BASE_CHANNELS"
  --decoder-channels "$DECODER_CHANNELS"
  --window-size "$WINDOW_SIZE"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --save-every "$SAVE_EVERY"
  --focal-gamma "$FOCAL_GAMMA"
  --loc-loss-weight "$LOC_LOSS_WEIGHT"
  --cls-loss-weight "$CLS_LOSS_WEIGHT"
  --aux-loc-weight "$AUX_LOC_WEIGHT"
  --minor-damage-boost "$MINOR_DAMAGE_BOOST"
  --major-damage-boost "$MAJOR_DAMAGE_BOOST"
  --max-damage-class-weight "$MAX_DAMAGE_CLASS_WEIGHT"
  --max-grad-norm "$MAX_GRAD_NORM"
  --postprocess-dilation none
  --dilation-kernel 3
  --ablation-thresholds $ABLATION_THRESHOLDS
  --ablation-dilations $ABLATION_DILATIONS
  --extra-photometric-aug
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"
