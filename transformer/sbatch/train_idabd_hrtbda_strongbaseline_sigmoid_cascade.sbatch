#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=idabd_sb_sigmoid
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

# =====================
# Paths
# =====================
IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd

# Reuse the same deterministic 70/9/8 IDA-BD split from your v2 cascade run.
SPLIT_FILE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/idabd_splits_seed42_80_10_10.json

# Reuse your already-trained IDA-BD Phase-I checkpoint by default.
# This isolates the Phase-II strong-baseline changes.
PHASE1_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/checkpoints/phase1_best.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_strongbaseline_sigmoid_cascade_80_10_10

# =====================
# Run mode
# =====================
# phase2_test = use existing Phase I, train strong-baseline Phase II, run val ablation, then real test.
# both        = train Phase I from scratch, then train Phase II, run val ablation, then real test.
PHASE=phase2_test

# =====================
# Training parameters
# =====================
PHASE1_EPOCHS=150
PHASE2_EPOCHS=80
EARLY_STOPPING_PATIENCE=20
WARMUP_EPOCHS=5

BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8
IMG_SIZE=1024

# Lower LR than the xBD run because IDA-BD is tiny and rare-class weighted training can become unstable.
LEARNING_RATE=5e-5
WEIGHT_DECAY=1e-4
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=0.5
FOCAL_GAMMA=2.0
SAVE_EVERY=1

# =====================
# Strong-baseline additions
# =====================
PHASE2_CROP_SIZE=608
CROP_CANDIDATES=16
RANDOM_CROP_MIN=529
RANDOM_CROP_MAX=715

# Rare-crop selection weights [no, minor, major, destroyed].
# Destroyed is heavily boosted because your current destroyed F1 is extremely low.
CROP_CLASS_WEIGHTS="1.0 10.0 10.0 30.0"

# Loss weighting boosts after inverse-frequency weighting.
AUX_LOC_WEIGHT=0.25
MINOR_BOOST=2.0
MAJOR_BOOST=2.0
DESTROYED_BOOST=8.0
MAX_CLASS_WEIGHT=25.0

# Validation ablation settings. The script selects best on validation, then tests once on real test.
ABLATION_THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"
ABLATION_DILATIONS="none minor destroyed minor_destroyed"
DILATION_KERNEL=3

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores" transformer/sbatch/logs

# =====================
# Conda
# =====================
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
echo "PHASE: $PHASE"
echo "IDABD_ROOT: $IDABD_ROOT"
echo "SPLIT_FILE: $SPLIT_FILE"
echo "PHASE1_CKPT: $PHASE1_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "MODEL: IDA-BD HRTBDA strong-baseline sigmoid cascade"
echo "PHASE2_CROP_SIZE: $PHASE2_CROP_SIZE"
echo "CROP_CANDIDATES: $CROP_CANDIDATES"
echo "CROP_CLASS_WEIGHTS: $CROP_CLASS_WEIGHTS"
echo "AUX_LOC_WEIGHT: $AUX_LOC_WEIGHT"
echo "MINOR_BOOST: $MINOR_BOOST"
echo "MAJOR_BOOST: $MAJOR_BOOST"
echo "DESTROYED_BOOST: $DESTROYED_BOOST"
echo "ABLATION_THRESHOLDS: $ABLATION_THRESHOLDS"
echo "ABLATION_DILATIONS: $ABLATION_DILATIONS"
echo "===================="

if [ "$PHASE" != "both" ] && [ ! -f "$PHASE1_CKPT" ]; then
  echo "ERROR: Phase-I checkpoint not found:"
  echo "$PHASE1_CKPT"
  exit 1
fi

CMD=(
  python -u transformer/scripts/train_idabd_hrtbda_strongbaseline_sigmoid_cascade.py
  --phase "$PHASE"
  --idabd-root "$IDABD_ROOT"
  --split-file "$SPLIT_FILE"
  --output-dir "$OUTPUT_DIR"
  --phase1-checkpoint "$PHASE1_CKPT"
  --phase1-epochs "$PHASE1_EPOCHS"
  --phase2-epochs "$PHASE2_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --img-size "$IMG_SIZE"
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
  --phase2-crop-size "$PHASE2_CROP_SIZE"
  --crop-candidates "$CROP_CANDIDATES"
  --random-crop-min "$RANDOM_CROP_MIN"
  --random-crop-max "$RANDOM_CROP_MAX"
  --crop-class-weights $CROP_CLASS_WEIGHTS
  --aux-loc-weight "$AUX_LOC_WEIGHT"
  --minor-boost "$MINOR_BOOST"
  --major-boost "$MAJOR_BOOST"
  --destroyed-boost "$DESTROYED_BOOST"
  --max-class-weight "$MAX_CLASS_WEIGHT"
  --ablation-thresholds $ABLATION_THRESHOLDS
  --ablation-dilations $ABLATION_DILATIONS
  --dilation-kernel "$DILATION_KERNEL"
  --extra-photometric-aug
  --run-final-test
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="

"${CMD[@]}"
