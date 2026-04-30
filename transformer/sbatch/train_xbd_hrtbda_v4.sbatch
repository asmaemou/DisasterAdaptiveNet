#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=hrtbda_v4_xbd
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# NOTE:
# Slurm needs the --output/--error directory to exist before sbatch opens the job log.
# Run this once before submitting if needed:
#   mkdir -p /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs

REPO_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$REPO_DIR"
mkdir -p "$REPO_DIR/transformer/sbatch/logs"

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v4

# ── Epoch counts ──────────────────────────────────────────────────────────────
PHASE1_EPOCHS=150
PHASE2_EPOCHS=80

# ── LR schedule ───────────────────────────────────────────────────────────────
WARMUP_EPOCHS=5

# ── Batch / image ─────────────────────────────────────────────────────────────
BATCH_SIZE=1
NUM_WORKERS=8
IMG_SIZE=1024

# ── Optimiser ─────────────────────────────────────────────────────────────────
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
SEED=42

# ── Architecture ──────────────────────────────────────────────────────────────
BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0

# ── Loss / augmentation ───────────────────────────────────────────────────────
FOCAL_GAMMA=2.0
LOC_LOSS_WEIGHT=1.0
CLS_LOSS_WEIGHT=1.0

# Phase II loc head weight. This is now a real localization head, not just aux.
AUX_LOC_WEIGHT=0.35

# Foreground-only damage-head class boosts.
MINOR_DAMAGE_BOOST=3.0
MAJOR_DAMAGE_BOOST=1.5
DESTROYED_DAMAGE_BOOST=1.0

# CutMix for rare damage crops.
CUTMIX_PROB=0.3

# Rare-damage weighted sampler for Phase II.
MINOR_SAMPLE_WEIGHT=4.0
MAJOR_SAMPLE_WEIGHT=2.5
DESTROYED_SAMPLE_WEIGHT=1.5

# Inference.
PHASE2_LOC_THRESHOLD=0.50
VAL_TTA=none
TEST_TTA=flip       # options: none, flip, d4. d4 is slower.

# ── Misc ──────────────────────────────────────────────────────────────────────
SAVE_EVERY=1
EARLY_STOPPING_PATIENCE=999

THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

# ── Conda ─────────────────────────────────────────────────────────────────────
if   [ -x "$HOME/miniforge3/bin/conda" ]; then CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda"  ]; then CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1;    then CONDA_EXE="$(command -v conda)"
else echo "ERROR: conda not found."; exit 1; fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== JOB INFO ====="
date; hostname; pwd; which python; python --version
echo "MODEL: HRTBDA v3"
echo "       4-stage HRNet-like backbone + DCSwin/DCMLP"
echo "       foreground-only damage head + loc head"
echo "       learnable pre/post difference fusion"
echo "       rare-damage weighted sampler + CutMix + test-time augmentation"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "WARMUP_EPOCHS: $WARMUP_EPOCHS"
echo "AUX_LOC_WEIGHT: $AUX_LOC_WEIGHT"
echo "MINOR_DAMAGE_BOOST: $MINOR_DAMAGE_BOOST"
echo "MAJOR_DAMAGE_BOOST: $MAJOR_DAMAGE_BOOST"
echo "CUTMIX_PROB: $CUTMIX_PROB"
echo "RARE SAMPLER weights: minor=$MINOR_SAMPLE_WEIGHT major=$MAJOR_SAMPLE_WEIGHT destroyed=$DESTROYED_SAMPLE_WEIGHT"
echo "VAL_TTA: $VAL_TTA"
echo "TEST_TTA: $TEST_TTA"
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v4.py
  --phase both
  --xbd-root "$XBD_ROOT"
  --train-split "${TRAIN_SPLITS[@]}"
  --val-split   "$VAL_SPLIT"
  --test-split  "$TEST_SPLIT"
  --output-dir  "$OUTPUT_DIR"
  --phase1-epochs "$PHASE1_EPOCHS"
  --phase2-epochs "$PHASE2_EPOCHS"
  --warmup-epochs "$WARMUP_EPOCHS"
  --batch-size  "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --img-size    "$IMG_SIZE"
  --lr          "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --seed        "$SEED"
  --base-channels    "$BASE_CHANNELS"
  --decoder-channels "$DECODER_CHANNELS"
  --window-size      "$WINDOW_SIZE"
  --max-grad-norm    "$MAX_GRAD_NORM"
  --save-every               "$SAVE_EVERY"
  --early-stopping-patience  "$EARLY_STOPPING_PATIENCE"
  --focal-gamma       "$FOCAL_GAMMA"
  --loc-loss-weight   "$LOC_LOSS_WEIGHT"
  --cls-loss-weight   "$CLS_LOSS_WEIGHT"
  --aux-loc-weight    "$AUX_LOC_WEIGHT"
  --minor-damage-boost "$MINOR_DAMAGE_BOOST"
  --major-damage-boost "$MAJOR_DAMAGE_BOOST"
  --destroyed-damage-boost "$DESTROYED_DAMAGE_BOOST"
  --cutmix-prob       "$CUTMIX_PROB"
  --rare-sampler
  --minor-sample-weight "$MINOR_SAMPLE_WEIGHT"
  --major-sample-weight "$MAJOR_SAMPLE_WEIGHT"
  --destroyed-sample-weight "$DESTROYED_SAMPLE_WEIGHT"
  --phase2-loc-threshold "$PHASE2_LOC_THRESHOLD"
  --val-tta "$VAL_TTA"
  --test-tta "$TEST_TTA"
  --amp
  --thresholds $THRESHOLDS
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"; echo
echo "==================="
"${CMD[@]}"
