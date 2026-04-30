#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --job-name=hrtbda_v2_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_SPLITS=(train tier3)
VAL_SPLIT=hold
TEST_SPLIT=test

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2

# ── Epoch counts ──────────────────────────────────────────────────────────────
PHASE1_EPOCHS=150
PHASE2_EPOCHS=80          # was 30; raised because best was at epoch 9

# ── LR schedule ───────────────────────────────────────────────────────────────
WARMUP_EPOCHS=10          # NEW: linear warmup before cosine decay

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
AUX_LOC_WEIGHT=0.3        # NEW: weight of aux localization head in Phase II
MINOR_DAMAGE_BOOST=3.0    # NEW: extra multiplier on minor-damage class weight
CUTMIX_PROB=0.3           # NEW: probability of applying CutMix per sample

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
echo "MODEL: HRTBDA v2  (4-stage backbone, fixed focal-pt, inv-freq weights,"
echo "       warmup+cosine LR, CutMix, aux-loc head, scale-jitter aug)"
echo "PHASE1_EPOCHS: $PHASE1_EPOCHS"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "WARMUP_EPOCHS: $WARMUP_EPOCHS"
echo "AUX_LOC_WEIGHT: $AUX_LOC_WEIGHT"
echo "MINOR_DAMAGE_BOOST: $MINOR_DAMAGE_BOOST"
echo "CUTMIX_PROB: $CUTMIX_PROB"
nvidia-smi || true
echo "===================="

CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v2.py
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
  --cutmix-prob       "$CUTMIX_PROB"
  --amp                                   # enable AMP — cuts memory; faster convergence
  --thresholds $THRESHOLDS
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"; echo
echo "==================="
"${CMD[@]}"
