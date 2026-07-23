#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=turkey_v5_p2_robust
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

# ---------------------------------------------------------------------------
# This is a NEW, separate experiment. It does not overwrite or resume the
# original run (train_earthquake_turkey_hrtbda_v5_full_finetune_xbdinit.sbatch,
# experiment name HRTBDA-v5-xBDInit-FullFineTune_EARTHQUAKE_TURKEY_resetbest).
#
# What changed vs. the original run, and why (see log
# transformer/sbatch/logs/turkey_v5_fullft_1083955.out from the original run):
#
#   1. Phase I is NOT retrained here. The original run already reached
#      test localization F1 = 0.932 -- that part works well. We reuse its
#      finished checkpoint (ORIGINAL_TURKEY_PHASE1_CKPT below) so this run
#      isolates the effect of the Phase II protocol changes and finishes in
#      hours instead of ~3 days.
#
#   2. Phase II uses train_earthquake_turkey_hrtbda_v5_phase2_robust_finetune.py
#      instead of train_earthquake_turkey_hrtbda_v5_phase2_finetune.py. It adds
#      --smoothing-window: "best epoch" is chosen by a trailing moving average
#      of the hold cascade score instead of a single-epoch argmax. In the
#      original run hold_score_cascade swung +/-0.13 between adjacent epochs
#      (94 val images, only a few thousand minor/major/destroyed pixels), and
#      the single epoch that won that lottery (epoch 19) generalized poorly to
#      test (hold major F1 0.315 -> test major F1 0.179).
#
#   3. PHASE2_LR is lowered (5e-5 -> 2e-5) and PHASE2_WARMUP_EPOCHS is raised
#      (3 -> 6, now out of 30 total Phase II epochs instead of shared with
#      Phase I's 150). In the original run the damage head collapsed to
#      near-zero minor/major/destroyed F1 for epochs 1-5 before recovering --
#      classic too-hot/too-short warmup applied to an already xBD-pretrained
#      checkpoint on a small, 97%-imbalanced target set.
#
#   4. FINETUNE_EPOCHS=0 disables the original's unconditional 3-epoch
#      "warm-restart" tail (LR reset back to 5e-5 after the cosine schedule
#      had already decayed to ~1.7e-7). On the original run this tail did not
#      beat the epoch-19 checkpoint and only added variance at the end.
#
#   5. MINOR_DAMAGE_BOOST / MAJOR_DAMAGE_BOOST raised 1.5 -> 2.5 and
#      MAX_DAMAGE_CLASS_WEIGHT raised 10 -> 16. Minor damage is ~1.7% and
#      major damage ~1.6% of building pixels in the Turkey train set; the
#      original boosts were being clipped by the weight cap before they
#      could meaningfully act.
#
# Architecture, data pipeline, and loss functions are otherwise byte-for-byte
# identical to the original run (same base/decoder channels, window size,
# image/crop sizes, crop-candidate rare-class sampling, focal gamma).
# ---------------------------------------------------------------------------

EXPERIMENT_NAME=HRTBDA-v5-xBDInit-Phase2Robust_EARTHQUAKE_TURKEY

TURKEY_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed

ORIGINAL_TURKEY_PHASE1_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/HRTBDA-v5-xBDInit-FullFineTune_EARTHQUAKE_TURKEY_resetbest/checkpoints/phase1_best.pt

XBD_PHASE2_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

PHASE2_WRAPPER=transformer/scripts/train_earthquake_turkey_hrtbda_v5_phase2_robust_finetune.py

OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

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

PHASE2_LR=2e-5
WEIGHT_DECAY=1e-4
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0
PHASE2_WARMUP_EPOCHS=6

FOCAL_GAMMA=2.0
AUX_LOC_WEIGHT=0.2

MINOR_DAMAGE_BOOST=2.5
MAJOR_DAMAGE_BOOST=2.5
MAX_DAMAGE_CLASS_WEIGHT=16.0

FINETUNE_EPOCHS=0
SMOOTHING_WINDOW=3

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

echo "===== EARTHQUAKE TURKEY PHASE II ROBUST FINE-TUNING JOB ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "TURKEY_ROOT: $TURKEY_ROOT"
echo "ORIGINAL_TURKEY_PHASE1_CKPT (reused, not retrained): $ORIGINAL_TURKEY_PHASE1_CKPT"
echo "XBD_PHASE2_CKPT: $XBD_PHASE2_CKPT"
echo "PHASE2_WRAPPER: $PHASE2_WRAPPER"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TRAIN_SPLITS: ${TRAIN_SPLITS[*]}"
echo "VAL_SPLIT: $VAL_SPLIT"
echo "TEST_SPLIT: $TEST_SPLIT"
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "PHASE2_LR: $PHASE2_LR"
echo "PHASE2_WARMUP_EPOCHS: $PHASE2_WARMUP_EPOCHS"
echo "FINETUNE_EPOCHS: $FINETUNE_EPOCHS"
echo "SMOOTHING_WINDOW: $SMOOTHING_WINDOW"
echo "MINOR_DAMAGE_BOOST: $MINOR_DAMAGE_BOOST"
echo "MAJOR_DAMAGE_BOOST: $MAJOR_DAMAGE_BOOST"
echo "MAX_DAMAGE_CLASS_WEIGHT: $MAX_DAMAGE_CLASS_WEIGHT"
echo "======================================================="

if [ ! -f "$ORIGINAL_TURKEY_PHASE1_CKPT" ]; then
  echo "ERROR: missing the original run's fine-tuned Turkey Phase-I checkpoint:"
  echo "$ORIGINAL_TURKEY_PHASE1_CKPT"
  echo "This script reuses that checkpoint instead of retraining Phase I."
  echo "Run train_earthquake_turkey_hrtbda_v5_full_finetune_xbdinit.sbatch first, or"
  echo "point ORIGINAL_TURKEY_PHASE1_CKPT at a different completed Phase I checkpoint."
  exit 1
fi

if [ ! -f "$XBD_PHASE2_CKPT" ]; then
  echo "ERROR: missing xBD Phase-II checkpoint:"
  echo "$XBD_PHASE2_CKPT"
  exit 1
fi

if [ ! -f "$PHASE2_WRAPPER" ]; then
  echo "ERROR: missing Phase-II robust wrapper:"
  echo "$PHASE2_WRAPPER"
  exit 1
fi

if [ ! -d "$TURKEY_ROOT/train/images" ]; then
  echo "ERROR: missing Turkey train/images:"
  echo "$TURKEY_ROOT/train/images"
  exit 1
fi

echo "===== CHECK TURKEY DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$TURKEY_ROOT" \
  --splits train val test

echo
echo "======================================================="
echo "Phase II robust fine-tune: xBD Phase-II init -> Earthquake Turkey"
echo "Localization mask comes from the already fine-tuned Turkey Phase-I model."
echo "======================================================="

python -u "$PHASE2_WRAPPER" \
  --phase phase2_test \
  --phase1-checkpoint "$ORIGINAL_TURKEY_PHASE1_CKPT" \
  --init-phase2-from "$XBD_PHASE2_CKPT" \
  --xbd-root "$TURKEY_ROOT" \
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
  --warmup-epochs "$PHASE2_WARMUP_EPOCHS" \
  --focal-gamma "$FOCAL_GAMMA" \
  --aux-loc-weight "$AUX_LOC_WEIGHT" \
  --minor-damage-boost "$MINOR_DAMAGE_BOOST" \
  --major-damage-boost "$MAJOR_DAMAGE_BOOST" \
  --max-damage-class-weight "$MAX_DAMAGE_CLASS_WEIGHT" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --smoothing-window "$SMOOTHING_WINDOW" \
  --postprocess-dilation "$POSTPROCESS_DILATION" \
  --dilation-kernel "$DILATION_KERNEL" \
  --amp

echo "Done."
