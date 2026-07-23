#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=turkey_v5_p2_smooth
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

# ---------------------------------------------------------------------------
# Third experiment in this series. Separate output dir; does not touch either
# prior run's output (HRTBDA-v5-xBDInit-FullFineTune_EARTHQUAKE_TURKEY_resetbest
# = original, overall score 0.509; HRTBDA-v5-xBDInit-Phase2Robust_EARTHQUAKE_TURKEY
# = first "robust" attempt, overall score 0.414 -- WORSE than original).
#
# Post-mortem on the first "robust" attempt (see
# logs/turkey_v5_p2_robust_1152796.out): it bundled three changes at once
# (lower LR + longer warmup, boosted minor/major class weights, and smoothed
# checkpoint selection) and the net effect was a regression. Unbundling the
# evidence from that log:
#
#   - Smoothed checkpoint selection (--smoothing-window) behaved exactly as
#     intended and cost almost nothing: it selected epoch 22 (raw hold score
#     0.4282) essentially tied with the run's true peak at epoch 29 (raw
#     0.4351). Not the source of the regression.
#
#   - Lowering LR (5e-5 -> 2e-5) and lengthening warmup (3 -> 6 epochs) did
#     NOT shorten the initial collapse into "predict everything no-damage" --
#     it roughly tripled it (epochs 1-13 stuck near the collapse floor,
#     vs epochs 1-5 in the original run). The run never reached the
#     original's peak hold score (0.435 vs 0.492) because a third of the
#     fixed 30-epoch budget was burned escaping a bad optimum instead of
#     refining a good one.
#
#   - Boosting minor/major class weights (1.5x -> 2.5x) silently cut
#     destroyed's relative weight by ~30% as an unintended side effect of
#     make_damage4_class_weights() renormalizing all four class weights by
#     their shared mean after the boost. Destroyed was the original run's
#     strongest damage class (F1 0.547) and it degraded the most (F1 0.468).
#
# This run isolates the one change that wasn't implicated: every
# hyperparameter below is set back to train_earthquake_turkey_hrtbda_v5_full_finetune_xbdinit.sbatch's
# exact original Phase II values (PHASE2_LR, PHASE2_WARMUP_EPOCHS,
# MINOR/MAJOR_DAMAGE_BOOST, MAX_DAMAGE_CLASS_WEIGHT, FINETUNE_EPOCHS,
# FINETUNE_LR all unchanged from the original run). Only SMOOTHING_WINDOW is
# new, via the same robust wrapper script used last time
# (train_earthquake_turkey_hrtbda_v5_phase2_robust_finetune.py). If this run
# beats or matches the original (0.509), smoothed selection is a real,
# isolated win. If it doesn't, drop this whole protocol-tuning direction.
#
# As before, Phase I is not retrained -- the already fine-tuned Turkey
# Phase I checkpoint (test loc F1 0.932) is reused directly.
# ---------------------------------------------------------------------------

EXPERIMENT_NAME=HRTBDA-v5-xBDInit-Phase2SmoothingOnly_EARTHQUAKE_TURKEY

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

# Unchanged from the original run (this is the reverted axis).
PHASE2_LR=5e-5
WEIGHT_DECAY=1e-4
SEED=42

BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0
PHASE2_WARMUP_EPOCHS=3

FOCAL_GAMMA=2.0
AUX_LOC_WEIGHT=0.2

# Unchanged from the original run (script defaults: 1.5 / 1.5 / 10.0).
MINOR_DAMAGE_BOOST=1.5
MAJOR_DAMAGE_BOOST=1.5
MAX_DAMAGE_CLASS_WEIGHT=10.0

# Unchanged from the original run (script defaults: 3 epochs @ 5e-5).
FINETUNE_EPOCHS=3
FINETUNE_LR=5e-5

# The one isolated change under test.
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

echo "===== EARTHQUAKE TURKEY PHASE II SMOOTHING-ONLY JOB ====="
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
echo "PHASE2_EPOCHS: $PHASE2_EPOCHS"
echo "PHASE2_LR (unchanged from original): $PHASE2_LR"
echo "PHASE2_WARMUP_EPOCHS (unchanged from original): $PHASE2_WARMUP_EPOCHS"
echo "MINOR_DAMAGE_BOOST (unchanged from original): $MINOR_DAMAGE_BOOST"
echo "MAJOR_DAMAGE_BOOST (unchanged from original): $MAJOR_DAMAGE_BOOST"
echo "MAX_DAMAGE_CLASS_WEIGHT (unchanged from original): $MAX_DAMAGE_CLASS_WEIGHT"
echo "FINETUNE_EPOCHS (unchanged from original): $FINETUNE_EPOCHS"
echo "FINETUNE_LR (unchanged from original): $FINETUNE_LR"
echo "SMOOTHING_WINDOW (the isolated change): $SMOOTHING_WINDOW"
echo "======================================================="

if [ ! -f "$ORIGINAL_TURKEY_PHASE1_CKPT" ]; then
  echo "ERROR: missing the original run's fine-tuned Turkey Phase-I checkpoint:"
  echo "$ORIGINAL_TURKEY_PHASE1_CKPT"
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
echo "Phase II smoothing-only fine-tune: xBD Phase-II init -> Earthquake Turkey"
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
  --finetune-lr "$FINETUNE_LR" \
  --smoothing-window "$SMOOTHING_WINDOW" \
  --postprocess-dilation "$POSTPROCESS_DILATION" \
  --dilation-kernel "$DILATION_KERNEL" \
  --amp

echo "Done."
