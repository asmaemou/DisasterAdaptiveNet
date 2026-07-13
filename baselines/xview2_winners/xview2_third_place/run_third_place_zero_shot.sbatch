#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=06:00:00
#SBATCH --job-name=xv2_3rd_zs
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_%x_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

if [ -z "${DATASET_ID:-}" ]; then
  echo "ERROR: DATASET_ID is missing"
  exit 1
fi

if [ -z "${SOURCE_TEST_ONLY:-}" ]; then
  echo "ERROR: SOURCE_TEST_ONLY is missing"
  exit 1
fi

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place
mkdir -p "$BASE/logs"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

cd "$BASE"

ln -sfn weights/extracted/ensemble models

THIRD_DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_${DATASET_ID}_TEST_ONLY
PRED_BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_${DATASET_ID}_TEST_ONLY_predictions
FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_${DATASET_ID}_TEST_ONLY_ZERO_SHOT_full_solution

echo "================================================"
echo "Create third-place TEST_ONLY view"
echo "================================================"
python -u make_third_place_test_view.py \
  --dataset-id "$DATASET_ID" \
  --source-test-only "$SOURCE_TEST_ONLY"

echo "================================================"
echo "Run 3rd-place xView2 weighted ensemble ZERO-SHOT"
echo "================================================"

TTA=flips-ms
if python -u predict_37_weighted.py \
    -dd "$THIRD_DATA" \
    -o "$PRED_BASE" \
    --size 512 \
    --tta "$TTA" \
    -p naive \
    --activation model \
    --fp16 \
    -b 1 \
    -w 0; then
  PRED_DIR="${PRED_BASE}_${TTA}"
else
  echo "TTA run failed. Retrying without TTA."
  python -u predict_37_weighted.py \
    -dd "$THIRD_DATA" \
    -o "$PRED_BASE" \
    --size 512 \
    -p naive \
    --activation model \
    --fp16 \
    -b 1 \
    -w 0
  PRED_DIR="$PRED_BASE"
fi

if [ ! -d "$PRED_DIR" ]; then
  echo "Expected prediction directory not found: $PRED_DIR"
  echo "Available nearby folders:"
  ls -ld ${PRED_BASE}*
  exit 2
fi

echo "================================================"
echo "Evaluate metrics"
echo "================================================"
python -u eval_third_place_predictions.py \
  --truth-dir "$THIRD_DATA" \
  --pred-dir "$PRED_DIR" \
  --out-dir "$FINAL_OUT" \
  --title "3rd-place xView2 weighted ensemble ZERO-SHOT on ${DATASET_ID} TEST"

echo "DONE: ${DATASET_ID}"
