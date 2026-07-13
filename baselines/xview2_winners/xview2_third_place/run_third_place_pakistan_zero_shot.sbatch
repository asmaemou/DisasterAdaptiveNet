#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=06:00:00
#SBATCH --job-name=xv2_3rd_pak_zs
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_pak_zs_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_pak_zs_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place
DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_pakistan_flooding_TEST_ONLY
PRED_BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_pakistan_flooding_TEST_ONLY_predictions
FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_pakistan_flooding_TEST_ONLY_ZERO_SHOT_full_solution

mkdir -p "$BASE/logs"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

cd "$BASE"

ln -sfn weights/extracted/ensemble models

echo "================================================"
echo "Prepare Pakistan Flooding TEST_ONLY for 3rd-place xView2"
echo "================================================"
python -u prepare_third_place_pakistan_test.py

echo "================================================"
echo "Run 3rd-place xView2 weighted ensemble ZERO-SHOT on Pakistan Flooding"
echo "================================================"

rm -rf "${PRED_BASE}" "${PRED_BASE}_flips-ms"

python -u predict_37_weighted.py \
  -dd "$DATA" \
  -o "$PRED_BASE" \
  --size 512 \
  --tta flips-ms \
  -p naive \
  --activation model \
  --fp16 \
  -b 1 \
  -w 0

PRED_DIR="${PRED_BASE}_flips-ms"

if [ ! -d "$PRED_DIR" ]; then
  echo "ERROR: expected prediction folder not found: $PRED_DIR"
  echo "Available folders:"
  ls -ld ${PRED_BASE}*
  exit 2
fi

echo "================================================"
echo "Evaluate 3rd-place predictions"
echo "================================================"

python -u eval_third_place_pakistan_predictions.py \
  --truth-dir "$DATA" \
  --pred-dir "$PRED_DIR" \
  --out-dir "$FINAL_OUT"

echo "DONE: 3rd-place xView2 Pakistan Flooding zero-shot"
