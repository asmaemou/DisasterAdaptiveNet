#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=08:00:00
#SBATCH --job-name=xv2_1st_delta_eval
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_delta_eval_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_delta_eval_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SUBMISSION_WORKERS=4

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place
CUSTOM=$BASE/custom
DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/first_place_hurricane_delta_FINE_TUNE_OFFICIAL_SPLIT
FT_WEIGHTS=$BASE/weights/hurricane_delta_finetuned_weights_official_split
PRED_SAVE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_hurricane_delta_FINE_TUNED_OFFICIAL_SPLIT_predictions
FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_hurricane_delta_FINE_TUNED_OFFICIAL_SPLIT_full_solution

mkdir -p "$BASE/sbatch/logs"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128
cd "$BASE"

if [ ! -d "$DATA/test/images" ]; then
  echo "ERROR: prepared Hurricane Delta test data is missing: $DATA/test/images"
  exit 1
fi
if [ ! -d "$FT_WEIGHTS" ]; then
  echo "ERROR: fine-tuned Hurricane Delta weights are missing: $FT_WEIGHTS"
  exit 1
fi

export MODELS_FOLDER="$FT_WEIGHTS"

echo "================================================"
echo "Validate and refresh aliases for existing fine-tuned weights"
echo "================================================"
python -u "$CUSTOM/setup_hurricane_delta_ft_weights.py"

rm -rf test masks folds.csv
ln -sfn "$DATA/test" test
ln -sfn "$DATA/masks" masks
ln -sfn "$DATA/folds.csv" folds.csv

rm -rf submission
rm -rf pred34_loc pred50_loc_tuned pred92_loc_tuned pred154_loc
rm -rf res34cls2_0_tuned res34cls2_1_tuned res34cls2_2_tuned
rm -rf res50cls_cce_0_tuned res50cls_cce_1_tuned res50cls_cce_2_tuned
rm -rf dpn92cls_cce_0_tuned dpn92cls_cce_1_tuned dpn92cls_cce_2_tuned
rm -rf se154cls_0_tuned se154cls_1_tuned se154cls_2_tuned
rm -rf "$PRED_SAVE" "$FINAL_OUT"

echo "================================================"
echo "Run prediction only with completed Hurricane Delta checkpoints"
echo "================================================"
bash predict.sh

SUB_COUNT=$(find submission -type f | wc -l)
echo "Submission file count: $SUB_COUNT"
if [ "$SUB_COUNT" -eq 0 ]; then
  echo "ERROR: submission is empty. Prediction failed."
  exit 1
fi

mkdir -p "$PRED_SAVE"
cp -a submission/. "$PRED_SAVE"/

echo "================================================"
echo "Evaluate on held-out Hurricane Delta test"
echo "================================================"
python -u eval_first_place_submission.py \
  --truth-dir "$DATA" \
  --pred-dir "$PRED_SAVE" \
  --out-dir "$FINAL_OUT" \
  --mode finetuned \
  --dataset-name "Hurricane Delta"

cat "$FINAL_OUT/metrics_summary.txt"
echo "DONE: recovered 1st-place Hurricane Delta prediction and evaluation"
