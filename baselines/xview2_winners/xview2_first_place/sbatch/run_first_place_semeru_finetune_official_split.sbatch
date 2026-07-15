#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xv2_1st_semeru_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_semeru_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_semeru_ftv_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place
CUSTOM=$BASE/custom
DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/first_place_mount_semeru_FINE_TUNE_OFFICIAL_SPLIT
FT_WEIGHTS=$BASE/weights/mount_semeru_finetuned_weights_official_split
PRED_SAVE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_mount_semeru_FINE_TUNED_OFFICIAL_SPLIT_predictions
FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_mount_semeru_FINE_TUNED_OFFICIAL_SPLIT_full_solution

mkdir -p "$BASE/sbatch/logs"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128
cd "$BASE"

echo "================================================"
echo "Prepare leak-free Mount Semeru official split"
echo "================================================"
python -u "$CUSTOM/prepare_semeru_official_split_1024.py"
python -u "$CUSTOM/patch_first_place_official_split.py"

rm -rf "$FT_WEIGHTS"
python -u "$CUSTOM/setup_semeru_ft_weights.py"

rm -rf train val test tier3 masks folds.csv
ln -sfn "$DATA/train" train
ln -sfn "$DATA/val" val
ln -sfn "$DATA/train" tier3
ln -sfn "$DATA/test" test
ln -sfn "$DATA/masks" masks
ln -sfn "$DATA/folds.csv" folds.csv

rm -rf submission pred_loc_val
rm -rf pred34_loc pred50_loc_tuned pred92_loc_tuned pred154_loc
rm -rf res34cls2_0_tuned res34cls2_1_tuned res34cls2_2_tuned
rm -rf res50cls_cce_0_tuned res50cls_cce_1_tuned res50cls_cce_2_tuned
rm -rf dpn92cls_cce_0_tuned dpn92cls_cce_1_tuned dpn92cls_cce_2_tuned
rm -rf se154cls_0_tuned se154cls_1_tuned se154cls_2_tuned
rm -rf "$PRED_SAVE" "$FINAL_OUT"

export MODELS_FOLDER="$FT_WEIGHTS"
export FT_EPOCHS=10
export USE_OFFICIAL_VAL=1

echo "================================================"
echo "Fine-tune Mount Semeru localization models"
echo "================================================"
python -u tune50_loc.py 0
python -u tune50_loc.py 1
python -u tune50_loc.py 2
python -u tune92_loc.py 0
python -u tune92_loc.py 1
python -u tune92_loc.py 2

echo "Refresh aliases after localization fine-tuning"
python -u "$CUSTOM/setup_semeru_ft_weights.py"
echo "Predict official Mount Semeru validation localization masks"
python -u predict_loc_val.py

echo "================================================"
echo "Fine-tune Mount Semeru damage classification models"
echo "================================================"
python -u tune34_cls.py 0
python -u tune34_cls.py 1
python -u tune34_cls.py 2
python -u tune50_cls_cce.py 0
python -u tune50_cls_cce.py 1
python -u tune50_cls_cce.py 2
python -u tune92_cls_cce.py 0
python -u tune92_cls_cce.py 1
python -u tune92_cls_cce.py 2
python -u tune154_cls_cce.py 0
python -u tune154_cls_cce.py 1
python -u tune154_cls_cce.py 2

echo "Refresh aliases after classification fine-tuning"
python -u "$CUSTOM/setup_semeru_ft_weights.py"

echo "================================================"
echo "Predict on unseen Mount Semeru test set"
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

python -u eval_first_place_submission.py \
  --truth-dir "$DATA" \
  --pred-dir "$PRED_SAVE" \
  --out-dir "$FINAL_OUT" \
  --mode finetuned \
  --dataset-name "Mount Semeru"

cat "$FINAL_OUT/metrics_summary.txt"
echo "DONE: 1st-place xView2 fine-tuned on Mount Semeru official train/val/test split"
