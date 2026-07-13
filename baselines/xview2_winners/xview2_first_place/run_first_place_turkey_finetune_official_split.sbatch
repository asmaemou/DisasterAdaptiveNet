#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xv2_1st_turkey_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/logs/xv2_1st_turkey_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/logs/xv2_1st_turkey_ftv_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place
DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT
SRC_WEIGHTS=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/weights/extracted_weights/weights
FT_WEIGHTS=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/weights/earthquake_turkey_finetuned_weights_official_split
PRED_SAVE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_earthquake_turkey_FINE_TUNED_OFFICIAL_SPLIT_predictions
FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_earthquake_turkey_FINE_TUNED_OFFICIAL_SPLIT_full_solution

mkdir -p "$BASE/logs"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

cd "$BASE"

if [ ! -f eval_first_place_submission.py ]; then
  echo "ERROR: missing eval_first_place_submission.py"
  exit 1
fi

echo "================================================"
echo "Prepare Earthquake Turkey official train/val/test data"
echo "================================================"

python -u prepare_first_place_turkey_finetune_official_split.py

echo "================================================"
echo "Patch 1st-place scripts for official train/val split"
echo "================================================"

python -u patch_first_place_official_split_safe_v2.py

echo "================================================"
echo "Verify official split patch"
echo "================================================"

grep -n "USE_OFFICIAL_VAL\|get_official_split_idxs\|train_dirs" train*.py tune*.py predict_loc_val.py | sed -n '1,80p'

echo "================================================"
echo "Create fine-tuning weight folder from xBD pretrained winner weights"
echo "================================================"

rm -rf "$FT_WEIGHTS"
mkdir -p "$FT_WEIGHTS"

python - <<'PY'
from pathlib import Path
import shutil

src = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/weights/extracted_weights/weights")
dst = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/weights/earthquake_turkey_finetuned_weights_official_split")

required = [
    "res34_loc_0_1_best",
    "res34_loc_1_1_best",
    "res34_loc_2_1_best",
    "res50_loc_0_tuned_best",
    "res50_loc_1_tuned_best",
    "res50_loc_2_tuned_best",
    "dpn92_loc_0_tuned_best",
    "dpn92_loc_1_tuned_best",
    "dpn92_loc_2_tuned_best",
    "se154_loc_0_1_best",
    "se154_loc_1_1_best",
    "se154_loc_2_1_best",
    "res34_cls2_0_tuned_best",
    "res34_cls2_1_tuned_best",
    "res34_cls2_2_tuned_best",
    "res50_cls_cce_0_tuned_best",
    "res50_cls_cce_1_tuned_best",
    "res50_cls_cce_2_tuned_best",
    "dpn92_cls_cce_0_tuned_best",
    "dpn92_cls_cce_1_tuned_best",
    "dpn92_cls_cce_2_tuned_best",
    "se154_cls_cce_0_tuned_best",
    "se154_cls_cce_1_tuned_best",
    "se154_cls_cce_2_tuned_best",
]

for name in required:
    s = src / name
    d = dst / name
    if not s.exists():
        raise FileNotFoundError(s)
    shutil.copy2(s, d)

aliases = {}

for seed in [0, 1, 2]:
    aliases[f"res50_loc_{seed}_best"] = f"res50_loc_{seed}_tuned_best"
    aliases[f"dpn92_loc_{seed}_best"] = f"dpn92_loc_{seed}_tuned_best"

    aliases[f"res34_cls2_{seed}_best"] = f"res34_cls2_{seed}_tuned_best"
    aliases[f"res50_cls_cce_{seed}_best"] = f"res50_cls_cce_{seed}_tuned_best"
    aliases[f"dpn92_cls_cce_{seed}_best"] = f"dpn92_cls_cce_{seed}_tuned_best"
    aliases[f"se154_cls_cce_{seed}_best"] = f"se154_cls_cce_{seed}_tuned_best"

for alias, original in aliases.items():
    src_file = dst / original
    dst_file = dst / alias
    if not src_file.exists():
        raise FileNotFoundError(src_file)
    shutil.copy2(src_file, dst_file)

print("Fine-tuning weights created:", dst)
print("Weight files:", len(list(dst.iterdir())))
PY

echo "================================================"
echo "Create missing fine-tuning weight aliases"
echo "================================================"

python -u fix_first_place_ft_weight_aliases.py

echo "================================================"
echo "Link official Earthquake Turkey train/val/test into 1st-place repo"
echo "================================================"

rm -rf train val test tier3 masks folds.csv
ln -sfn "$DATA/train" train
ln -sfn "$DATA/val" val
ln -sfn "$DATA/train" tier3
ln -sfn "$DATA/test" test
ln -sfn "$DATA/masks" masks
ln -sfn "$DATA/folds.csv" folds.csv

echo "Train images:"
ls -lah train/images | sed -n '1,10p'

echo "Val images:"
ls -lah val/images | sed -n '1,10p'

echo "Test images:"
ls -lah test/images | sed -n '1,10p'

echo "Evaluator folds.csv must be TEST only:"
head folds.csv
wc -l folds.csv

echo "================================================"
echo "Clean old prediction folders"
echo "================================================"

rm -rf submission
rm -rf pred34_loc pred50_loc_tuned pred92_loc_tuned pred154_loc
rm -rf res34cls2_0_tuned res34cls2_1_tuned res34cls2_2_tuned
rm -rf res50cls_cce_0_tuned res50cls_cce_1_tuned res50cls_cce_2_tuned
rm -rf dpn92cls_cce_0_tuned dpn92cls_cce_1_tuned dpn92cls_cce_2_tuned
rm -rf se154cls_0_tuned se154cls_1_tuned se154cls_2_tuned
rm -rf "$PRED_SAVE"
rm -rf "$FINAL_OUT"

export MODELS_FOLDER="$FT_WEIGHTS"
export FT_EPOCHS=10
export USE_OFFICIAL_VAL=1

echo "================================================"
echo "Fine-tune 1st-place xView2 winner on Earthquake Turkey"
echo "Official split mode:"
echo "  train = Earthquake Turkey train"
echo "  val   = Earthquake Turkey val"
echo "  test  = Earthquake Turkey test"
echo "FT_EPOCHS=$FT_EPOCHS"
echo "MODELS_FOLDER=$MODELS_FOLDER"
echo "USE_OFFICIAL_VAL=$USE_OFFICIAL_VAL"
echo "================================================"

python -u tune50_loc.py 0
python -u tune50_loc.py 1
python -u tune50_loc.py 2

python -u tune92_loc.py 0
python -u tune92_loc.py 1
python -u tune92_loc.py 2

echo "Skipping predict_loc_val.py for official-split run."

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

echo "================================================"
echo "Run prediction using official-split fine-tuned weights"
echo "================================================"

bash predict.sh

echo "================================================"
echo "Inspect submission"
echo "================================================"

SUB_COUNT=$(find submission -type f | wc -l)
echo "Submission file count: $SUB_COUNT"
find submission -type f | sed -n '1,40p'

if [ "$SUB_COUNT" -eq 0 ]; then
  echo "ERROR: submission is empty. Prediction failed."
  exit 1
fi

echo "================================================"
echo "Save predictions"
echo "================================================"

mkdir -p "$PRED_SAVE"
cp -a submission/. "$PRED_SAVE"/

echo "Saved prediction count:"
find "$PRED_SAVE" -type f | wc -l

echo "================================================"
echo "Evaluate on unseen Earthquake Turkey TEST split"
echo "================================================"

python -u eval_first_place_submission.py --truth-dir "$DATA" --pred-dir "$PRED_SAVE" --out-dir "$FINAL_OUT"

echo "================================================"
echo "Final metrics"
echo "================================================"

cat "$FINAL_OUT/metrics_summary.txt"

echo "DONE: 1st-place xView2 Earthquake Turkey fine-tuning with official train/val/test split"
