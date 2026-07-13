#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=06:00:00
#SBATCH --job-name=xv2_1st_texas_zs
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/logs/xv2_1st_texas_zs_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/logs/xv2_1st_texas_zs_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place
DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/first_place_texas_tornadoes_TEST_ONLY
PRED_SAVE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_texas_tornadoes_TEST_ONLY_predictions
FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/first_place_texas_tornadoes_TEST_ONLY_ZERO_SHOT_full_solution
WEIGHT_SRC=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/weights/extracted_weights/weights

mkdir -p "$BASE/logs"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

cd "$BASE"

echo "================================================"
echo "Prepare Texas Tornadoes TEST_ONLY for 1st-place xView2"
echo "================================================"

python -u prepare_first_place_texas_tornadoes_test.py

echo "================================================"
echo "Patch torch.load for newer PyTorch compatibility"
echo "================================================"

python - <<'PY'
from pathlib import Path
import re

for p in Path(".").glob("predict*.py"):
    s = p.read_text()
    old = s
    s = re.sub(
        r"torch\.load\((path\.join\(models_folder,\s*snap_to_load\)),\s*map_location='cpu'\)",
        r"torch.load(\1, map_location='cpu', weights_only=False)",
        s,
    )
    if s != old:
        p.write_text(s)
        print("patched", p)
PY

echo "================================================"
echo "Create correct absolute weight symlinks"
echo "================================================"

python - <<'PY'
from pathlib import Path
import os

BASE = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place")
src_dir = BASE / "weights" / "extracted_weights" / "weights"
dst_dir = BASE / "weights"

if not src_dir.exists():
    raise FileNotFoundError(f"Missing correct weights folder: {src_dir}")

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

missing = []

for name in required:
    src = src_dir / name
    dst = dst_dir / name

    if not src.exists():
        missing.append(str(src))
        continue

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    os.symlink(src.resolve(), dst)

bad = []
for name in required:
    p = dst_dir / name
    if not p.exists():
        bad.append(str(p))

if missing or bad:
    print("Missing source weights:")
    for x in missing:
        print(x)
    print("Bad destination links:")
    for x in bad:
        print(x)
    raise SystemExit(1)

print("All required weight symlinks are valid.")
for name in required[:8]:
    p = dst_dir / name
    print(name, "->", os.readlink(p))
PY

echo "================================================"
echo "Link DATA/test into 1st-place repo"
echo "================================================"

rm -rf test
ln -sfn "$DATA/test" test

echo "Check test images:"
ls -lah test/images | sed -n '1,12p'

echo "================================================"
echo "Clean old 1st-place prediction folders"
echo "================================================"

rm -rf submission
rm -rf pred34_loc pred50_loc_tuned pred92_loc_tuned pred154_loc
rm -rf res34cls2_0_tuned res34cls2_1_tuned res34cls2_2_tuned
rm -rf res50cls_cce_0_tuned res50cls_cce_1_tuned res50cls_cce_2_tuned
rm -rf dpn92cls_cce_0_tuned dpn92cls_cce_1_tuned dpn92cls_cce_2_tuned
rm -rf se154cls_0_tuned se154cls_1_tuned se154cls_2_tuned
rm -rf "$PRED_SAVE"
rm -rf "$FINAL_OUT"

echo "================================================"
echo "Run 1st-place xView2 winner ZERO-SHOT on Texas Tornadoes"
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
echo "Save Texas predictions"
echo "================================================"

mkdir -p "$PRED_SAVE"
cp -a submission/. "$PRED_SAVE"/

echo "Saved prediction count:"
find "$PRED_SAVE" -type f | wc -l

echo "================================================"
echo "Evaluate 1st-place predictions on Texas Tornadoes"
echo "================================================"

python -u eval_first_place_submission.py --truth-dir "$DATA" --pred-dir "$PRED_SAVE" --out-dir "$FINAL_OUT"

echo "================================================"
echo "Final metrics"
echo "================================================"

cat "$FINAL_OUT/metrics_summary.txt"

echo "DONE: 1st-place xView2 Texas Tornadoes zero-shot"
