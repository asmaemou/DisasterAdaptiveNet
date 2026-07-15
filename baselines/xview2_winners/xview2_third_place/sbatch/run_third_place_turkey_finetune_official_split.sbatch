#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xv2_3rd_turkey_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/sbatch/logs/xv2_3rd_turkey_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/sbatch/logs/xv2_3rd_turkey_ftv_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

REPO=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place
CUSTOM=$REPO/custom
MANIFEST=$CUSTOM/third_place_turkey_finetune_models.tsv
ORIGINAL_MODELS=$REPO/weights/extracted/ensemble

DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_earthquake_turkey_OFFICIAL_SPLIT
EXP_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_earthquake_turkey_FULL_SOLUTION_finetune_official_split
WORK=$EXP_OUT/work
FT_MODELS=$EXP_OUT/models
PRED_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_earthquake_turkey_FINE_TUNED_OFFICIAL_SPLIT_predictions
FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_earthquake_turkey_FINE_TUNED_OFFICIAL_SPLIT_full_solution

FT_EPOCHS=10
TRAIN_BATCH_SIZE=4
RESET_EXPERIMENT=${RESET_EXPERIMENT:-0}

mkdir -p "$REPO/sbatch/logs"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "================================================"
echo "Prepare leak-free Earthquake Turkey official split"
echo "================================================"

python -u "$CUSTOM/prepare_third_place_turkey_official_split.py"

echo "================================================"
echo "Validate the 13 released third-place ensemble checkpoints"
echo "================================================"

python - <<PY
import csv
from pathlib import Path

manifest = Path("$MANIFEST")
models = Path("$ORIGINAL_MODELS")
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))

if len(rows) != 13:
    raise RuntimeError(f"Expected 13 third-place ensemble checkpoints, found {len(rows)}")

missing = [row["checkpoint"] for row in rows if not (models / row["checkpoint"]).is_file()]
if missing:
    raise FileNotFoundError(
        "Missing released third-place checkpoints:\n"
        + "\n".join(f"  - {name}" for name in missing)
    )

print("Validated third-place checkpoints:", len(rows))
PY

echo "================================================"
echo "Prepare resumable fine-tuning output"
echo "================================================"

if [ "$RESET_EXPERIMENT" = "1" ]; then
  rm -rf "$EXP_OUT"
fi
rm -rf "$PRED_DIR" "$FINAL_OUT"
mkdir -p "$WORK" "$FT_MODELS"

cd "$WORK"

echo "================================================"
echo "Fine-tune all 13 third-place ensemble models"
echo "================================================"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r CHECKPOINT MODEL SEED CRITERION EXTRA_CRITERION; do
  SOURCE="$ORIGINAL_MODELS/$CHECKPOINT"
  TARGET="$FT_MODELS/$CHECKPOINT"
  TEMP_TARGET="$TARGET.partial"

  if [ -f "$TARGET" ]; then
    echo "Already completed; skipping: $TARGET"
    continue
  fi

  rm -f "$TEMP_TARGET"

  CMD=(
    python -u "$CUSTOM/finetune_third_place_turkey_native.py"
    --seed "$SEED"
    --data-dir "$DATA"
    --source-checkpoint "$SOURCE"
    --output-checkpoint "$TEMP_TARGET"
    --model "$MODEL"
    --batch-size "$TRAIN_BATCH_SIZE"
    --epochs "$FT_EPOCHS"
    --learning-rate 0.0001
    --criterion "$CRITERION"
    --extra-criterion "$EXTRA_CRITERION"
    --workers 8
    --augmentations medium
    --fold 0
    --weight-decay 0.0001
    --size 512
    --post-transform
    --amp
  )

  "${CMD[@]}"

  if [ ! -f "$TEMP_TARGET" ]; then
    echo "ERROR: fine-tuned checkpoint was not created: $TEMP_TARGET"
    exit 1
  fi

  mv "$TEMP_TARGET" "$TARGET"
  echo "Saved fine-tuned ensemble checkpoint: $TARGET"
done

FT_COUNT=$(find "$FT_MODELS" -maxdepth 1 -type f -name '*.pth' | wc -l)
echo "Fine-tuned checkpoint count: $FT_COUNT"
if [ "$FT_COUNT" -ne 13 ]; then
  echo "ERROR: expected 13 fine-tuned checkpoints, found $FT_COUNT"
  exit 1
fi

echo "================================================"
echo "Run the original weighted ensemble with fine-tuned checkpoints"
echo "================================================"

cd "$REPO"
ln -sfn "$FT_MODELS" models

python -u predict_37_weighted.py \
  --data-dir "$DATA" \
  --output-dir "$PRED_DIR" \
  --size 512 \
  --postprocessing naive \
  --activation model \
  --fp16 \
  --batch-size 1 \
  --workers 0

echo "================================================"
echo "Evaluate on the 95 held-out Earthquake Turkey test samples"
echo "================================================"

python -u eval_third_place_turkey_predictions.py \
  --truth-dir "$DATA" \
  --pred-dir "$PRED_DIR" \
  --out-dir "$FINAL_OUT" \
  --mode finetuned

cat "$FINAL_OUT/metrics_summary.txt"

echo "DONE: 3rd-place xView2 fine-tuned on Earthquake Turkey official train/val/test split"
