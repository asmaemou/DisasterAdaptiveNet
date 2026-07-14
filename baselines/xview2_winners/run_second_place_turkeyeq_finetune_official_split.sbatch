#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xv2_2nd_teq_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_teq_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_teq_ftv_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners
REPO=$BASE/xview2_second_place
MANIFEST=$BASE/second_place_full_solution_manifest.tsv

DATA_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_earthquake_turkey
TRAIN_FOLDS=$DATA_OUT/folds_train_val.csv
EXP_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_earthquake_turkey_FULL_SOLUTION_finetune_official_split

FT_EPOCHS=10

mkdir -p "$BASE/logs"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

echo "================================================"
echo "Prepare Earthquake Turkey official train/val/test data"
echo "================================================"

cd "$BASE"
python -u prepare_second_place_turkeyeq.py
python -u make_second_place_turkeyeq_test_view.py

echo "================================================"
echo "Reset experiment output and create 10-epoch fine-tuning configs"
echo "================================================"

rm -rf "$EXP_OUT"
mkdir -p "$EXP_OUT/init"
mkdir -p "$EXP_OUT/configs"
mkdir -p "$EXP_OUT/weights_localization"
mkdir -p "$EXP_OUT/weights_damage"
mkdir -p "$EXP_OUT/logs_localization"
mkdir -p "$EXP_OUT/logs_damage"
mkdir -p "$EXP_OUT/oof_localization"
mkdir -p "$EXP_OUT/oof_damage"

python - <<PY
import csv
import json
from pathlib import Path

manifest = Path("$MANIFEST")
config_out = Path("$EXP_OUT/configs")
epochs = int("$FT_EPOCHS")

rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
if len(rows) != 21:
    raise RuntimeError(f"Expected 21 ensemble rows, found {len(rows)}")

for row in rows:
    config = Path(row["config"])
    weight = Path(row["weight"])
    if not config.is_file():
        raise FileNotFoundError(config)
    if not weight.is_file():
        raise FileNotFoundError(weight)

    destination = config_out / config.name
    if not destination.exists():
        data = json.loads(config.read_text())
        data["optimizer"]["schedule"]["epochs"] = epochs
        destination.write_text(json.dumps(data, indent=2) + "\n")
        print("Fine-tuning config:", destination, "epochs:", epochs)

print("Validated manifest models:", len(rows))
print("Localization models:", sum(row["task"] == "loc" for row in rows))
print("Damage models:", sum(row["task"] == "damage" for row in rows))
PY

echo "================================================"
echo "Verify the official split is leak-free"
echo "================================================"

python - <<PY
from pathlib import Path
import pandas as pd

folds = Path("$TRAIN_FOLDS")
df = pd.read_csv(folds, dtype={"id": str})

train = set(df[df["fold"] != 0]["id"])
val = set(df[df["fold"] == 0]["id"])

if train & val:
    raise RuntimeError("Train/validation overlap detected")
if set(df[df["fold"] != 0]["split"]) != {"train"}:
    raise RuntimeError("Non-training samples would enter the training loader")
if set(df[df["fold"] == 0]["split"]) != {"val"}:
    raise RuntimeError("Validation fold contains non-validation samples")
if "test" in set(df["split"]):
    raise RuntimeError("Test samples are present in the training folds file")

print("Official training samples:", len(train))
print("Official validation samples:", len(val))
print("Test leakage: none")
PY

cd "$REPO"

echo "================================================"
echo "Fine-tune all second-place localization models"
echo "================================================"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r TASK FAMILY TAG CONFIG WEIGHT; do
  if [ "$TASK" != "loc" ]; then
    continue
  fi

  RUNTIME_CONFIG="$EXP_OUT/configs/$(basename "$CONFIG")"
  INIT="$EXP_OUT/init/${TAG}_epoch0_init"
  MODEL_OUT="$EXP_OUT/weights_localization/$TAG"

  python - <<PY
import torch
from pathlib import Path

src = Path("$WEIGHT")
dst = Path("$INIT")
checkpoint = torch.load(src, map_location="cpu", weights_only=False)
checkpoint["epoch"] = 0
checkpoint["dice_best"] = 0
checkpoint["xview_best"] = 0
for key in ["optimizer", "scheduler", "amp"]:
    checkpoint.pop(key, None)
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, dst)
print("Saved localization initialization:", dst)
PY

  mkdir -p "$MODEL_OUT"
  mkdir -p "$EXP_OUT/logs_localization/$TAG"
  mkdir -p "$EXP_OUT/oof_localization/$TAG"

  python -u train_localization.py \
    --config "$RUNTIME_CONFIG" \
    --workers 8 \
    --gpu 0 \
    --output-dir "$MODEL_OUT/" \
    --resume "$INIT" \
    --fold 0 \
    --prefix "turkey_official_${TAG}_" \
    --data-dir "$DATA_OUT" \
    --folds-csv "$TRAIN_FOLDS" \
    --logdir "$EXP_OUT/logs_localization/$TAG" \
    --predictions "$EXP_OUT/oof_localization/$TAG" \
    --opt-level O0 \
    --freeze-epochs 1 \
    --test_every 1
done

echo "================================================"
echo "Fine-tune all second-place damage models"
echo "================================================"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r TASK FAMILY TAG CONFIG WEIGHT; do
  if [ "$TASK" != "damage" ]; then
    continue
  fi

  RUNTIME_CONFIG="$EXP_OUT/configs/$(basename "$CONFIG")"
  INIT="$EXP_OUT/init/${TAG}_epoch0_init"
  MODEL_OUT="$EXP_OUT/weights_damage/$TAG"

  python - <<PY
import torch
from pathlib import Path

src = Path("$WEIGHT")
dst = Path("$INIT")
checkpoint = torch.load(src, map_location="cpu", weights_only=False)
checkpoint["epoch"] = 0
checkpoint["dice_best"] = 0
checkpoint["xview_best"] = 0
for key in ["optimizer", "scheduler", "amp"]:
    checkpoint.pop(key, None)
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, dst)
print("Saved damage initialization:", dst)
PY

  mkdir -p "$MODEL_OUT"
  mkdir -p "$EXP_OUT/logs_damage/$TAG"
  mkdir -p "$EXP_OUT/oof_damage/$TAG"

  python -u train.py \
    --config "$RUNTIME_CONFIG" \
    --workers 8 \
    --gpu 0 \
    --output-dir "$MODEL_OUT/" \
    --resume "$INIT" \
    --fold 0 \
    --prefix "turkey_official_${TAG}_" \
    --data-dir "$DATA_OUT" \
    --folds-csv "$TRAIN_FOLDS" \
    --logdir "$EXP_OUT/logs_damage/$TAG" \
    --predictions "$EXP_OUT/oof_damage/$TAG" \
    --opt-level O0 \
    --freeze-epochs 1 \
    --test_every 1
done

echo "================================================"
echo "Evaluate the fine-tuned full ensemble on held-out Turkey test"
echo "================================================"

cd "$BASE"
python -u eval_second_place_turkeyeq_test_full.py --mode finetuned

METRICS=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_earthquake_turkey_FINE_TUNED_OFFICIAL_SPLIT_full_solution/metrics_summary.txt
cat "$METRICS"

echo "DONE: 2nd-place xView2 fine-tuned on Earthquake Turkey official train/val/test split"
