#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=idabd_v2_th_ablate
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

# =========================
# Paths
# =========================
IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd

BASE_RUN=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10
PHASE1_ORIG=$BASE_RUN/checkpoints/phase1_best.pt
PHASE2_BEST=$BASE_RUN/checkpoints/phase2_best.pt
SPLIT_ORIG=$BASE_RUN/idabd_splits_seed42_80_10_10.json

ABLATION_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_threshold_ablation
THRESH_CKPT_DIR=$ABLATION_ROOT/threshold_checkpoints
VAL_SPLIT_FILE=$ABLATION_ROOT/idabd_splits_seed42_80_10_10_val_as_test.json
BEST_JSON=$ABLATION_ROOT/best_validation_threshold.json

# Set to 1 to automatically run final test once using best validation threshold.
# Set to 0 if you only want validation ablation.
RUN_FINAL_TEST=1

# Thresholds to test on validation.
THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"

# =========================
# Same architecture/training params as IDA-BD v2 cascade
# =========================
BATCH_SIZE=1
NUM_WORKERS=8
IMG_SIZE=1024
SEED=42
BASE_CHANNELS=48
DECODER_CHANNELS=128
WINDOW_SIZE=8
MAX_GRAD_NORM=1.0
FOCAL_GAMMA=2.0

mkdir -p "$ABLATION_ROOT" "$THRESH_CKPT_DIR"

# =========================
# Conda
# =========================
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

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "IDABD_ROOT: $IDABD_ROOT"
echo "BASE_RUN: $BASE_RUN"
echo "PHASE1_ORIG: $PHASE1_ORIG"
echo "PHASE2_BEST: $PHASE2_BEST"
echo "SPLIT_ORIG: $SPLIT_ORIG"
echo "ABLATION_ROOT: $ABLATION_ROOT"
echo "THRESHOLDS: $THRESHOLDS"
echo "RUN_FINAL_TEST: $RUN_FINAL_TEST"
echo "===================="

if [ ! -f "$PHASE1_ORIG" ]; then
  echo "ERROR: Phase I checkpoint not found: $PHASE1_ORIG"
  exit 1
fi
if [ ! -f "$PHASE2_BEST" ]; then
  echo "ERROR: Phase II checkpoint not found: $PHASE2_BEST"
  exit 1
fi
if [ ! -f "$SPLIT_ORIG" ]; then
  echo "ERROR: Split file not found: $SPLIT_ORIG"
  exit 1
fi

# =========================
# Prepare threshold checkpoints and validation-as-test split file
# =========================
python - <<PY
import json
import torch
from pathlib import Path

phase1_src = Path("$PHASE1_ORIG")
out_dir = Path("$THRESH_CKPT_DIR")
out_dir.mkdir(parents=True, exist_ok=True)

thresholds = [float(x) for x in "$THRESHOLDS".split()]
for th in thresholds:
    ckpt = torch.load(phase1_src, map_location="cpu")
    ckpt["best_threshold"] = float(th)
    dst = out_dir / f"phase1_best_threshold_{th:.2f}.pt"
    torch.save(ckpt, dst)
    print(f"Wrote threshold checkpoint: {dst}")

split_orig = Path("$SPLIT_ORIG")
with open(split_orig, "r", encoding="utf-8") as f:
    splits = json.load(f)

# The IDA-BD adapter always evaluates the split named 'test'.
# For validation ablation, we create a temporary split file where test = val.
val_as_test = {
    "train": splits["train"],
    "val": splits["val"],
    "test": splits["val"],
}
val_file = Path("$VAL_SPLIT_FILE")
val_file.parent.mkdir(parents=True, exist_ok=True)
with open(val_file, "w", encoding="utf-8") as f:
    json.dump(val_as_test, f, indent=2)
print(f"Wrote validation-as-test split file: {val_file}")
print(f"Original split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
print(f"Val-as-test sizes: train={len(val_as_test['train'])}, val={len(val_as_test['val'])}, test={len(val_as_test['test'])}")
PY

# =========================
# Validation threshold ablation
# =========================
for TH in $THRESHOLDS; do
  PHASE1_CKPT=$THRESH_CKPT_DIR/phase1_best_threshold_${TH}.pt
  OUTDIR=$ABLATION_ROOT/val_th${TH}

  mkdir -p "$OUTDIR" "$OUTDIR/checkpoints" "$OUTDIR/scores"

  echo
  echo "=================================================="
  echo "Running IDA-BD VALIDATION threshold ablation: threshold=$TH"
  echo "PHASE1_CKPT=$PHASE1_CKPT"
  echo "PHASE2_BEST=$PHASE2_BEST"
  echo "OUTDIR=$OUTDIR"
  echo "=================================================="

  python -u transformer/scripts/train_idabd_hrtbda_v2_cascaded_phase1mask_full.py \
    --phase test \
    --idabd-root "$IDABD_ROOT" \
    --split-file "$VAL_SPLIT_FILE" \
    --phase1-checkpoint "$PHASE1_CKPT" \
    --phase2-checkpoint "$PHASE2_BEST" \
    --output-dir "$OUTDIR" \
    --train-split train80 \
    --val-split val10 \
    --test-split val10 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --img-size "$IMG_SIZE" \
    --seed "$SEED" \
    --base-channels "$BASE_CHANNELS" \
    --decoder-channels "$DECODER_CHANNELS" \
    --window-size "$WINDOW_SIZE" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --focal-gamma "$FOCAL_GAMMA" \
    --amp

done

# =========================
# Summarize validation results and select best threshold by Overall Score
# =========================
python - <<PY
import json
import re
from pathlib import Path

root = Path("$ABLATION_ROOT")
rows = []
for summary in sorted(root.glob("val_th*/scores/summary_hrtbda_v2_cascaded_phase1mask.txt")):
    text = summary.read_text(encoding="utf-8")
    th_match = re.search(r"Phase I threshold used for mask:\s*([0-9.]+)", text)
    loc_match = re.search(r"Test Localization F1 from Phase I mask:\s*([0-9.]+)", text)
    no_match = re.search(r"No Damage F1:\s*([0-9.]+)", text)
    minor_match = re.search(r"Minor Damage F1:\s*([0-9.]+)", text)
    major_match = re.search(r"Major Damage F1:\s*([0-9.]+)", text)
    dest_match = re.search(r"Destroyed F1:\s*([0-9.]+)", text)
    dmg_match = re.search(r"Damage F1:\s*([0-9.]+)", text)
    overall_match = re.search(r"Overall Score:\s*([0-9.]+)", text)
    if not (th_match and overall_match):
        continue
    rows.append({
        "threshold": float(th_match.group(1)),
        "loc_f1": float(loc_match.group(1)) if loc_match else None,
        "no_f1": float(no_match.group(1)) if no_match else None,
        "minor_f1": float(minor_match.group(1)) if minor_match else None,
        "major_f1": float(major_match.group(1)) if major_match else None,
        "destroyed_f1": float(dest_match.group(1)) if dest_match else None,
        "damage_f1": float(dmg_match.group(1)) if dmg_match else None,
        "overall": float(overall_match.group(1)),
        "summary": str(summary),
    })

rows.sort(key=lambda r: r["overall"], reverse=True)
print("\n================ IDA-BD V2 VALIDATION THRESHOLD ABLATION ================")
print("Rank | Threshold | Loc F1 | No F1 | Minor F1 | Major F1 | Destroyed F1 | Damage F1 | Overall")
for i, r in enumerate(rows, start=1):
    print(
        f"{i:>4} | {r['threshold']:.2f} | {r['loc_f1']:.6f} | {r['no_f1']:.6f} | "
        f"{r['minor_f1']:.6f} | {r['major_f1']:.6f} | {r['destroyed_f1']:.6f} | "
        f"{r['damage_f1']:.6f} | {r['overall']:.6f}"
    )
print("========================================================================")

if not rows:
    raise SystemExit("No ablation summaries found.")

best = rows[0]
with open("$BEST_JSON", "w", encoding="utf-8") as f:
    json.dump(best, f, indent=2)
print(f"\nBest validation threshold: {best['threshold']:.2f}")
print(f"Best validation overall:   {best['overall']:.6f}")
print(f"Wrote: $BEST_JSON")
PY

# =========================
# Optional final test using best validation threshold
# =========================
if [ "$RUN_FINAL_TEST" = "1" ]; then
  BEST_TH=$(python - <<PY
import json
with open("$BEST_JSON", "r", encoding="utf-8") as f:
    print(f"{json.load(f)['threshold']:.2f}")
PY
)

  PHASE1_BEST_TH=$THRESH_CKPT_DIR/phase1_best_threshold_${BEST_TH}.pt
  FINAL_OUTDIR=$ABLATION_ROOT/final_test_selected_th${BEST_TH}
  mkdir -p "$FINAL_OUTDIR" "$FINAL_OUTDIR/checkpoints" "$FINAL_OUTDIR/scores"

  echo
  echo "=================================================="
  echo "Running FINAL TEST using validation-selected threshold=$BEST_TH"
  echo "PHASE1_BEST_TH=$PHASE1_BEST_TH"
  echo "PHASE2_BEST=$PHASE2_BEST"
  echo "FINAL_OUTDIR=$FINAL_OUTDIR"
  echo "=================================================="

  python -u transformer/scripts/train_idabd_hrtbda_v2_cascaded_phase1mask_full.py \
    --phase test \
    --idabd-root "$IDABD_ROOT" \
    --split-file "$SPLIT_ORIG" \
    --phase1-checkpoint "$PHASE1_BEST_TH" \
    --phase2-checkpoint "$PHASE2_BEST" \
    --output-dir "$FINAL_OUTDIR" \
    --train-split train80 \
    --val-split val10 \
    --test-split test10 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --img-size "$IMG_SIZE" \
    --seed "$SEED" \
    --base-channels "$BASE_CHANNELS" \
    --decoder-channels "$DECODER_CHANNELS" \
    --window-size "$WINDOW_SIZE" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --focal-gamma "$FOCAL_GAMMA" \
    --amp

  echo
  echo "================ FINAL TEST SUMMARY ================"
  cat "$FINAL_OUTDIR/scores/summary_hrtbda_v2_cascaded_phase1mask.txt"
  echo "===================================================="
fi

echo "Done."
