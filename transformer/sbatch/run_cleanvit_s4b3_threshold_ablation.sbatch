#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=cleanvit_th_ablate
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet

mkdir -p transformer/sbatch/logs

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

# =========================
# PATHS
# =========================
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2

EXP=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_cascaded_phase1mask_cleanvit_s4b3

PHASE1_ORIG=$EXP/checkpoints/phase1_best.pt
PHASE2_BEST=$EXP/checkpoints/phase2_best.pt

ABLATION_ROOT=$EXP/test_threshold_ablation
THRESH_CKPT_DIR=$ABLATION_ROOT/threshold_checkpoints

SCRIPT=transformer/scripts/train_xbd_hrtbda_v2_cascaded_phase1mask_vit_clean.py

mkdir -p "$ABLATION_ROOT" "$THRESH_CKPT_DIR"

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "XBD_ROOT: $XBD_ROOT"
echo "EXP: $EXP"
echo "PHASE1_ORIG: $PHASE1_ORIG"
echo "PHASE2_BEST: $PHASE2_BEST"
echo "ABLATION_ROOT: $ABLATION_ROOT"
echo "SCRIPT: $SCRIPT"
echo "===================="

if [ ! -f "$PHASE1_ORIG" ]; then
  echo "ERROR: Phase I checkpoint not found:"
  echo "$PHASE1_ORIG"
  exit 1
fi

if [ ! -f "$PHASE2_BEST" ]; then
  echo "ERROR: Phase II checkpoint not found:"
  echo "$PHASE2_BEST"
  exit 1
fi

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: Python script not found:"
  echo "$SCRIPT"
  exit 1
fi

# =========================
# CREATE PHASE I CHECKPOINT COPIES WITH DIFFERENT THRESHOLDS
# =========================
python - <<PY
import torch
from pathlib import Path

src = Path("$PHASE1_ORIG")
out_dir = Path("$THRESH_CKPT_DIR")
out_dir.mkdir(parents=True, exist_ok=True)

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

for th in thresholds:
    ckpt = torch.load(src, map_location="cpu")
    ckpt["best_threshold"] = float(th)
    dst = out_dir / f"phase1_best_threshold_{th:.2f}.pt"
    torch.save(ckpt, dst)
    print(f"Wrote {dst}")
PY

# =========================
# TEST EACH THRESHOLD
# No retraining happens here.
# It only changes the Phase I mask threshold during cascaded testing.
# =========================
for TH in 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70; do

  PHASE1_CKPT=$THRESH_CKPT_DIR/phase1_best_threshold_${TH}.pt
  OUTDIR=$ABLATION_ROOT/th${TH}

  mkdir -p "$OUTDIR" "$OUTDIR/checkpoints" "$OUTDIR/scores"

  echo
  echo "=================================================="
  echo "Running CleanViT-S4B3 threshold ablation"
  echo "Threshold: $TH"
  echo "PHASE1_CKPT: $PHASE1_CKPT"
  echo "PHASE2_BEST: $PHASE2_BEST"
  echo "OUTDIR: $OUTDIR"
  echo "=================================================="

  python -u "$SCRIPT" \
    --phase test \
    --phase1-checkpoint "$PHASE1_CKPT" \
    --phase2-checkpoint "$PHASE2_BEST" \
    --xbd-root "$XBD_ROOT" \
    --train-split train tier3 \
    --val-split hold \
    --test-split test \
    --output-dir "$OUTDIR" \
    --batch-size 1 \
    --num-workers 8 \
    --img-size 1024 \
    --base-channels 48 \
    --decoder-channels 128 \
    --window-size 8 \
    --phase1-vit-stages stage4_b3 \
    --phase2-vit-stages stage4_b3 \
    --amp

done

echo
echo "================ THRESHOLD ABLATION SUMMARIES ================"

for f in "$ABLATION_ROOT"/th*/scores/summary_hrtbda_v2_cascaded_phase1mask.txt; do
  echo
  echo "----- $f -----"
  grep -E "Phase I threshold|Localization F1|No Damage|Minor Damage|Major Damage|Destroyed|Damage F1|Overall Score" "$f" || true
done

echo "==============================================================="
echo "Done."