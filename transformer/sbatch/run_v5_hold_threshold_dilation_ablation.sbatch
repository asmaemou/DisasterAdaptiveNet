#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=hrtbda_v5_hold_ablate
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

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

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2

PHASE1_ORIG=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp/checkpoints/phase1_best.pt

PHASE2_BEST=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

ABLATION_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_hold_ablation
THRESH_CKPT_DIR=$ABLATION_ROOT/threshold_checkpoints

mkdir -p "$ABLATION_ROOT" "$THRESH_CKPT_DIR"

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "PHASE1_ORIG: $PHASE1_ORIG"
echo "PHASE2_BEST: $PHASE2_BEST"
echo "ABLATION_ROOT: $ABLATION_ROOT"
echo "===================="

python - <<PY
import torch
from pathlib import Path

src = Path("$PHASE1_ORIG")
out_dir = Path("$THRESH_CKPT_DIR")
out_dir.mkdir(parents=True, exist_ok=True)

for th in [0.40, 0.45, 0.50, 0.55]:
    ckpt = torch.load(src, map_location="cpu")
    ckpt["best_threshold"] = float(th)
    dst = out_dir / f"phase1_best_threshold_{th:.2f}.pt"
    torch.save(ckpt, dst)
    print(f"Wrote {dst}")
PY

for TH in 0.40 0.45 0.50 0.55; do
  for DIL in none minor; do

    PHASE1_CKPT=$THRESH_CKPT_DIR/phase1_best_threshold_${TH}.pt
    OUTDIR=$ABLATION_ROOT/th${TH}_dil${DIL}

    mkdir -p "$OUTDIR" "$OUTDIR/checkpoints" "$OUTDIR/scores"

    echo
    echo "=================================================="
    echo "Running HOLD ablation: threshold=$TH dilation=$DIL"
    echo "PHASE1_CKPT=$PHASE1_CKPT"
    echo "OUTDIR=$OUTDIR"
    echo "=================================================="

    python -u transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py \
      --phase test \
      --phase1-checkpoint "$PHASE1_CKPT" \
      --phase2-checkpoint "$PHASE2_BEST" \
      --xbd-root "$XBD_ROOT" \
      --train-split train tier3 \
      --val-split hold \
      --test-split test \
      --output-dir "$OUTDIR" \
      --batch-size 2 \
      --eval-batch-size 1 \
      --grad-accum-steps 4 \
      --num-workers 8 \
      --img-size 1024 \
      --phase2-crop-size 608 \
      --base-channels 48 \
      --decoder-channels 128 \
      --window-size 8 \
      --postprocess-dilation "$DIL" \
      --dilation-kernel 3 \
      --amp

  done
done

echo
echo "================ HOLD ABLATION SUMMARIES ================"
for f in "$ABLATION_ROOT"/th*/scores/summary_hrtbda_v5_multilabel_crop_cascade.txt; do
  echo
  echo "----- $f -----"
  grep -E "Phase I threshold|Damage post-processing|Localization F1|No Damage|Minor Damage|Major Damage|Destroyed|Damage F1|Overall Score" "$f" || true
done
echo "=========================================================="