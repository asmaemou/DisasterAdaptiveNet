#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=hrtbda_v5_final_test
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

PHASE2_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_FINAL_TEST_th040_dilminor

THRESH_CKPT_DIR=$OUTPUT_DIR/threshold_checkpoints
PHASE1_CKPT=$THRESH_CKPT_DIR/phase1_best_threshold_0.40.pt

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores" "$THRESH_CKPT_DIR"

echo "Creating Phase-I threshold checkpoint with threshold=0.40"
python - <<PY
import torch
from pathlib import Path

src = Path("$PHASE1_ORIG")
dst = Path("$PHASE1_CKPT")
dst.parent.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(src, map_location="cpu")
ckpt["best_threshold"] = 0.40
torch.save(ckpt, dst)

print(f"Wrote {dst}")
print(f"best_threshold={ckpt['best_threshold']}")
PY

echo "===== FINAL XBD TEST JOB ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "XBD_ROOT: $XBD_ROOT"
echo "PHASE1_CKPT: $PHASE1_CKPT"
echo "PHASE2_CKPT: $PHASE2_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TEST_SPLIT: test"
echo "THRESHOLD: 0.40"
echo "DILATION: minor"
echo "DILATION_KERNEL: 3"
echo "=============================="

python -u transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py \
  --phase test \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --phase2-checkpoint "$PHASE2_CKPT" \
  --xbd-root "$XBD_ROOT" \
  --train-split train tier3 \
  --val-split hold \
  --test-split test \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 2 \
  --eval-batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --img-size 1024 \
  --phase2-crop-size 608 \
  --base-channels 48 \
  --decoder-channels 128 \
  --window-size 8 \
  --postprocess-dilation minor \
  --dilation-kernel 3 \
  --amp

echo "Done."