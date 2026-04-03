#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=xbd_rescuenet_to_ianbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
RESCUENET_ROOT=/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd
IANBD_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xbd_plus_rescuenet_to_ianbd

XBD_VAL_SPLIT=hold

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniconda3/bin/conda"
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
echo "XBD_ROOT:       $XBD_ROOT"
echo "RESCUENET_ROOT: $RESCUENET_ROOT"
echo "IANBD_ROOT:     $IANBD_ROOT"
echo "RUN_DIR:        $RUN_DIR"
nvidia-smi || true
echo "===================="

mkdir -p "$RUN_DIR"

python -u train_xbd_plus_rescuenet_test_ianbd.py \
  --xbd-root "$XBD_ROOT" \
  --rescuenet-root "$RESCUENET_ROOT" \
  --target-root "$IANBD_ROOT" \
  --xbd-train-split train \
  --xbd-val-split "$XBD_VAL_SPLIT" \
  --rescuenet-train-split train \
  --rescuenet-val-split val \
  --target-test-split test \
  --output-dir "$RUN_DIR" \
  --epochs 40 \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512 \
  --amp