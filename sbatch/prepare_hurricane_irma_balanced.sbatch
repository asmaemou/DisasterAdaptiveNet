#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=prep_irma_bal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

SCRIPT_PATH=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/prepare_hurricane_irma_balanced.py


if [ ! -f "$SCRIPT_PATH" ]; then
  echo "ERROR: script not found at $SCRIPT_PATH"
  exit 1
fi

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
echo "SCRIPT_PATH: $SCRIPT_PATH"
echo "RAW_ROOT:    /homes/j244s673/documents/wsu/phd/HURRICANE-IRMA"
echo "IMG_DIR:     /homes/j244s673/documents/wsu/phd/HURRICANE-IRMA/images"
echo "MSK_DIR:     /homes/j244s673/documents/wsu/phd/HURRICANE-IRMA/masks"
echo "OUT_ROOT:    /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet"
echo "===================="

python -u "$SCRIPT_PATH"

echo ""
echo "===== OUTPUT CHECK ====="
echo "Output root:"
echo "  /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet"
echo ""
echo "Metadata:"
echo "  /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet/metadata.json"
echo ""
echo "Train images count:"
find /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet/train/images -type f | wc -l
echo "Val images count:"
find /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet/val/images -type f | wc -l
echo "Test images count:"
find /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet/test/images -type f | wc -l
echo ""
echo "Train masks count:"
find /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet/train/masks -type f | wc -l
echo "Val masks count:"
find /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet/val/masks -type f | wc -l
echo "Test masks count:"
find /homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet/test/masks -type f | wc -l