#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=disasteradaptivenet_metrics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/disasteradaptivenet_event_1067383

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

mkdir -p "$RUN_DIR/scores"

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "XBD_ROOT: $XBD_ROOT"
echo "RUN_DIR: $RUN_DIR"
echo "===================="

python -u compute_metrics.py \
  -c disasteradaptivenet_event \
  -o "$RUN_DIR" \
  -d "$XBD_ROOT"