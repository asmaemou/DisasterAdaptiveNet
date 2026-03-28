#!/bin/bash
#SBATCH -p killable.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=disasteradaptivenet_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -w gpu202401


set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/uda_two_stage/xbd

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
conda activate disasteradaptivenet

# Recommended on HPC to avoid wandb connection timeout
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "XBD_ROOT: $XBD_ROOT"
nvidia-smi || true
echo "===================="

python -u train_network.py \
  -c disasteradaptivenet_event \
  -o output \
  -d "$XBD_ROOT"