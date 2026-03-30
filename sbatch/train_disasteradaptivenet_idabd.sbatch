#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=disasteradaptivenet_idabd_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs
mkdir -p output

IDABD_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/disasteradaptivenet_idabd_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate disasteradaptivenet_cu128

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_DISABLE_SERVICE=true

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "IDABD_ROOT: $IDABD_ROOT"
echo "RUN_DIR: $RUN_DIR"
nvidia-smi || true
echo "===================="

python -u train_network_idabd.py \
  -c disasteradaptivenet_idabd \
  -o "$RUN_DIR" \
  -d "$IDABD_ROOT"