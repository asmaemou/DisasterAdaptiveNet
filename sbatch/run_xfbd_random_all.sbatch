#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --job-name=disasteradaptivenet_xfbd_random_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs
mkdir -p output

XFBD_ROOT=/homes/j244s673/documents/wsu/phd/xfbd_random_disasteradaptivenet
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/disasteradaptivenet_xfbd_random_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/scores"

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

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_DISABLE_SERVICE=true

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "XFBD_ROOT: $XFBD_ROOT"
echo "RUN_DIR: $RUN_DIR"
nvidia-smi || true
echo "===================="

echo "===== STEP 0: PREPARE XFBD RANDOM ====="
python -u prepare_xfbd_random.py

echo "===== STEP 1: TRAINING ====="
python -u train_network_xfbd.py \
  -c disasteradaptivenet_xfbd_random \
  -o "$RUN_DIR" \
  -d "$XFBD_ROOT"

echo "===== STEP 2: INFERENCE ====="
python -u create_submission_xfbd.py \
  -c disasteradaptivenet_xfbd_random \
  -o "$RUN_DIR" \
  -d "$XFBD_ROOT"

echo "===== STEP 3: METRICS ====="
python -u compute_metrics_xfbd.py \
  -c disasteradaptivenet_xfbd_random \
  -o "$RUN_DIR" \
  -d "$XFBD_ROOT"

echo "===== ALL DONE ====="
date