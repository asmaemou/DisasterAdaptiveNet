#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=disasteradaptivenet_idabd_metrics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

IDABD_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/disasteradaptivenet_idabd_JOBID_REPLACE
mkdir -p "$RUN_DIR/scores"

eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate disasteradaptivenet_cu128

python -u compute_metrics_idabd.py \
  -c disasteradaptivenet_idabd \
  -o "$RUN_DIR" \
  -d "$IDABD_ROOT"