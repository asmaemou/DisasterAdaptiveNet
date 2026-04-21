#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=stca_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Edit these if needed
PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
DATA_ROOT="/homes/j244s673/documents/wsu/phd/xview2"
RUN_NAME="tier3_to_train_stca"
CONDA_ENV="disasteradaptivenet"

cd "${PROJECT_DIR}"
mkdir -p logs runs

source ~/.bashrc
conda activate "${CONDA_ENV}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "DATA_ROOT: ${DATA_ROOT}"

python scripts/train_stca_xview2.py \
  --root "${DATA_ROOT}" \
  --work-dir "${PROJECT_DIR}/runs/${RUN_NAME}" \
  --cache-dir "${PROJECT_DIR}/runs/${RUN_NAME}/mask_cache" \
  --source-split tier3 \
  --target-split train \
  --val-split test \
  --eval-split hold \
  --crop-size 512 \
  --batch-size 4 \
  --workers 8 \
  --source-epochs 12 \
  --adapt-epochs 6 \
  --source-lr 1e-4 \
  --adapt-lr 5e-5 \
  --stca-weight 0.2 \
  --target-threshold 0.5 \
  --target-feature-samples 64 \
  --source-feature-samples 64
