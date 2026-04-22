#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=changeos_stca_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
PREPARED_ROOT="${PROJECT_DIR}/data/xview2_torchange"
WORK_DIR="${PROJECT_DIR}/runs/tier3_to_train_changeos_r18_stca"
CONDA_ENV="stca_changeos_py311"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"
cd "${PROJECT_DIR}"

source /homes/j244s673/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
export WANDB_MODE=disabled

python scripts/train_changeos_stca.py \
  --prepared-root "${PREPARED_ROOT}" \
  --source-split tier3 \
  --target-split train \
  --val-split test \
  --eval-split hold \
  --backbone r18 \
  --epochs-source 10 \
  --epochs-stca 5 \
  --batch-size 4 \
  --lr 3e-4 \
  --stca-loss lovasz \
  --stca-weight 0.2 \
  --target-threshold 0.5 \
  --source-feature-samples 256 \
  --target-feature-samples 256 \
  --work-dir "${WORK_DIR}" \
  --seed 42