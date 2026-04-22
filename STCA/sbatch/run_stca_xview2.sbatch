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

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
DATA_ROOT="/homes/j244s673/documents/wsu/phd/xview2"
CONDA_ENV="disasteradaptivenet_cu128"
RUN_NAME="tier3_to_train_stca"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p runs
mkdir -p "runs/${RUN_NAME}"

source /homes/j244s673/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "=================================================="
echo "JOB_ID:      ${SLURM_JOB_ID}"
echo "HOSTNAME:    $(hostname)"
echo "START TIME:  $(date)"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "DATA_ROOT:   ${DATA_ROOT}"
echo "CONDA_ENV:   ${CONDA_ENV}"
echo "RUN_NAME:    ${RUN_NAME}"
echo "=================================================="

which python
python --version
nvidia-smi || true

python scripts/train_stca_xview2.py \
  --root "${DATA_ROOT}" \
  --source-split tier3 \
  --target-train-split train \
  --target-val-split test \
  --target-test-split hold \
  --work-dir "runs/${RUN_NAME}"

EXIT_CODE=$?

echo "=================================================="
echo "END TIME:    $(date)"
echo "EXIT CODE:   ${EXIT_CODE}"
echo "=================================================="

exit ${EXIT_CODE}