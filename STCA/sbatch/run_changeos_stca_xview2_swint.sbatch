#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=changeos_stca_swint
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
DATA_ROOT="/homes/j244s673/documents/wsu/phd/xview2"
CONDA_ENV="disasteradaptivenet_cu128"
RUN_NAME="tier3_to_train_changeos_swint_stca"

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

python scripts/prepare_xview2_for_changeos.py \
  --root "${DATA_ROOT}" \
  --out-dir "runs/${RUN_NAME}/prepared_xview2"

python scripts/train_changeos_stca.py \
  --root "runs/${RUN_NAME}/prepared_xview2" \
  --work-dir "runs/${RUN_NAME}" \
  --source-split tier3 \
  --target-split train \
  --val-split test \
  --eval-split hold \
  --backbone swint \
  --seed 42

EXIT_CODE=$?

echo "=================================================="
echo "END TIME:    $(date)"
echo "EXIT CODE:   ${EXIT_CODE}"
echo "=================================================="

exit ${EXIT_CODE}