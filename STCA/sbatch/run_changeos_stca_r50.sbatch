#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=changeos_stca_r50
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
PREPARED_ROOT="/homes/j244s673/documents/wsu/phd/xview2"
WORK_DIR="${PROJECT_DIR}/runs/tier3_to_train_changeos_r50_stca"
ENV_PY="/homes/j244s673/miniforge3/envs/stca_changeos_py311/bin/python"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"
cd "${PROJECT_DIR}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled

echo "===== ENV CHECK ====="
"${ENV_PY}" -V
nvidia-smi || true

"${ENV_PY}" - <<'PY'
import sys, torch, torchange, ever, numpy
print("python exe:", sys.executable)
print("torch file:", torch.__file__)
print("torchange file:", torchange.__file__)
print("numpy file:", numpy.__file__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job.")
PY

"${ENV_PY}" scripts/train_changeos_stca.py \
  --prepared-root "${PREPARED_ROOT}" \
  --source-split tier3 \
  --target-split train \
  --val-split test \
  --eval-split hold \
  --backbone r50 \
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