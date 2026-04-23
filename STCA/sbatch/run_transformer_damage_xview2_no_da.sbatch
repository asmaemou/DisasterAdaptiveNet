#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=trans_no_da_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
DATA_ROOT="/homes/j244s673/documents/wsu/phd/xview2"
WORK_DIR="${PROJECT_DIR}/runs/transformer_damage_xview2_no_da"
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
import sys, torch, numpy
print("python exe:", sys.executable)
print("torch file:", torch.__file__)
print("numpy file:", numpy.__file__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job.")
PY

"${ENV_PY}" scripts/transformer_damage_xview2_no_da.py   --train-root "${DATA_ROOT}"   --train-split tier3   --val-split test   --eval-split hold   --epochs 20   --batch-size 2   --lr 1e-4   --weight-decay 1e-4   --crop-size 512   --stride 256   --loc-threshold 0.5   --work-dir "${WORK_DIR}"   --seed 42
