#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=damformer_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
DATA_ROOT="/homes/j244s673/documents/wsu/phd/xview2"
WORK_DIR="${PROJECT_DIR}/runs/damformer_xview2_tier3_test_hold"
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

"${ENV_PY}" scripts/damformer_xview2.py   --train-root "${DATA_ROOT}"   --train-split tier3   --val-split test   --eval-split hold   --epochs 20   --batch-size 2   --lr 6e-5   --weight-decay 5e-3   --alpha 1.0   --crop-size 512   --stride 256   --loc-threshold 0.5   --work-dir "${WORK_DIR}"   --seed 42
