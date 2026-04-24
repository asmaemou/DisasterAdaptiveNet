#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=idabd_loc_ft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/new_approach"
OUT_DIR="${PROJECT_DIR}/idabd_stage1_loc_ft_checkpoints"

# Use direct python from the conda env, same style as your working sbatch.
ENV_PY="/homes/j244s673/miniforge3/envs/disasteradaptivenet_cu128/bin/python"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled

echo "START: $(date)"
echo "HOST:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID}"
echo "PWD:   $(pwd)"
echo "OUT_DIR: ${OUT_DIR}"

echo "===== ENV CHECK ====="
"${ENV_PY}" -V
nvidia-smi || true

"${ENV_PY}" - <<'PYTORCH'
import sys
import torch
import numpy
import cv2

print("python exe:", sys.executable)
print("torch file:", torch.__file__)
print("numpy file:", numpy.__file__)
print("cv2 file:", cv2.__file__)
print("cuda available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available inside the Slurm job.")
PYTORCH

"${ENV_PY}" -u train_idabd_stage1_loc_da_calib_fusion_DANet.py \
  --train_all \
  --input_split pre \
  --gt_split post \
  --epochs 5 \
  --batch 2 \
  --lr 2e-4 \
  --wd 1e-4 \
  --workers 8 \
  --crop 512 \
  --fusion_p 0.5 \
  --use_pos_weight \
  --amp \
  --thresh_sweep \
  --out_dir "${OUT_DIR}"

echo "END: $(date)"