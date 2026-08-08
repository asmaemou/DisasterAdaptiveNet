#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=xbd_r34_swin_film
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Driver 595/CUDA 13.2 on this cluster has been observed opening
# $HOME/.nv/ComputeCache/index even when CUDA_CACHE_PATH points at node-local
# storage. On the shared home filesystem that leaves Python blocked in
# rpc_wait_bit_killable before its first batch. Disable the disk cache for
# this job; kernels remain cached in-process for the lifetime of training.
export CUDA_CACHE_DISABLE=1

LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_JOB_TMP/torchinductor
export HF_HOME=$LOCAL_JOB_TMP/huggingface
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
OUTPUT_DIR=$PROJECT_ROOT/output/ResNet34SwinTinyGatedFiLM_xBD_TrainTier3

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== RESNET34 + SWIN-TINY GATED FUSION + FILM ====="
echo "Protocol: xBD train+tier3 -> hold selection -> untouched test"
echo "Output: $OUTPUT_DIR"
echo "CUDA disk cache disabled: $CUDA_CACHE_DISABLE"
echo "Job-local runtime cache root: $LOCAL_JOB_TMP"
date
hostname
nvidia-smi

# Synchronous loading is intentional on this shared HPC filesystem; prior
# xBD jobs have blocked inside remote filesystem RPC calls with workers > 0.
python -u transformer/scripts/train_xbd_resnet34_swin_film_gated.py \
  --xbd-root "$XBD_ROOT" \
  --train-split train+tier3 \
  --val-split hold \
  --test-split test \
  --output-dir "$OUTPUT_DIR" \
  --epochs 60 \
  --batch-size 2 \
  --num-workers 0 \
  --img-size 896 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --seed 42 \
  --save-every 5 \
  --early-stopping-patience 12 \
  --loc-bce-weight 1.0 \
  --loc-dice-weight 1.0 \
  --dmg-ce-weight 1.0 \
  --thresholds 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 \
  --amp

echo "DONE: ResNet34 + Swin-Tiny gated fusion + FiLM on xBD"
date
