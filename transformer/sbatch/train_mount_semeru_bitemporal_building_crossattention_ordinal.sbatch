#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=sem_bg_xattn_ord
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_CACHE_DISABLE=1

LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_JOB_TMP/torchinductor
export HF_HOME=$LOCAL_JOB_TMP/huggingface
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/mount_semeru_eruption_preprocessed
OUTPUT_DIR=$PROJECT_ROOT/output/HBG_CAON_MOUNT_SEMERU_ERUPTION

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

echo "===== HBG-CAON ON MOUNT SEMERU ERUPTION ====="
echo "Architecture: identical to xBD job xbd_bg_xattn_ord"
echo "Backbones: ImageNet-pretrained Siamese ResNet34 U-Net + Swin-Tiny"
echo "Fusion: bidirectional windowed cross-attention"
echo "Heads: soft building guidance + 4-class damage + ordinal supervision"
echo "Protocol: Mount Semeru train -> val selection -> untouched test"
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "CUDA disk cache disabled: $CUDA_CACHE_DISABLE"
date
hostname
nvidia-smi

python -u transformer/scripts/train_mount_semeru_bitemporal_building_crossattention_ordinal.py \
  --xbd-root "$DATA_ROOT" \
  --train-split train \
  --val-split val \
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

echo "DONE: HBG-CAON trained and tested on Mount Semeru eruption"
date
