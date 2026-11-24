#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=xbd_rsbd_simple_fusion
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_CACHE_DISABLE=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/xview2
OUTPUT_DIR=$PROJECT_ROOT/output/RSBDNet_ablation_xBD/simple_fusion
DATASET_LOADER=xbd_multisplit
TRAIN_SPLIT=train+tier3
VAL_SPLIT=hold

LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_JOB_TMP/torchinductor
export HF_HOME=$LOCAL_JOB_TMP/huggingface
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs
if [ ! -d "$DATA_ROOT" ]; then
    echo "Dataset does not exist: $DATA_ROOT" >&2
    exit 1
fi
if [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge/bin/conda"
else
    CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

# Fail rather than overwrite any previous simple-fusion experiment.
mkdir -p "$(dirname "$OUTPUT_DIR")"
if ! mkdir "$OUTPUT_DIR"; then
    echo "Output directory exists or cannot be created: $OUTPUT_DIR" >&2
    exit 1
fi

echo "===== RS-BDNet SIMPLE FUSION ====="
echo "Dataset: $DATA_ROOT; output: $OUTPUT_DIR"
echo "Fusion: parameter-free mean of ResNet and Swin change features"
date
hostname
nvidia-smi
python -u transformer/scripts/train_xbd_rsbdnet_simple_fusion.py --check-fusion
python -u transformer/scripts/train_xbd_rsbdnet_simple_fusion.py \
  --dataset-loader "$DATASET_LOADER" \
  --xbd-root "$DATA_ROOT" \
  --train-split "$TRAIN_SPLIT" \
  --val-split "$VAL_SPLIT" \
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

echo "DONE: $OUTPUT_DIR"
date
