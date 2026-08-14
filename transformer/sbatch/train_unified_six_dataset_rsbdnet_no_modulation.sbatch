#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=six_no_mod
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
OUTPUT_DIR=$PROJECT_ROOT/output/UnifiedRSBDNet_NoModulation_SixDatasets
LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_JOB_TMP/torchinductor
export HF_HOME=$LOCAL_JOB_TMP/huggingface
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR"

if [ -x "$HOME/miniforge3/bin/conda" ]; then CONDA_EXE="$HOME/miniforge3/bin/conda";
elif [ -x "$HOME/miniforge/bin/conda" ]; then CONDA_EXE="$HOME/miniforge/bin/conda";
else CONDA_EXE="$(command -v conda)"; fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== UNIFIED SIX-DATASET RS-BDNET CONTROL (NO MODULATION) ====="
echo "Same shared architecture, data, sampler, losses, and selection protocol"
echo "Only DA-CTDM is disabled, providing a controlled ablation"
echo "Output: $OUTPUT_DIR"
date; hostname; nvidia-smi

RESUME_ARGS=()
if [ -f "$OUTPUT_DIR/checkpoints/last.pt" ]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/checkpoints/last.pt")
  echo "Resuming from existing last checkpoint."
fi

python -u transformer/scripts/train_unified_six_dataset_rsbdnet_dactdm.py \
  --data-parent /homes/j244s673/documents/wsu/phd \
  --xview2-root /homes/j244s673/documents/wsu/phd/xview2 \
  --output-dir "$OUTPUT_DIR" \
  --mode none --epochs 40 --patience 8 --batch-size 2 --eval-batch-size 1 \
  --num-workers 0 --samples-per-epoch 9168 --img-size 896 --lr 2e-5 \
  --weight-decay 1e-4 --seed 42 --save-every 10 --amp \
  "${RESUME_ARGS[@]}"

echo "DONE: unified six-dataset no-modulation control"; date
