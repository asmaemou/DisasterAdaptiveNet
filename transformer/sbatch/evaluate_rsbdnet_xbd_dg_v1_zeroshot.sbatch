#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=1-12:00:00
#SBATCH --job-name=rsbd_dg_ebd_v1
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONHASHSEED=42
export CUDA_CACHE_DISABLE=1
LOCAL_JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-manual}}"
export CUDA_CACHE_PATH="$LOCAL_JOB_TMP/cuda_cache"
export XDG_CACHE_HOME="$LOCAL_JOB_TMP/xdg_cache"
export TORCH_EXTENSIONS_DIR="$LOCAL_JOB_TMP/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_TMP/torchinductor"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PROJECT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
TRAIN_DIR="$PROJECT/output/RSBDNet_xBD_DomainGeneralization_v1"
OUTPUT="$PROJECT/output/RSBDNet_xBD_DomainGeneralization_v1_ZeroShot_EBD"
SCRIPT="$PROJECT/transformer/scripts/evaluate_rsbdnet_xbd_dg_v1_zeroshot.py"
cd "$PROJECT"
for REQUIRED in "$SCRIPT" "$TRAIN_DIR/checkpoints/best.pt" "$TRAIN_DIR/experiment_manifest.json" \
    "$PROJECT/transformer/scripts/evaluate_rsbdnet_xbd_zeroshot_ebd.py"; do
    [ -f "$REQUIRED" ] || { echo "Missing required file: $REQUIRED" >&2; exit 1; }
done
if [ -d "$OUTPUT" ] && [ -n "$(ls -A "$OUTPUT")" ]; then
    echo "ERROR: Zero-shot DG results output already exists, refusing overwrite: $OUTPUT" >&2
    exit 1
fi
if [ -x "$HOME/miniforge3/bin/conda" ]; then CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then CONDA_EXE="$HOME/miniforge/bin/conda"
else CONDA_EXE="$(command -v conda)"; fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== Frozen NEW DG checkpoint -> EBD zero-shot evaluation ====="
echo "SLURM Job ID: ${SLURM_JOB_ID:-manual} | UTC: $(date -u -Is)"
echo "READ ONLY: $TRAIN_DIR/checkpoints/best.pt"
echo "WRITE ONLY: $OUTPUT"
nvidia-smi
python -u "$SCRIPT" \
    --checkpoint "$TRAIN_DIR/checkpoints/best.pt" \
    --training-manifest "$TRAIN_DIR/experiment_manifest.json" \
    --output-dir "$OUTPUT" \
    --xbd-root /homes/j244s673/documents/wsu/phd/xview2 \
    --ebd-base /homes/j244s673/documents/wsu/phd \
    --batch-size 2 --num-workers 0 --hash-input-files

echo "===== DG ZERO-SHOT EVALUATION COMPLETE ====="
echo "Comparison: $OUTPUT/comparison_to_original_exploratory.csv"
date -u -Is
