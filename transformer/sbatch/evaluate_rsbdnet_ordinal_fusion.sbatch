#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=rsbd_ordinal_fusion
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_CACHE_DISABLE=1

LOCAL_JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-manual}}"
export CUDA_CACHE_PATH="$LOCAL_JOB_TMP/cuda_cache"
export XDG_CACHE_HOME="$LOCAL_JOB_TMP/xdg_cache"
export TORCH_EXTENSIONS_DIR="$LOCAL_JOB_TMP/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_TMP/torchinductor"
export HF_HOME="$LOCAL_JOB_TMP/huggingface"
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

PROJECT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SCRIPT="$PROJECT/transformer/scripts/evaluate_rsbdnet_ordinal_fusion.py"
OUTPUT="$PROJECT/output/RSBDNet_xBD_ZeroShot_OrdinalFusion"
SOURCE="$PROJECT/output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"
BASELINE="$PROJECT/output/RSBDNet_xBD_ZeroShot_EBD"
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
EBD_BASE=/homes/j244s673/documents/wsu/phd
EXPECTED_HASH=c781c1d9539618e2d5c67bc6365daa24557e9c01e64f87fc7d92282fe90aea77

cd "$PROJECT"
for required in "$SCRIPT" "$PROJECT/transformer/scripts/evaluate_rsbdnet_xbd_zeroshot_ebd.py" "$SOURCE" "$BASELINE/experiment_manifest.json" "$BASELINE/cross_dataset_summary.csv"; do
  test -f "$required" || { echo "Missing required file: $required" >&2; exit 1; }
done

echo "$EXPECTED_HASH  $SOURCE" | sha256sum -c - || { echo 'Original source checkpoint hash mismatch' >&2; exit 1; }

if [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge/bin/conda"
else
    CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

printf '=== RS-BDNet ordinal fusion (READ-ONLY source checkpoint) ===\n'
printf 'SLURM job: %s\n' "${SLURM_JOB_ID:-manual}"
printf 'Source checkpoint: %s\n' "$SOURCE"
printf 'Output destination (NEW ONLY): %s\n' "$OUTPUT"
nvidia-smi

python -u "$SCRIPT" \
    --source-checkpoint "$SOURCE" \
    --output-dir "$OUTPUT" \
    --baseline-output "$BASELINE" \
    --xbd-root "$XBD_ROOT" \
    --ebd-base "$EBD_BASE" \
    --image-size 896 \
    --batch-size 2 \
    --num-workers 0 \
    --seed 42

echo "$EXPECTED_HASH  $SOURCE" | sha256sum -c -
echo 'COMPLETED. xBD Hold comparison + EBD results (if requested) saved in NEW folder.'
