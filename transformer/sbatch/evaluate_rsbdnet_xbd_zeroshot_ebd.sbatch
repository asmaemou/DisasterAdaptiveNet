#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=rsbd_zeroshot_ebd
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
export HF_HOME="$LOCAL_JOB_TMP/huggingface"
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
EBD_BASE=/homes/j244s673/documents/wsu/phd
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
CHECKPOINT="$PROJECT_ROOT/output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"
OUTPUT_DIR="$PROJECT_ROOT/output/RSBDNet_xBD_ZeroShot_EBD"
SCRIPT="$PROJECT_ROOT/transformer/scripts/evaluate_rsbdnet_xbd_zeroshot_ebd.py"

cd "$PROJECT_ROOT"

# Important: SLURM creates the output/error log files before the script begins.
# Create transformer/sbatch/logs manually *before* submitting this job.
for REQUIRED in "$SCRIPT" "$CHECKPOINT"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "ERROR: Missing file: $REQUIRED" >&2
        exit 1
    fi
done
for REQUIRED in "$XBD_ROOT/test" \
    "$EBD_BASE/earthquake_turkey_preprocessed/test" \
    "$EBD_BASE/mount_semeru_eruption_preprocessed/test" \
    "$EBD_BASE/texas_tornadoes_preprocessed/test" \
    "$EBD_BASE/hurricane_delta_preprocessed/test" \
    "$EBD_BASE/pakistan_flooding_preprocessed/test"; do
    if [ ! -d "$REQUIRED" ]; then
        echo "ERROR: Missing dataset test directory: $REQUIRED" >&2
        exit 1
    fi
done

if [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge/bin/conda"
else
    CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR")" ]; then
    echo "ERROR: Output directory already contains files: $OUTPUT_DIR" >&2
    echo "Use a fresh output directory to preserve existing results." >&2
    exit 1
fi

echo "===== RS-BDNet: xBD -> EBD strict zero-shot evaluation ====="
echo "Date UTC: $(date -u -Is)"
echo "Hostname: $(hostname)"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"
echo "Python: $(command -v python)"
echo "Source checkpoint: $CHECKPOINT"
echo "Data base: $EBD_BASE"
echo "Output: $OUTPUT_DIR"
echo "Protocol: one frozen xBD checkpoint; xBD Hold threshold; all 5 EBD test sets"
echo "Metrics: localization P/R/F1, four damage-class P/R/F1, macro P/R/F1"
echo "Verification: reproduce xBD Test 0.7756 macro-F1 before EBD"
nvidia-smi

python -u "$SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --ebd-base "$EBD_BASE" \
    --xbd-root "$XBD_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --verify-xbd-test \
    --expected-xbd-macro-f1 0.7756 \
    --expected-xbd-localization-f1 0.8447 \
    --xbd-reproduction-tolerance 0.00015 \
    --expected-epoch 35 \
    --expected-threshold 0.50 \
    --batch-size 2 \
    --num-workers 0 \
    --image-size 896 \
    --seed 42 \
    --hash-input-files

echo "===== COMPLETED SUCCESSFULLY ====="
echo "Summary: $OUTPUT_DIR/cross_dataset_summary.csv"
date -u -Is
