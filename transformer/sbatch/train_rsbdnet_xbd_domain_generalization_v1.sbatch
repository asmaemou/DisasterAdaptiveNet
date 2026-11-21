#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=rsbd_dg_train_v1
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
# Persistent pretrained backbone cache is helpful if nodes cannot access HF.
# Neither cache is under any of the protected experiment output folders.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PROJECT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SCRIPT="$PROJECT/transformer/scripts/train_rsbdnet_xbd_domain_generalization_v1.py"
XBD=/homes/j244s673/documents/wsu/phd/xview2
OUTPUT="$PROJECT/output/RSBDNet_xBD_DomainGeneralization_v1"
ORIGINAL="$PROJECT/output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"
cd "$PROJECT"

if [ ! -f "$SCRIPT" ] || [ ! -f "$ORIGINAL" ]; then
    echo "ERROR: Missing source training script or original protected checkpoint" >&2
    exit 1
fi
for SPLIT in train tier3; do
    [ -d "$XBD/$SPLIT/images" ] && [ -d "$XBD/$SPLIT/targets" ] || {
        echo "ERROR: Missing xBD $SPLIT directory" >&2; exit 1;
    }
done
if [ -x "$HOME/miniforge3/bin/conda" ]; then CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then CONDA_EXE="$HOME/miniforge/bin/conda"
else CONDA_EXE="$(command -v conda)"; fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

# JOBS must never silently overwrite prior results. To resume a DG v1 run,
# submit with: sbatch --export=ALL,RESUME=1 <script.sbatch>
RESUME="${RESUME:-0}"
RESUME_ARGS=()
if [ "$RESUME" = "1" ]; then
    [ -f "$OUTPUT/checkpoints/last.pt" ] || {
        echo "ERROR: RESUME=1 but no DG v1 last.pt found" >&2; exit 1;
    }
    RESUME_ARGS=(--resume)
elif [ "$RESUME" != "0" ]; then
    echo "ERROR: RESUME must be 0 or 1" >&2; exit 1
elif [ -d "$OUTPUT" ] && [ -n "$(ls -A "$OUTPUT")" ]; then
    echo "ERROR: DG output exists and is nonempty: $OUTPUT" >&2
    echo "If continuing an interrupted job, submit with RESUME=1 instead." >&2
    exit 1
fi

echo "===== xBD-ONLY EVENT-DISJOINT DOMAIN-GENERALIZATION TRAINING ====="
echo "UTC: $(date -u -Is) | SLURM job: ${SLURM_JOB_ID:-manual}"
echo "Starting FRESH ImageNet backbone initialization; NOT importing old model weights"
echo "Old checkpoint: READ-ONLY $ORIGINAL"
echo "New checkpoint/output: $OUTPUT"
echo "Validation: withheld entire source xBD events; NO EBD DATA"
echo "RESUME=$RESUME"
nvidia-smi

python -u "$SCRIPT" \
    --xbd-root "$XBD" \
    --output-dir "$OUTPUT" \
    --epochs 60 \
    --batch-size 2 \
    --num-workers 0 \
    --image-size 896 \
    --seed 42 \
    --learning-rate 2e-5 \
    --weight-decay 1e-4 \
    --early-stop-patience 12 \
    --val-event-fraction 0.25 \
    --event-balance-alpha 0.75 \
    --minor-sampling-boost 0.50 \
    --major-sampling-boost 0.60 \
    --destroyed-sampling-boost 0.20 \
    --minor-loss-multiplier 1.25 \
    --major-loss-multiplier 1.25 \
    --aux-ce-weight 0.25 \
    --thresholds 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 \
    --amp \
    "${RESUME_ARGS[@]}"

echo "===== DG TRAINING FINISHED ====="
echo "Manifest: $OUTPUT/experiment_manifest.json"
echo "Best checkpoint: $OUTPUT/checkpoints/best.pt"
date -u -Is
