#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=rsbd_source_calibration
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_CACHE_DISABLE=1

CAL_STAGE="${CAL_STAGE:-source}"
case "$CAL_STAGE" in
  source|target) ;;
  *) echo "Invalid CAL_STAGE=$CAL_STAGE; expected source or target" >&2; exit 2 ;;
esac

LOCAL_JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID:-manual}}"
export CUDA_CACHE_PATH="$LOCAL_JOB_TMP/cuda_cache"
export XDG_CACHE_HOME="$LOCAL_JOB_TMP/xdg_cache"
export TORCH_EXTENSIONS_DIR="$LOCAL_JOB_TMP/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_TMP/torchinductor"
export HF_HOME="$LOCAL_JOB_TMP/huggingface"
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

PROJECT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SOURCE="$PROJECT/output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"
SCRIPT="$PROJECT/transformer/scripts/evaluate_rsbdnet_source_calibration.py"
OUTPUT="$PROJECT/output/RSBDNet_xBD_ZeroShot_Calibration"
HASH=c781c1d9539618e2d5c67bc6365daa24557e9c01e64f87fc7d92282fe90aea77

cd "$PROJECT"
for file in "$SOURCE" "$SCRIPT" "$PROJECT/transformer/scripts/evaluate_rsbdnet_xbd_zeroshot_ebd.py"; do
  test -f "$file" || { echo "Missing: $file" >&2; exit 1; }
done
printf '%s  %s\n' "$HASH" "$SOURCE" | sha256sum -c -

if [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge/bin/conda"
else
    CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128
nvidia-smi

echo "[STAGE] $CAL_STAGE; [NEW OUTPUT ONLY] $OUTPUT"
python -u "$SCRIPT" --stage "$CAL_STAGE" --output-dir "$OUTPUT" \
  --batch-size 2 --num-workers 0 --seed 42 \
  --sample-per-image-class 96 --bias-bound 2.0 --min-validation-gain 0.005

printf '%s  %s\n' "$HASH" "$SOURCE" | sha256sum -c -
echo "[DONE] $CAL_STAGE stage finished."
