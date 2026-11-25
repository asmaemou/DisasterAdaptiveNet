#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=12:00:00
#SBATCH --job-name=reviewer3_comment16_figure
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/reviewer3_comment16_figure_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/reviewer3_comment16_figure_%j.err

# Reviewer #3, Comment (16): xBD matched-area RS-BDNet / 3 winner ensembles.
# Every job writes a NEW directory, and all model inference stages run in
# separate Python processes so VRAM is released between ensembles.
set -euo pipefail
ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SCRIPT="$ROOT/transformer/scripts/generate_reviewer3_comment16_figure.py"
OUT="$ROOT/output/reviewer3_comment16_figure_${SLURM_JOB_ID:?}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_CACHE_DISABLE=1
export HF_HUB_OFFLINE=1

# Activate Conda BEFORE the first Python invocation.
if [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge/bin/conda"
else
    CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

cd "$ROOT"
test -f "$SCRIPT"
test ! -e "$OUT" || { echo "Refusing to overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"

echo "Reviewer #3 Comment (16), unique output: $OUT"
echo "Python: $(command -v python)"
date; hostname; nvidia-smi

# Reuse the qualitative selection logic and full benchmark backends.
for STAGE in rsbdnet first second third assemble; do
    echo "===== STAGE: $STAGE ====="
    date
    python -u "$SCRIPT" --stage "$STAGE" --output-dir "$OUT" \
        --xbd-root /homes/j244s673/documents/wsu/phd/xview2
    echo "===== COMPLETE: $STAGE ====="
done

echo "SUCCESS: $OUT/figures/"
date
