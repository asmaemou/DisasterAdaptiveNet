#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=30:00:00
#SBATCH --job-name=reviewer3_comment11_v2
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/reviewer3_comment11_v2_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/reviewer3_comment11_v2_%j.err

# Reviewer #3, Comment (11) — full xBD pixel-level error audit.
# Preserves original weights and scripts; results are unique per Slurm job ID.
set -euo pipefail

ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SCRIPT="$ROOT/transformer/scripts/reviewer3_comment11_analysis_v2.py"
export REVIEWER3_COMMENT11_OUTPUT_DIR="$ROOT/output/reviewer3_comment11_v2_${SLURM_JOB_ID:?}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
test -f "$SCRIPT" || { echo "Missing $SCRIPT" >&2; exit 1; }
if [ -e "$REVIEWER3_COMMENT11_OUTPUT_DIR" ]; then
    echo "Refusing overwrite: $REVIEWER3_COMMENT11_OUTPUT_DIR" >&2
    exit 1
fi

# Python must be available BEFORE the first Python command.
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
test -f "$CONDA_SH" || { echo "Missing $CONDA_SH" >&2; exit 1; }
source "$CONDA_SH"
conda activate disasteradaptivenet_cu128

echo "Reviewer #3, Comment (11): final full-test audit"
echo "Job ID: $SLURM_JOB_ID"
echo "Results: $REVIEWER3_COMMENT11_OUTPUT_DIR"
echo "Python: $(command -v python)"
date
hostname
nvidia-smi
sha256sum "$SCRIPT" "$ROOT/output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"

python -u "$SCRIPT"

echo "SUCCESS: saved $REVIEWER3_COMMENT11_OUTPUT_DIR"
ls -lh "$REVIEWER3_COMMENT11_OUTPUT_DIR"
