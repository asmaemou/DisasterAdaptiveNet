#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=03:00:00
#SBATCH --job-name=rsbd_r2_c7_bench
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$ROOT"
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
# Unique, additive location; never overwrite any trained checkpoints/results.
OUT="$ROOT/output/Reviewer2_Comment7_InferenceBench/rsbdnet_${SLURM_JOB_ID}"
python -u "$ROOT/transformer/scripts/benchmark_rsbdnet_comment7.py" \
  --backend rsbdnet \
  --xbd-root /homes/j244s673/documents/wsu/phd/xview2 \
  --checkpoint "$ROOT/output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt" \
  --samples 100 --warmup 20 --repeats 100 \
  --output-dir "$OUT"
