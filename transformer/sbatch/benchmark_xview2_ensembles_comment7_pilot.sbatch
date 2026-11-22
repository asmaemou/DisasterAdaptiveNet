#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --job-name=xv2_r2_c7_pilot
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%A_%a.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%A_%a.err
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE
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
BACKENDS=(first second third)
BACKEND="${BACKENDS[$SLURM_ARRAY_TASK_ID]}"
echo "Pilot: complete $BACKEND ensemble (all checkpoints, 1 warmup + 2 timed pairs)"
python -u transformer/scripts/benchmark_xview2_full_ensembles_comment7.py \
 --backend "$BACKEND" --mode full \
 --xbd-root /homes/j244s673/documents/wsu/phd/xview2 \
 --samples 2 --warmup 1 --repeats 2 \
 --output-name "xview2_${BACKEND}_pilot_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
