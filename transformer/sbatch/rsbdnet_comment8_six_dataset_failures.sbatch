#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=rsbd_c8_sixfail
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_CACHE_DISABLE=1
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$ROOT"
mkdir -p transformer/sbatch/logs
if [ -x "$HOME/miniforge3/bin/conda" ]; then
 CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
 CONDA_EXE="$HOME/miniforge/bin/conda"
else
 CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128
OUT="$ROOT/output/RSBDNet_Comment8_SixDataset_FailureAnalysis"
SCRIPT="$ROOT/transformer/scripts/rsbdnet_comment8_six_dataset_failures.py"
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT" >&2; exit 1; }
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "ERROR: New experiment output folder is not empty: $OUT" >&2
  echo "Do not delete existing data just to rerun; select a new experiment version." >&2
  exit 1
fi
printf 'Comment 8 six-dataset target-trained inference\n'
printenv SLURM_JOB_ID || true
nvidia-smi
python -u "$SCRIPT" --output-dir "$OUT" --min-area 64 --min-purity 0.85 --crop-size 448
