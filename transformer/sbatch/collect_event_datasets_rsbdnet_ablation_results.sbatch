#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=collect_event_ablation
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
ABLATION_JOB_ID=${ABLATION_JOB_ID:?Submit with --export=ALL,ABLATION_JOB_ID=<job-id>}

cd "$PROJECT_ROOT"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

python -u transformer/scripts/collect_event_datasets_rsbdnet_ablation_results.py \
  --project-root "$PROJECT_ROOT" \
  --job-id "$ABLATION_JOB_ID"
