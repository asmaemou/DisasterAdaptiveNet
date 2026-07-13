#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=prepare_hurricane_irma
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/DS_preprocessing/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/DS_preprocessing/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet"
SBATCH_DIR="$PROJECT_ROOT/DS_preprocessing/sbatch"
SCRIPT_PATH="$PROJECT_ROOT/DS_preprocessing/python_script/prepare_hurricane_irma.py"

mkdir -p "$SBATCH_DIR/logs"

eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda activate disasteradaptivenet_cu128

python "$SCRIPT_PATH"
