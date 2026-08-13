#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=damage_dist
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_PARENT=/homes/j244s673/documents/wsu/phd
OUTPUT_DIR=$PROJECT_ROOT/output/RSBDNet_dataset_statistics

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== DAMAGE-CLASS CONNECTED-COMPONENT DISTRIBUTION ====="
echo "Datasets: xBD + five evaluated EBD events"
echo "Counting all train/validation/test masks; each mask is scanned once."
echo "Output: $OUTPUT_DIR"
date
hostname

python -u transformer/scripts/plot_damage_class_distribution.py \
  --data-parent "$DATA_PARENT" \
  --output-dir "$OUTPUT_DIR" \
  --minimum-component-pixels 1

echo "DONE: damage-class distribution figure generated"
date
