#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=reviewer3_comment16_assemble
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

# Reviewer #3, Comment (16): render figures only from already saved predictions.
# Do NOT rerun the four GPU inference stages.
set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
OUT="$ROOT/output/reviewer3_comment16_figure_34996"
PY="$ROOT/transformer/scripts/generate_reviewer3_comment16_figure.py"
cd "$ROOT"

if [ ! -f "$PY" ]; then
  echo "Missing figure-generation Python script: $PY" >&2
  exit 1
fi
if [ ! -f "$OUT/selection_manifest.csv" ]; then
  echo "Missing selection manifest; original predictions cannot be verified: $OUT" >&2
  exit 1
fi

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

# Missing directory caused original job 34996 to fail after all predictions finished.
mkdir -p "$OUT/figures" "$ROOT/transformer/sbatch/logs"

echo "Assembling only; reusing existing predictions from job 34996"
echo "Output directory: $OUT"
python --version
python -u "$PY" --stage assemble --output-dir "$OUT"
echo "SUCCESS: Figure generation completed without rerunning inference."
find "$OUT/figures" -maxdepth 1 -type f -printf '%f\n' | sort
