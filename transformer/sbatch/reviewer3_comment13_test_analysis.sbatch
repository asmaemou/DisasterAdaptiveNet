#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=30:00:00
#SBATCH --job-name=reviewer3_comment13
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_CACHE_DISABLE=1
export HF_HUB_OFFLINE=1
ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$ROOT"
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    echo 'ERROR: miniforge3 conda.sh not found' >&2; exit 1
fi
conda activate disasteradaptivenet_cu128
SOURCE="$ROOT/transformer/scripts/reviewer3_comment13_test_error_analysis.py"
OUTPUT="$ROOT/output/reviewer3_comment13_${SLURM_JOB_ID}"
test -f "$SOURCE" || { echo "ERROR: missing $SOURCE" >&2; exit 1; }
test ! -e "$OUTPUT" || { echo "ERROR: refusing to overwrite $OUTPUT" >&2; exit 1; }
echo "Reviewer #3 Comment 13 | job $SLURM_JOB_ID"
echo "Output directory: $OUTPUT"
date
which python
python --version
nvidia-smi
python -u "$SOURCE" --datasets semeru texas --output-dir "$OUTPUT" --amp --disable-cudnn
echo "SUCCESS: $OUTPUT"
date
