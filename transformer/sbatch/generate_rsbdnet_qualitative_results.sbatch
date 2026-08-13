#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=04:00:00
#SBATCH --job-name=rsbd_qual
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_CACHE_DISABLE=1
export HF_HUB_OFFLINE=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_PARENT=/homes/j244s673/documents/wsu/phd
OUTPUT_DIR=$PROJECT_ROOT/output/RSBDNet_qualitative_results

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

echo "===== RS-BDNET SIX-DATASET QUALITATIVE EVALUATION ====="
echo "Protocol: validation-selected best checkpoints; untouched test sets only"
echo "Selection: ground-truth severity coverage; never prediction quality"
echo "xBD checkpoint: BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3 (not concatenation ablation)"
echo "Output: $OUTPUT_DIR"
date
hostname
nvidia-smi

python -u transformer/scripts/generate_rsbdnet_qualitative_results.py \
  --project-root "$PROJECT_ROOT" \
  --data-parent "$DATA_PARENT" \
  --output-dir "$OUTPUT_DIR" \
  --image-size 896 \
  --examples-per-dataset 4 \
  --amp

echo "DONE: RS-BDNet qualitative figures generated"
date
