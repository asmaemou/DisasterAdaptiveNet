#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=r3_gerard_xbd
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/reviewer3/gerard_xbd_official_v1/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/reviewer3/gerard_xbd_official_v1/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
RUN_DIR="$PROJECT/output/reviewer3_gerard_xbd_official_${SLURM_JOB_ID}"

# Refuse to reuse any existing output directory.
if [[ -e "$RUN_DIR" ]]; then
    echo "STOP: output directory already exists: $RUN_DIR"
    exit 1
fi
mkdir "$RUN_DIR"

source /homes/j244s673/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_DIR="$RUN_DIR"

cd "$PROJECT"
echo "Job=$SLURM_JOB_ID; host=$(hostname); output=$RUN_DIR"
which python
nvidia-smi

python -u reviewer3/gerard_xbd_official_v1/train_official.py --preflight \
  -c strong_baseline_xview2 -o "$RUN_DIR" -d "$XBD_ROOT"
python -u reviewer3/gerard_xbd_official_v1/train_official.py \
  -c strong_baseline_xview2 -o "$RUN_DIR" -d "$XBD_ROOT"
