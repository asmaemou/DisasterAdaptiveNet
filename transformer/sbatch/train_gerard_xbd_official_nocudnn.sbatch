#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=r3_gerard_nc
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
RUN_DIR="$PROJECT/output/reviewer3_gerard_xbd_nocudnn_${SLURM_JOB_ID}"

# Never reuse another experiment's output directory.
if [[ -e "$RUN_DIR" ]]; then
    echo "STOP: experiment output already exists: $RUN_DIR"
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
nvidia-smi || true

# Preflight, then train; zero loader workers and more frequent progress logs.
python -u transformer/scripts/train_official_nocudnn.py --preflight \
  -c strong_baseline_xview2 -o "$RUN_DIR" -d "$XBD_ROOT" \
  DATALOADER.NUM_WORKER 0 LOG_FREQ 10

python -u transformer/scripts/train_official_nocudnn.py \
  -c strong_baseline_xview2 -o "$RUN_DIR" -d "$XBD_ROOT" \
  DATALOADER.NUM_WORKER 0 LOG_FREQ 10
