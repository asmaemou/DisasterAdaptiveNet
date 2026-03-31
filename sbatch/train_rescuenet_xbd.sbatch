#!/bin/bash
#SBATCH --job-name=dan_rescuenet_xbd
#SBATCH --output=logs/dan_rescuenet_xbd_%j.out
#SBATCH --error=logs/dan_rescuenet_xbd_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs output/rescuenet_xbd_disasteradaptivenet

# Activate your environment here.
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate disasteradaptivenet

python train_network_rescuenet_xbd.py \
  --dataset-root /homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd \
  --output-dir /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/rescuenet_xbd_disasteradaptivenet \
  --epochs 40 \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512 \
  --amp