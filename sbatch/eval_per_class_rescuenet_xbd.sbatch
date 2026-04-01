#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=eval_rescuenet_f1
#SBATCH --output=logs/eval_rescuenet_f1_%j.out
#SBATCH --error=logs/eval_rescuenet_f1_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

set -euo pipefail

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

python -u evaluate_per_class_rescuenet_xbd.py \
  --dataset-root /homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd \
  --checkpoint /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/rescuenet_xbd_disasteradaptivenet/checkpoints/best.pt \
  --split test \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512