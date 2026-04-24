#!/bin/bash
#SBATCH --job-name=idabd_loc_ft
#SBATCH --partition=wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/idabd_loc_ft_%j.out
#SBATCH --error=logs/idabd_loc_ft_%j.err

# Do NOT use set -u before sourcing ~/.bashrc on this cluster.
# /etc/bashrc references BASHRCSOURCED before it is defined.
set -eo pipefail

RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/new_approach
OUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/new_approach/idabd_stage1_loc_ft_checkpoints

cd "$RUN_DIR"
mkdir -p logs "$OUT_DIR"

echo "START: $(date)"
echo "HOST:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID}"
echo "RUN_DIR: $RUN_DIR"
echo "OUT_DIR: $OUT_DIR"

# Temporarily disable unbound-variable checking while loading bashrc/conda.
set +u
source ~/.bashrc
conda activate disasteradaptivenet_cu128
set -u

python - <<'PYTORCH'
import torch
print('cuda available:', torch.cuda.is_available())
print('torch version:', torch.__version__)
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PYTORCH

python train_idabd_stage1_loc_da_calib_fusion_DANet.py \
  --train_all \
  --input_split pre \
  --gt_split post \
  --epochs 5 \
  --batch 2 \
  --lr 2e-4 \
  --wd 1e-4 \
  --workers 4 \
  --crop 512 \
  --fusion_p 0.5 \
  --use_pos_weight \
  --amp \
  --thresh_sweep \
  --out_dir "$OUT_DIR"

echo "END: $(date)"
