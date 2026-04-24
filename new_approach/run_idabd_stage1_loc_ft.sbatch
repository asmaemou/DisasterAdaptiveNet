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

echo "START: $(date)"
echo "HOST:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID}"

cd /homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach
mkdir -p logs

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

python train_idabd_stage1_loc_da_calib_fusion_modified.py \
  --train_all \
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
  --out_dir /homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd_stage1_loc_ft_checkpoints

echo "END: $(date)"
