#!/bin/bash
#SBATCH --job-name=idabd_loc_eval
#SBATCH --partition=wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/idabd_loc_eval_%j.out
#SBATCH --error=logs/idabd_loc_eval_%j.err

# Do NOT use set -u before sourcing ~/.bashrc on this cluster.
# /etc/bashrc references BASHRCSOURCED before it is defined.
set -eo pipefail

RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/new_approach
XVIEW2_CODE_DIR=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach

cd "$RUN_DIR"
mkdir -p logs

echo "START: $(date)"
echo "HOST:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID}"
echo "PWD:   $(pwd)"

# Temporarily disable unbound-variable checking while loading bashrc/conda.
set +u
source ~/.bashrc
conda activate disasteradaptivenet_cu128
set -u

# Let Python find zoo either from DisasterAdaptiveNet/new_approach or xview2_1st_place_solution/new_approach.
export PYTHONPATH="$RUN_DIR:$XVIEW2_CODE_DIR:${PYTHONPATH:-}"

python - <<'PYTORCH'
import torch
print('cuda available:', torch.cuda.is_available())
print('torch version:', torch.__version__)
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PYTORCH

python eval_idabd_stage1_loc_pre_with_postgt_DANet.py \
  --input_split pre \
  --gt_split post \
  --thresh 0.1 \
  --thresh_sweep \
  --debug \
  --debug_first 5 \
  --debug_every 20

# To save prediction masks too, add this to the python command above:
#   --save_preds --out_dir /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/new_approach/pred_idabd_loc_stage1

echo "END: $(date)"
