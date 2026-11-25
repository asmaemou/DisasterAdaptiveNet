#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=r3_gerard_eval
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/r3_gerard_eval_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/r3_gerard_eval_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
source /homes/j244s673/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

# Unique job directory; never reuse an existing folder.
OUT="$PWD/output/reviewer3_gerard_eval_${SLURM_JOB_ID}"
if [[ -e "$OUT" ]]; then
  echo "ERROR: output directory already exists: $OUT"
  exit 1
fi

python -u transformer/scripts/evaluate_gerard_official_xbd.py \
  --data-dir /homes/j244s673/documents/wsu/phd/xview2 \
  --checkpoint "$PWD/output/reviewer3_gerard_xbd_nocudnn_32359/networks/strong_baseline_xview2.pt" \
  --output-dir "$OUT" \
  --image-size 896
