#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=changeos_stca_xbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_DIR="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/STCA"
RAW_XVIEW2_ROOT="/homes/j244s673/documents/wsu/phd/xview2"
PREPARED_ROOT="${PROJECT_DIR}/data/xview2_torchange"
WORK_DIR="${PROJECT_DIR}/runs/tier3_to_train_changeos_r18_stca"
CONDA_ENV="disasteradaptivenet_cu128"

mkdir -p "${PROJECT_DIR}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs" "${PROJECT_DIR}/data"
cd "${PROJECT_DIR}"

source /homes/j244s673/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# Install authors' trainable ChangeOS code path and its training engine if missing.
python - <<'PY'
import importlib.util, sys
mods = ["torchange", "ever"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
raise SystemExit(1 if missing else 0)
PY
if [ $? -ne 0 ]; then
  pip install ever-beta
  pip install torchange
fi

python scripts/prepare_xview2_for_changeos.py \
  --raw-root "${RAW_XVIEW2_ROOT}" \
  --out-root "${PREPARED_ROOT}" \
  --splits tier3 train test hold

python scripts/train_changeos_stca.py \
  --prepared-root "${PREPARED_ROOT}" \
  --source-split tier3 \
  --target-split train \
  --val-split test \
  --eval-split hold \
  --backbone r18 \
  --epochs-source 10 \
  --epochs-stca 5 \
  --batch-size 4 \
  --lr 3e-4 \
  --stca-loss lovasz \
  --stca-weight 0.2 \
  --target-threshold 0.5 \
  --source-feature-samples 256 \
  --target-feature-samples 256 \
  --work-dir "${WORK_DIR}"
