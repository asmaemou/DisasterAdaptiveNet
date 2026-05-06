#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=idabd_v7_studa_sweep
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found."
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

PHASE1_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/checkpoints/phase1_best.pt

PHASE2_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_to_idabd_v7_msdf_studa/checkpoints/phase2_studa_teacher_latest.pt

IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd

SPLIT_JSON=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_to_idabd_v7_msdf_studa/idabd_splits_seed42_80_10_10.json

OUTDIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_to_idabd_v7_msdf_studa

echo "===== ST-UDA THRESHOLD SWEEP ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "PHASE1_CKPT: $PHASE1_CKPT"
echo "PHASE2_CKPT: $PHASE2_CKPT"
echo "IDABD_ROOT: $IDABD_ROOT"
echo "SPLIT_JSON: $SPLIT_JSON"
echo "OUTDIR: $OUTDIR"
echo "==================================="

python -u transformer/scripts/sweep_idabd_v7_msdf_thresholds.py \
  --direct-script transformer/scripts/eval_idabd_v7_msdf_direct_transfer.py \
  --v7-script transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --phase2-checkpoint "$PHASE2_CKPT" \
  --idabd-root "$IDABD_ROOT" \
  --source-xbd-root /homes/j244s673/documents/wsu/phd/xview2 \
  --output-dir "$OUTDIR" \
  --split-json "$SPLIT_JSON" \
  --eval-split test \
  --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
  --dilations none minor minor_major \
  --dilation-kernel 3 \
  --eval-batch-size 1 \
  --num-workers 8 \
  --img-size 1024 \
  --base-channels 48 \
  --decoder-channels 128 \
  --window-size 8

echo "Done."