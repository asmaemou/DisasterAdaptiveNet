#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=idabd_v7_dann
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

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd

PHASE1_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/checkpoints/phase1_best.pt
PHASE2_CKPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/checkpoints/phase2_best.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_to_idabd_v7_msdf_foreground_dann

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

echo "===== HRTBDA v7-MSDF FOREGROUND DANN JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "XBD_ROOT: $XBD_ROOT"
echo "IDABD_ROOT: $IDABD_ROOT"
echo "PHASE1_CKPT: $PHASE1_CKPT"
echo "PHASE2_CKPT: $PHASE2_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "==================================================="

python -u transformer/scripts/train_xbd_to_idabd_v7_msdf_foreground_dann.py \
  --v7-script transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --phase2-checkpoint "$PHASE2_CKPT" \
  --xbd-root "$XBD_ROOT" \
  --idabd-root "$IDABD_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 15 \
  --steps-per-epoch 500 \
  --source-batch-size 2 \
  --target-batch-size 2 \
  --eval-batch-size 1 \
  --num-workers 8 \
  --img-size 1024 \
  --phase2-crop-size 608 \
  --crop-candidate-count 8 \
  --lr 2e-5 \
  --disc-lr 1e-4 \
  --weight-decay 1e-4 \
  --seed 42 \
  --base-channels 48 \
  --decoder-channels 128 \
  --window-size 8 \
  --max-grad-norm 1.0 \
  --focal-gamma 2.0 \
  --aux-loc-weight 0.2 \
  --domain-loss-weight 0.05 \
  --max-dann-lambda 1.0 \
  --target-layer change_fusion.0 \
  --postprocess-dilation minor \
  --dilation-kernel 3 \
  --amp

