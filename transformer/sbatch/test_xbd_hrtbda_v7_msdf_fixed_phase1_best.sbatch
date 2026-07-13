#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=hrtbda_v7_test
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

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_fixed_phase1

PHASE1_CKPT=$OUTPUT_DIR/threshold_checkpoints/phase1_best_threshold_0.40.pt
PHASE2_CKPT=$OUTPUT_DIR/checkpoints/phase2_best.pt

echo "===== V7-MSDF FIXED PHASE-I TEST ONLY ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "PHASE1_CKPT: $PHASE1_CKPT"
echo "PHASE2_CKPT: $PHASE2_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TEST_SPLIT: test"
echo "=========================================="

python -u transformer/scripts/train_xbd_hrtbda_v7_msdf_fixed_phase1.py \
  --phase test \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --phase2-checkpoint "$PHASE2_CKPT" \
  --xbd-root "$XBD_ROOT" \
  --train-split train tier3 \
  --val-split hold \
  --test-split test \
  --output-dir "$OUTPUT_DIR" \
  --phase2-epochs 60 \
  --phase2-batch-size 2 \
  --eval-batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --img-size 1024 \
  --phase2-crop-size 608 \
  --crop-candidate-count 8 \
  --crop-weight-no-damage 1.0 \
  --crop-weight-minor 12.0 \
  --crop-weight-major 12.0 \
  --crop-weight-destroyed 4.0 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --seed 42 \
  --base-channels 48 \
  --decoder-channels 128 \
  --window-size 8 \
  --max-grad-norm 1.0 \
  --warmup-epochs 5 \
  --focal-gamma 2.0 \
  --cls-loss-weight 1.0 \
  --aux-loc-weight 0.2 \
  --minor-damage-boost 1.5 \
  --major-damage-boost 1.5 \
  --max-damage-class-weight 10.0 \
  --finetune-epochs 3 \
  --finetune-lr 5e-5 \
  --postprocess-dilation minor \
  --dilation-kernel 3 \
  --amp
