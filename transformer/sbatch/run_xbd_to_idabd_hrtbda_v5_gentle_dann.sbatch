#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=idabd_dann_gentle
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

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
SPLIT_FILE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/idabd_splits_seed42_80_10_10.json

XBD_PHASE1=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp/checkpoints/phase1_best.pt
XBD_PHASE2=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_to_idabd_hrtbda_v5_foreground_dann_GENTLE
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

echo "===== GENTLE DANN JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "==============================="

python -u transformer/scripts/train_xbd_to_idabd_hrtbda_v5_foreground_dann.py \
  --phase adapt \
  --xbd-root "$XBD_ROOT" \
  --source-train-split train tier3 \
  --idabd-root "$IDABD_ROOT" \
  --split-file "$SPLIT_FILE" \
  --phase1-checkpoint "$XBD_PHASE1" \
  --phase2-checkpoint "$XBD_PHASE2" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 8 \
  --steps-per-epoch 150 \
  --source-batch-size 1 \
  --target-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  --img-size 1024 \
  --phase2-crop-size 608 \
  --crop-candidate-count 8 \
  --base-channels 48 \
  --decoder-channels 128 \
  --window-size 8 \
  --lr 5e-6 \
  --weight-decay 1e-4 \
  --warmup-epochs 2 \
  --max-grad-norm 0.5 \
  --early-stopping-patience 3 \
  --focal-gamma 2.0 \
  --cls-loss-weight 1.0 \
  --aux-loc-weight 0.2 \
  --minor-damage-boost 1.5 \
  --major-damage-boost 1.5 \
  --destroyed-damage-boost 1.0 \
  --max-damage-class-weight 10.0 \
  --lambda-domain 0.005 \
  --lambda-domain-loss 0.05 \
  --domain-warmup-epochs 3 \
  --lambda-entropy 0.0 \
  --domain-hidden-dim 128 \
  --domain-dropout 0.3 \
  --dilation-options none minor \
  --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 \
  --amp
