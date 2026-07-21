#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=hrtbda_v5_tx_xai
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
EXPERIMENT_DIR=$PROJECT_ROOT/output/HRTBDA-v5-xBDInit-FullFineTune_TEXAS_TORNADOES_resetbest
PHASE1_CHECKPOINT=$EXPERIMENT_DIR/checkpoints/phase1_best.pt
PHASE2_CHECKPOINT=$EXPERIMENT_DIR/checkpoints/phase2_best.pt
DATA_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
OUTPUT_DIR=$EXPERIMENT_DIR/xai_gradcam_test

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found"
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== HRTBDA-v5 TEXAS TORNADOES GRAD-CAM XAI ====="
date
hostname
nvidia-smi || true
echo "Phase I: $PHASE1_CHECKPOINT"
echo "Phase II: $PHASE2_CHECKPOINT"
echo "Test data: $DATA_ROOT/test"
echo "Output: $OUTPUT_DIR"

python -u transformer/scripts/xai_texas_tornadoes_hrtbda_v5_gradcam.py \
  --phase1-checkpoint "$PHASE1_CHECKPOINT" \
  --phase2-checkpoint "$PHASE2_CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --test-split test \
  --output-dir "$OUTPUT_DIR" \
  --sample-count 0 \
  --num-workers 2 \
  --img-size 1024 \
  --postprocess-dilation minor \
  --dilation-kernel 3 \
  --device cuda

echo "DONE: HRTBDA-v5 Texas Tornadoes Grad-CAM XAI"
date
