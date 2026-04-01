#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=dan_rescuenet_train_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

DATA_ROOT=/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/rescuenet_xbd_disasteradaptivenet_cleanrerun

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniconda3/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found."
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "DATA_ROOT: $DATA_ROOT"
echo "RUN_DIR:   $RUN_DIR"
nvidia-smi || true
echo "===================="

mkdir -p "$RUN_DIR"

echo ""
echo "===== STEP 1: TRAIN CLEAN RERUN ====="
python -u train_network_rescuenet_xbd_clean.py \
  --dataset-root "$DATA_ROOT" \
  --output-dir "$RUN_DIR" \
  --epochs 20 \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512 \
  --amp

echo ""
echo "===== STEP 2: EVALUATE BEST CHECKPOINT ON TEST SET ====="
python -u evaluate_test_f1_rescuenet_xbd.py \
  --dataset-root "$DATA_ROOT" \
  --checkpoint "$RUN_DIR/checkpoints/best.pt" \
  --split test \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512 \
  --output-dir "$RUN_DIR/scores"

echo ""
echo "===== DONE ====="
echo "Training outputs:"
echo "  $RUN_DIR/checkpoints/best.pt"
echo "  $RUN_DIR/history.json"
echo "  $RUN_DIR/test_metrics.json"
echo ""
echo "Per-class test F1 outputs:"
echo "  $RUN_DIR/scores/scores_rescuenet_xbd_test.json"
echo "  $RUN_DIR/scores/scores_rescuenet_xbd_test.txt"