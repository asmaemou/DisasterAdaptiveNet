#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=xbd_ida_r34_dann
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
IDA_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/resnet34_dann_xbd_to_ida

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
echo "XBD_ROOT:  $XBD_ROOT"
echo "IDA_ROOT:  $IDA_ROOT"
echo "RUN_DIR:   $RUN_DIR"
nvidia-smi || true
echo "===================="

mkdir -p "$RUN_DIR"

python -u train_resnet34_dann_xbd_to_ida.py \
  --xbd-root "$XBD_ROOT" \
  --ida-root "$IDA_ROOT" \
  --xbd-train-split train \
  --xbd-val-split hold \
  --ida-train-split train \
  --ida-val-split val \
  --ida-test-split test \
  --output-dir "$RUN_DIR" \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512 \
  --save-every 1 \
  --early-stopping-patience 10 \
  --domain-weight 0.2 \
  --amp

echo ""
echo "===== FINAL OUTPUT FILES ====="
echo "Best checkpoint:"
echo "  $RUN_DIR/checkpoints/best.pt"
echo ""
echo "Last checkpoint:"
echo "  $RUN_DIR/checkpoints/last.pt"
echo ""
echo "Training history:"
echo "  $RUN_DIR/history.json"
echo ""
echo "Final target test metrics:"
echo "  $RUN_DIR/target_test_metrics.json"
echo ""
echo "Readable score file:"
echo "  $RUN_DIR/scores/scores_idabd_test.txt"
echo ""
echo "JSON score file:"
echo "  $RUN_DIR/scores/scores_idabd_test.json"
