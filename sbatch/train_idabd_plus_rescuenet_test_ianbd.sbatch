#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=idabd_rescuenet_to_ianbd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

IDA_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet
RESCUENET_ROOT=/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd
IANBD_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/idabd_plus_rescuenet_to_ianbd

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
echo "IDA_ROOT:       $IDA_ROOT"
echo "RESCUENET_ROOT: $RESCUENET_ROOT"
echo "IANBD_ROOT:     $IANBD_ROOT"
echo "RUN_DIR:        $RUN_DIR"
nvidia-smi || true
echo "===================="

mkdir -p "$RUN_DIR"

python -u train_idabd_plus_rescuenet_test_ianbd.py \
  --ida-root "$IDA_ROOT" \
  --rescuenet-root "$RESCUENET_ROOT" \
  --target-root "$IANBD_ROOT" \
  --ida-train-split train \
  --ida-val-split val \
  --rescuenet-train-split train \
  --rescuenet-val-split val \
  --target-test-split test \
  --output-dir "$RUN_DIR" \
  --epochs 40 \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512 \
  --amp

echo ""
echo "===== FINAL OUTPUT FILES ====="
echo "Best checkpoint:"
echo "  $RUN_DIR/checkpoints/best.pt"
echo ""
echo "Training history:"
echo "  $RUN_DIR/history.json"
echo ""
echo "Final target test metrics:"
echo "  $RUN_DIR/target_test_metrics.json"
echo ""
echo "Readable score file:"
echo "  $RUN_DIR/scores/scores_ianbd_test.txt"
echo ""
echo "JSON score file:"
echo "  $RUN_DIR/scores/scores_ianbd_test.json"