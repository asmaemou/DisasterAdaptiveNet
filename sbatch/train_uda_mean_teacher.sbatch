#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=uda_mt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

if [ $# -lt 2 ]; then
  echo "Usage: sbatch train_uda_mean_teacher.sbatch <source_dataset> <target_dataset>"
  echo "Valid dataset names: xbd ida ian rescuenet"
  exit 1
fi

SOURCE_DATASET="$1"
TARGET_DATASET="$2"

if [ "$SOURCE_DATASET" = "$TARGET_DATASET" ]; then
  echo "ERROR: source and target must be different"
  exit 1
fi

case "$SOURCE_DATASET" in
  xbd|ida|ian|rescuenet) ;;
  *) echo "ERROR: invalid source dataset '$SOURCE_DATASET'"; exit 1 ;;
esac

case "$TARGET_DATASET" in
  xbd|ida|ian|rescuenet) ;;
  *) echo "ERROR: invalid target dataset '$TARGET_DATASET'"; exit 1 ;;
esac

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
IDA_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet
IAN_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
RESCUENET_ROOT=/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/uda_${SOURCE_DATASET}_to_${TARGET_DATASET}

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
echo "SOURCE_DATASET:   $SOURCE_DATASET"
echo "TARGET_DATASET:   $TARGET_DATASET"
echo "XBD_ROOT:         $XBD_ROOT"
echo "IDA_ROOT:         $IDA_ROOT"
echo "IAN_ROOT:         $IAN_ROOT"
echo "RESCUENET_ROOT:   $RESCUENET_ROOT"
echo "RUN_DIR:          $RUN_DIR"
nvidia-smi || true
echo "===================="

mkdir -p "$RUN_DIR"

python -u train_uda_mean_teacher.py \
  --source-dataset "$SOURCE_DATASET" \
  --target-dataset "$TARGET_DATASET" \
  --xbd-root "$XBD_ROOT" \
  --ida-root "$IDA_ROOT" \
  --ian-root "$IAN_ROOT" \
  --rescuenet-root "$RESCUENET_ROOT" \
  --output-dir "$RUN_DIR" \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 8 \
  --img-size 512 \
  --save-every 1 \
  --early-stopping-patience 10 \
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
echo "Target test metrics:"
echo "  $RUN_DIR/target_test_metrics.json"
echo ""
echo "Readable score file:"
echo "  $RUN_DIR/scores/scores_${TARGET_DATASET}_test.txt"
echo ""
echo "JSON score file:"
echo "  $RUN_DIR/scores/scores_${TARGET_DATASET}_test.json"
