#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=dan_mada
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

if [ $# -lt 2 ]; then
  echo "Usage: sbatch train_disasteradaptivenet_mada_style.sbatch <comma_separated_sources> <target>"
  echo "Example: sbatch train_disasteradaptivenet_mada_style.sbatch xbd,ida rescuenet"
  echo "Valid dataset names: xbd ida ian rescuenet"
  exit 1
fi

SOURCES="$1"
TARGET="$2"

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
IDA_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet
IAN_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
RESCUENET_ROOT=/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/disasteradaptivenet_mada_${SOURCES//,/plus}_to_${TARGET}

# Optional: if you later train CycleGAN/CycleGAN+classifier translations, point these at translated roots.
TRANSLATED_XBD_ROOT=""
TRANSLATED_IDA_ROOT=""
TRANSLATED_IAN_ROOT=""
TRANSLATED_RESCUENET_ROOT=""

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
echo "SOURCES:                $SOURCES"
echo "TARGET:                 $TARGET"
echo "XBD_ROOT:               $XBD_ROOT"
echo "IDA_ROOT:               $IDA_ROOT"
echo "IAN_ROOT:               $IAN_ROOT"
echo "RESCUENET_ROOT:         $RESCUENET_ROOT"
echo "RUN_DIR:                $RUN_DIR"
nvidia-smi || true
echo "===================="

mkdir -p "$RUN_DIR"

python -u train_disasteradaptivenet_mada_style.py \
  --sources "$SOURCES" \
  --target "$TARGET" \
  --xbd-root "$XBD_ROOT" \
  --ida-root "$IDA_ROOT" \
  --ian-root "$IAN_ROOT" \
  --rescuenet-root "$RESCUENET_ROOT" \
  --translated-xbd-root "$TRANSLATED_XBD_ROOT" \
  --translated-ida-root "$TRANSLATED_IDA_ROOT" \
  --translated-ian-root "$TRANSLATED_IAN_ROOT" \
  --translated-rescuenet-root "$TRANSLATED_RESCUENET_ROOT" \
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
echo "  $RUN_DIR/scores/scores_${TARGET}_test.txt"
echo ""
echo "JSON score file:"
echo "  $RUN_DIR/scores/scores_${TARGET}_test.json"
