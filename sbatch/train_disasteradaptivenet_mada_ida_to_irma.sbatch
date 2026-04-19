#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --job-name=dan_mada_ida_irma
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

IDA_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet
# EDIT THIS PATH if your standardized IRMA dataset is elsewhere.
IRMA_ROOT=/homes/j244s673/documents/wsu/phd/HURRICANE-IRMA_disasteradaptivenet
# OPTIONAL: if you later create a CycleGAN-translated IDA source in IRMA style,
# set this path; otherwise leave it empty.
TRANSLATED_IDA_ROOT=
RUN_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/disasteradaptivenet_mada_ida_to_irma

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

eval "$($CONDA_EXE shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== JOB INFO ====="
date
hostname
pwd
which python
python --version
echo "IDA_ROOT:            $IDA_ROOT"
echo "IRMA_ROOT:           $IRMA_ROOT"
echo "TRANSLATED_IDA_ROOT: ${TRANSLATED_IDA_ROOT:-<empty>}"
echo "RUN_DIR:             $RUN_DIR"
nvidia-smi || true
echo "===================="

mkdir -p "$RUN_DIR"

PY_ARGS=(
  --ida-root "$IDA_ROOT"
  --irma-root "$IRMA_ROOT"
  --output-dir "$RUN_DIR"
  --epochs 100
  --batch-size 8
  --num-workers 8
  --img-size 512
  --save-every 1
  --early-stopping-patience 10
  --amp
)

if [ -n "${TRANSLATED_IDA_ROOT}" ]; then
  PY_ARGS+=(--translated-ida-root "$TRANSLATED_IDA_ROOT")
fi

python -u train_disasteradaptivenet_mada_ida_to_irma.py "${PY_ARGS[@]}"

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
echo "  $RUN_DIR/scores/scores_irma_test.txt"
echo ""
echo "JSON score file:"
echo "  $RUN_DIR/scores/scores_irma_test.json"
