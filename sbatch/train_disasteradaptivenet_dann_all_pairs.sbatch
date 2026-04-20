#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=dan_dann_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

PY_SCRIPT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/train_disasteradaptivenet_dann_with_irma.py

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
IDA_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet
IAN_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
IRMA_ROOT=/homes/j244s673/documents/wsu/phd/irma_disasteradaptivenet

BASE_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output

if [ ! -f "$PY_SCRIPT" ]; then
  echo "ERROR: Python script not found: $PY_SCRIPT"
  exit 1
fi

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
echo "PY_SCRIPT:  $PY_SCRIPT"
echo "XBD_ROOT:   $XBD_ROOT"
echo "IDA_ROOT:   $IDA_ROOT"
echo "IAN_ROOT:   $IAN_ROOT"
echo "IRMA_ROOT:  $IRMA_ROOT"
echo "BASE_OUT:   $BASE_OUT"
nvidia-smi || true
echo "===================="

SUMMARY_TXT="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_summary.txt"
SUMMARY_CSV="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_summary.csv"

echo "job_id,source,target,status,exit_code,run_dir" > "$SUMMARY_CSV"
echo "Summary for job $SLURM_JOB_ID" > "$SUMMARY_TXT"
echo "" >> "$SUMMARY_TXT"

DATASETS=(xbd ida ian irma)

run_one_pair () {
  local SRC="$1"
  local TGT="$2"
  local RUN_DIR="${BASE_OUT}/disasteradaptivenet_dann_${SRC}_to_${TGT}"
  local PAIR_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SRC}_to_${TGT}.log"

  mkdir -p "$RUN_DIR"

  echo ""
  echo "======================================================"
  echo "RUNNING: ${SRC} -> ${TGT}"
  echo "RUN_DIR:  ${RUN_DIR}"
  echo "PAIR_LOG: ${PAIR_LOG}"
  echo "START:    $(date)"
  echo "======================================================"

  {
    echo "RUNNING: ${SRC} -> ${TGT}"
    echo "START: $(date)"
    echo "RUN_DIR: ${RUN_DIR}"
    echo ""
  } > "$PAIR_LOG"

  set +e
  python -u "$PY_SCRIPT" \
    --source-dataset "$SRC" \
    --target-dataset "$TGT" \
    --xbd-root "$XBD_ROOT" \
    --ida-root "$IDA_ROOT" \
    --ian-root "$IAN_ROOT" \
    --irma-root "$IRMA_ROOT" \
    --output-dir "$RUN_DIR" \
    --epochs 100 \
    --batch-size 8 \
    --num-workers 8 \
    --img-size 512 \
    --save-every 1 \
    --early-stopping-patience 10 \
    --amp 2>&1 | tee -a "$PAIR_LOG"

  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="success"
  else
    STATUS="failed"
  fi

  echo ""
  echo "END: $(date)" | tee -a "$PAIR_LOG"
  echo "STATUS: $STATUS" | tee -a "$PAIR_LOG"
  echo "EXIT_CODE: $EXIT_CODE" | tee -a "$PAIR_LOG"

  echo "SOURCE=${SRC} TARGET=${TGT} STATUS=${STATUS} EXIT_CODE=${EXIT_CODE} RUN_DIR=${RUN_DIR}" >> "$SUMMARY_TXT"
  echo "${SLURM_JOB_ID},${SRC},${TGT},${STATUS},${EXIT_CODE},${RUN_DIR}" >> "$SUMMARY_CSV"

  echo "Finished ${SRC} -> ${TGT} with status=${STATUS}"

  return 0
}

for SRC in "${DATASETS[@]}"; do
  for TGT in "${DATASETS[@]}"; do
    if [ "$SRC" = "$TGT" ]; then
      continue
    fi
    run_one_pair "$SRC" "$TGT"
  done
done

echo ""
echo "===== ALL RUNS FINISHED ====="
echo "Summary text: $SUMMARY_TXT"
echo "Summary csv:  $SUMMARY_CSV"