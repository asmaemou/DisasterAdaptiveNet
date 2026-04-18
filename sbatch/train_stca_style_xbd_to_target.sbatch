#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=stca_xbd_multi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

MODE="${1:-all}"

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p logs

XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
IDA_ROOT=/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet
IAN_ROOT=/homes/j244s673/documents/wsu/phd/idabd_disasteradaptivenet
RESCUENET_ROOT=/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd
BASE_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/stca_style

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
echo "MODE:            $MODE"
echo "XBD_ROOT:        $XBD_ROOT"
echo "IDA_ROOT:        $IDA_ROOT"
echo "IAN_ROOT:        $IAN_ROOT"
echo "RESCUENET_ROOT:  $RESCUENET_ROOT"
echo "BASE_OUT:        $BASE_OUT"
nvidia-smi || true
echo "===================="

mkdir -p "$BASE_OUT"

declare -a TARGETS
if [ "$MODE" = "all" ]; then
  TARGETS=(ida ian rescuenet)
else
  case "$MODE" in
    ida|ian|rescuenet) TARGETS=($MODE) ;;
    *)
      echo "ERROR: target must be one of: ida ian rescuenet all"
      exit 1
      ;;
  esac
fi

SUMMARY_TXT="$BASE_OUT/summary_job_${SLURM_JOB_ID}.txt"
SUMMARY_JSONL="$BASE_OUT/summary_job_${SLURM_JOB_ID}.jsonl"
: > "$SUMMARY_TXT"
: > "$SUMMARY_JSONL"

for TARGET in "${TARGETS[@]}"; do
  RUN_DIR="$BASE_OUT/xbd_to_${TARGET}"
  mkdir -p "$RUN_DIR"

  echo ""
  echo "===== RUNNING TARGET: $TARGET ====="
  echo "===== RUNNING TARGET: $TARGET =====" >> "$SUMMARY_TXT"

  python -u train_stca_style_xbd_to_target.py \
    --target-dataset "$TARGET" \
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
    --amp || true

  if [ -f "$RUN_DIR/target_test_metrics.json" ]; then
    echo "STATUS: completed" >> "$SUMMARY_TXT"
    cat "$RUN_DIR/target_test_metrics.json" >> "$SUMMARY_JSONL"
    echo "" >> "$SUMMARY_JSONL"
    echo "Readable score file: $RUN_DIR/scores/scores_${TARGET}_test.txt" >> "$SUMMARY_TXT"
    cat "$RUN_DIR/scores/scores_${TARGET}_test.txt" >> "$SUMMARY_TXT"
  elif [ -f "$RUN_DIR/stca_skip.json" ]; then
    echo "STATUS: skipped" >> "$SUMMARY_TXT"
    cat "$RUN_DIR/stca_skip.json" >> "$SUMMARY_JSONL"
    echo "" >> "$SUMMARY_JSONL"
    cat "$RUN_DIR/stca_skip.json" >> "$SUMMARY_TXT"
  else
    echo "STATUS: failed before producing metrics or skip file" >> "$SUMMARY_TXT"
  fi
  echo "" >> "$SUMMARY_TXT"
done

echo "===== FINAL SUMMARY ====="
cat "$SUMMARY_TXT"
echo "Summary TXT:   $SUMMARY_TXT"
echo "Summary JSONL: $SUMMARY_JSONL"
