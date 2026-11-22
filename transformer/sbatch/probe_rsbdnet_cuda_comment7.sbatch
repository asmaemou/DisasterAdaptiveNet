#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=00:08:00
#SBATCH --job-name=rsbd_r2_c7_probe
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err
set -uo pipefail
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$ROOT" || exit 1
if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)" || exit 1
conda activate disasteradaptivenet_cu128 || exit 1
nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu --format=csv
GPU_LOG="$ROOT/transformer/sbatch/logs/rsbd_r2_c7_probe_${SLURM_JOB_ID}_gpu.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 3 > "$GPU_LOG" 2>&1 &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT
for test in conv_default conv_no_cudnn model_default model_no_cudnn; do
  echo
  echo "========== START ${test} =========="
  timeout --signal=TERM --kill-after=5s 65s \
    python -u "$ROOT/transformer/scripts/probe_rsbdnet_cuda_comment7.py" --test "$test"
  rc=$?
  echo "========== END ${test}; exit_code=${rc} =========="
  if [ "$rc" -eq 124 ]; then
    echo "TEST_TIMED_OUT: ${test} (65-second per-test safety limit)"
  elif [ "$rc" -ne 0 ]; then
    echo "TEST_FAILED: ${test}; inspect .err log"
  fi
done
printf '\nDiagnostic complete. GPU sampling log: %s\n' "$GPU_LOG"
