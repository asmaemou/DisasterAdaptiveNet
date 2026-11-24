#!/usr/bin/env bash
#SBATCH --job-name=RSBD_Rev3_C7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=rsbdnet_comment7_%j.out
#SBATCH --error=rsbdnet_comment7_%j.err

# Reviewer #3, Comment (7): FULL xBD test-set analysis, both trained models.
# No retraining. Run using: sbatch reviewer3_comment7_xbd_analysis.sbatch
set -euo pipefail

PROJECT="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet"
XBD="/homes/j244s673/documents/wsu/phd/xview2"
OUT="$PROJECT/output/Reviewer3_Comment7"
SCRIPT="$PROJECT/transformer/scripts/reviewer3_comment7_xbd_analysis.py"
PYTHON_BIN="/homes/j244s673/miniforge3/envs/disasteradaptivenet_cu128/bin/python"

cd "$PROJECT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Use the exact working training environment even when SLURM doesn't inherit
# an interactive Conda activation (system python imports a broken 'torch').
"$PYTHON_BIN" - <<'PY'
import sys
import torch, cv2, numpy, PIL
print('Python executable:', sys.executable, flush=True)
print('Environment check OK: torch=', torch.__version__,
      'cv2=', cv2.__version__, 'cuda=', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('No usable CUDA GPU visible in this SLURM job.')
PY

# --disable-cudnn is required: the earlier full CUDA forward pass stalled.
# No --limit: processes all 933 xBD test image pairs for BOTH models.
# The existing manual worksheet is an input, not overwritten.
"$PYTHON_BIN" "$SCRIPT" \
  --project-root "$PROJECT" \
  --xbd-root "$XBD" \
  --output-dir "$OUT" \
  --manual-sheet "$OUT/manual_review_worksheet.csv" \
  --disable-cudnn

printf '\nReviewer #3 Comment (7) analysis finished. Output: %s\n' "$OUT"
