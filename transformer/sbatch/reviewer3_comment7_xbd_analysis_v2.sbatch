#!/bin/bash
#SBATCH --job-name=RSBD_Rev3_C7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=rsbdnet_comment7_%j.out
#SBATCH --error=rsbdnet_comment7_%j.err

# Run: sbatch reviewer3_comment7_xbd_analysis.sbatch
# Submit from the DisasterAdaptiveNet repository directory. This job performs
# inference on xBD test data; it DOES NOT train or modify model checkpoints.
set -euo pipefail

PROJECT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
XBD=/homes/j244s673/documents/wsu/phd/xview2
OUT="$PROJECT/output/Reviewer3_Comment7"
SCRIPT="$PROJECT/transformer/scripts/reviewer3_comment7_xbd_analysis.py"

# If your Slurm job does NOT inherit an environment with torch/cv2/Pillow,
# uncomment and edit ONE of the following to use your existing training env:
# source /homes/j244s673/path/to/your/venv/bin/activate
# module load anaconda
# conda activate your-rsbdnet-environment

cd "$PROJECT"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
PYTHON_BIN="${PYTHON_BIN:-/homes/j244s673/miniforge3/envs/disasteradaptivenet_cu128/bin/python}"

"$PYTHON_BIN" - <<'PY'
import torch, cv2, numpy, PIL
print('Environment check OK: torch=', torch.__version__,
      'cv2=', cv2.__version__, 'cuda=', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('Slurm did not allocate a usable GPU.')
PY

"$PYTHON_BIN" "$SCRIPT" \
  --project-root "$PROJECT" \
  --xbd-root "$XBD" \
  --output-dir "$OUT" \
  --disable-cudnn

# cuDNN is disabled because the first GPU forward stalled with it enabled.
# If the model or original preprocessing differ, the Python script stops with an
# explicit error rather than silently reporting misleading statistics.
