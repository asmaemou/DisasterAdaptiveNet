#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=53-00:00:00
#SBATCH --job-name=reviewer3_comment11
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/reviewer3_comment11_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/reviewer3_comment11_%j.err

# Reviewer #3, Comment (11): pixel-level minor-damage error breakdown.
# This NEW job has an isolated output directory and does NOT modify/overwrite
# the original analysis file, checkpoint, or previous job's output directory.

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SOURCE="$ROOT/transformer/scripts/reviewer3_comment11_analysis.py"
RUN_NAME="reviewer3_comment11_${SLURM_JOB_ID:?Submit with sbatch}"
export RUN_OUTPUT_DIR="$ROOT/output/$RUN_NAME"
RUN_SCRIPT_DIR="$ROOT/output/reviewer3_comment11_launchers"
export RUN_SCRIPT="$RUN_SCRIPT_DIR/${RUN_NAME}_analysis.py"

cd "$ROOT"
test -f "$SOURCE" || { echo "Missing source script: $SOURCE" >&2; exit 1; }
if [ -e "$RUN_OUTPUT_DIR" ]; then
    echo "Refusing to overwrite existing output directory: $RUN_OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_SCRIPT_DIR"

# Make a job-specific copy. The original analysis has a hard-coded output path;
# changing only the SLURM log name would NOT protect its result files.
# Disable cuDNN in this isolated copy for the RTX PRO 6000 Blackwell GPU,
# matching the configuration used by the working RS-BDNet GPU benchmark.
python - <<'PY'
from pathlib import Path
import os

root = Path('/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet')
original = root / 'transformer/scripts/reviewer3_comment11_analysis.py'
copy = Path(os.environ['RUN_SCRIPT'])
output = Path(os.environ['RUN_OUTPUT_DIR'])
source = original.read_text(encoding='utf-8')
old = 'output_dir = Path("output/Reviewer3_Comment11")'
new = f'output_dir = Path({str(output)!r})'
if source.count(old) != 1:
    raise RuntimeError('Expected original output_dir assignment was not found exactly once. Refusing unsafe run.')
source = source.replace(old, new, 1)
marker = 'import torch\n'
if source.count(marker) != 1:
    raise RuntimeError('Expected torch import was not found exactly once. Refusing unsafe run.')
source = source.replace(marker,
    marker + 'torch.backends.cudnn.enabled = False\n'
           + 'torch.backends.cudnn.benchmark = False\n', 1)
copy.write_text(source, encoding='utf-8')
print('Preserved original script:', original)
print('Job-specific script:', copy)
print('Job-specific output directory:', output)
PY

# Ensure imports work when the isolated script is under output/, not transformer/.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
    CONDA_EXE="$HOME/miniforge/bin/conda"
else
    CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo '===== Reviewer #3, Comment (11): ISOLATED LONG RUN ====='
echo "Job ID:       $SLURM_JOB_ID"
echo "Run output:   $RUN_OUTPUT_DIR"
echo "Frozen script: $RUN_SCRIPT"
echo 'Wall time:    56 days (partition maximum; subject to QOS/account limits)'
date
hostname
nvidia-smi
sha256sum "$SOURCE" "$RUN_SCRIPT" \
  "$ROOT/output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"

# No output directory is created until the script finishes evaluating and its
# exact reference metric validation passes. It refuses to overwrite results.
python -u "$RUN_SCRIPT"

echo "SUCCESS: outputs saved under $RUN_OUTPUT_DIR"
date
