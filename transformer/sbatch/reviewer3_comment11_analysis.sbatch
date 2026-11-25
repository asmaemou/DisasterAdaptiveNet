#!/usr/bin/env bash
# =============================================================================
# Reviewer #3, Comment (11) — RS-BDNet xBD minor-damage error analysis
# =============================================================================
# Goal:
#   Re-evaluate the existing, frozen epoch-35 RS-BDNet checkpoint on xBD Test.
#   Produce pixel-level minor-damage precision, recall, F1, and a breakdown of
#   missed localization vs. confusion with no/major/destroyed damage.
#
# No training or checkpoint modification occurs.
# Original scores are reproduced and checked by the Python analysis script.
# =============================================================================
#SBATCH --job-name=R3C11_xBD_minor
#SBATCH --partition=wsu_soc_kshirsagar.q
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --chdir=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
#SBATCH --output=slurm-R3C11-%j.out
#SBATCH --error=slurm-R3C11-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail

PROJECT_ROOT="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet"
ENV_NAME="disasteradaptivenet_cu128"
ANALYSIS_MODULE="transformer.scripts.reviewer3_comment11_analysis"
ANALYSIS_FILE="transformer/scripts/reviewer3_comment11_analysis.py"
CHECKPOINT="output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"
EXPECTED_DATA_DIR="/homes/j244s673/documents/wsu/phd/xview2"
RESULT_DIR="output/Reviewer3_Comment11"

on_exit() {
    rc=$?
    echo "------------------------------------------------------------------"
    echo "Job ended (exit code: ${rc}) at $(date --iso-8601=seconds)"
    if [[ ${rc} -eq 0 ]]; then
        echo "SUCCESS: ${PROJECT_ROOT}/${RESULT_DIR}"
    else
        echo "FAILURE: Review slurm-R3C11-${SLURM_JOB_ID:-unknown}.err and .out"
    fi
}
trap on_exit EXIT

cd "$PROJECT_ROOT"
echo "=================================================================="
echo "Reviewer #3 Comment (11) | xBD minor-damage error analysis"
echo "Start time:       $(date --iso-8601=seconds)"
echo "Slurm job ID:     ${SLURM_JOB_ID:-not_set}"
echo "Compute node:     $(hostname)"
echo "Partition:        ${SLURM_JOB_PARTITION:-unknown}"
echo "Allocated GPUs:   ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPU cores:        ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Project:          $PROJECT_ROOT"
echo "Python module:    $ANALYSIS_MODULE"
echo "Checkpoint:       $CHECKPOINT"
echo "Ground-truth set: $EXPECTED_DATA_DIR (Test split, 933 pairs)"
echo "Epoch:            35"
echo "Loc. threshold:   0.50 (selected on Hold)"
echo "Outputs:          $RESULT_DIR"
echo "Evaluation unit:  pixels, NOT individual building polygons"
echo "=================================================================="

# Check inputs before loading model or writing analysis outputs.
[[ -f "$ANALYSIS_FILE" ]] || { echo "ERROR: Missing $ANALYSIS_FILE" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "ERROR: Missing $CHECKPOINT" >&2; exit 2; }
[[ -d "$EXPECTED_DATA_DIR/test" ]] || { echo "ERROR: Missing xBD Test directory" >&2; exit 2; }
[[ ! -e "$RESULT_DIR" ]] || {
    echo "ERROR: $RESULT_DIR already exists. The analysis refuses to overwrite it." >&2
    exit 2
}

# The Miniforge environment is installed in the user's home directory.
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || { echo "ERROR: Missing $CONDA_SH" >&2; exit 2; }
source "$CONDA_SH"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

printf '\n--- Environment / GPU ---\n'
which python
python --version
nvidia-smi
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('PyTorch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: No CUDA GPU visible inside this Slurm job')
print('GPU:', torch.cuda.get_device_name(0))
PY

printf '\n--- Input SHA256 checksums (for reproducibility) ---\n'
sha256sum "$CHECKPOINT" "$ANALYSIS_FILE"

printf '\n--- Running frozen-checkpoint evaluation ---\n'
python -m "$ANALYSIS_MODULE"

printf '\n--- Generated results ---\n'
ls -lah "$RESULT_DIR"
cat "$RESULT_DIR/minor_damage_analysis.json"
printf '\n--- Confusion matrix ---\n'
cat "$RESULT_DIR/confusion_matrix.csv"
