#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=idabd_gate_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found."
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd
SPLIT_FILE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/idabd_splits_seed42_80_10_10.json

BASE_PHASE1=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/checkpoints/phase1_best.pt
BASE_PHASE2=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/checkpoints/phase2_best.pt

DANN_PHASE1=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp/checkpoints/phase1_best.pt
DANN_PHASE2=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_xbd_to_idabd_hrtbda_v5_foreground_dann/checkpoints/phase2_best_da.pt

OUTPUT_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/eval_idabd_v2_plus_dann_gated_ensemble
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/scores"

python -u transformer/scripts/eval_idabd_v2_dann_gated_ensemble.py \
  --idabd-root "$IDABD_ROOT" \
  --split-file "$SPLIT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --base-phase1-checkpoint "$BASE_PHASE1" \
  --base-phase2-checkpoint "$BASE_PHASE2" \
  --dann-phase1-checkpoint "$DANN_PHASE1" \
  --dann-phase2-checkpoint "$DANN_PHASE2" \
  --base-channels 48 \
  --decoder-channels 128 \
  --window-size 8 \
  --img-size 1024 \
  --eval-batch-size 1 \
  --num-workers 8 \
  --base-thresholds 0.40 0.45 0.50 0.55 0.60 0.65 \
  --dann-thresholds 0.30 0.40 0.50 \
  --minor-conf-thresholds 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90
