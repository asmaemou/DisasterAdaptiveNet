#!/bin/bash
# Submit only after all five training array tasks completed successfully.
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=tx_swindf_eval
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
ENSEMBLE_ROOT=$PROJECT_ROOT/output/SwinDamageFormerEnsemble_TEXAS
OUTPUT_DIR=$ENSEMBLE_ROOT/ensemble_scores
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

CHECKPOINTS=(
  "$ENSEMBLE_ROOT/member_42/checkpoints/best.pt"
  "$ENSEMBLE_ROOT/member_3407/checkpoints/best.pt"
  "$ENSEMBLE_ROOT/member_7919/checkpoints/best.pt"
  "$ENSEMBLE_ROOT/member_11003/checkpoints/best.pt"
  "$ENSEMBLE_ROOT/member_15401/checkpoints/best.pt"
)
for checkpoint in "${CHECKPOINTS[@]}"; do
  test -s "$checkpoint" || { echo "ERROR: missing ensemble checkpoint: $checkpoint"; exit 1; }
done

echo "===== VALIDATION-CALIBRATED FIVE-MODEL, EIGHT-VIEW TTA ENSEMBLE ====="
date
hostname
nvidia-smi || true

python -u transformer/scripts/eval_swin_damageformer_ensemble.py \
  --data-root "$DATA_ROOT" --val-split val --test-split test \
  --checkpoints "${CHECKPOINTS[@]}" \
  --output-dir "$OUTPUT_DIR" --img-size 896 --batch-size 1 --num-workers 0 \
  --tta identity hflip vflip rot90 rot180 rot270 transpose anti_transpose \
  --thresholds 0.35 0.45 0.55 0.65 0.75 \
  --ordinal-blends 0.0 0.25 0.5 \
  --morph-kernels 0 3 5 --min-building-pixels 0 8 16 --object-vote 0 1 \
  --class-boosts 1 1 1 1 \
  --class-boosts 1 1.25 1.25 1 \
  --class-boosts 1 1.5 1.5 1.1

echo "DONE: Texas SwinDamageFormer ensemble evaluation"
date
