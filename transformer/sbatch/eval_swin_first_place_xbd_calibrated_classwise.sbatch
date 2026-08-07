#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=xbd_1st_swin_cal
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SOURCE=$PROJECT_ROOT/output/SwinFirstPlaceSoftEnsemble_xBD_TrainTier3/probabilities
OUTPUT=$PROJECT_ROOT/output/SwinFirstPlaceCalibratedClasswise_xBD_TrainTier3/results

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate disasteradaptivenet_cu128

for required in \
  "$SOURCE/swin/val" "$SOURCE/swin/test" \
  "$SOURCE/first_place/val/_SUCCESS" "$SOURCE/first_place/test/_SUCCESS"
do
  test -e "$required" || {
    echo "ERROR: required completed probability cache missing: $required"
    echo "Finish run_swin_first_place_xbd_soft_ensemble.sbatch first."
    exit 1
  }
done

echo "===== xBD SWIN-T + FIRST-PLACE CALIBRATED CLASS-WISE FUSION ====="
echo "No training or inference is repeated; complete cached probabilities are reused."
echo "All calibration, weights, and fallback decisions use xBD hold only."
echo "The untouched xBD test split is evaluated once."
echo "Output: $OUTPUT"
date
hostname

python -u transformer/scripts/eval_swin_first_place_xbd_calibrated_classwise.py \
  --swin-root "$SOURCE/swin" \
  --first-place-root "$SOURCE/first_place" \
  --output-dir "$OUTPUT" \
  --expected-val-samples 933 --expected-test-samples 933 \
  --selection-stride 8

cat "$OUTPUT/calibrated_classwise_metrics.txt"
echo "DONE: xBD calibrated class-wise Swin-T + first-place experiment"
date
