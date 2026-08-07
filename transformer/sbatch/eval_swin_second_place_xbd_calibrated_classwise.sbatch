#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=xbd_2nd_swin_cal
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SOURCE=$PROJECT_ROOT/output/SwinSecondPlaceSoftEnsemble_xBD_TrainTier3/probabilities
OUTPUT=$PROJECT_ROOT/output/SwinSecondPlaceCalibratedClasswise_xBD_TrainTier3/results
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate disasteradaptivenet_cu128
for required in "$SOURCE/swin_xbd_train_tier3/val" "$SOURCE/swin_xbd_train_tier3/test" \
  "$SOURCE/second_place_released_xbd/val/_SUCCESS_V1" "$SOURCE/second_place_released_xbd/test/_SUCCESS_V1"
do
  test -e "$required" || { echo "ERROR: missing completed cache: $required"; exit 1; }
done
for cache in "$SOURCE/swin_xbd_train_tier3/val" "$SOURCE/swin_xbd_train_tier3/test" \
  "$SOURCE/second_place_released_xbd/val" "$SOURCE/second_place_released_xbd/test"
do
  count=$(find "$cache" -maxdepth 1 -type f -name '*.npz' | wc -l)
  test "$count" -eq 933 || { echo "ERROR: expected 933 cached samples under $cache, found $count"; exit 1; }
done
python -u transformer/scripts/eval_swin_npz_winner_xbd_calibrated_classwise.py \
  --swin-root "$SOURCE/swin_xbd_train_tier3" \
  --winner-root "$SOURCE/second_place_released_xbd" \
  --winner-name second_place --winner-mode threshold \
  --output-dir "$OUTPUT" --expected-val-samples 933 --expected-test-samples 933 --selection-stride 8
cat "$OUTPUT/calibrated_classwise_metrics.txt"
echo "DONE: xBD calibrated class-wise Swin-T + second-place experiment"
