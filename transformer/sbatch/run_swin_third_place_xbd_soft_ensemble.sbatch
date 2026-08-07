#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xbd_swin_3rd_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
THIRD_BASE=$PROJECT_ROOT/baselines/xview2_winners/xview2_third_place
THIRD_CHECKPOINTS=$THIRD_BASE/weights/extracted/ensemble
SWIN_EXPERIMENT=$PROJECT_ROOT/output/SwinImageNet-DirectxBD_TrainTier3
EXPERIMENT_ROOT=$PROJECT_ROOT/output/SwinThirdPlaceSoftEnsemble_xBD_TrainTier3
THIRD_DATA=$EXPERIMENT_ROOT/third_place_xbd_hold_test_view
SWIN_DATA=$EXPERIMENT_ROOT/swin_xbd_hold_test_view
THIRD_RAW=$EXPERIMENT_ROOT/probabilities/third_place_released_xbd
SWIN_RAW=$EXPERIMENT_ROOT/probabilities/swin_xbd_train_tier3
RESULTS=$EXPERIMENT_ROOT/results

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$THIRD_RAW/val" "$THIRD_RAW/test" "$SWIN_RAW" "$RESULTS"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate disasteradaptivenet_cu128

LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCH_HOME=$LOCAL_JOB_TMP/torch
export HF_HOME=$LOCAL_JOB_TMP/huggingface
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$HF_HOME"

for required in \
  "$XBD_ROOT/hold/images" "$XBD_ROOT/hold/targets" \
  "$XBD_ROOT/test/images" "$XBD_ROOT/test/targets" \
  "$THIRD_BASE/predict_37_weighted.py" "$THIRD_CHECKPOINTS" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt"
do
  test -e "$required" || { echo "ERROR: required input missing: $required"; exit 1; }
done

THIRD_COUNT=$(find "$THIRD_CHECKPOINTS" -maxdepth 1 -type f -name '*.pth' | wc -l)
test "$THIRD_COUNT" -eq 13 || {
  echo "ERROR: expected 13 released third-place xBD checkpoints, found $THIRD_COUNT"
  exit 1
}

echo "===== xBD SWIN-T + RELEASED THIRD-PLACE xVIEW2 SOFT ENSEMBLE ====="
echo "Branch A: ImageNet-pretrained Swin-Tiny trained on xBD train+tier3"
echo "Branch B: released 13-model third-place xView2 xBD ensemble"
echo "Selection: xBD hold only (expected 933 samples)"
echo "Final evaluation: untouched xBD test once (expected 933 samples)"
echo "Objective: official 0.3 localization + 0.7 harmonic damage F1"
echo "Output: $EXPERIMENT_ROOT"
date
hostname
nvidia-smi || true

python -u transformer/scripts/prepare_xbd_third_place_ensemble_view.py \
  --xbd-root "$XBD_ROOT" --output-root "$THIRD_DATA"
python -u transformer/scripts/prepare_xbd_first_place_ensemble_view.py \
  --xbd-root "$XBD_ROOT" --output-root "$SWIN_DATA"
MANIFEST=$SWIN_DATA/official_split_manifest.csv

VAL_COUNT=$(tr -d '\r' < "$MANIFEST" | awk -F, 'NR>1 && $2=="val" {n++} END {print n+0}')
TEST_COUNT=$(tr -d '\r' < "$MANIFEST" | awk -F, 'NR>1 && $2=="test" {n++} END {print n+0}')
test "$VAL_COUNT" -eq 933 || { echo "ERROR: expected 933 xBD hold samples, found $VAL_COUNT"; exit 1; }
test "$TEST_COUNT" -eq 933 || { echo "ERROR: expected 933 xBD test samples, found $TEST_COUNT"; exit 1; }

VAL_CACHE=$(find "$THIRD_RAW/val" -maxdepth 1 -type f -name '*.npz' | wc -l)
TEST_CACHE=$(find "$THIRD_RAW/test" -maxdepth 1 -type f -name '*.npz' | wc -l)
if [ -f "$THIRD_RAW/val/_SUCCESS_BGR_V1" ] && [ -f "$THIRD_RAW/test/_SUCCESS_BGR_V1" ] && \
   [ "$VAL_CACHE" -eq "$VAL_COUNT" ] && [ "$TEST_CACHE" -eq "$TEST_COUNT" ]; then
  echo "Reusing complete released third-place xBD probability cache."
else
  python -u transformer/scripts/export_third_place_texas_probabilities.py \
    --winner-root "$THIRD_BASE" --checkpoint-root "$THIRD_CHECKPOINTS" \
    --data-root "$THIRD_DATA" --output-root "$THIRD_RAW" \
    --splits val test --input-size 512 --output-size 1024 \
    --expected-val-samples "$VAL_COUNT" --expected-test-samples "$TEST_COUNT"
fi

if python -u transformer/scripts/validate_swin_probability_cache.py \
     --root "$SWIN_RAW/val" --manifest "$MANIFEST" --split val \
     --data-root "$SWIN_DATA" --output-size 1024 && \
   python -u transformer/scripts/validate_swin_probability_cache.py \
     --root "$SWIN_RAW/test" --manifest "$MANIFEST" --split test \
     --data-root "$SWIN_DATA" --output-size 1024; then
  echo "Reusing complete validated xBD Swin probability cache."
else
  python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
    --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
    --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
    --data-root "$SWIN_DATA" --output-root "$SWIN_RAW" \
    --split-manifest "$MANIFEST" --splits val test \
    --output-size 1024 --geometry top-left-pad \
    --num-workers 0 --device cuda --overwrite
fi

python -u transformer/scripts/eval_swin_third_place_texas_soft_ensemble.py \
  --swin-root "$SWIN_RAW" --third-place-root "$THIRD_RAW" \
  --output-dir "$RESULTS" \
  --expected-val-samples "$VAL_COUNT" --expected-test-samples "$TEST_COUNT" \
  --minimum-third-place-val-loc-f1 0.8 \
  --selection-objective official --minor-dilation-kernel 1 --selection-stride 8 \
  --alphas 0,0.25,0.5,0.75,1 \
  --betas 0,0.25,0.5,0.75,1 \
  --thresholds 0.3,0.4,0.5,0.6,0.7 \
  --experiment-label "xBD train+tier3 Swin-Tiny + released third-place xView2 xBD soft ensemble" \
  --selection-label "Fusion selected only on xBD hold; untouched xBD test evaluated once."

cat "$RESULTS/ensemble_metrics.txt"
echo "DONE: xBD Swin-T + released third-place xView2 soft ensemble"
date
