#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=turkey_swin_1stxbd
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
FIRST_BASE=$PROJECT_ROOT/baselines/xview2_winners/xview2_first_place
FIRST_DATA=$PROJECT_ROOT/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT
FIRST_XBD_SOURCE=$FIRST_BASE/weights/extracted_weights/weights
SWIN_EXPERIMENT=$PROJECT_ROOT/output/HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY

HYBRID_ROOT=$PROJECT_ROOT/output/hybrid_swin_hrtbda_first_place_xbd_zero_shot_turkey
FIRST_XBD_ALIASES=$HYBRID_ROOT/first_place_xbd_inference_weights
FIRST_RAW=$HYBRID_ROOT/probabilities/first_place_xbd_zero_shot
SWIN_RAW=$HYBRID_ROOT/probabilities/swin_hrtbda
RESULTS=$HYBRID_ROOT/results

PREDICTION_DIRS=(
  pred50_loc_tuned pred92_loc_tuned pred34_loc pred154_loc
  dpn92cls_cce_0_tuned dpn92cls_cce_1_tuned dpn92cls_cce_2_tuned
  res34cls2_0_tuned res34cls2_1_tuned res34cls2_2_tuned
  res50cls_cce_0_tuned res50cls_cce_1_tuned res50cls_cce_2_tuned
  se154cls_0_tuned se154cls_1_tuned se154cls_2_tuned
)

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$FIRST_RAW" "$SWIN_RAW" "$RESULTS"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found"
  exit 1
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== SWIN-HRTBDA + ORIGINAL FIRST-PLACE xBD LATE FUSION ====="
date
hostname
nvidia-smi || true
echo "Branch A: ImageNet-initialized Swin-HRTBDA trained directly on Earthquake Turkey"
echo "Branch B: original first-place xView2 xBD checkpoints, zero-shot on Earthquake Turkey"
echo "Fusion selection: 93-sample Turkey validation set only"
echo "Final evaluation: untouched 95-sample Turkey test set"
echo "Output: $HYBRID_ROOT"

for required in \
  "$FIRST_DATA/val/images" \
  "$FIRST_DATA/test/images" \
  "$FIRST_DATA/official_split_manifest.csv" \
  "$FIRST_XBD_SOURCE" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt"
do
  if [ ! -e "$required" ]; then
    echo "ERROR: required input is missing: $required"
    exit 1
  fi
done

echo "===== Map predictor names to ORIGINAL first-place xBD weights ====="
python -u transformer/scripts/setup_first_place_xbd_inference_weights.py \
  --source "$FIRST_XBD_SOURCE" \
  --output "$FIRST_XBD_ALIASES"

echo "===== Prepare first-place inference scripts ====="
cd "$FIRST_BASE"
python -u custom/patch_first_place_official_split.py
export MODELS_FOLDER="$FIRST_XBD_ALIASES"
export SUBMISSION_WORKERS=4

cleanup_prediction_dirs() {
  rm -rf submission
  for directory in "${PREDICTION_DIRS[@]}"; do
    rm -rf "$directory"
  done
}

restore_test_link() {
  cd "$FIRST_BASE"
  rm -rf test
  ln -sfn "$FIRST_DATA/test" test
}
trap restore_test_link EXIT

generate_first_place_probabilities() {
  local split=$1
  local destination=$FIRST_RAW/$split
  if [ -f "$destination/_SUCCESS" ]; then
    echo "Reusing completed original first-place xBD probabilities: $destination"
    return
  fi

  echo "===== Original first-place xBD zero-shot inference: $split ====="
  cleanup_prediction_dirs
  rm -rf test
  ln -sfn "$FIRST_DATA/$split" test
  bash predict.sh

  rm -rf "$destination"
  mkdir -p "$destination"
  for directory in "${PREDICTION_DIRS[@]}"; do
    if [ ! -d "$directory" ]; then
      echo "ERROR: first-place probability folder missing: $directory"
      exit 1
    fi
    mv "$directory" "$destination/$directory"
  done
  touch "$destination/_SUCCESS"
  rm -rf submission
}

generate_first_place_probabilities val
generate_first_place_probabilities test
restore_test_link
trap - EXIT

echo "===== Export Swin-HRTBDA DirectTurkey validation/test probabilities ====="
cd "$PROJECT_ROOT"
python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
  --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
  --data-root /homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed \
  --output-root "$SWIN_RAW" \
  --split-manifest "$FIRST_DATA/official_split_manifest.csv" \
  --splits val test \
  --output-size 1024 \
  --num-workers 2 \
  --device cuda

echo "===== Select a SMALL fusion grid on validation; evaluate test once ====="
python -u transformer/scripts/eval_hrtbda_first_place_turkey_soft_ensemble.py \
  --hrtbda-root "$SWIN_RAW" \
  --first-place-root "$FIRST_RAW" \
  --output-dir "$RESULTS" \
  --alphas 0.25,0.5,0.75 \
  --betas 0.25,0.5,0.75 \
  --thresholds 0.4,0.5,0.6,0.7 \
  --expected-val-samples 93 \
  --expected-test-samples 95 \
  --experiment-label "Turkey-trained Swin-HRTBDA + original first-place xView2 xBD zero-shot soft ensemble" \
  --hrtbda-label "ImageNet-initialized Swin-HRTBDA trained directly on Earthquake Turkey" \
  --first-place-label "Original first-place xView2 CNN ensemble trained on xBD (Turkey zero-shot)"

cat "$RESULTS/hybrid_metrics_summary.txt"
echo "DONE: Swin-HRTBDA DirectTurkey + original first-place xBD probability ensemble"
date
