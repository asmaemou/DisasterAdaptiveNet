#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=turkey_hybrid_1st
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
FIRST_BASE=$PROJECT_ROOT/baselines/xview2_winners/xview2_first_place
FIRST_DATA=$PROJECT_ROOT/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT
FIRST_WEIGHTS=$FIRST_BASE/weights/earthquake_turkey_finetuned_weights_official_split
HRT_EXPERIMENT=$PROJECT_ROOT/output/HRTBDA-v5-xBDInit-FullFineTune_EARTHQUAKE_TURKEY_resetbest
HYBRID_ROOT=$PROJECT_ROOT/output/hybrid_hrtbda_first_place_turkey
FIRST_RAW=$HYBRID_ROOT/probabilities/first_place
HRT_RAW=$HYBRID_ROOT/probabilities/hrtbda
RESULTS=$HYBRID_ROOT/results

PREDICTION_DIRS=(
  pred50_loc_tuned pred92_loc_tuned pred34_loc pred154_loc
  dpn92cls_cce_0_tuned dpn92cls_cce_1_tuned dpn92cls_cce_2_tuned
  res34cls2_0_tuned res34cls2_1_tuned res34cls2_2_tuned
  res50cls_cce_0_tuned res50cls_cce_1_tuned res50cls_cce_2_tuned
  se154cls_0_tuned se154cls_1_tuned se154cls_2_tuned
)

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$FIRST_RAW" "$HRT_RAW" "$RESULTS"

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

echo "===== HRTBDA-v5 + FIRST-PLACE xView2 TURKEY SOFT ENSEMBLE ====="
date
hostname
nvidia-smi || true
echo "Selection split: Turkey validation"
echo "Final evaluation split: Turkey test"
echo "Hybrid output: $HYBRID_ROOT"

for required in \
  "$FIRST_DATA/val/images" \
  "$FIRST_DATA/test/images" \
  "$FIRST_DATA/official_split_manifest.csv" \
  "$FIRST_WEIGHTS" \
  "$HRT_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$HRT_EXPERIMENT/checkpoints/phase2_best.pt"
do
  if [ ! -e "$required" ]; then
    echo "ERROR: required input is missing: $required"
    exit 1
  fi
done

echo "===== Ensure first-place prediction scripts use fine-tuned weights ====="
cd "$FIRST_BASE"
python -u custom/patch_first_place_official_split.py
export MODELS_FOLDER="$FIRST_WEIGHTS"
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
    echo "Reusing completed first-place probabilities: $destination"
    return
  fi

  echo "===== First-place probability inference: $split ====="
  cleanup_prediction_dirs
  rm -rf test
  ln -sfn "$FIRST_DATA/$split" test
  bash predict.sh

  rm -rf "$destination"
  mkdir -p "$destination"
  for directory in "${PREDICTION_DIRS[@]}"; do
    if [ ! -d "$directory" ]; then
      echo "ERROR: first-place prediction folder missing: $directory"
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

echo "===== Export HRTBDA-v5 validation/test probabilities ====="
cd "$PROJECT_ROOT"
python -u transformer/scripts/export_hrtbda_v5_turkey_probabilities.py \
  --phase1-checkpoint "$HRT_EXPERIMENT/checkpoints/phase1_best.pt" \
  --phase2-checkpoint "$HRT_EXPERIMENT/checkpoints/phase2_best.pt" \
  --data-root /homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed \
  --output-root "$HRT_RAW" \
  --split-manifest "$FIRST_DATA/official_split_manifest.csv" \
  --splits val test \
  --img-size 1024 \
  --num-workers 2 \
  --device cuda

echo "===== Select fusion on validation and evaluate test once ====="
python -u transformer/scripts/eval_hrtbda_first_place_turkey_soft_ensemble.py \
  --hrtbda-root "$HRT_RAW" \
  --first-place-root "$FIRST_RAW" \
  --output-dir "$RESULTS" \
  --alphas 0,0.25,0.5,0.75,1 \
  --betas 0,0.25,0.5,0.75,1 \
  --thresholds 0.3,0.4,0.5,0.6,0.7 \
  --expected-val-samples 93 \
  --expected-test-samples 95

cat "$RESULTS/hybrid_metrics_summary.txt"
echo "DONE: validation-tuned HRTBDA-v5 + first-place xView2 Turkey soft ensemble"
date
