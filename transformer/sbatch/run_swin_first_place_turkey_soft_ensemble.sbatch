#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tr_swin_1st_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
TURKEY_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed
FIRST_BASE=$PROJECT_ROOT/baselines/xview2_winners/xview2_first_place
FIRST_DATA=$PROJECT_ROOT/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT
FIRST_WEIGHTS=$FIRST_BASE/weights/earthquake_turkey_finetuned_weights_official_split
SWIN_EXPERIMENT=$PROJECT_ROOT/output/HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY
# Isolated from both the old xBD-initialized HRTBDA hybrid and zero-shot run.
EXPERIMENT_ROOT=$PROJECT_ROOT/output/SwinFirstPlaceSoftEnsembleAlignedOfficial_EARTHQUAKE_TURKEY
FIRST_RAW=$EXPERIMENT_ROOT/probabilities/first_place_turkey_finetuned
SWIN_RAW=$EXPERIMENT_ROOT/probabilities/swin_direct_turkey
RESULTS=$EXPERIMENT_ROOT/results
MANIFEST=$FIRST_DATA/official_split_manifest.csv

PREDICTION_DIRS=(
  pred50_loc_tuned pred92_loc_tuned pred34_loc pred154_loc
  dpn92cls_cce_0_tuned dpn92cls_cce_1_tuned dpn92cls_cce_2_tuned
  res34cls2_0_tuned res34cls2_1_tuned res34cls2_2_tuned
  res50cls_cce_0_tuned res50cls_cce_1_tuned res50cls_cce_2_tuned
  se154cls_0_tuned se154cls_1_tuned se154cls_2_tuned
)

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$FIRST_RAW" "$SWIN_RAW" "$RESULTS"
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
  "$TURKEY_ROOT/val/images" "$TURKEY_ROOT/test/images" \
  "$FIRST_DATA/val/images" "$FIRST_DATA/test/images" "$MANIFEST" \
  "$FIRST_WEIGHTS" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt"
do
  test -e "$required" || { echo "ERROR: required input missing: $required"; exit 1; }
done

echo "===== TURKEY DIRECT SWIN-T + TURKEY-FINETUNED FIRST-PLACE xVIEW2 ====="
echo "Branch A: ImageNet-pretrained Swin-Tiny fine-tuned on Turkey train"
echo "Branch B: first-place xView2 ensemble fine-tuned on Turkey train"
echo "Selection: Turkey validation only (93 samples)"
echo "Final evaluation: untouched Turkey test once (95 samples)"
echo "Objective: official 0.3 localization + 0.7 harmonic damage F1"
echo "Minor dilation: kernel 3 (matches DirectTurkey checkpoint evaluation)"
echo "Output: $EXPERIMENT_ROOT"
date
hostname
nvidia-smi || true

cd "$FIRST_BASE"
python -u custom/patch_first_place_official_split.py
python -u custom/setup_turkey_ft_weights.py
export MODELS_FOLDER="$FIRST_WEIGHTS"
export SUBMISSION_WORKERS=4

cleanup_predictor_outputs() {
  rm -rf submission
  for directory in "${PREDICTION_DIRS[@]}"; do rm -rf "$directory"; done
}
restore_test_link() {
  cd "$FIRST_BASE"
  rm -rf test
  ln -sfn "$FIRST_DATA/test" test
}
trap restore_test_link EXIT

normalize_probability_names() {
  local root=$1 directory legacy normalized
  for directory in "${PREDICTION_DIRS[@]}"; do
    [ -d "$root/$directory" ] || continue
    while IFS= read -r -d '' legacy; do
      normalized=${legacy%.png}
      test ! -e "$normalized" || { echo "ERROR: normalized file already exists: $normalized"; exit 1; }
      mv "$legacy" "$normalized"
    done < <(find "$root/$directory" -maxdepth 1 -type f -name '*.png.png' -print0)
  done
}

generate_first_place_probabilities() {
  local split=$1
  local destination=$FIRST_RAW/$split
  normalize_probability_names "$destination"
  if python -u "$PROJECT_ROOT/transformer/scripts/validate_first_place_probability_cache.py" \
      --root "$destination" --manifest "$MANIFEST" --split "$split" --require-success-marker; then
    echo "Reusing complete Turkey first-place cache: $destination"
    return
  fi
  cleanup_predictor_outputs
  rm -rf test
  ln -sfn "$FIRST_DATA/$split" test
  bash -e predict.sh
  normalize_probability_names "$FIRST_BASE"
  rm -rf "$destination"; mkdir -p "$destination"
  for directory in "${PREDICTION_DIRS[@]}"; do
    test -d "$directory" || { echo "ERROR: missing first-place output: $directory"; exit 1; }
    mv "$directory" "$destination/$directory"
  done
  touch "$destination/_SUCCESS"
  python -u "$PROJECT_ROOT/transformer/scripts/validate_first_place_probability_cache.py" \
    --root "$destination" --manifest "$MANIFEST" --split "$split" --require-success-marker
  rm -rf submission
}

generate_first_place_probabilities val
generate_first_place_probabilities test
restore_test_link
trap - EXIT

cd "$PROJECT_ROOT"
python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
  --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
  --data-root "$TURKEY_ROOT" --output-root "$SWIN_RAW" \
  --split-manifest "$MANIFEST" --splits val test \
  --output-size 1024 --geometry top-left-pad \
  --num-workers 0 --device cuda --overwrite

python -u transformer/scripts/eval_swin_first_place_texas_soft_ensemble.py \
  --swin-root "$SWIN_RAW" --first-place-root "$FIRST_RAW" \
  --first-place-data-root "$FIRST_DATA" --output-dir "$RESULTS" \
  --expected-val-samples 93 --expected-test-samples 95 \
  --minimum-first-place-val-loc-f1 0.8 \
  --selection-objective official --minor-dilation-kernel 3 \
  --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --betas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --experiment-label "Turkey-trained ImageNet Swin-Tiny + Turkey-fine-tuned first-place xView2 soft ensemble" \
  --selection-label "Fusion selected only on Turkey validation; untouched Turkey test evaluated once."

cat "$RESULTS/ensemble_metrics.txt"
echo "DONE: Turkey DirectSwin-T + Turkey-fine-tuned first-place xView2 ensemble"
date
