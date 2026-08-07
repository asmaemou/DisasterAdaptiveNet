#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xbd_swin_1st_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
FIRST_BASE=$PROJECT_ROOT/baselines/xview2_winners/xview2_first_place
FIRST_XBD_SOURCE=$FIRST_BASE/weights/weights
SWIN_EXPERIMENT=$PROJECT_ROOT/output/SwinImageNet-DirectxBD_TrainTier3
EXPERIMENT_ROOT=$PROJECT_ROOT/output/SwinFirstPlaceSoftEnsemble_xBD_TrainTier3
COMMON_VIEW=$EXPERIMENT_ROOT/common_xbd_hold_test_view
FIRST_ALIASES=$EXPERIMENT_ROOT/first_place_xbd_inference_weights
FIRST_RAW=$EXPERIMENT_ROOT/probabilities/first_place
SWIN_RAW=$EXPERIMENT_ROOT/probabilities/swin
RESULTS=$EXPERIMENT_ROOT/results

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
  "$XBD_ROOT/hold/images" "$XBD_ROOT/hold/targets" \
  "$XBD_ROOT/test/images" "$XBD_ROOT/test/targets" \
  "$FIRST_XBD_SOURCE" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt"
do
  test -e "$required" || { echo "ERROR: required input missing: $required"; exit 1; }
done

echo "===== xBD SWIN-T + RELEASED FIRST-PLACE xVIEW2 SOFT ENSEMBLE ====="
echo "Branch A: ImageNet-pretrained Swin-Tiny trained on xBD train+tier3"
echo "Branch B: released first-place xView2 ensemble trained on xBD"
echo "Selection: xBD hold split only"
echo "Final evaluation: xBD test split once"
echo "Official objective: 0.3 localization F1 + 0.7 harmonic damage F1"
echo "Damage dilation: none (kernel 1)"
echo "Output: $EXPERIMENT_ROOT"
date
hostname
nvidia-smi || true

# All released first-place scripts read the same mutable FIRST_BASE/test link.
# Use an atomic directory lock because this shared filesystem does not support
# POSIX flock(2) locks (it returns errno 37: No locks available).
LOCK_DIR=$FIRST_BASE/.first_place_inference.lockdir
release_workspace_lock() {
  rm -f "$LOCK_DIR/job_id"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
acquire_workspace_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$SLURM_JOB_ID" > "$LOCK_DIR/job_id"
    return 0
  fi
  local owner=""
  owner=$(cat "$LOCK_DIR/job_id" 2>/dev/null || true)
  if [ -n "$owner" ] && squeue -h -j "$owner" 2>/dev/null | grep -q .; then
    echo "ERROR: active first-place job $owner owns the shared legacy workspace."
    return 1
  fi
  echo "Removing stale first-place workspace lock (recorded owner: ${owner:-unknown})."
  rm -f "$LOCK_DIR/job_id"
  rmdir "$LOCK_DIR" 2>/dev/null || return 1
  mkdir "$LOCK_DIR"
  echo "$SLURM_JOB_ID" > "$LOCK_DIR/job_id"
}
acquire_workspace_lock || { echo "Wait for the active first-place job, then resubmit."; exit 1; }
trap release_workspace_lock EXIT
echo "Acquired exclusive first-place inference workspace lock."

python -u transformer/scripts/prepare_xbd_first_place_ensemble_view.py \
  --xbd-root "$XBD_ROOT" --output-root "$COMMON_VIEW"
MANIFEST=$COMMON_VIEW/official_split_manifest.csv
VAL_COUNT=$(tr -d '\r' < "$MANIFEST" | awk -F, 'NR>1 && $2=="val" {n++} END {print n+0}')
TEST_COUNT=$(tr -d '\r' < "$MANIFEST" | awk -F, 'NR>1 && $2=="test" {n++} END {print n+0}')
test "$VAL_COUNT" -gt 0 || { echo "ERROR: manifest contains zero validation samples"; exit 1; }
test "$TEST_COUNT" -gt 0 || { echo "ERROR: manifest contains zero test samples"; exit 1; }
echo "Common samples: hold/val=$VAL_COUNT test=$TEST_COUNT"

python -u transformer/scripts/setup_first_place_xbd_inference_weights.py \
  --source "$FIRST_XBD_SOURCE" --output "$FIRST_ALIASES"

cd "$FIRST_BASE"
python -u custom/patch_first_place_official_split.py
export MODELS_FOLDER="$FIRST_ALIASES"
export SUBMISSION_WORKERS=4

cleanup_predictor_outputs() {
  rm -rf submission
  for directory in "${PREDICTION_DIRS[@]}"; do rm -rf "$directory"; done
}
restore_test_link() {
  cd "$FIRST_BASE"
  rm -rf test
  ln -sfn "$COMMON_VIEW/test" test
  release_workspace_lock
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
    echo "Reusing complete first-place xBD cache: $destination"
    return
  fi
  cleanup_predictor_outputs
  rm -rf test
  ln -sfn "$COMMON_VIEW/$split" test
  test "$(readlink -f test)" = "$(readlink -f "$COMMON_VIEW/$split")" || {
    echo "ERROR: first-place test link does not point to requested split=$split"; exit 1;
  }
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
if python -u transformer/scripts/validate_swin_probability_cache.py \
     --root "$SWIN_RAW/val" --manifest "$MANIFEST" --split val && \
   python -u transformer/scripts/validate_swin_probability_cache.py \
     --root "$SWIN_RAW/test" --manifest "$MANIFEST" --split test; then
  echo "Reusing complete validated xBD Swin probability cache."
else
  python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
    --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
    --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
    --data-root "$COMMON_VIEW" --output-root "$SWIN_RAW" \
    --split-manifest "$MANIFEST" --splits val test \
    --output-size 1024 --geometry top-left-pad \
    --num-workers 0 --device cuda --overwrite
fi

python -u transformer/scripts/eval_swin_first_place_texas_soft_ensemble.py \
  --swin-root "$SWIN_RAW" --first-place-root "$FIRST_RAW" \
  --first-place-data-root "$COMMON_VIEW" --output-dir "$RESULTS" \
  --expected-val-samples "$VAL_COUNT" --expected-test-samples "$TEST_COUNT" \
  --minimum-first-place-val-loc-f1 0.8 \
  --selection-objective official --minor-dilation-kernel 1 --selection-stride 8 \
  --alphas 0,0.25,0.5,0.75,1 \
  --betas 0,0.25,0.5,0.75,1 \
  --thresholds 0.3,0.4,0.5,0.6,0.7 \
  --experiment-label "xBD train+tier3 Swin-Tiny + released first-place xView2 xBD soft ensemble" \
  --selection-label "Fusion selected only on xBD hold; untouched xBD test evaluated once."

cat "$RESULTS/ensemble_metrics.txt"
echo "DONE: xBD Swin-T + released first-place xView2 soft ensemble"
date
