#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tr_swin_2nd_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
TURKEY_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed
SECOND_DATA=$PROJECT_ROOT/output/xview2_baseline_datasets/second_place_earthquake_turkey
SECOND_FT=$PROJECT_ROOT/output/xview2_baselines/second_place_earthquake_turkey_FULL_SOLUTION_finetune_official_split
SWIN_EXPERIMENT=$PROJECT_ROOT/output/HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY
EXPERIMENT_ROOT=$PROJECT_ROOT/output/SwinSecondPlaceSoftEnsembleAlignedOfficial_EARTHQUAKE_TURKEY
SECOND_RAW=$EXPERIMENT_ROOT/probabilities/second_place_turkey_finetuned
SWIN_RAW=$EXPERIMENT_ROOT/probabilities/swin_direct_turkey
RESULTS=$EXPERIMENT_ROOT/results

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$SECOND_RAW/val" "$SECOND_RAW/test" "$SWIN_RAW" "$RESULTS"
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
  "$SECOND_DATA/folds.csv" "$SECOND_DATA/images" "$SECOND_DATA/masks" \
  "$SECOND_FT/weights_localization" "$SECOND_FT/weights_damage" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt"
do
  test -e "$required" || { echo "ERROR: required input missing: $required"; exit 1; }
done

VAL_COUNT=$(awk -F, 'NR>1 && $3=="val" {n++} END {print n+0}' "$SECOND_DATA/folds.csv")
TEST_COUNT=$(awk -F, 'NR>1 && $3=="test" {n++} END {print n+0}' "$SECOND_DATA/folds.csv")
test "$VAL_COUNT" -eq 93 || { echo "ERROR: expected 93 Turkey validation samples, found $VAL_COUNT"; exit 1; }
test "$TEST_COUNT" -eq 95 || { echo "ERROR: expected 95 Turkey test samples, found $TEST_COUNT"; exit 1; }

echo "===== TURKEY DIRECT SWIN-T + TURKEY-FINETUNED SECOND-PLACE xVIEW2 ====="
echo "Branch A: ImageNet-pretrained Swin-Tiny fine-tuned on Turkey train"
echo "Branch B: complete second-place ensemble fine-tuned on Turkey train"
echo "Second-place ensemble: 6 localization + 15 damage models"
echo "Selection: Turkey validation only ($VAL_COUNT samples)"
echo "Final evaluation: untouched Turkey test once ($TEST_COUNT samples)"
echo "Objective: official 0.3 localization + 0.7 harmonic damage F1"
echo "This job does not use the first-place shared test link."
echo "Output: $EXPERIMENT_ROOT"
date
hostname
nvidia-smi || true

VAL_CACHE=$(find "$SECOND_RAW/val" -maxdepth 1 -type f -name '*.npz' | wc -l)
TEST_CACHE=$(find "$SECOND_RAW/test" -maxdepth 1 -type f -name '*.npz' | wc -l)
if [ -f "$SECOND_RAW/val/_SUCCESS_V1" ] && [ -f "$SECOND_RAW/test/_SUCCESS_V1" ] && \
   [ "$VAL_CACHE" -eq "$VAL_COUNT" ] && [ "$TEST_CACHE" -eq "$TEST_COUNT" ]; then
  echo "Reusing complete second-place Turkey probability cache."
else
  python -u transformer/scripts/export_second_place_texas_probabilities.py \
    --data-root "$SECOND_DATA" --finetune-root "$SECOND_FT" \
    --output-root "$SECOND_RAW" --splits val test \
    --expected-val-samples "$VAL_COUNT" --expected-test-samples "$TEST_COUNT"
fi

python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
  --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
  --data-root "$TURKEY_ROOT" --output-root "$SWIN_RAW" \
  --split-manifest "$SECOND_DATA/folds.csv" --splits val test \
  --output-size 512 --geometry stretch \
  --num-workers 0 --device cuda --overwrite

python -u transformer/scripts/eval_swin_second_place_texas_soft_ensemble.py \
  --swin-root "$SWIN_RAW" --second-place-root "$SECOND_RAW" \
  --output-dir "$RESULTS" \
  --expected-val-samples "$VAL_COUNT" --expected-test-samples "$TEST_COUNT" \
  --minimum-second-place-val-loc-f1 0.8 \
  --selection-objective official --minor-dilation-kernel 3 \
  --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --betas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --experiment-label "Turkey-trained ImageNet Swin-Tiny + Turkey-fine-tuned second-place xView2 soft ensemble" \
  --selection-label "Fusion selected only on Turkey validation; untouched Turkey test evaluated once."

cat "$RESULTS/ensemble_metrics.txt"
echo "DONE: Turkey DirectSwin-T + Turkey-fine-tuned second-place xView2 ensemble"
date
