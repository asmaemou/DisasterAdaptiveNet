#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tr_swin_3rd_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
TURKEY_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed
THIRD_BASE=$PROJECT_ROOT/baselines/xview2_winners/xview2_third_place
THIRD_CHECKPOINTS=$PROJECT_ROOT/output/xview2_baselines/third_place_earthquake_turkey_FULL_SOLUTION_finetune_official_split/models
SWIN_EXPERIMENT=$PROJECT_ROOT/output/HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY
MANIFEST=$PROJECT_ROOT/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT/official_split_manifest.csv
EXPERIMENT_ROOT=$PROJECT_ROOT/output/SwinThirdPlaceSoftEnsembleAlignedOfficial_EARTHQUAKE_TURKEY
THIRD_RAW=$EXPERIMENT_ROOT/probabilities/third_place_turkey_finetuned
SWIN_RAW=$EXPERIMENT_ROOT/probabilities/swin_direct_turkey
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
  "$TURKEY_ROOT/val/images" "$TURKEY_ROOT/test/images" \
  "$THIRD_BASE/predict_37_weighted.py" "$THIRD_CHECKPOINTS" \
  "$MANIFEST" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt"
do
  test -e "$required" || { echo "ERROR: required input missing: $required"; exit 1; }
done

THIRD_COUNT=$(find "$THIRD_CHECKPOINTS" -maxdepth 1 -type f -name '*.pth' | wc -l)
test "$THIRD_COUNT" -eq 13 || { echo "ERROR: expected 13 Turkey-fine-tuned third-place checkpoints, found $THIRD_COUNT"; exit 1; }

echo "===== TURKEY DIRECT SWIN-T + TURKEY-FINETUNED THIRD-PLACE xVIEW2 ====="
echo "Branch A: ImageNet-pretrained Swin-Tiny fine-tuned on Turkey train"
echo "Branch B: complete 13-model third-place xView2 ensemble fine-tuned on Turkey train"
echo "Selection: Turkey validation only (93 samples)"
echo "Final evaluation: untouched Turkey test once (95 samples)"
echo "Objective: official 0.3 localization + 0.7 harmonic damage F1"
echo "This job does not use the first-place shared test link."
echo "Output: $EXPERIMENT_ROOT"
date
hostname
nvidia-smi || true

VAL_CACHE=$(find "$THIRD_RAW/val" -maxdepth 1 -type f -name '*.npz' | wc -l)
TEST_CACHE=$(find "$THIRD_RAW/test" -maxdepth 1 -type f -name '*.npz' | wc -l)
if [ -f "$THIRD_RAW/val/_SUCCESS_BGR_V1" ] && [ -f "$THIRD_RAW/test/_SUCCESS_BGR_V1" ] && \
   [ "$VAL_CACHE" -eq 93 ] && [ "$TEST_CACHE" -eq 95 ]; then
  echo "Reusing complete third-place Turkey probability cache."
else
  python -u transformer/scripts/export_third_place_texas_probabilities.py \
    --winner-root "$THIRD_BASE" --checkpoint-root "$THIRD_CHECKPOINTS" \
    --data-root "$TURKEY_ROOT" --output-root "$THIRD_RAW" \
    --splits val test --input-size 512 --output-size 1024 \
    --expected-val-samples 93 --expected-test-samples 95
fi

python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
  --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
  --data-root "$TURKEY_ROOT" --output-root "$SWIN_RAW" \
  --split-manifest "$MANIFEST" \
  --splits val test --output-size 1024 --geometry stretch \
  --num-workers 0 --device cuda --overwrite

python -u transformer/scripts/eval_swin_third_place_texas_soft_ensemble.py \
  --swin-root "$SWIN_RAW" --third-place-root "$THIRD_RAW" \
  --output-dir "$RESULTS" \
  --expected-val-samples 93 --expected-test-samples 95 \
  --minimum-third-place-val-loc-f1 0.8 \
  --selection-objective official --minor-dilation-kernel 3 --selection-stride 4 \
  --alphas 0,0.25,0.5,0.75,1 \
  --betas 0,0.25,0.5,0.75,1 \
  --thresholds 0.3,0.4,0.5,0.6,0.7 \
  --experiment-label "Turkey-trained ImageNet Swin-Tiny + Turkey-fine-tuned third-place xView2 soft ensemble" \
  --selection-label "Fusion selected only on Turkey validation; untouched Turkey test evaluated once."

cat "$RESULTS/ensemble_metrics.txt"
echo "DONE: Turkey DirectSwin-T + Turkey-fine-tuned third-place xView2 ensemble"
date
