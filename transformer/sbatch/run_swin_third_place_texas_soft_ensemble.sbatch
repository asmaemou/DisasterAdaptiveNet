#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tx_swin_3rd_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
TEXAS_DATA=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
THIRD_BASE=$PROJECT_ROOT/baselines/xview2_winners/xview2_third_place
THIRD_CHECKPOINTS=$PROJECT_ROOT/output/xview2_baselines/third_place_texas_tornadoes_FULL_SOLUTION_finetune_official_split/models
COMMON_MANIFEST=$PROJECT_ROOT/output/xview2_baseline_datasets/first_place_texas_tornadoes_FINE_TUNE_OFFICIAL_SPLIT/official_split_manifest.csv
SWIN_EXPERIMENT=$PROJECT_ROOT/output/SwinImageNet-DirectTexas_TEXAS_TORNADOES
# New isolated location: no prior experiment, prediction, or script is removed.
EXPERIMENT_ROOT=$PROJECT_ROOT/output/SwinThirdPlaceSoftEnsembleAlignedMacro_TEXAS_TORNADOES
THIRD_RAW=$EXPERIMENT_ROOT/probabilities/third_place
SWIN_RAW=$EXPERIMENT_ROOT/probabilities/swin
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
  "$TEXAS_DATA/val/images" "$TEXAS_DATA/test/images" \
  "$THIRD_BASE/predict_37_weighted.py" "$THIRD_CHECKPOINTS" \
  "$COMMON_MANIFEST" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt"
do
  test -e "$required" || { echo "ERROR: required input missing: $required"; exit 1; }
done

THIRD_COUNT=$(find "$THIRD_CHECKPOINTS" -maxdepth 1 -type f -name '*.pth' | wc -l)
test "$THIRD_COUNT" -eq 13 || { echo "ERROR: expected exactly 13 fine-tuned third-place checkpoints, found $THIRD_COUNT"; exit 1; }

echo "===== TEXAS SWIN-T + THIRD-PLACE xVIEW2 SOFT ENSEMBLE ====="
echo "Branch A: ImageNet-pretrained Swin-Tiny fine-tuned on Texas train"
echo "Branch B: complete 13-model third-place xView2 ensemble fine-tuned on Texas train"
echo "Selection: Texas validation only (45 samples)"
echo "Final evaluation: untouched Texas test once (46 samples)"
echo "Safety: exact truth alignment + third-place validation reproduction gate"
echo "Geometry: both branches evaluated at 1024x1024"
echo "Output: $EXPERIMENT_ROOT"
date
hostname
nvidia-smi || true

THIRD_VAL_COUNT=$(find "$THIRD_RAW/val" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)
THIRD_TEST_COUNT=$(find "$THIRD_RAW/test" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)
if [ -f "$THIRD_RAW/val/_SUCCESS" ] && [ -f "$THIRD_RAW/test/_SUCCESS" ] && \
   [ "$THIRD_VAL_COUNT" -eq 45 ] && [ "$THIRD_TEST_COUNT" -eq 46 ]; then
  echo "Reusing complete third-place probability cache: $THIRD_RAW"
else
  echo "Third-place cache absent or incomplete; exporting all 13 models"
  python -u transformer/scripts/export_third_place_texas_probabilities.py \
    --winner-root "$THIRD_BASE" \
    --checkpoint-root "$THIRD_CHECKPOINTS" \
    --data-root "$TEXAS_DATA" \
    --output-root "$THIRD_RAW" \
    --splits val test --input-size 512 --output-size 1024 \
    --expected-val-samples 45 --expected-test-samples 46
fi

python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
  --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
  --data-root "$TEXAS_DATA" --output-root "$SWIN_RAW" \
  --split-manifest "$COMMON_MANIFEST" --splits val test \
  --output-size 1024 --geometry stretch \
  --num-workers 0 --device cuda --overwrite

python -u transformer/scripts/eval_swin_third_place_texas_soft_ensemble.py \
  --swin-root "$SWIN_RAW" --third-place-root "$THIRD_RAW" \
  --output-dir "$RESULTS" \
  --minimum-third-place-val-loc-f1 0.8 \
  --expected-val-samples 45 --expected-test-samples 46

cat "$RESULTS/ensemble_metrics.txt"
echo "DONE: Texas Swin-T + third-place xView2 soft ensemble"
date
