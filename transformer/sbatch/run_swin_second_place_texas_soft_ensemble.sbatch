#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tx_swin_2nd_ens
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SECOND_DATA=$PROJECT_ROOT/output/xview2_baseline_datasets/second_place_texas_tornadoes
SECOND_FT=$PROJECT_ROOT/output/xview2_baselines/second_place_texas_tornadoes_FULL_SOLUTION_finetune_official_split
SWIN_EXPERIMENT=$PROJECT_ROOT/output/SwinImageNet-DirectTexas_TEXAS_TORNADOES
EXPERIMENT_ROOT=$PROJECT_ROOT/output/SwinSecondPlaceSoftEnsembleAlignedMacro_TEXAS_TORNADOES
SECOND_RAW=$EXPERIMENT_ROOT/probabilities/second_place
SWIN_RAW=$EXPERIMENT_ROOT/probabilities/swin
RESULTS=$EXPERIMENT_ROOT/results

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$SECOND_RAW" "$SWIN_RAW" "$RESULTS"
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

echo "===== TEXAS SWIN-T + SECOND-PLACE xVIEW2 SOFT ENSEMBLE ====="
echo "Branch A: ImageNet-pretrained Swin-Tiny fine-tuned on Texas train"
echo "Branch B: complete second-place xView2 ensemble fine-tuned on Texas train"
echo "Second-place ensemble: 6 localization + 15 damage models"
echo "Selection: Texas validation only (45 samples)"
echo "Final evaluation: untouched Texas test once (46 samples)"
echo "Geometry: second-place native 512x512 stretch"
echo "Output: $EXPERIMENT_ROOT"
date
hostname
nvidia-smi || true

python -u transformer/scripts/export_second_place_texas_probabilities.py \
  --data-root "$SECOND_DATA" \
  --finetune-root "$SECOND_FT" \
  --output-root "$SECOND_RAW" \
  --splits val test \
  --expected-val-samples 45 --expected-test-samples 46

python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
  --phase1-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  --phase2-checkpoint "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
  --data-root /homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed \
  --output-root "$SWIN_RAW" \
  --split-manifest "$SECOND_DATA/folds.csv" \
  --splits val test --output-size 512 --geometry stretch \
  --num-workers 0 --device cuda --overwrite

python -u transformer/scripts/eval_swin_second_place_texas_soft_ensemble.py \
  --swin-root "$SWIN_RAW" \
  --second-place-root "$SECOND_RAW" \
  --output-dir "$RESULTS" \
  --minimum-second-place-val-loc-f1 0.8 \
  --expected-val-samples 45 --expected-test-samples 46

cat "$RESULTS/ensemble_metrics.txt"
echo "DONE: Texas Swin-T + second-place xView2 soft ensemble"
date
