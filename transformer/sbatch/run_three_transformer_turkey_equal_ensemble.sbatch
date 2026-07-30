#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=turkey_3tf_equal
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed
MANIFEST=$PROJECT_ROOT/output/xview2_baseline_datasets/first_place_earthquake_turkey_FINE_TUNE_OFFICIAL_SPLIT/official_split_manifest.csv

SWIN_EXPERIMENT=$PROJECT_ROOT/output/HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY
PVT_EXPERIMENT=$PROJECT_ROOT/output/Transformer-PVTv2B2-ImageNet-DirectTurkey_EARTHQUAKE_TURKEY
TWINS_EXPERIMENT=$PROJECT_ROOT/output/Transformer-TwinsSVTS-ImageNet-DirectTurkey_EARTHQUAKE_TURKEY

ENSEMBLE_ROOT=$PROJECT_ROOT/output/Swin-PVTv2-Twins-EqualEnsemble_EARTHQUAKE_TURKEY
SWIN_RAW=$ENSEMBLE_ROOT/probabilities/swin
PVT_RAW=$ENSEMBLE_ROOT/probabilities/pvtv2
TWINS_RAW=$ENSEMBLE_ROOT/probabilities/twins
RESULTS=$ENSEMBLE_ROOT/results

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$SWIN_RAW" "$PVT_RAW" "$TWINS_RAW" "$RESULTS"

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

echo "===== THREE-TRANSFORMER EQUAL-WEIGHT ENSEMBLE ====="
date
hostname
nvidia-smi || true
echo "Members: Swin-T, PVTv2-B2, Twins-SVT-S"
echo "Weights: 1/3, 1/3, 1/3"
echo "Dataset: Earthquake Turkey"
echo "Output: $ENSEMBLE_ROOT"

for required in \
  "$MANIFEST" \
  "$SWIN_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$SWIN_EXPERIMENT/checkpoints/phase2_best.pt" \
  "$PVT_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$PVT_EXPERIMENT/checkpoints/phase2_best.pt" \
  "$TWINS_EXPERIMENT/checkpoints/phase1_best.pt" \
  "$TWINS_EXPERIMENT/checkpoints/phase2_best.pt"
do
  if [ ! -f "$required" ]; then
    echo "ERROR: required checkpoint/input is missing: $required"
    exit 1
  fi
done

export_member() {
  local experiment=$1
  local destination=$2
  local name=$3
  echo "===== Export $name probabilities ====="
  python -u transformer/scripts/export_swin_hrtbda_v5_turkey_probabilities.py \
    --phase1-checkpoint "$experiment/checkpoints/phase1_best.pt" \
    --phase2-checkpoint "$experiment/checkpoints/phase2_best.pt" \
    --data-root "$DATA_ROOT" \
    --output-root "$destination" \
    --split-manifest "$MANIFEST" \
    --splits val test \
    --output-size 1024 \
    --num-workers 2 \
    --device cuda
}

export_member "$SWIN_EXPERIMENT" "$SWIN_RAW" "Swin-T"
export_member "$PVT_EXPERIMENT" "$PVT_RAW" "PVTv2-B2"

# Twins features_only support comes from the isolated timm 1.x runtime created
# by its training task.  Use it only for the Twins export; the shared Conda
# environment is not modified.
TIMM_RUNTIME=$PROJECT_ROOT/output/three_transformer_timm_runtime
if [ ! -d "$TIMM_RUNTIME/timm" ]; then
  echo "ERROR: isolated Twins timm runtime is missing: $TIMM_RUNTIME"
  exit 1
fi
export PYTHONPATH="$TIMM_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
python -c "import timm; print('Twins export timm version:', timm.__version__)"
export_member "$TWINS_EXPERIMENT" "$TWINS_RAW" "Twins-SVT-S"

echo "===== Equal probability averaging and evaluation ====="
python -u transformer/scripts/eval_three_transformer_turkey_equal_ensemble.py \
  --swin-root "$SWIN_RAW" \
  --pvtv2-root "$PVT_RAW" \
  --twins-root "$TWINS_RAW" \
  --output-dir "$RESULTS" \
  --thresholds 0.4,0.5,0.6,0.7 \
  --expected-val-samples 93 \
  --expected-test-samples 95

cat "$RESULTS/three_transformer_equal_ensemble_summary.txt"
echo "DONE: Swin-T + PVTv2-B2 + Twins-SVT-S equal-weight Turkey ensemble"
date
