#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=semeru_swin_direct
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
SEMERU_ROOT=/homes/j244s673/documents/wsu/phd/mount_semeru_eruption_preprocessed
EXPERIMENT_NAME=SwinImageNet-DirectSemeru_MOUNT_SEMERU_ERUPTION
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

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

# Keep any ImageNet-weight download locks off the shared HPC filesystem.
LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export HF_HOME=$LOCAL_JOB_TMP/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
mkdir -p "$HF_HUB_CACHE"

echo "===== SWIN-T DIRECT TRAINING ON MOUNT SEMERU ====="
date
hostname
nvidia-smi || true
echo "Architecture: ImageNet-pretrained Swin-Tiny two-phase localization/damage cascade"
echo "Initialization: ImageNet only; no xBD and no xView2-winner weights"
echo "Dataset: $SEMERU_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Hugging Face cache: $HF_HUB_CACHE"

for required in \
  "$SEMERU_ROOT/train/images" \
  "$SEMERU_ROOT/train/targets" \
  "$SEMERU_ROOT/val/images" \
  "$SEMERU_ROOT/val/targets" \
  "$SEMERU_ROOT/test/images" \
  "$SEMERU_ROOT/test/targets"
do
  if [ ! -d "$required" ]; then
    echo "ERROR: required dataset directory is missing: $required"
    exit 1
  fi
done

if ! python -c "import timm" >/dev/null 2>&1; then
  echo "ERROR: timm is missing from disasteradaptivenet_cu128"
  exit 1
fi
python -c "import timm; print('timm version:', timm.__version__)"

echo "===== CHECK MOUNT SEMERU DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$SEMERU_ROOT" \
  --splits train val test

# These are the same architecture and training settings used by
# HRTBDA-v5-SwinImageNet-DirectTurkey_EARTHQUAKE_TURKEY. Only the dataset and
# output directory are different. Phase II is initialized from the newly
# Semeru-trained Phase-I Swin backbone.
CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py
  --phase both
  --xbd-root "$SEMERU_ROOT"
  --train-split train
  --val-split val
  --test-split test
  --output-dir "$OUTPUT_DIR"
  --phase1-epochs 150
  --phase2-epochs 30
  --batch-size 2
  --eval-batch-size 1
  --grad-accum-steps 4
  --num-workers 8
  --img-size 896
  --phase2-crop-size 672
  --crop-candidate-count 8
  --lr 5e-5
  --weight-decay 1e-4
  --seed 42
  --swin-variant swin_tiny_patch4_window7_224
  --swin-patch-size 4
  --swin-window-size 7
  --decoder-channels 128
  --max-grad-norm 1.0
  --warmup-epochs 3
  --focal-gamma 2.0
  --aux-loc-weight 0.2
  --minor-damage-boost 1.5
  --major-damage-boost 1.5
  --max-damage-class-weight 10.0
  --finetune-epochs 3
  --finetune-lr 5e-5
  --postprocess-dilation minor
  --dilation-kernel 3
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="
"${CMD[@]}"

echo "===== FINAL MOUNT SEMERU RESULT ====="
SUMMARY_FILE=$OUTPUT_DIR/scores/summary_hrtbda_v5_multilabel_crop_cascade.txt
if [ -f "$SUMMARY_FILE" ]; then
  cat "$SUMMARY_FILE"
else
  echo "WARNING: expected summary was not found: $SUMMARY_FILE"
  find "$OUTPUT_DIR/scores" -maxdepth 1 -type f -print
fi
echo "DONE: ImageNet-pretrained Swin-Tiny trained directly on Mount Semeru"
date
