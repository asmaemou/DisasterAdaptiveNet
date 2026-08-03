#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=xbd_swin_best_test
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
XBD_ROOT=/homes/j244s673/documents/wsu/phd/xview2
TRAIN_OUTPUT_DIR=$PROJECT_ROOT/output/SwinImageNet-DirectxBD_TrainTier3
PHASE1_CKPT=$TRAIN_OUTPUT_DIR/checkpoints/phase1_best.pt
PHASE2_CKPT=$TRAIN_OUTPUT_DIR/checkpoints/phase2_best.pt
TEST_OUTPUT_DIR=$PROJECT_ROOT/output/SwinImageNet-DirectxBD_TrainTier3_best_checkpoint_test

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$TEST_OUTPUT_DIR/checkpoints" "$TEST_OUTPUT_DIR/scores"

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

# Keep CUDA and model caches off the NFS-mounted home directory.
LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export HF_HOME=$LOCAL_JOB_TMP/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
mkdir -p \
  "$CUDA_CACHE_PATH" \
  "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$HF_HUB_CACHE"

python -u -c '
import sys
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is unavailable on the allocated compute node")
print("GPU:", torch.cuda.get_device_name(0))
'

for required in \
  "$XBD_ROOT/train/images" \
  "$XBD_ROOT/train/targets" \
  "$XBD_ROOT/tier3/images" \
  "$XBD_ROOT/tier3/targets" \
  "$XBD_ROOT/hold/images" \
  "$XBD_ROOT/hold/targets" \
  "$XBD_ROOT/test/images" \
  "$XBD_ROOT/test/targets" \
  "$PHASE1_CKPT" \
  "$PHASE2_CKPT"
do
  if [ ! -e "$required" ]; then
    echo "ERROR: required test input is missing: $required"
    exit 1
  fi
done

echo "===== TEST xBD SWIN BEST CHECKPOINTS ONLY ====="
date
hostname
nvidia-smi || true
echo "Dataset root: $XBD_ROOT"
echo "Test split: test"
echo "Phase-I checkpoint: $PHASE1_CKPT"
echo "Phase-II checkpoint: $PHASE2_CKPT"
echo "Test output: $TEST_OUTPUT_DIR"
echo "No training or short fine-tuning will run."

python -u transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py \
  --phase test \
  --phase1-checkpoint "$PHASE1_CKPT" \
  --phase2-checkpoint "$PHASE2_CKPT" \
  --xbd-root "$XBD_ROOT" \
  --train-split train tier3 \
  --val-split hold \
  --test-split test \
  --output-dir "$TEST_OUTPUT_DIR" \
  --batch-size 2 \
  --eval-batch-size 1 \
  --num-workers 0 \
  --img-size 896 \
  --phase2-crop-size 672 \
  --swin-variant swin_tiny_patch4_window7_224 \
  --swin-patch-size 4 \
  --swin-window-size 7 \
  --decoder-channels 128 \
  --postprocess-dilation none \
  --amp

echo "===== FINAL xBD BEST-CHECKPOINT TEST RESULT ====="
SUMMARY_FILE=$TEST_OUTPUT_DIR/scores/summary_hrtbda_v5_multilabel_crop_cascade.txt
if [ -f "$SUMMARY_FILE" ]; then
  cat "$SUMMARY_FILE"
else
  echo "ERROR: expected test summary was not produced: $SUMMARY_FILE"
  exit 1
fi
date
