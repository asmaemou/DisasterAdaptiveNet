#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=21:00:00
#SBATCH --job-name=texas_swin_xbdft
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
TEXAS_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
TEXAS_PHASE1_CKPT=$PROJECT_ROOT/output/SwinImageNet-DirectTexas_TEXAS_TORNADOES/checkpoints/phase1_best.pt
XBD_PHASE2_CKPT=$PROJECT_ROOT/output/SwinImageNet-DirectxBD_TrainTier3/checkpoints/phase2_best.pt
EXPERIMENT_NAME=SwinxBDPhase2-FineTuneTexas_TEXAS_TORNADOES
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

# Keep runtime caches off the NFS-mounted home directory. CUDA otherwise uses
# ~/.nv/ComputeCache; job 1163216 blocked there in rpc_wait_bit_killable during
# the first forward pass.
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

# Fail immediately if Slurm placed the job on a node where PyTorch cannot use
# the requested GPU. This check runs inside the allocation, not on a head node.
python -u -c '
import sys
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is unavailable on the allocated compute node")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU count:", torch.cuda.device_count())
'

echo "===== xBD-PRETRAINED SWIN-T PHASE-II FINE-TUNING ON TEXAS ====="
date
hostname
nvidia-smi || true
echo "Architecture: ImageNet-pretrained Swin-Tiny two-phase localization/damage cascade"
echo "Initialization: complete xBD-trained Swin Phase-II checkpoint"
echo "Texas Phase-I mask: $TEXAS_PHASE1_CKPT"
echo "xBD Phase-II initialization: $XBD_PHASE2_CKPT"
echo "Dataset: $TEXAS_ROOT"
echo "Training split: train"
echo "Validation/model selection split: val"
echo "Final evaluation split: test"
echo "Output: $OUTPUT_DIR"
echo "Hugging Face cache: $HF_HUB_CACHE"

for required in \
  "$TEXAS_ROOT/train/images" \
  "$TEXAS_ROOT/train/targets" \
  "$TEXAS_ROOT/val/images" \
  "$TEXAS_ROOT/val/targets" \
  "$TEXAS_ROOT/test/images" \
  "$TEXAS_ROOT/test/targets"
do
  if [ ! -d "$required" ]; then
    echo "ERROR: required dataset directory is missing: $required"
    exit 1
  fi
done

if [ ! -f "$TEXAS_PHASE1_CKPT" ]; then
  echo "ERROR: Texas Phase-I checkpoint is missing: $TEXAS_PHASE1_CKPT"
  echo "Run the direct Texas experiment first."
  exit 1
fi

if [ ! -f "$XBD_PHASE2_CKPT" ]; then
  echo "ERROR: xBD Swin Phase-II checkpoint is missing: $XBD_PHASE2_CKPT"
  echo "First submit: transformer/sbatch/train_xbd_hrtbda_v5_swin_pretrained_cascade.sbatch"
  exit 1
fi

if ! python -c "import timm" >/dev/null 2>&1; then
  echo "ERROR: timm is missing from disasteradaptivenet_cu128"
  exit 1
fi
python -c "import timm; print('timm version:', timm.__version__)"

echo "===== CHECK TEXAS TORNADOES DATASET FORMAT ====="
python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$TEXAS_ROOT" \
  --splits train val test

# Fine-tune the complete xBD-trained Swin damage model on Texas. The existing
# high-quality Texas Phase-I model supplies the localization mask at validation
# and test time; Phase I is not retrained in this experiment.
CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py
  --phase phase2_test
  --phase1-checkpoint "$TEXAS_PHASE1_CKPT"
  --init-phase2-from "$XBD_PHASE2_CKPT"
  --xbd-root "$TEXAS_ROOT"
  --train-split train
  --val-split val
  --test-split test
  --output-dir "$OUTPUT_DIR"
  --phase2-epochs 60
  --batch-size 2
  --eval-batch-size 1
  --grad-accum-steps 4
  # Load samples in the training process. Multiprocess workers stalled before
  # the first batch on the shared HPC filesystem in job 1163214.
  --num-workers 0
  --img-size 896
  --phase2-crop-size 672
  --crop-candidate-count 32
  --crop-weight-no-damage 1.0
  --crop-weight-minor 16.0
  --crop-weight-major 16.0
  --crop-weight-destroyed 4.0
  --extra-photometric-aug
  --lr 2e-5
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
  --minor-damage-boost 3.0
  --major-damage-boost 3.0
  --max-damage-class-weight 15.0
  --finetune-epochs 8
  --finetune-lr 5e-6
  --postprocess-dilation minor
  --dilation-kernel 3
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="
"${CMD[@]}"

echo "===== FINAL TEXAS TORNADOES RESULT ====="
SUMMARY_FILE=$OUTPUT_DIR/scores/summary_hrtbda_v5_multilabel_crop_cascade.txt
if [ -f "$SUMMARY_FILE" ]; then
  cat "$SUMMARY_FILE"
else
  echo "WARNING: expected summary was not found: $SUMMARY_FILE"
  find "$OUTPUT_DIR/scores" -maxdepth 1 -type f -print
fi
echo "DONE: complete xBD Swin Phase-II model fine-tuned on Texas Tornadoes"
date
