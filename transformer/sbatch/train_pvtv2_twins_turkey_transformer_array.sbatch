#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=turkey_tf_members
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%A_%a.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
TURKEY_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

# The existing Swin member was trained with this same protocol.  This array
# changes only the ImageNet-pretrained hierarchical Transformer backbone.
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    BACKBONE_VARIANT=pvt_v2_b2
    MEMBER_NAME=PVTv2-B2
    EXPERIMENT_NAME=Transformer-PVTv2B2-ImageNet-DirectTurkey_EARTHQUAKE_TURKEY
    ;;
  1)
    BACKBONE_VARIANT=twins_svt_small
    MEMBER_NAME=Twins-SVT-S
    EXPERIMENT_NAME=Transformer-TwinsSVTS-ImageNet-DirectTurkey_EARTHQUAKE_TURKEY
    ;;
  *)
    echo "ERROR: unsupported array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

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

echo "===== EARTHQUAKE TURKEY TRANSFORMER MEMBER TRAINING ====="
date
hostname
nvidia-smi || true
echo "Member: $MEMBER_NAME"
echo "timm backbone: $BACKBONE_VARIANT"
echo "Initialization: ImageNet-1K pretrained"
echo "Training data: Earthquake Turkey train"
echo "Model/threshold selection: Earthquake Turkey validation"
echo "Final evaluation: Earthquake Turkey test"
echo "Output: $OUTPUT_DIR"

if [ ! -d "$TURKEY_ROOT/train/images" ]; then
  echo "ERROR: missing Turkey dataset: $TURKEY_ROOT/train/images"
  exit 1
fi

if ! python -c "import timm" >/dev/null 2>&1; then
  echo "ERROR: timm is missing from disasteradaptivenet_cu128"
  echo "Install timm before resubmitting: pip install 'timm>=1.0,<2'"
  exit 1
fi
python -c "import timm; print('timm version:', timm.__version__)"

python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$TURKEY_ROOT" \
  --splits train val test

# These settings match the completed Swin DirectTurkey member.
CMD=(
  python -u transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py
  --phase both
  --xbd-root "$TURKEY_ROOT"
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
  --swin-variant "$BACKBONE_VARIANT"
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

echo "DONE: $MEMBER_NAME trained, validated, and tested on Earthquake Turkey"
date
