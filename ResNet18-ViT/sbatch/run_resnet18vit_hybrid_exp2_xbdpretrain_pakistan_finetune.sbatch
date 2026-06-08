#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=r18vit_pak_ft
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/ResNet18-ViT/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/ResNet18-ViT/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p ResNet18-ViT/sbatch/logs

PAKISTAN_ROOT=/homes/j244s673/documents/wsu/phd/pakistan_flooding_preprocessed

XBD_HYBRID_CKPT=$PROJECT_ROOT/output/ResNet18ViT-HybridDamage_xBD_train_tier3_test_xBD/checkpoints/best_model.pt

EXPERIMENT_NAME=ResNet18ViT-HybridDamage_xBDPretrain_PAKISTAN_FLOODING_finetune
OUTPUT_DIR=$PROJECT_ROOT/output/$EXPERIMENT_NAME

TRAIN_SPLITS=(train)
VAL_SPLIT=val
TEST_SPLIT=test

EPOCHS=60
BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8
IMG_SIZE=1024

LR=5e-5
WEIGHT_DECAY=1e-4
SEED=42

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== EXP 2: ResNet18-ViT Hybrid | xBD pretrained -> Pakistan fine-tune ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "PAKISTAN_ROOT: $PAKISTAN_ROOT"
echo "XBD_HYBRID_CKPT: $XBD_HYBRID_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "==========================================================================="

if [ ! -f "$XBD_HYBRID_CKPT" ]; then
  echo "ERROR: xBD hybrid checkpoint not found:"
  echo "$XBD_HYBRID_CKPT"
  echo "Run Experiment 1 first."
  exit 1
fi

python -u ResNet18-ViT/script/train_resnet18_vit_hybrid_damage_segmentation.py \
  --dataset-root "$PAKISTAN_ROOT" \
  --train-split "${TRAIN_SPLITS[@]}" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --resume-checkpoint "$XBD_HYBRID_CKPT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" \
  --img-size "$IMG_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --amp

echo "Done."