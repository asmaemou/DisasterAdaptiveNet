#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=texas_swindf_v2
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
OUTPUT_DIR=$PROJECT_ROOT/output/SwinDamageFormerV2_TEXAS_TORNADOES
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCH_HOME=$LOCAL_JOB_TMP/torch
export HF_HOME=$LOCAL_JOB_TMP/huggingface
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$HF_HOME"

python - <<'PY'
import sys, torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: allocated node has no usable GPU")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== SWINDAMAGEFORMER V2: CALIBRATED RARE-CLASS INFERENCE ====="
echo "Training: Texas train"
echo "Checkpoint and inference calibration: Texas val"
echo "Final evaluation: untouched Texas test once"
echo "Validation compares pixel inference and object voting"
echo "Validation calibrates categorical/ordinal fusion and rare-class boosts"
date
hostname
nvidia-smi || true

python -u transformer/scripts/train_texas_swin_damageformer.py \
  --phase train_test --data-root "$DATA_ROOT" \
  --train-split train --val-split val --test-split test \
  --output-dir "$OUTPUT_DIR" \
  --epochs 100 --patience 20 --freeze-backbone-epochs 5 --warmup-epochs 5 \
  --batch-size 2 --eval-batch-size 1 --grad-accum-steps 4 --num-workers 0 \
  --img-size 896 --crop-size 672 --crop-candidates 48 \
  --minor-crop-weight 20 --major-crop-weight 20 \
  --rare-tile-sampling --minor-tile-weight 4 --major-tile-weight 3 --destroyed-tile-weight 2 \
  --swin-variant swin_tiny_patch4_window7_224 \
  --decoder-channels 192 --temporal-heads 6 --temporal-window 7 \
  --lr 8e-5 --backbone-lr 3e-6 --weight-decay 1e-4 --max-grad-norm 1.0 \
  --ordinal-weight 0.35 --instance-weight 0.75 --min-instance-pixels 8 \
  --damage-focal-gamma 2.0 --damage-label-smoothing 0.02 --ema-decay 0.995 \
  --thresholds 0.35 0.45 0.55 0.65 0.75 \
  --seed 42 --amp

echo "===== FINAL SWINDAMAGEFORMER V2 TEXAS RESULT ====="
cat "$OUTPUT_DIR/scores/summary.txt"
date
