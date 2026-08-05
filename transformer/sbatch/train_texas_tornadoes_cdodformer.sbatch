#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=texas_cdod
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
OUTPUT_DIR=$PROJECT_ROOT/output/CDODFormer_TEXAS_TORNADOES
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

for directory in train/images train/targets val/images val/targets test/images test/targets; do
  test -d "$DATA_ROOT/$directory" || { echo "ERROR: missing $DATA_ROOT/$directory"; exit 1; }
done

echo "===== CDOD-FORMER ON TEXAS TORNADOES ====="
echo "Shared Siamese Swin-Tiny encoder"
echo "Alignment-aware local cross-attention + global subsampled temporal attention"
echo "DamFormer-inspired task-specific adapters and dual decoders"
echo "Localization-conditioned building-token reasoning"
echo "Categorical + ordinal + instance + boundary supervision"
echo "Validation-selected inference; untouched Texas test evaluated once"
date
hostname
nvidia-smi || true

python -u transformer/scripts/train_texas_cdodformer.py \
  --phase train_test --data-root "$DATA_ROOT" \
  --train-split train --val-split val --test-split test \
  --output-dir "$OUTPUT_DIR" \
  --epochs 100 --patience 20 --freeze-backbone-epochs 5 \
  --batch-size 1 --eval-batch-size 1 --grad-accum-steps 8 --num-workers 0 \
  --img-size 896 --crop-size 672 --crop-candidates 48 \
  --minor-crop-weight 20 --major-crop-weight 20 --destroyed-crop-weight 8 \
  --swin-variant swin_tiny_patch4_window7_224 \
  --decoder-channels 192 --temporal-heads 6 --temporal-window 7 \
  --global-grid 12 --reasoner-grid 14 \
  --lr 8e-5 --backbone-lr 3e-6 --weight-decay 1e-4 --max-grad-norm 1.0 \
  --dice-weight 0.5 --ordinal-weight 0.35 --consistency-weight 0.2 \
  --instance-weight 0.75 --boundary-weight 0.2 --min-instance-pixels 8 \
  --thresholds 0.35 0.45 0.55 0.65 0.75 \
  --ordinal-blends 0.0 0.25 0.5 --object-vote 0 1 \
  --seed 42 --amp

echo "===== FINAL CDOD-FORMER TEXAS RESULT ====="
cat "$OUTPUT_DIR/scores/summary.txt"
date
