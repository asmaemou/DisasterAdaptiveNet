#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=texas_damformer
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
OUTPUT_DIR=$PROJECT_ROOT/output/DamFormerPaper_TEXAS_TORNADOES

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

# Node-local caches prevent the NFS CUDA cache stall observed previously.
LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCH_HOME=$LOCAL_JOB_TMP/torch
export HF_HOME=$LOCAL_JOB_TMP/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCH_HOME" "$HF_HUB_CACHE"

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

for required in \
  "$DATA_ROOT/train/images" "$DATA_ROOT/train/targets" \
  "$DATA_ROOT/val/images" "$DATA_ROOT/val/targets" \
  "$DATA_ROOT/test/images" "$DATA_ROOT/test/targets"
do
  if [ ! -d "$required" ]; then
    echo "ERROR: required dataset directory is missing: $required"
    exit 1
  fi
done

python -u -c '
import timm
models = timm.list_models("mit_b2")
print("timm:", timm.__version__)
print("MiT-B2 available:", models)
if "mit_b2" not in models:
    raise SystemExit("ERROR: installed timm does not provide mit_b2")
'

python -u transformer/scripts/check_idabd_xbd_format.py \
  --root "$DATA_ROOT" \
  --splits train val test

echo "===== PAPER-FAITHFUL DAMFORMER ON TEXAS TORNADOES ====="
date
hostname
nvidia-smi || true
echo "Encoder: weight-shared ImageNet-pretrained MiT-B2, depths [3,4,6,3]"
echo "Fusion: task-specific concat + 3x3 convolution + CBAM channel attention"
echo "Decoder: dual All-MLP cross-layer fusion with localization feature addition"
echo "Loss: localization BCE+Dice; damage CE+Lovasz-Softmax"
echo "Paper optimizer: AdamW, lr=6e-5, weight_decay=5e-3"
echo "Dataset: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"

CMD=(
  python -u transformer/scripts/train_texas_damformer_paper.py
  --phase train_test
  --data-root "$DATA_ROOT"
  --train-split train
  --val-split val
  --test-split test
  --output-dir "$OUTPUT_DIR"
  --epochs 80
  --patience 15
  --batch-size 1
  --eval-batch-size 1
  --grad-accum-steps 4
  --num-workers 0
  --img-size 1024
  --decoder-width 256
  --lr 6e-5
  --weight-decay 5e-3
  --damage-weight 1.0
  --max-grad-norm 1.0
  --thresholds 0.35 0.45 0.55 0.65 0.75
  --seed 42
  --imagenet-pretrained
  --amp
)

echo "===== COMMAND ====="
printf '%q ' "${CMD[@]}"
echo
echo "==================="
"${CMD[@]}"

echo "===== FINAL DAMFORMER TEXAS RESULT ====="
SUMMARY_FILE=$OUTPUT_DIR/scores/summary.txt
if [ -f "$SUMMARY_FILE" ]; then
  cat "$SUMMARY_FILE"
else
  echo "WARNING: expected summary was not found: $SUMMARY_FILE"
fi
date
