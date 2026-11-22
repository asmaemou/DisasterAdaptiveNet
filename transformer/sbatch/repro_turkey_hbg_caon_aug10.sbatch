#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tr_bg_xattn_ord_repro
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

# Faithful rerun of original HBG-CAON Earthquake Turkey training settings.
# August 10 source job: 1173291; reported Macro-F1 0.6115.
# New result directory per Slurm job prevents overwriting older data.
set -euo pipefail
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export CUDA_CACHE_DISABLE=1

LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
export CUDA_CACHE_PATH=$LOCAL_JOB_TMP/cuda_cache
export XDG_CACHE_HOME=$LOCAL_JOB_TMP/xdg_cache
export TORCH_EXTENSIONS_DIR=$LOCAL_JOB_TMP/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_JOB_TMP/torchinductor
export HF_HOME=$LOCAL_JOB_TMP/huggingface
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed
OUTPUT_DIR="$PROJECT_ROOT/output/HBG_CAON_EARTHQUAKE_TURKEY_REPRO_${SLURM_JOB_ID}"
SCRIPT="transformer/scripts/train_turkey_bitemporal_building_crossattention_ordinal.py"

cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT_DIR"

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

{
  echo "===== HBG-CAON Earthquake Turkey AUGUST-SETTINGS REPRODUCTION ====="
  echo "Historical job: 1173291"
  echo "SLURM_JOB_ID: $SLURM_JOB_ID"
  echo "Data root: $DATA_ROOT"
  echo "Output root: $OUTPUT_DIR"
  echo "Script: $SCRIPT"
  echo "Git HEAD: $(git rev-parse HEAD)"
  echo "Python executable: $(command -v python)"
  python --version
  date
  hostname
  nvidia-smi
} | tee "$OUTPUT_DIR/run_metadata.txt"

# Save provenance so this rerun itself remains auditable.
sha256sum "$SCRIPT" > "$OUTPUT_DIR/training_script_sha256.txt"
git status --short -- transformer > "$OUTPUT_DIR/transformer_git_status.txt"
git diff -- transformer > "$OUTPUT_DIR/transformer_uncommitted_diff.patch"
git diff 656593e -- transformer > "$OUTPUT_DIR/transformer_diff_vs_aug10_commit.patch"
python -m pip freeze > "$OUTPUT_DIR/pip_freeze.txt"
find "$DATA_ROOT" -type f -printf '%P\t%s\n' | LC_ALL=C sort \
  > "$OUTPUT_DIR/data_paths_and_sizes.tsv"
cp "$0" "$OUTPUT_DIR/launch_sbatch_copy.sbatch"

# All training arguments are unchanged from the original SBATCH file.
# Epoch and threshold are selected on validation; do not force historical choices.
python -u "$SCRIPT" \
  --xbd-root "$DATA_ROOT" \
  --train-split train \
  --val-split val \
  --test-split test \
  --output-dir "$OUTPUT_DIR" \
  --epochs 60 \
  --batch-size 2 \
  --num-workers 0 \
  --img-size 896 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --seed 42 \
  --save-every 5 \
  --early-stopping-patience 12 \
  --loc-bce-weight 1.0 \
  --loc-dice-weight 1.0 \
  --dmg-ce-weight 1.0 \
  --thresholds 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 \
  --amp

echo "DONE: HBG-CAON Earthquake Turkey isolated reproduction"
date
