#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=gradcam_v7_msdf
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found."
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

echo "===== GRAD-CAM JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "============================="

python -u transformer/scripts/gradcam_xbd_hrtbda_v7_msdf_full.py \
  --v7-script transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py \
  --phase1-checkpoint /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/checkpoints/phase1_best.pt \
  --phase2-checkpoint /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/checkpoints/phase2_best.pt \
  --xbd-root /homes/j244s673/documents/wsu/phd/xview2 \
  --output-dir /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v7_msdf_full_two_stage/gradcam_visuals \
  --test-split test \
  --sample-count 8 \
  --cam-classes minor major destroyed \
  --target-layer change_fusion.0 \
  --score-mask pred_class \
  --postprocess-dilation minor \
  --dilation-kernel 3 \
  --num-workers 8 \
  --img-size 1024 \
  --base-channels 48 \
  --decoder-channels 128 \
  --window-size 8

echo "Done."
SH