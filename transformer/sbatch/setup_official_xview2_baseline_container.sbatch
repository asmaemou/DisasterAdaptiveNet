#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=xview2_setup
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
IMAGE=$PROJECT_ROOT/containers/xview2_baseline_v1.0.sif
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs containers
if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer is not installed on this cluster. Ask HPC support for the Apptainer module name."
  exit 1
fi
echo "Building the official legacy xView2 v1.0 runtime: $IMAGE"
if ! apptainer build --fakeroot --ignore-fakeroot-command --force \
  "$IMAGE" transformer/containers/xview2_baseline.def; then
  echo "ERROR: container build failed. If fakeroot is disabled, ask HPC support to build this definition file once."
  exit 1
fi
apptainer exec "$IMAGE" test -s /home/appuser/code/xview-2/weights/localization.h5
apptainer exec "$IMAGE" test -s /home/appuser/code/xview-2/weights/classification.hdf5
echo "READY: $IMAGE"
