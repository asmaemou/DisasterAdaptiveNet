#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs

TRAIN_JOB_RAW=$(sbatch --parsable \
  transformer/sbatch/train_pvtv2_twins_turkey_transformer_array.sbatch)
TRAIN_JOB=${TRAIN_JOB_RAW%%;*}

ENSEMBLE_JOB_RAW=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB}" \
  transformer/sbatch/run_three_transformer_turkey_equal_ensemble.sbatch)
ENSEMBLE_JOB=${ENSEMBLE_JOB_RAW%%;*}

echo "Submitted PVTv2/Twins training array: $TRAIN_JOB"
echo "  PVTv2-B2 task: ${TRAIN_JOB}_0"
echo "  Twins-SVT-S task: ${TRAIN_JOB}_1"
echo "Submitted dependent equal-weight ensemble: $ENSEMBLE_JOB"
echo "The ensemble starts only after both training tasks complete successfully."
