#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=collect_rsbd_ablation
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
ABLATION_JOB_ID=${ABLATION_JOB_ID:?Submit with --export=ALL,ABLATION_JOB_ID=<job-id>}

cd "$PROJECT_ROOT"

python -u transformer/scripts/collect_rsbdnet_ablation_results.py \
  --project-root "$PROJECT_ROOT" \
  --job-id "$ABLATION_JOB_ID"
