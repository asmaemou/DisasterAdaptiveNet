#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xv2_3rd_texas_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/sbatch/logs/xv2_3rd_texas_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/sbatch/logs/xv2_3rd_texas_ftv_%j.err

set -euo pipefail
set -x

REPO=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place

export TARGET_DATASET=texas_tornadoes
export DATASET_LABEL="Texas Tornadoes"
export PREP_SCRIPT=prepare_third_place_texas_tornadoes_official_split.py

bash "$REPO/sbatch/run_third_place_semeru_finetune_official_split.sbatch"
