#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xv2_1st_delta_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_delta_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_delta_ftv_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place

export TARGET_DATASET=hurricane_delta
export DATASET_LABEL="Hurricane Delta"
export PREP_SCRIPT=prepare_hurricane_delta_official_split_1024.py
export SETUP_SCRIPT=setup_hurricane_delta_ft_weights.py

bash "$BASE/sbatch/run_first_place_texas_tornadoes_finetune_official_split.sbatch"
