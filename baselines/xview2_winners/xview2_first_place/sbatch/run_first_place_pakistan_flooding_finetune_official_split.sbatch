#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=72:00:00
#SBATCH --job-name=xv2_1st_pakf_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_pakf_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place/sbatch/logs/xv2_1st_pakf_ftv_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_first_place

export TARGET_DATASET=pakistan_flooding
export DATASET_LABEL="Pakistan Flooding"
export PREP_SCRIPT=prepare_pakistan_flooding_official_split_1024.py
export SETUP_SCRIPT=setup_pakistan_flooding_ft_weights.py

bash "$BASE/sbatch/run_first_place_texas_tornadoes_finetune_official_split.sbatch"
