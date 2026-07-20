#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=72:00:00
#SBATCH --job-name=xv2_2nd_pakf_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_pakf_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_pakf_ftv_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners

export TARGET_DATASET=pakistan_flooding
export DATASET_LABEL="Pakistan Flooding"
export TRAIN_PREFIX=pakistan_flooding_official
export PREP_SCRIPT=prepare_second_place_pakistan_flooding.py
export TEST_VIEW_SCRIPT=make_second_place_pakistan_flooding_test_view.py

bash "$BASE/run_second_place_semeru_finetune_official_split.sbatch"
