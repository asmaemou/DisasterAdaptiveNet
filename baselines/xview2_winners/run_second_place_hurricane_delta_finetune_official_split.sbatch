#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=72:00:00
#SBATCH --job-name=xv2_2nd_delta_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_delta_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_delta_ftv_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners

export TARGET_DATASET=hurricane_delta
export DATASET_LABEL="Hurricane Delta"
export TRAIN_PREFIX=delta_official
export PREP_SCRIPT=prepare_second_place_hurricane_delta.py
export TEST_VIEW_SCRIPT=make_second_place_hurricane_delta_test_view.py

bash "$BASE/run_second_place_semeru_finetune_official_split.sbatch"
