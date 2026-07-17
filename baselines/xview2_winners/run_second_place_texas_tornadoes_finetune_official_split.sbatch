#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=24:00:00
#SBATCH --job-name=xv2_2nd_texas_ftv
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_texas_ftv_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_texas_ftv_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners

export TARGET_DATASET=texas_tornadoes
export DATASET_LABEL="Texas Tornadoes"
export TRAIN_PREFIX=texas_official
export PREP_SCRIPT=prepare_second_place_texas_tornadoes.py
export TEST_VIEW_SCRIPT=make_second_place_texas_tornadoes_test_view.py

bash "$BASE/run_second_place_semeru_finetune_official_split.sbatch"
