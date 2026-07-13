#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=04:00:00
#SBATCH --job-name=xv2_2nd_pak_zs
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_pak_zs_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/logs/xv2_2nd_pak_zs_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners
mkdir -p "$BASE/logs"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

cd "$BASE"

echo "================================================"
echo "Create Pakistan Flooding TEST-ONLY view"
echo "================================================"
python -u make_second_place_pakistan_test_view.py

echo "================================================"
echo "Run 2nd-place xView2 full available ensemble ZERO-SHOT on Pakistan Flooding TEST set"
echo "================================================"
python -u eval_second_place_pakistan_test_full.py --mode zero_shot

echo "DONE: Pakistan Flooding TEST zero-shot evaluation"
