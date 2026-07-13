#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=inspect_train_args
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/inspect_train_args_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/inspect_train_args_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners

echo "================================================"
echo "SECOND-PLACE train_localization.py args"
echo "================================================"
cd "$BASE/xview2_second_place"
sed -n '60,130p' train_localization.py

echo "================================================"
echo "SECOND-PLACE train.py args"
echo "================================================"
sed -n '60,130p' train.py

echo "================================================"
echo "SECOND-PLACE dataset loading"
echo "================================================"
grep -R "folds.csv\|train.csv\|data_dir\|data-dir\|images\|masks\|labels\|json" -n train.py train_localization.py dataset configs | head -250 || true

echo "================================================"
echo "SECOND-PLACE run.sh"
echo "================================================"
cat run.sh || true

echo "================================================"
echo "THIRD-PLACE finetune.py args"
echo "================================================"
cd "$BASE/xview2_third_place"
sed -n '40,130p' finetune.py

echo "================================================"
echo "THIRD-PLACE fit_predict.py args"
echo "================================================"
sed -n '40,130p' fit_predict.py

echo "================================================"
echo "THIRD-PLACE dataset loading"
echo "================================================"
grep -R "train.csv\|train_folds.csv\|data-dir\|data_dir\|pre_disaster\|post_disaster\|mask\|target" -n finetune.py fit_predict.py xview/dataset.py | head -300 || true

echo "================================================"
echo "FIRST-PLACE train.sh and hardcoded paths"
echo "================================================"
cd "$BASE/xview2_first_place"
cat train.sh

echo "================================================"
echo "FIRST-PLACE dataset/path variables"
echo "================================================"
grep -R "train_dir\|masks_dir\|images_dir\|folds\|json\|pre_disaster\|post_disaster\|models_folder\|path.join" -n train*.py tune*.py | head -300 || true

echo "================================================"
echo "DONE"
echo "================================================"