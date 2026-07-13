#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=inspect_dorian_winners
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/inspect_dorian_winners_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/inspect_dorian_winners_%j.err

set -euo pipefail
set -x

BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners
DORIAN=/homes/j244s673/documents/wsu/phd/hurricane_dorian_preprocessed

echo "================================================"
echo "CHECK HURRICANE DORIAN DATASET"
echo "================================================"

echo "DORIAN=$DORIAN"

find "$DORIAN" -maxdepth 3 -type d | sort

echo "Sample train files:"
find "$DORIAN/train" -maxdepth 3 -type f | sort | head -40 || true

echo "Sample val files:"
find "$DORIAN/val" -maxdepth 3 -type f | sort | head -40 || true

echo "Sample test files:"
find "$DORIAN/test" -maxdepth 3 -type f | sort | head -40 || true

echo "Counts:"
find "$DORIAN/train" -type f | wc -l
find "$DORIAN/val" -type f | wc -l
find "$DORIAN/test" -type f | wc -l

echo "================================================"
echo "FIRST-PLACE SOLUTION: scripts and weights"
echo "================================================"

cd "$BASE/xview2_first_place"

echo "First-place repo:"
pwd
ls -lh

echo "First-place weights:"
find "$BASE/xview2_first_place/weights/extracted_weights/weights" -maxdepth 2 -type f | sort || true

echo "Possible train/finetune/predict scripts:"
find . -maxdepth 3 -type f \( -name "*.py" -o -name "*.sh" \) | grep -Ei "train|fine|predict|infer|submission|test" | sort || true

echo "Search for argparse/sys.argv in first-place scripts:"
grep -R "argparse\|sys.argv\|ArgumentParser\|add_argument\|weights\|checkpoint\|resume\|fine" -n . \
  --include="*.py" --include="*.sh" | head -300 || true

echo "================================================"
echo "SECOND-PLACE SOLUTION: scripts and weights"
echo "================================================"

cd "$BASE/xview2_second_place"

echo "Second-place repo:"
pwd
ls -lh

echo "Second-place weights:"
find "$BASE/xview2_second_place/weights" -maxdepth 2 -type f | sort || true

echo "Possible train/finetune/predict scripts:"
find . -maxdepth 3 -type f \( -name "*.py" -o -name "*.sh" \) | grep -Ei "train|fine|predict|infer|submission|test" | sort || true

echo "Search for argparse/sys.argv in second-place scripts:"
grep -R "argparse\|sys.argv\|ArgumentParser\|add_argument\|weights\|checkpoint\|resume\|fine" -n . \
  --include="*.py" --include="*.sh" | head -300 || true

echo "================================================"
echo "THIRD-PLACE SOLUTION: scripts and weights"
echo "================================================"

cd "$BASE/xview2_third_place"

echo "Third-place repo:"
pwd
ls -lh

echo "Third-place weights:"
find "$BASE/xview2_third_place/weights/extracted/ensemble" -maxdepth 2 -type f -name "*.pth" | sort || true

echo "Possible train/finetune/predict scripts:"
find . -maxdepth 3 -type f \( -name "*.py" -o -name "*.sh" \) | grep -Ei "train|fine|predict|infer|submission|test" | sort || true

echo "Search for argparse/sys.argv in third-place scripts:"
grep -R "argparse\|sys.argv\|ArgumentParser\|add_argument\|weights\|checkpoint\|resume\|fine" -n . \
  --include="*.py" --include="*.sh" | head -300 || true

echo "================================================"
echo "DONE"
echo "Paste this log back so we can write the actual Dorian fine-tuning sbatch."
echo "================================================"