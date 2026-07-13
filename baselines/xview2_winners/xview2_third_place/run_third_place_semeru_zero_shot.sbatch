#!/bin/bash

#SBATCH -p wsu_soc_kshirsagar.q

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=96G

#SBATCH --gres=gpu:rtx_pro_6000:1

#SBATCH --time=04:00:00

#SBATCH --job-name=xv2_3rd_semeru_zs

#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_semeru_zs_%j.out

#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_semeru_zs_%j.err



set -euo pipefail

set -x

export PYTHONUNBUFFERED=1



BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place

DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_mount_semeru_TEST_ONLY

PRED_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_mount_semeru_TEST_ONLY_predictions_fast

EVAL_PRED_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_mount_semeru_TEST_ONLY_predictions_fast_naive

FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_mount_semeru_TEST_ONLY_ZERO_SHOT_full_solution



mkdir -p "$BASE/logs"



source ~/miniforge3/etc/profile.d/conda.sh

conda activate disasteradaptivenet_cu128



cd "$BASE"



ln -sfn weights/extracted/ensemble models



echo "================================================"

echo "Prepare Mount Semeru TEST_ONLY for 3rd-place xView2"

echo "================================================"

python -u prepare_third_place_semeru_test.py



echo "================================================"

echo "Run 3rd-place xView2 weighted ensemble ZERO-SHOT on Mount Semeru"

echo "================================================"



rm -rf "$PRED_DIR" "$EVAL_PRED_DIR"



python -u predict_37_weighted.py -dd "$DATA" -o "$PRED_DIR" --size 512 -p naive --activation model --fp16 -b 1 -w 0



echo "================================================"

echo "Evaluate 3rd-place predictions on Mount Semeru"

echo "================================================"



python -u eval_third_place_semeru_predictions.py --truth-dir "$DATA" --pred-dir "$EVAL_PRED_DIR" --out-dir "$FINAL_OUT"



echo "DONE: 3rd-place xView2 Mount Semeru zero-shot"

