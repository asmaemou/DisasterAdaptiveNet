#!/bin/bash

#SBATCH -p wsu_soc_kshirsagar.q

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=96G

#SBATCH --gres=gpu:rtx_pro_6000:1

#SBATCH --time=04:00:00

#SBATCH --job-name=xv2_3rd_texas_zs

#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_texas_zs_%j.out

#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place/logs/xv2_3rd_texas_zs_%j.err



set -euo pipefail

set -x

export PYTHONUNBUFFERED=1



BASE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners/xview2_third_place

DATA=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_texas_tornadoes_TEST_ONLY

PRED_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_texas_tornadoes_TEST_ONLY_predictions_fast

EVAL_PRED_DIR=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_texas_tornadoes_TEST_ONLY_predictions_fast_naive

FINAL_OUT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/third_place_texas_tornadoes_TEST_ONLY_ZERO_SHOT_full_solution



mkdir -p "$BASE/logs"



source ~/miniforge3/etc/profile.d/conda.sh

conda activate disasteradaptivenet_cu128



cd "$BASE"



ln -sfn weights/extracted/ensemble models



echo "================================================"

echo "Prepare Texas Tornadoes TEST_ONLY for 3rd-place xView2"

echo "================================================"

python -u prepare_third_place_texas_tornadoes_test.py



echo "================================================"

echo "Run 3rd-place xView2 weighted ensemble ZERO-SHOT on Texas Tornadoes"

echo "================================================"



rm -rf "$PRED_DIR" "$EVAL_PRED_DIR"



python -u predict_37_weighted.py -dd "$DATA" -o "$PRED_DIR" --size 512 -p naive --activation model --fp16 -b 1 -w 0



echo "================================================"

echo "Create mask aliases for evaluator"

echo "================================================"



python - <<'PY2'

from pathlib import Path

import pandas as pd

import os



DATA = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/third_place_texas_tornadoes_TEST_ONLY")

MASKS = DATA / "masks"

df = pd.read_csv(DATA / "folds.csv")



made = 0

missing = []



for tile_id in df["id"].astype(str):

    loc_src = MASKS / f"test_localization_{tile_id}_target.png"

    dmg_src = MASKS / f"test_damage_{tile_id}_target.png"



    loc_dst = MASKS / f"{tile_id}_pre_disaster.png"

    dmg_dst = MASKS / f"{tile_id}_post_disaster.png"



    if not loc_src.exists():

        missing.append(str(loc_src))

        continue

    if not dmg_src.exists():

        missing.append(str(dmg_src))

        continue



    for src, dst in [(loc_src, loc_dst), (dmg_src, dmg_dst)]:

        if dst.exists() or dst.is_symlink():

            dst.unlink()

        os.symlink(src, dst)

        made += 1



print("Aliases created:", made)

if missing:

    print("Missing source masks:")

    for x in missing[:20]:

        print(x)

    raise SystemExit(1)

PY2



echo "================================================"

echo "Evaluate 3rd-place predictions on Texas Tornadoes"

echo "================================================"



python -u eval_third_place_texas_tornadoes_predictions.py --truth-dir "$DATA" --pred-dir "$EVAL_PRED_DIR" --out-dir "$FINAL_OUT"



echo "DONE: 3rd-place xView2 Texas Tornadoes zero-shot"

