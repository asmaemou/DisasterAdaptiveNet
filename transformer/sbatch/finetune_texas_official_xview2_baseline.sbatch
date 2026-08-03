#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=texas_xv2_ft
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
IMAGE=$PROJECT_ROOT/containers/xview2_baseline_v1.0.sif
PREPARED=$PROJECT_ROOT/output/OfficialxView2-TexasPrepared
OUTPUT=$PROJECT_ROOT/output/OfficialxView2-FineTuneTexas
OFFICIAL=/home/appuser/code/xview-2
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$PREPARED" "$OUTPUT/localization" "$OUTPUT/classification" "$OUTPUT/predictions" "$OUTPUT/scores"
test -s "$IMAGE" || { echo "ERROR: run setup_official_xview2_baseline_container.sbatch first"; exit 1; }

python -u transformer/scripts/check_idabd_xbd_format.py --root "$DATA_ROOT" --splits train val test
python -u transformer/scripts/prepare_texas_for_official_xview2.py \
  --data-root "$DATA_ROOT" --output-root "$PREPARED"

echo "===== FINE-TUNE OFFICIAL LOCALIZATION MODEL ====="
apptainer exec --nv \
  --bind "$PREPARED:/data:ro" \
  --bind "$OUTPUT/localization:/results" \
  "$IMAGE" bash -lc \
  "cd $OFFICIAL/spacenet/src/models && python3 train_model.py /data/localization/dataSet /data/localization/images /data/localization/labels -e 100 -g 0 -r $OFFICIAL/weights/localization.h5 -o /results"

LOCALIZATION_MODEL=$(find "$OUTPUT/localization" -type f \( -name '*.npz' -o -name '*.h5' \) -print | sort | tail -n 1)
if [ -z "$LOCALIZATION_MODEL" ]; then
  echo "ERROR: official localization training produced no checkpoint"
  exit 1
fi
echo "Localization checkpoint: $LOCALIZATION_MODEL"

echo "===== FINE-TUNE OFFICIAL DAMAGE CLASSIFIER ====="
apptainer exec --nv \
  --bind "$PREPARED:/data:ro" \
  --bind "$OUTPUT/classification:/results" \
  "$IMAGE" bash -lc \
  "cd $OFFICIAL/model && python3 damage_classification.py --train_data /data/classification/train --train_csv /data/classification/train.csv --test_data /data/classification/val --test_csv /data/classification/val.csv --model_in $OFFICIAL/weights/classification.hdf5 --model_out /results/texas"

apptainer exec --nv \
  --bind "$PROJECT_ROOT:/project:ro" \
  --bind "$PREPARED:/data:ro" \
  --bind "$OUTPUT/classification:/results" \
  "$IMAGE" python3 /project/transformer/scripts/select_official_xview2_classifier.py \
  --official-model-dir "$OFFICIAL/model" \
  --checkpoint-glob '/results/texas-saved-model-*.hdf5' \
  --val-data /data/classification/val \
  --val-csv /data/classification/val.csv \
  --output /results/best_on_texas_val.hdf5

echo "===== UNTOUCHED TEXAS TEST ====="
for pre_host in "$DATA_ROOT"/test/images/*_pre_disaster.*; do
  name=$(basename "$pre_host")
  stem=${name%_pre_disaster.*}
  ext=${name##*.}
  apptainer exec --nv \
    --bind "$DATA_ROOT/test/images:/submission:ro" \
    --bind "$OUTPUT/predictions:/predictions" \
    --bind "$LOCALIZATION_MODEL:/tmp/texas_localization:ro" \
    --bind "$OUTPUT/classification/best_on_texas_val.hdf5:/tmp/texas_classification.hdf5:ro" \
    "$IMAGE" "$OFFICIAL/utils/inference.sh" \
    -x "$OFFICIAL" \
    -i "/submission/${stem}_pre_disaster.$ext" \
    -p "/submission/${stem}_post_disaster.$ext" \
    -l /tmp/texas_localization \
    -c /tmp/texas_classification.hdf5 \
    -o "/predictions/${stem}_damage.png" \
    -y
done

python -u transformer/scripts/score_official_xview2_texas.py \
  --data-root "$DATA_ROOT" \
  --prediction-dir "$OUTPUT/predictions" \
  --output "$OUTPUT/scores/test_results.json"
echo "DONE: official xBD-pretrained xView2 baseline fine-tuned on Texas"
