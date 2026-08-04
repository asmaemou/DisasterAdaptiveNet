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
ENV_DIR=$PROJECT_ROOT/.conda/xview2_baseline_py37_v2
PREPARED=$PROJECT_ROOT/output/OfficialxView2-TexasPrepared
OUTPUT=$PROJECT_ROOT/output/OfficialxView2-FineTuneTexas
OFFICIAL=$PROJECT_ROOT/third_party/xview2_baseline
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$PREPARED" "$OUTPUT/localization" "$OUTPUT/classification" "$OUTPUT/predictions" "$OUTPUT/scores"
mkdir -p "$OUTPUT/internal_logs"
test -x "$ENV_DIR/bin/python" || { echo "ERROR: run setup_official_xview2_baseline_container.sbatch first"; exit 1; }
test -s "$OFFICIAL/weights/localization.h5" || { echo "ERROR: official localization weights missing"; exit 1; }
test -s "$OFFICIAL/weights/classification.hdf5" || { echo "ERROR: official classification weights missing"; exit 1; }
export PATH=$ENV_DIR/bin:$PATH
export PYTHONPATH=$OFFICIAL:$OFFICIAL/model:$OFFICIAL/spacenet/src
# The released 2019 TensorFlow/Chainer runtime cannot execute on a Blackwell
# GPU. Run the official code on CPU rather than changing framework or weights.
export CUDA_VISIBLE_DEVICES=""

python -u transformer/scripts/check_idabd_xbd_format.py --root "$DATA_ROOT" --splits train val test
python -u transformer/scripts/prepare_texas_for_official_xview2.py \
  --data-root "$DATA_ROOT" --output-root "$PREPARED"

echo "===== FINE-TUNE OFFICIAL LOCALIZATION MODEL ====="
(cd "$OFFICIAL/spacenet/src/models" && \
  python train_model.py \
    "$PREPARED/localization/dataSet" \
    "$PREPARED/localization/images" \
    "$PREPARED/localization/labels" \
    -e 100 -g -1 -r "$OFFICIAL/weights/localization.h5" -o "$OUTPUT/localization")

LOCALIZATION_MODEL=$(find "$OUTPUT/localization" -type f \( -name '*.npz' -o -name '*.h5' \) -print | sort | tail -n 1)
if [ -z "$LOCALIZATION_MODEL" ]; then
  echo "ERROR: official localization training produced no checkpoint"
  exit 1
fi
echo "Localization checkpoint: $LOCALIZATION_MODEL"

echo "===== FINE-TUNE OFFICIAL DAMAGE CLASSIFIER ====="
(cd "$OFFICIAL/model" && \
  python damage_classification.py \
    --train_data "$PREPARED/classification/train" \
    --train_csv "$PREPARED/classification/train.csv" \
    --test_data "$PREPARED/classification/val" \
    --test_csv "$PREPARED/classification/val.csv" \
    --model_in "$OFFICIAL/weights/classification.hdf5" \
    --model_out "$OUTPUT/classification/texas")

python transformer/scripts/select_official_xview2_classifier.py \
  --official-model-dir "$OFFICIAL/model" \
  --checkpoint-glob "$OUTPUT/classification/texas-saved-model-*.hdf5" \
  --val-data "$PREPARED/classification/val" \
  --val-csv "$PREPARED/classification/val.csv" \
  --output "$OUTPUT/classification/best_on_texas_val.hdf5"

echo "===== UNTOUCHED TEXAS TEST ====="
for pre_host in "$DATA_ROOT"/test/images/*_pre_disaster.*; do
  name=$(basename "$pre_host")
  stem=${name%_pre_disaster.*}
  ext=${name##*.}
  : > /tmp/inference_log
  set +e
  "$OFFICIAL/utils/inference.sh" \
    -x "$OFFICIAL" \
    -i "$pre_host" \
    -p "$DATA_ROOT/test/images/${stem}_post_disaster.$ext" \
    -l "$LOCALIZATION_MODEL" \
    -c "$OUTPUT/classification/best_on_texas_val.hdf5" \
    -o "$OUTPUT/predictions/${stem}_damage.png" \
    -y
  inference_status=$?
  set -e
  cp /tmp/inference_log "$OUTPUT/internal_logs/${stem}.log" 2>/dev/null || true
  if [ "$inference_status" -ne 0 ]; then
    echo "ERROR: official inference failed for $stem with exit code $inference_status"
    cat /tmp/inference_log || true
    exit "$inference_status"
  fi
done

python -u transformer/scripts/score_official_xview2_texas.py \
  --data-root "$DATA_ROOT" \
  --prediction-dir "$OUTPUT/predictions" \
  --output "$OUTPUT/scores/test_results.json"
echo "DONE: official xBD-pretrained xView2 baseline fine-tuned on Texas"
