#!/bin/bash
# Transparent equivalent of the official utils/inference.sh pipeline. It keeps
# every intermediate file and does not hide exceptions behind signal traps.
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 OFFICIAL_REPO PRE_IMAGE POST_IMAGE LOC_WEIGHTS CLASS_WEIGHTS OUTPUT_PNG WORK_DIR"
  exit 2
fi

OFFICIAL=$1
PRE_IMAGE=$2
POST_IMAGE=$3
LOC_WEIGHTS=$4
CLASS_WEIGHTS=$5
OUTPUT_PNG=$6
WORK_DIR=$7

input_name=$(basename "$PRE_IMAGE")
stem=${input_name%.*}
RUN_DIR=$WORK_DIR/$stem
LABEL_DIR=$RUN_DIR/labels
POLYGON_DIR=$RUN_DIR/output_polygons
mkdir -p "$LABEL_DIR" "$POLYGON_DIR" "$(dirname "$OUTPUT_PNG")"

echo "[1/5] Official SpaceNet localization"
(
  cd "$OFFICIAL/spacenet/inference"
  python3 ./inference.py \
    --input "$PRE_IMAGE" \
    --weights "$LOC_WEIGHTS" \
    --mean "$OFFICIAL/weights/mean.npy" \
    --output "$LABEL_DIR/$stem.json"
)

echo "[2/5] Extract post-disaster building chips"
(
  cd "$OFFICIAL/model"
  python3 ./process_data_inference.py \
    --input_img "$POST_IMAGE" \
    --label_path "$LABEL_DIR/$stem.json" \
    --output_dir "$POLYGON_DIR" \
    --output_csv "$RUN_DIR/output.csv"
)

echo "[3/5] Official ResNet50 + shallow-CNN damage classification"
(
  cd "$OFFICIAL/model"
  python3 ./damage_inference.py \
    --test_data "$POLYGON_DIR" \
    --test_csv "$RUN_DIR/output.csv" \
    --model_weights "$CLASS_WEIGHTS" \
    --output_json "$RUN_DIR/classification_inference.json"
)

echo "[4/5] Combine localization polygons and damage labels"
python3 "$OFFICIAL/utils/combine_jsons.py" \
  --polys "$LABEL_DIR/$stem.json" \
  --classes "$RUN_DIR/classification_inference.json" \
  --output "$RUN_DIR/inference.json"

echo "[5/5] Rasterize official prediction"
python3 "$OFFICIAL/utils/inference_image_output.py" \
  --input "$RUN_DIR/inference.json" \
  --output "$OUTPUT_PNG"

test -s "$OUTPUT_PNG"
echo "Official xView2 inference complete: $OUTPUT_PNG"
