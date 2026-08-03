#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=texas_xv2_zero
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
DATA_ROOT=/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed
IMAGE=$PROJECT_ROOT/containers/xview2_baseline_v1.0.sif
OUTPUT=$PROJECT_ROOT/output/OfficialxView2-ZeroShotTexas
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs "$OUTPUT/predictions" "$OUTPUT/scores"
test -s "$IMAGE" || { echo "ERROR: run setup_official_xview2_baseline_container.sbatch first"; exit 1; }
command -v apptainer >/dev/null || { echo "ERROR: apptainer unavailable"; exit 1; }

echo "STRICT ZERO-SHOT: released xBD weights; Texas test images only"
for pre_host in "$DATA_ROOT"/test/images/*_pre_disaster.*; do
  name=$(basename "$pre_host")
  stem=${name%_pre_disaster.*}
  ext=${name##*.}
  post_host=$DATA_ROOT/test/images/${stem}_post_disaster.$ext
  test -f "$post_host" || { echo "ERROR: missing post image for $stem"; exit 1; }
  echo "Inference: $stem"
  apptainer run --nv \
    --bind "$DATA_ROOT/test/images:/submission:ro" \
    --bind "$OUTPUT/predictions:/output" \
    "$IMAGE" \
    "/submission/${stem}_pre_disaster.$ext" \
    "/submission/${stem}_post_disaster.$ext" \
    "/output/${stem}_localization.png" \
    "/output/${stem}_damage.png"
done

python -u transformer/scripts/score_official_xview2_texas.py \
  --data-root "$DATA_ROOT" \
  --prediction-dir "$OUTPUT/predictions" \
  --output "$OUTPUT/scores/test_results.json"
echo "DONE: official xView2 v1.0 zero-shot Texas test"
