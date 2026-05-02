#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=xbdv5_idabd_zs
#SBATCH --output=transformer/sbatch/logs/%x_%j.out
#SBATCH --error=transformer/sbatch/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
mkdir -p transformer/sbatch/logs

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found."
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate disasteradaptivenet_cu128

# ---------------------------------------------------------------------
# IDA-BD input and fixed split
# ---------------------------------------------------------------------
IDABD_ROOT=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/idabd

# Use the exact same IDA-BD 80/10/10 split from your previous v2 run.
# This keeps the real test split at 8 images.
IDABD_SPLIT_FILE=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_idabd_hrtbda_v2_cascaded_phase1mask_80_10_10/idabd_splits_seed42_80_10_10.json

# Converted xBD-style root used only for evaluation.
IDABD_XBDSTYLE_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/idabd_80_10_10_xbdstyle_for_xbd_v5_zero_shot

# ---------------------------------------------------------------------
# xBD-trained model checkpoints: no retraining, zero-shot only
# ---------------------------------------------------------------------
XBD_PHASE1_ORIG=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v2_4branch_warmup_amp/checkpoints/phase1_best.pt
XBD_PHASE2_BEST=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/train_plus_tier3_test_xbd_hrtbda_v5_multilabel_crop_cascade/checkpoints/phase2_best.pt

# Output for zero-shot evaluation.
OUTPUT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/zero_shot_xbd_v5_on_idabd_test8
THRESH_CKPT_DIR=$OUTPUT_ROOT/threshold_checkpoints

# The xBD hold-selected setting was threshold=0.40, dilation=minor.
# We also sweep thresholds/dilation on the real IDA-BD test split for diagnosis.
THRESHOLDS="0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"
DILATIONS="none minor"

mkdir -p "$OUTPUT_ROOT" "$THRESH_CKPT_DIR"

echo "===== ZERO-SHOT xBD v5 -> IDA-BD JOB INFO ====="
date
hostname
pwd
which python
python --version
nvidia-smi || true
echo "IDABD_ROOT: $IDABD_ROOT"
echo "IDABD_SPLIT_FILE: $IDABD_SPLIT_FILE"
echo "IDABD_XBDSTYLE_ROOT: $IDABD_XBDSTYLE_ROOT"
echo "XBD_PHASE1_ORIG: $XBD_PHASE1_ORIG"
echo "XBD_PHASE2_BEST: $XBD_PHASE2_BEST"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "THRESHOLDS: $THRESHOLDS"
echo "DILATIONS: $DILATIONS"
echo "================================================"

if [ ! -f "$XBD_PHASE1_ORIG" ]; then
  echo "ERROR: missing xBD Phase-I checkpoint: $XBD_PHASE1_ORIG"
  exit 1
fi
if [ ! -f "$XBD_PHASE2_BEST" ]; then
  echo "ERROR: missing xBD Phase-II checkpoint: $XBD_PHASE2_BEST"
  exit 1
fi
if [ ! -f "$IDABD_SPLIT_FILE" ]; then
  echo "ERROR: missing IDA-BD split file: $IDABD_SPLIT_FILE"
  exit 1
fi

# ---------------------------------------------------------------------
# 1) Prepare IDA-BD as xBD-style data, preserving the existing 70/9/8 split
# ---------------------------------------------------------------------
python -u transformer/scripts/prepare_idabd_xbdstyle_from_split.py \
  --idabd-root "$IDABD_ROOT" \
  --out-root "$IDABD_XBDSTYLE_ROOT" \
  --split-file "$IDABD_SPLIT_FILE" \
  --seed 42 \
  --image-mode symlink

# ---------------------------------------------------------------------
# 2) Create copied Phase-I checkpoints with different stored thresholds
# ---------------------------------------------------------------------
python - <<PY
import torch
from pathlib import Path

src = Path("$XBD_PHASE1_ORIG")
out_dir = Path("$THRESH_CKPT_DIR")
out_dir.mkdir(parents=True, exist_ok=True)

for th in [float(x) for x in "$THRESHOLDS".split()]:
    ckpt = torch.load(src, map_location="cpu")
    ckpt["best_threshold"] = float(th)
    dst = out_dir / f"phase1_xbd_best_threshold_{th:.2f}.pt"
    torch.save(ckpt, dst)
    print(f"Wrote threshold checkpoint: {dst}")
PY

# ---------------------------------------------------------------------
# 3) Run diagnostic real-test sweep on the IDA-BD 8-image test split
# ---------------------------------------------------------------------
for TH in $THRESHOLDS; do
  for DIL in $DILATIONS; do
    PHASE1_CKPT=$THRESH_CKPT_DIR/phase1_xbd_best_threshold_${TH}.pt
    OUTDIR=$OUTPUT_ROOT/test_th${TH}_dil${DIL}
    mkdir -p "$OUTDIR" "$OUTDIR/checkpoints" "$OUTDIR/scores"

    echo
    echo "=================================================="
    echo "Zero-shot xBD v5 on IDA-BD real test: threshold=$TH dilation=$DIL"
    echo "PHASE1_CKPT=$PHASE1_CKPT"
    echo "PHASE2_BEST=$XBD_PHASE2_BEST"
    echo "OUTDIR=$OUTDIR"
    echo "=================================================="

    python -u transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py \
      --phase test \
      --phase1-checkpoint "$PHASE1_CKPT" \
      --phase2-checkpoint "$XBD_PHASE2_BEST" \
      --xbd-root "$IDABD_XBDSTYLE_ROOT" \
      --train-split train \
      --val-split hold \
      --test-split test \
      --output-dir "$OUTDIR" \
      --batch-size 2 \
      --eval-batch-size 1 \
      --grad-accum-steps 4 \
      --num-workers 8 \
      --img-size 1024 \
      --phase2-crop-size 608 \
      --base-channels 48 \
      --decoder-channels 128 \
      --window-size 8 \
      --postprocess-dilation "$DIL" \
      --dilation-kernel 3 \
      --amp
  done
done

# ---------------------------------------------------------------------
# 4) Summarize results from JSON files
# ---------------------------------------------------------------------
python - <<PY
import json
from pathlib import Path

root = Path("$OUTPUT_ROOT")
rows = []
for p in sorted(root.glob("test_th*_dil*/scores/scores_xbd_test_hrtbda_v5_multilabel_crop_cascade.json")):
    name = p.parents[1].name  # test_th0.40_dilminor
    parts = name.replace("test_th", "").split("_dil")
    th = float(parts[0])
    dil = parts[1]
    with open(p, "r", encoding="utf-8") as f:
        r = json.load(f)
    rows.append({
        "threshold": th,
        "dilation": dil,
        "loc_f1": float(r.get("localization_f1", 0.0)),
        "no_f1": float(r.get("damage_f1_no_damage", 0.0)),
        "minor_f1": float(r.get("damage_f1_minor_damage", 0.0)),
        "major_f1": float(r.get("damage_f1_major_damage", 0.0)),
        "destroyed_f1": float(r.get("damage_f1_destroyed", 0.0)),
        "damage_f1": float(r.get("damage_f1", 0.0)),
        "overall": float(r.get("score", 0.0)),
        "json": str(p),
    })

rows.sort(key=lambda x: x["overall"], reverse=True)

print()
print("================ ZERO-SHOT xBD V5 -> IDA-BD REAL 8-IMAGE TEST SWEEP ================")
print("Rank | Threshold | Dilation | Loc F1 | No F1 | Minor F1 | Major F1 | Destroyed F1 | Damage F1 | Overall")
for i, r in enumerate(rows, 1):
    print(
        f"{i:4d} | {r['threshold']:.2f} | {r['dilation']:<5s} | "
        f"{r['loc_f1']:.6f} | {r['no_f1']:.6f} | {r['minor_f1']:.6f} | "
        f"{r['major_f1']:.6f} | {r['destroyed_f1']:.6f} | "
        f"{r['damage_f1']:.6f} | {r['overall']:.6f}"
    )
print("=====================================================================================")

out_json = root / "zero_shot_xbd_v5_on_idabd_real_test_sweep_results.json"
out_txt = root / "zero_shot_xbd_v5_on_idabd_real_test_sweep_summary.txt"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("ZERO-SHOT xBD V5 -> IDA-BD REAL 8-IMAGE TEST SWEEP\n")
    for i, r in enumerate(rows, 1):
        f.write(
            f"{i:4d} | th={r['threshold']:.2f} | dil={r['dilation']} | "
            f"loc={r['loc_f1']:.6f} | no={r['no_f1']:.6f} | minor={r['minor_f1']:.6f} | "
            f"major={r['major_f1']:.6f} | destroyed={r['destroyed_f1']:.6f} | "
            f"damage={r['damage_f1']:.6f} | overall={r['overall']:.6f}\n"
        )
print(f"Wrote: {out_json}")
print(f"Wrote: {out_txt}")
PY

echo "Done."
