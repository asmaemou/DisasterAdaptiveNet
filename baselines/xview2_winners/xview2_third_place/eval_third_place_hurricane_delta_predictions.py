from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--truth-dir", required=True)
parser.add_argument("--pred-dir", required=True)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()

TRUTH = Path(args.truth_dir)
PRED = Path(args.pred_dir)
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

folds = pd.read_csv(TRUTH / "folds.csv")
ids = list(folds["id"].astype(str))

def read_any(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def decode_mask(path: Path, loc=False):
    img = read_any(path)

    if img.ndim == 3:
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        out = np.zeros(rgb.shape[:2], dtype=np.uint8)

        color_map = {
            (0, 0, 0): 0,
            (255, 255, 255): 1,
            (0, 255, 0): 1,
            (255, 255, 0): 2,
            (0, 0, 255): 2,
            (255, 128, 0): 3,
            (255, 165, 0): 3,
            (255, 0, 0): 4,
        }

        for color, cls in color_map.items():
            m = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
            out[m] = cls

        if out.max() > 0:
            return out

        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    img = img.astype(np.uint8)

    if loc:
        return (img > 0).astype(np.uint8)

    vals = sorted([int(v) for v in np.unique(img)])
    if len(vals) <= 5 and max(vals) <= 4:
        return img

    out = np.zeros_like(img, dtype=np.uint8)
    nonzero = [v for v in vals if v != 0]

    if len(nonzero) <= 4:
        for i, v in enumerate(nonzero, start=1):
            out[img == v] = i
        return out

    out[img > 0] = 1
    return out

def resize_like(a, shape):
    if a.shape[:2] == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

def f1_binary(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)

def find_pred(tile_id, kind):
    patterns = [
        f"*{kind}*{tile_id}*prediction*.png",
        f"*{tile_id}*{kind}*prediction*.png",
        f"*{kind}*{tile_id}*.png",
        f"*{tile_id}*{kind}*.png",
    ]
    hits = []
    for pat in patterns:
        hits.extend(PRED.rglob(pat))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"No {kind} prediction found for {tile_id} under {PRED}")
    return hits[0]

loc_scores = []
class_scores = {1: [], 2: [], 3: [], 4: []}
errors = []

for tile_id in ids:
    try:
        gt_pre = decode_mask(TRUTH / "masks" / f"{tile_id}_pre_disaster.png", loc=True)
        gt_post = decode_mask(TRUTH / "masks" / f"{tile_id}_post_disaster.png", loc=False)

        pred_loc = decode_mask(find_pred(tile_id, "localization"), loc=True)
        pred_dmg = decode_mask(find_pred(tile_id, "damage"), loc=False)

        gt_pre = resize_like(gt_pre, pred_loc.shape[:2])
        gt_post = resize_like(gt_post, pred_dmg.shape[:2])

        loc_scores.append(f1_binary(pred_loc > 0, gt_pre > 0))

        for cls in [1, 2, 3, 4]:
            class_scores[cls].append(f1_binary(pred_dmg == cls, gt_post == cls))

    except Exception as e:
        errors.append((tile_id, str(e)))

if errors:
    print("ERROR: failed to evaluate some predictions.")
    for e in errors[:30]:
        print(e)
    print("Total errors:", len(errors))
    raise SystemExit(2)

metrics = {
    "Localization_F1": float(np.mean(loc_scores)),
    "No_damage_F1": float(np.mean(class_scores[1])),
    "Minor_damage_F1": float(np.mean(class_scores[2])),
    "Major_damage_F1": float(np.mean(class_scores[3])),
    "Destroyed_F1": float(np.mean(class_scores[4])),
}
metrics["Macro_F1_damage_classes"] = float(np.mean([
    metrics["No_damage_F1"],
    metrics["Minor_damage_F1"],
    metrics["Major_damage_F1"],
    metrics["Destroyed_F1"],
]))

with open(OUT / "metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(OUT / "metrics_summary.txt", "w") as f:
    f.write("3rd-place xView2 weighted ensemble ZERO-SHOT on Hurricane Delta TEST\n")
    f.write("No Hurricane Delta fine-tuning used.\n\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.6f}\n")

print("Metrics TXT:", OUT / "metrics_summary.txt")
print("Metrics JSON:", OUT / "metrics_summary.json")
print()
print((OUT / "metrics_summary.txt").read_text())