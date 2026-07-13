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
    tile_id_l = str(tile_id).lower()
    tile_num_l = str(tile_id).split("_")[-1].lower()

    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        files.extend(PRED.rglob(ext))

    candidates = []
    for f in files:
        name = f.name.lower()
        if tile_id_l in name or tile_num_l in name:
            candidates.append(f)

    hits = []

    if kind == "localization":
        key_words = ["localization", "localisation", "loc", "building", "pre_disaster"]
        reject_words = ["damage", "dmg", "post_disaster"]
    else:
        key_words = ["damage", "dmg", "post_disaster"]
        reject_words = []

    for f in candidates:
        name = f.name.lower()
        if any(k in name for k in key_words) and not any(r in name for r in reject_words):
            hits.append(f)

    if not hits:
        sample = "\n".join(str(f) for f in candidates[:30])
        raise FileNotFoundError(
            f"No {kind} prediction found for {tile_id} under {PRED}\n"
            f"Candidate files:\n{sample}"
        )

    return sorted(set(hits))[0]
