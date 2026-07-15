import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def read_mask(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 0] == arr[:, :, 2]):
            arr = arr[:, :, 0]
        else:
            arr = arr[:, :, 0]
    return arr


def normalize_loc(arr):
    return (arr > 0).astype(np.uint8)


def normalize_damage(arr, loc=None):
    arr = arr.astype(np.int64)
    out = arr.copy()
    out[(out < 0) | (out > 4)] = 0
    if loc is not None:
        out[loc == 0] = 0
    return out.astype(np.uint8)


def f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else (2 * tp) / den


def find_file(root, tile_id, kind):
    tile_id = str(tile_id)
    patterns = []

    if kind == "loc_pred":
        patterns = [
            f"{tile_id}_localization_disaster_prediction.png",
            f"*{tile_id}*localization*prediction*.png",
            f"*{tile_id}*loc*prediction*.png",
        ]
    elif kind == "dmg_pred":
        patterns = [
            f"{tile_id}_damage_disaster_prediction.png",
            f"*{tile_id}*damage*prediction*.png",
            f"*{tile_id}*dmg*prediction*.png",
        ]
    elif kind == "loc_true":
        patterns = [
            f"{tile_id}_pre_disaster.png",
            f"test_localization_{tile_id}_target.png",
            f"*{tile_id}*localization*target*.png",
        ]
    elif kind == "dmg_true":
        patterns = [
            f"{tile_id}_post_disaster.png",
            f"test_damage_{tile_id}_target.png",
            f"*{tile_id}*damage*target*.png",
        ]

    hits = []
    for pat in patterns:
        hits.extend(root.rglob(pat))

    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"Missing {kind} for {tile_id} under {root}")

    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["zero_shot", "finetuned"], default="zero_shot")
    ap.add_argument("--dataset-name", default="target dataset")
    args = ap.parse_args()

    truth = Path(args.truth_dir)
    pred = Path(args.pred_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    folds = truth / "folds.csv"
    masks = truth / "masks"

    if not folds.exists():
        raise FileNotFoundError(f"Missing folds.csv: {folds}")
    if not masks.exists():
        raise FileNotFoundError(f"Missing truth masks folder: {masks}")
    if not pred.exists():
        raise FileNotFoundError(f"Missing prediction folder: {pred}")

    df = pd.read_csv(folds)
    ids = df["id"].astype(str).tolist()

    loc_tp = loc_fp = loc_fn = 0
    dmg_counts = {1: {"tp": 0, "fp": 0, "fn": 0},
                  2: {"tp": 0, "fp": 0, "fn": 0},
                  3: {"tp": 0, "fp": 0, "fn": 0},
                  4: {"tp": 0, "fp": 0, "fn": 0}}

    errors = []

    for tile_id in ids:
        try:
            loc_true_p = find_file(masks, tile_id, "loc_true")
            dmg_true_p = find_file(masks, tile_id, "dmg_true")
            loc_pred_p = find_file(pred, tile_id, "loc_pred")
            dmg_pred_p = find_file(pred, tile_id, "dmg_pred")

            loc_true = normalize_loc(read_mask(loc_true_p))
            loc_pred = normalize_loc(read_mask(loc_pred_p))

            dmg_true_raw = read_mask(dmg_true_p)
            dmg_pred_raw = read_mask(dmg_pred_p)

            dmg_true = normalize_damage(dmg_true_raw, loc_true)
            dmg_pred = normalize_damage(dmg_pred_raw, loc_pred)

            if loc_true.shape != loc_pred.shape:
                raise ValueError(f"Shape mismatch loc for {tile_id}: true {loc_true.shape}, pred {loc_pred.shape}")
            if dmg_true.shape != dmg_pred.shape:
                raise ValueError(f"Shape mismatch dmg for {tile_id}: true {dmg_true.shape}, pred {dmg_pred.shape}")

            loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum())
            loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum())
            loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum())

            valid = loc_true == 1

            for cls in [1, 2, 3, 4]:
                t = (dmg_true == cls) & valid
                p = (dmg_pred == cls) & valid

                dmg_counts[cls]["tp"] += int((p & t).sum())
                dmg_counts[cls]["fp"] += int((p & ~t).sum())
                dmg_counts[cls]["fn"] += int((~p & t).sum())

        except Exception as e:
            errors.append(f"{tile_id}: {e}")

    if errors:
        (out / "eval_errors.txt").write_text("\n".join(errors) + "\n")
        print("ERRORS found:", len(errors))
        for e in errors[:20]:
            print(e)
        raise SystemExit(1)

    loc_f1 = f1(loc_tp, loc_fp, loc_fn)

    class_names = {
        1: "No_damage_F1",
        2: "Minor_damage_F1",
        3: "Major_damage_F1",
        4: "Destroyed_F1",
    }

    metrics = {
        "Localization_F1": loc_f1,
    }

    class_f1s = []
    for cls in [1, 2, 3, 4]:
        c = dmg_counts[cls]
        val = f1(c["tp"], c["fp"], c["fn"])
        metrics[class_names[cls]] = val
        class_f1s.append(val)

    metrics["Macro_F1_damage_classes"] = float(np.mean(class_f1s))
    metrics["Overall_xView2_style_score_0.3loc_0.7damage"] = 0.3 * metrics["Localization_F1"] + 0.7 * metrics["Macro_F1_damage_classes"]

    lines = []
    if args.mode == "finetuned":
        lines.append(f"1st-place xView2 winner FINE-TUNED on {args.dataset_name} official split")
        lines.append(
            f"Fine-tuned on {args.dataset_name} train, selected on validation, "
            f"evaluated on held-out test."
        )
    else:
        lines.append("1st-place xView2 winner ZERO-SHOT evaluation")
        lines.append(f"No {args.dataset_name} fine-tuning used.")
    lines.append(f"Truth: {truth}")
    lines.append(f"Pred:  {pred}")
    lines.append(f"Samples: {len(ids)}")
    lines.append("")
    for k, v in metrics.items():
        lines.append(f"{k}: {v:.6f}")

    summary = "\n".join(lines) + "\n"
    print(summary)

    (out / "metrics_summary.txt").write_text(summary)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Wrote:", out / "metrics_summary.txt")


if __name__ == "__main__":
    main()
