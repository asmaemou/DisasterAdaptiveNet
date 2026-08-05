#!/usr/bin/env python3
"""Score official xView2 baseline raster predictions against Texas masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def f1(tp: int, fp: int, fn: int) -> float:
    return 0.0 if 2 * tp + fp + fn == 0 else 2.0 * tp / (2 * tp + fp + fn)


def harmonic(values, eps: float = 1e-6) -> float:
    return len(values) / sum(1.0 / max(v, eps) for v in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    targets = Path(args.data_root) / "test" / "targets"
    predictions = Path(args.prediction_dir)
    loc_counts = [0, 0, 0]
    class_counts = {c: [0, 0, 0] for c in range(1, 5)}
    files = sorted(predictions.glob("*_damage.png"))
    if not files:
        raise RuntimeError(f"No *_damage.png predictions found in {predictions}")
    for prediction_path in files:
        # The official xView2 runtime uses Python 3.7; str.removesuffix was
        # introduced in Python 3.9.
        stem = prediction_path.name[:-len("_damage.png")]
        loc_path = targets / f"{stem}_pre_disaster_target.png"
        damage_path = targets / f"{stem}_post_disaster_target.png"
        if not (loc_path.exists() and damage_path.exists()):
            raise FileNotFoundError(f"Ground truth missing for prediction {stem}")
        pred = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
        truth_loc = cv2.imread(str(loc_path), cv2.IMREAD_UNCHANGED)
        truth_damage = cv2.imread(str(damage_path), cv2.IMREAD_UNCHANGED)
        if pred.ndim == 3: pred = pred[..., 0]
        if truth_loc.ndim == 3: truth_loc = truth_loc[..., 0]
        if truth_damage.ndim == 3: truth_damage = truth_damage[..., 0]
        if pred.shape != truth_loc.shape:
            pred = cv2.resize(pred, (truth_loc.shape[1], truth_loc.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_loc, truth_loc = pred > 0, truth_loc > 0
        loc_counts[0] += int((pred_loc & truth_loc).sum())
        loc_counts[1] += int((pred_loc & ~truth_loc).sum())
        loc_counts[2] += int((~pred_loc & truth_loc).sum())
        valid = np.isin(truth_damage, [1, 2, 3, 4]) & truth_loc
        pv, tv = pred[valid], truth_damage[valid]
        for cls in range(1, 5):
            class_counts[cls][0] += int(((pv == cls) & (tv == cls)).sum())
            class_counts[cls][1] += int(((pv == cls) & (tv != cls)).sum())
            class_counts[cls][2] += int(((pv != cls) & (tv == cls)).sum())
    loc_f1 = f1(*loc_counts)
    class_f1 = [f1(*class_counts[c]) for c in range(1, 5)]
    damage_f1 = harmonic(class_f1)
    result = {
        "test_images": len(files), "localization_f1": loc_f1,
        "no_damage_f1": class_f1[0], "minor_damage_f1": class_f1[1],
        "major_damage_f1": class_f1[2], "destroyed_f1": class_f1[3],
        "damage_f1": damage_f1, "overall_score": 0.3 * loc_f1 + 0.7 * damage_f1,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
