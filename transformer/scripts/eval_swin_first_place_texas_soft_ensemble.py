#!/usr/bin/env python3
"""Validation-selected fusion of Texas Swin-T and first-place xView2 models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import eval_hrtbda_first_place_turkey_soft_ensemble as fusion


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def harmonic(values, epsilon: float = 1e-6) -> float:
    return len(values) / sum(1.0 / max(float(value), epsilon) for value in values)


def evaluate(samples, mode: str, alpha=0.5, beta=0.5, threshold=0.5):
    loc_tp = loc_fp = loc_fn = 0
    counts = {class_id: [0, 0, 0] for class_id in range(1, 5)}
    for sample in samples:
        loc_pred, damage_pred = fusion.predictions(sample, mode, alpha, beta, threshold)
        loc_true = sample["loc_true"] > 0
        damage_true = sample["damage_true"]
        loc_tp += int(((loc_pred == 1) & loc_true).sum())
        loc_fp += int(((loc_pred == 1) & ~loc_true).sum())
        loc_fn += int(((loc_pred == 0) & loc_true).sum())
        valid = (damage_true >= 1) & (damage_true <= 4)
        for class_id in range(1, 5):
            truth = (damage_true == class_id) & valid
            prediction = (damage_pred == class_id) & valid
            counts[class_id][0] += int((truth & prediction).sum())
            counts[class_id][1] += int((~truth & prediction & valid).sum())
            counts[class_id][2] += int((truth & ~prediction).sum())
    localization = f1(loc_tp, loc_fp, loc_fn)
    classes = [f1(*counts[class_id]) for class_id in range(1, 5)]
    macro = float(np.mean(classes))
    damage_harmonic = harmonic(classes)
    return {
        "localization_f1": localization,
        "no_damage_f1": classes[0], "minor_damage_f1": classes[1],
        "major_damage_f1": classes[2], "destroyed_f1": classes[3],
        "macro_damage_f1": macro, "harmonic_damage_f1": damage_harmonic,
        "official_xview2_score": 0.3 * localization + 0.7 * damage_harmonic,
        "macro_composite_score": 0.3 * localization + 0.7 * macro,
    }


def values(text: str):
    return [float(value) for value in text.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swin-root", type=Path, required=True)
    parser.add_argument("--first-place-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--expected-val-samples", type=int, default=45)
    parser.add_argument("--expected-test-samples", type=int, default=46)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = fusion.load_split(args.swin_root / "val", args.first_place_root / "val")
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(f"Expected {args.expected_val_samples} validation samples, found {len(validation)}")
    rows = []
    for alpha in values(args.alphas):
        for beta in values(args.betas):
            for threshold in values(args.thresholds):
                rows.append({
                    "swin_localization_weight": alpha, "swin_damage_weight": beta,
                    "localization_threshold": threshold,
                    **evaluate(validation, "hybrid", alpha, beta, threshold),
                })
    rows.sort(
        key=lambda row: (row["official_xview2_score"], row["harmonic_damage_f1"], row["macro_damage_f1"]),
        reverse=True,
    )
    selected = rows[0]
    with (args.output_dir / "validation_fusion_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected))
        writer.writeheader()
        writer.writerows(rows)
    print("Selected on Texas validation only:", json.dumps(selected, indent=2), flush=True)

    test = fusion.load_split(args.swin_root / "test", args.first_place_root / "test")
    if len(test) != args.expected_test_samples:
        raise RuntimeError(f"Expected {args.expected_test_samples} test samples, found {len(test)}")
    alpha = float(selected["swin_localization_weight"])
    beta = float(selected["swin_damage_weight"])
    threshold = float(selected["localization_threshold"])
    validation_metrics = {
        "swin": evaluate(validation, "hrtbda"),
        "first_place": evaluate(validation, "first_place"),
        "equal_ensemble": evaluate(validation, "hybrid", 0.5, 0.5, 0.5),
        "selected_ensemble": evaluate(validation, "hybrid", alpha, beta, threshold),
    }
    test_metrics = {
        "swin": evaluate(test, "hrtbda"),
        "first_place": evaluate(test, "first_place"),
        "equal_ensemble": evaluate(test, "hybrid", 0.5, 0.5, 0.5),
        "selected_ensemble": evaluate(test, "hybrid", alpha, beta, threshold),
    }
    fusion.save_predictions(test, args.output_dir / "selected_test_predictions", alpha, beta, threshold)
    summary = {
        "experiment": "Texas-fine-tuned ImageNet Swin-T + Texas-fine-tuned first-place xView2 soft ensemble",
        "selection": "Fusion weights and localization threshold selected only on Texas validation; Texas test evaluated once.",
        "alpha_definition": "Swin localization probability weight",
        "beta_definition": "Swin damage probability weight",
        "selected_parameters": {
            "swin_localization_weight": alpha,
            "first_place_localization_weight": 1.0 - alpha,
            "swin_damage_weight": beta,
            "first_place_damage_weight": 1.0 - beta,
            "localization_threshold": threshold,
        },
        "validation_samples": len(validation), "test_samples": len(test),
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
    }
    (args.output_dir / "ensemble_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [summary["experiment"], summary["selection"], "", json.dumps(summary["selected_parameters"], indent=2), ""]
    for name, metrics in test_metrics.items():
        lines.extend([
            f"TEST - {name}",
            *(f"{key}: {value:.6f}" for key, value in metrics.items()), "",
        ])
    (args.output_dir / "ensemble_metrics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
