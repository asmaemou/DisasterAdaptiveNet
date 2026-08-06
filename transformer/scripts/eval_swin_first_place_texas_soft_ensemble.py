#!/usr/bin/env python3
"""Validation-selected fusion of Texas Swin-T and first-place xView2 models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import cv2

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


def verify_truth_alignment(samples, reference_root: Path, split: str) -> None:
    mismatches = []
    for sample in samples:
        candidates = [
            reference_root / split / "masks" / f"{sample['stem']}_pre_disaster.png",
            reference_root / split / "targets" / f"{sample['stem']}_pre_disaster_target.png",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        reference = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if reference is None:
            mismatches.append(f"missing reference mask: {path}")
            continue
        if reference.ndim == 3:
            reference = reference[..., 0]
        reference = (reference > 0).astype(np.uint8)
        truth = (sample["loc_true"] > 0).astype(np.uint8)
        if reference.shape != truth.shape:
            mismatches.append(
                f"{sample['stem']}: shape reference={reference.shape}, exported={truth.shape}"
            )
        elif not np.array_equal(reference, truth):
            differing = int(np.count_nonzero(reference != truth))
            mismatches.append(f"{sample['stem']}: {differing} localization pixels differ")
    if mismatches:
        preview = "\n".join(f"  - {item}" for item in mismatches[:20])
        raise RuntimeError(
            f"FAIL-FAST: {split} truth geometry does not match first-place coordinates "
            f"for {len(mismatches)} sample(s):\n{preview}"
        )
    print(
        f"PASS alignment gate: split={split}, samples={len(samples)}, "
        "exported truth exactly matches first-place prepared masks",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swin-root", type=Path, required=True)
    parser.add_argument("--first-place-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-place-data-root", type=Path, required=True)
    parser.add_argument("--minimum-first-place-val-loc-f1", type=float, default=0.8)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--expected-val-samples", type=int, default=45)
    parser.add_argument("--expected-test-samples", type=int, default=46)
    parser.add_argument("--selection-objective", choices=["macro", "official"], default="macro")
    parser.add_argument("--minor-dilation-kernel", type=int, default=3)
    parser.add_argument("--experiment-label", default="Texas-fine-tuned ImageNet Swin-T + Texas-fine-tuned first-place xView2 soft ensemble")
    parser.add_argument("--selection-label", default="Fusion weights and localization threshold selected only on Texas validation; Texas test evaluated once.")
    return parser.parse_args()


def main():
    args = parse_args()
    fusion.MINOR_DILATION_KERNEL = args.minor_dilation_kernel
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = fusion.load_split(args.swin_root / "val", args.first_place_root / "val")
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(f"Expected {args.expected_val_samples} validation samples, found {len(validation)}")
    verify_truth_alignment(validation, args.first_place_data_root, "val")
    first_place_validation = evaluate(validation, "first_place")
    print(
        "First-place validation preflight:",
        json.dumps(first_place_validation, indent=2),
        flush=True,
    )
    if first_place_validation["localization_f1"] < args.minimum_first_place_val_loc_f1:
        raise RuntimeError(
            "FAIL-FAST: first-place validation localization F1 "
            f"{first_place_validation['localization_f1']:.6f} is below required "
            f"{args.minimum_first_place_val_loc_f1:.6f}; refusing to tune/report fusion"
        )
    rows = []
    for alpha in values(args.alphas):
        for beta in values(args.betas):
            for threshold in values(args.thresholds):
                rows.append({
                    "swin_localization_weight": alpha, "swin_damage_weight": beta,
                    "localization_threshold": threshold,
                    **evaluate(validation, "hybrid", alpha, beta, threshold),
                })
    # Match the metric used by the standalone Texas winner evaluations and in
    # the paper tables: 0.3 * localization F1 + 0.7 * macro damage-class F1.
    # The harmonic score remains a diagnostic but must not select a different
    # model than the reported macro-composite objective.
    objective = "official_xview2_score" if args.selection_objective == "official" else "macro_composite_score"
    rows.sort(
        key=lambda row: (
            row[objective],
            row["macro_damage_f1"],
            row["localization_f1"],
        ),
        reverse=True,
    )
    selected = rows[0]
    with (args.output_dir / "validation_fusion_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected))
        writer.writeheader()
        writer.writerows(rows)
    print("Selected on validation only:", json.dumps(selected, indent=2), flush=True)

    test = fusion.load_split(args.swin_root / "test", args.first_place_root / "test")
    if len(test) != args.expected_test_samples:
        raise RuntimeError(f"Expected {args.expected_test_samples} test samples, found {len(test)}")
    verify_truth_alignment(test, args.first_place_data_root, "test")
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
        "experiment": args.experiment_label,
        "selection": args.selection_label,
        "selection_objective": objective,
        "alpha_definition": "Swin localization probability weight",
        "beta_definition": "Swin damage probability weight",
        "selected_parameters": {
            "swin_localization_weight": alpha,
            "first_place_localization_weight": 1.0 - alpha,
            "swin_damage_weight": beta,
            "first_place_damage_weight": 1.0 - beta,
            "localization_threshold": threshold,
            "minor_dilation_kernel": args.minor_dilation_kernel,
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
