#!/usr/bin/env python3
"""Validation-selected soft fusion of Texas Swin-T and xView2 second place."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def harmonic(values, epsilon=1e-6):
    return len(values) / sum(1.0 / max(float(value), epsilon) for value in values)


def dilate_minor(damage, loc, kernel_size):
    if kernel_size <= 1:
        return damage
    output = damage.copy()
    minor = cv2.dilate((damage == 2).astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)) > 0
    output[minor & loc & (output == 1)] = 2
    return output


def load_split(swin_root: Path, second_root: Path):
    samples = []
    swin_files = sorted(swin_root.glob("*.npz"))
    second_files = {path.stem: path for path in second_root.glob("*.npz")}
    if not swin_files:
        raise RuntimeError(f"No Swin probabilities found: {swin_root}")
    if {path.stem for path in swin_files} != set(second_files):
        raise RuntimeError("Swin and second-place probability IDs do not match")
    for swin_path in swin_files:
        with np.load(swin_path) as swin, np.load(second_files[swin_path.stem]) as second:
            swin_truth = (swin["loc_true"] > 0).astype(np.uint8)
            second_truth = (second["loc_true"] > 0).astype(np.uint8)
            if not np.array_equal(swin_truth, second_truth):
                raise RuntimeError(f"FAIL-FAST: localization truth mismatch for {swin_path.stem}")
            if not np.array_equal(swin["damage_true"], second["damage_true"]):
                raise RuntimeError(f"FAIL-FAST: damage truth mismatch for {swin_path.stem}")
            samples.append({
                "stem": swin_path.stem,
                "s_loc": swin["loc_probability"].astype(np.float32),
                "s_damage": swin["damage_probability"].astype(np.float32),
                "s_threshold": float(swin["phase1_threshold"]),
                "w_loc": second["loc_probability"].astype(np.float32),
                "w_damage": second["damage_probability"].astype(np.float32),
                "w_loc_prediction": second["loc_prediction"].astype(np.uint8),
                "w_damage_prediction": second["damage_prediction"].astype(np.uint8),
                "loc_true": second_truth,
                "damage_true": second["damage_true"].astype(np.uint8),
            })
    print(f"PASS probability/truth alignment: samples={len(samples)}", flush=True)
    return samples


def predict(sample, mode, alpha=0.5, beta=0.5, threshold=0.5, minor_dilation_kernel=1):
    if mode == "swin":
        loc = sample["s_loc"] > sample["s_threshold"]
        damage = sample["s_damage"].argmax(axis=0).astype(np.uint8) + 1
    elif mode == "second_place":
        # The winner's raw localization ensemble is reliable, but its legacy
        # combined loc/damage post-processing is not calibrated consistently
        # across target datasets. Select this threshold on validation only.
        loc = sample["w_loc"] > threshold
        damage = sample["w_damage_prediction"].astype(np.uint8)
    elif mode == "hybrid":
        loc_probability = alpha * sample["s_loc"] + (1.0 - alpha) * sample["w_loc"]
        damage_probability = beta * sample["s_damage"] + (1.0 - beta) * sample["w_damage"]
        loc = loc_probability > threshold
        damage = damage_probability.argmax(axis=0).astype(np.uint8) + 1
    else:
        raise ValueError(mode)
    damage = dilate_minor(damage, loc, minor_dilation_kernel)
    final_damage = np.zeros_like(damage, dtype=np.uint8)
    final_damage[loc] = damage[loc]
    return loc.astype(np.uint8), final_damage


def evaluate(samples, mode, alpha=0.5, beta=0.5, threshold=0.5, minor_dilation_kernel=1):
    loc_tp = loc_fp = loc_fn = 0
    counts = {class_id: [0, 0, 0] for class_id in range(1, 5)}
    for sample in samples:
        loc, damage = predict(sample, mode, alpha, beta, threshold, minor_dilation_kernel)
        loc_true = sample["loc_true"] > 0
        damage_true = sample["damage_true"]
        loc_tp += int(((loc == 1) & loc_true).sum())
        loc_fp += int(((loc == 1) & ~loc_true).sum())
        loc_fn += int(((loc == 0) & loc_true).sum())
        valid = (damage_true >= 1) & (damage_true <= 4)
        for class_id in range(1, 5):
            truth = (damage_true == class_id) & valid
            prediction = (damage == class_id) & valid
            counts[class_id][0] += int((truth & prediction).sum())
            counts[class_id][1] += int((~truth & prediction & valid).sum())
            counts[class_id][2] += int((truth & ~prediction).sum())
    localization = f1(loc_tp, loc_fp, loc_fn)
    classes = [f1(*counts[class_id]) for class_id in range(1, 5)]
    macro = float(np.mean(classes))
    harmonic_damage = harmonic(classes)
    return {
        "localization_f1": localization,
        "no_damage_f1": classes[0],
        "minor_damage_f1": classes[1],
        "major_damage_f1": classes[2],
        "destroyed_f1": classes[3],
        "macro_damage_f1": macro,
        "harmonic_damage_f1": harmonic_damage,
        "harmonic_composite_score": 0.3 * localization + 0.7 * harmonic_damage,
        "macro_composite_score": 0.3 * localization + 0.7 * macro,
    }


def grid(text):
    return [float(value) for value in text.split(",")]


def save_predictions(samples, output, alpha, beta, threshold, minor_dilation_kernel):
    output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        loc, damage = predict(sample, "hybrid", alpha, beta, threshold, minor_dilation_kernel)
        cv2.imwrite(str(output / f"{sample['stem']}_localization.png"), loc)
        cv2.imwrite(str(output / f"{sample['stem']}_damage.png"), damage)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swin-root", type=Path, required=True)
    parser.add_argument("--second-place-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--betas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--expected-val-samples", type=int, default=45)
    parser.add_argument("--expected-test-samples", type=int, default=46)
    parser.add_argument("--minimum-second-place-val-loc-f1", type=float, default=0.8)
    parser.add_argument("--selection-objective", choices=["macro", "official"], default="macro")
    parser.add_argument("--minor-dilation-kernel", type=int, default=1)
    parser.add_argument("--experiment-label", default="Texas-fine-tuned ImageNet Swin-T + Texas-fine-tuned second-place xView2 soft ensemble")
    parser.add_argument("--selection-label", default="Fusion selected only on Texas validation; held-out Texas test evaluated once.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = load_split(args.swin_root / "val", args.second_place_root / "val")
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(f"Expected {args.expected_val_samples} validation samples, found {len(validation)}")
    calibration_rows = []
    for threshold in grid("0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95"):
        metrics = evaluate(
            validation, "second_place", threshold=threshold,
            minor_dilation_kernel=args.minor_dilation_kernel,
        )
        calibration_rows.append({"second_place_localization_threshold": threshold, **metrics})
    calibration_rows.sort(key=lambda row: (row["localization_f1"], row["harmonic_damage_f1"]), reverse=True)
    second_calibration = calibration_rows[0]
    second_threshold = float(second_calibration["second_place_localization_threshold"])
    second_validation = {key: value for key, value in second_calibration.items() if key != "second_place_localization_threshold"}
    with (args.output_dir / "second_place_localization_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibration_rows[0]))
        writer.writeheader(); writer.writerows(calibration_rows)
    print("Second-place validation preflight:", json.dumps(second_validation, indent=2), flush=True)
    print(f"Second-place localization threshold selected on validation: {second_threshold:.2f}", flush=True)
    if second_validation["localization_f1"] < args.minimum_second_place_val_loc_f1:
        raise RuntimeError(
            f"FAIL-FAST: second-place validation localization F1 "
            f"{second_validation['localization_f1']:.6f} is below "
            f"{args.minimum_second_place_val_loc_f1:.6f}"
        )

    rows = []
    for alpha in grid(args.alphas):
        for beta in grid(args.betas):
            for threshold in grid(args.thresholds):
                rows.append({
                    "swin_localization_weight": alpha,
                    "swin_damage_weight": beta,
                    "localization_threshold": threshold,
                    **evaluate(validation, "hybrid", alpha, beta, threshold, args.minor_dilation_kernel),
                })
    objective = "harmonic_composite_score" if args.selection_objective == "official" else "macro_composite_score"
    rows.sort(
        key=lambda row: (row[objective], row["macro_damage_f1"], row["localization_f1"]),
        reverse=True,
    )
    selected = rows[0]
    with (args.output_dir / "validation_fusion_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected))
        writer.writeheader()
        writer.writerows(rows)
    print("Selected on validation only:", json.dumps(selected, indent=2), flush=True)

    test = load_split(args.swin_root / "test", args.second_place_root / "test")
    if len(test) != args.expected_test_samples:
        raise RuntimeError(f"Expected {args.expected_test_samples} test samples, found {len(test)}")
    alpha = float(selected["swin_localization_weight"])
    beta = float(selected["swin_damage_weight"])
    threshold = float(selected["localization_threshold"])
    test_metrics = {
        "swin": evaluate(test, "swin", minor_dilation_kernel=args.minor_dilation_kernel),
        "second_place": evaluate(test, "second_place", threshold=second_threshold, minor_dilation_kernel=args.minor_dilation_kernel),
        "equal_ensemble": evaluate(test, "hybrid", 0.5, 0.5, 0.5, args.minor_dilation_kernel),
        "selected_ensemble": evaluate(test, "hybrid", alpha, beta, threshold, args.minor_dilation_kernel),
    }
    save_predictions(test, args.output_dir / "selected_test_predictions", alpha, beta, threshold, args.minor_dilation_kernel)
    summary = {
        "experiment": args.experiment_label,
        "selection": args.selection_label,
        "selection_objective": objective,
        "selected_parameters": {
            "swin_localization_weight": alpha,
            "second_place_localization_weight": 1.0 - alpha,
            "swin_damage_weight": beta,
            "second_place_damage_weight": 1.0 - beta,
            "localization_threshold": threshold,
            "minor_dilation_kernel": args.minor_dilation_kernel,
        },
        "validation_samples": len(validation),
        "test_samples": len(test),
        "validation_second_place_preflight": second_validation,
        "second_place_localization_threshold_selected_on_validation": second_threshold,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "ensemble_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [summary["experiment"], summary["selection"], "", json.dumps(summary["selected_parameters"], indent=2), ""]
    for name, metrics in test_metrics.items():
        lines.extend([f"TEST - {name}", *(f"{key}: {value:.6f}" for key, value in metrics.items()), ""])
    text = "\n".join(lines) + "\n"
    (args.output_dir / "ensemble_metrics.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
