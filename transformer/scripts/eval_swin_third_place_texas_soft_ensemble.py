#!/usr/bin/env python3
"""Validation-selected soft fusion of Texas Swin-T and xView2 third place."""
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


def load_split(swin_root: Path, winner_root: Path):
    swin_files = sorted(swin_root.glob("*.npz"))
    winner_files = {path.stem: path for path in winner_root.glob("*.npz")}
    if not swin_files or {path.stem for path in swin_files} != set(winner_files):
        raise RuntimeError("Swin and third-place probability IDs do not match")
    samples = []
    for swin_path in swin_files:
        with np.load(swin_path) as swin, np.load(winner_files[swin_path.stem]) as winner:
            if not np.array_equal(swin["loc_true"] > 0, winner["loc_true"] > 0):
                raise RuntimeError(f"FAIL-FAST localization truth mismatch: {swin_path.stem}")
            if not np.array_equal(swin["damage_true"], winner["damage_true"]):
                raise RuntimeError(f"FAIL-FAST damage truth mismatch: {swin_path.stem}")
            samples.append({
                "stem": swin_path.stem,
                "s_loc": swin["loc_probability"].astype(np.float32),
                "s_damage": swin["damage_probability"].astype(np.float32),
                "s_threshold": float(swin["phase1_threshold"]),
                "w_loc": winner["loc_probability"].astype(np.float32),
                "w_damage": winner["damage_probability"].astype(np.float32),
                "w_loc_prediction": winner["loc_prediction"].astype(np.uint8),
                "w_damage_prediction": winner["damage_prediction"].astype(np.uint8),
                "loc_true": (winner["loc_true"] > 0).astype(np.uint8),
                "damage_true": winner["damage_true"].astype(np.uint8),
            })
    print(f"PASS exact probability/truth alignment: samples={len(samples)}", flush=True)
    return samples


def predict(sample, mode, alpha=.5, beta=.5, threshold=.5, minor_dilation_kernel=1):
    if mode == "swin":
        loc = sample["s_loc"] > sample["s_threshold"]
        damage = sample["s_damage"].argmax(0).astype(np.uint8) + 1
    elif mode == "third_place":
        loc = sample["w_loc_prediction"] > 0
        damage = sample["w_damage_prediction"]
    elif mode == "hybrid":
        loc = alpha * sample["s_loc"] + (1-alpha) * sample["w_loc"] > threshold
        damage = (beta * sample["s_damage"] + (1-beta) * sample["w_damage"]).argmax(0).astype(np.uint8) + 1
    else:
        raise ValueError(mode)
    damage = dilate_minor(damage, loc, minor_dilation_kernel)
    final = np.zeros_like(damage, np.uint8)
    final[loc] = damage[loc]
    return loc.astype(np.uint8), final


def evaluate(samples, mode, alpha=.5, beta=.5, threshold=.5, minor_dilation_kernel=1):
    loc_counts = [0, 0, 0]
    damage_counts = {class_id: [0, 0, 0] for class_id in range(1, 5)}
    for sample in samples:
        loc, damage = predict(sample, mode, alpha, beta, threshold, minor_dilation_kernel)
        truth_loc = sample["loc_true"] > 0
        loc_counts[0] += int(((loc == 1) & truth_loc).sum())
        loc_counts[1] += int(((loc == 1) & ~truth_loc).sum())
        loc_counts[2] += int(((loc == 0) & truth_loc).sum())
        truth_damage = sample["damage_true"]
        valid = truth_loc
        for class_id in range(1, 5):
            truth = (truth_damage == class_id) & valid
            pred = (damage == class_id) & valid
            damage_counts[class_id][0] += int((truth & pred).sum())
            damage_counts[class_id][1] += int((~truth & pred & valid).sum())
            damage_counts[class_id][2] += int((truth & ~pred).sum())
    localization = f1(*loc_counts)
    classes = [f1(*damage_counts[i]) for i in range(1, 5)]
    macro = float(np.mean(classes))
    harmonic_damage = harmonic(classes)
    return {
        "localization_f1": localization, "no_damage_f1": classes[0],
        "minor_damage_f1": classes[1], "major_damage_f1": classes[2],
        "destroyed_f1": classes[3], "macro_damage_f1": macro,
        "harmonic_damage_f1": harmonic_damage,
        "official_xview2_score": .3 * localization + .7 * harmonic_damage,
        "macro_composite_score": .3 * localization + .7 * macro,
    }


def values(text):
    return [float(item) for item in text.split(",")]


def strided_samples(samples, stride):
    if stride <= 1:
        return samples
    output = []
    for sample in samples:
        reduced = dict(sample)
        for key in ("s_loc", "w_loc", "w_loc_prediction", "w_damage_prediction", "loc_true", "damage_true"):
            reduced[key] = sample[key][::stride, ::stride]
        for key in ("s_damage", "w_damage"):
            reduced[key] = sample[key][:, ::stride, ::stride]
        output.append(reduced)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swin-root", type=Path, required=True)
    parser.add_argument("--third-place-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--betas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--expected-val-samples", type=int, default=45)
    parser.add_argument("--expected-test-samples", type=int, default=46)
    parser.add_argument("--minimum-third-place-val-loc-f1", type=float, default=.8)
    parser.add_argument("--selection-objective", choices=["macro", "official"], default="macro")
    parser.add_argument("--minor-dilation-kernel", type=int, default=1)
    parser.add_argument("--experiment-label", default="Texas-fine-tuned ImageNet Swin-T + Texas-fine-tuned third-place xView2 soft ensemble")
    parser.add_argument("--selection-label", default="Fusion selected only on Texas validation; held-out Texas test evaluated once.")
    parser.add_argument("--selection-stride", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = load_split(args.swin_root / "val", args.third_place_root / "val")
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(f"Expected {args.expected_val_samples} validation samples, found {len(validation)}")
    preflight = evaluate(validation, "third_place", minor_dilation_kernel=args.minor_dilation_kernel)
    print("Third-place validation reproduction gate:", json.dumps(preflight, indent=2), flush=True)
    if preflight["localization_f1"] < args.minimum_third_place_val_loc_f1:
        raise RuntimeError(f"FAIL-FAST: third-place validation localization F1 {preflight['localization_f1']:.6f} is below {args.minimum_third_place_val_loc_f1:.6f}; test was not evaluated")

    rows = []
    selection_validation = strided_samples(validation, args.selection_stride)
    print(f"Fusion-grid validation pixel stride: {args.selection_stride}", flush=True)
    for alpha in values(args.alphas):
        for beta in values(args.betas):
            for threshold in values(args.thresholds):
                rows.append({"swin_localization_weight": alpha, "swin_damage_weight": beta,
                             "localization_threshold": threshold,
                             **evaluate(selection_validation, "hybrid", alpha, beta, threshold, args.minor_dilation_kernel)})
    objective = "official_xview2_score" if args.selection_objective == "official" else "macro_composite_score"
    rows.sort(key=lambda row: (row[objective], row["macro_damage_f1"], row["localization_f1"]), reverse=True)
    selected = rows[0]
    with (args.output_dir / "validation_fusion_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected)); writer.writeheader(); writer.writerows(rows)
    print("Selected using Texas validation only:", json.dumps(selected, indent=2), flush=True)

    test = load_split(args.swin_root / "test", args.third_place_root / "test")
    if len(test) != args.expected_test_samples:
        raise RuntimeError(f"Expected {args.expected_test_samples} test samples, found {len(test)}")
    alpha, beta, threshold = (selected["swin_localization_weight"], selected["swin_damage_weight"], selected["localization_threshold"])
    metrics = {"swin": evaluate(test, "swin", minor_dilation_kernel=args.minor_dilation_kernel),
               "third_place": evaluate(test, "third_place", minor_dilation_kernel=args.minor_dilation_kernel),
               "equal_ensemble": evaluate(test, "hybrid", .5, .5, .5, args.minor_dilation_kernel),
               "selected_ensemble": evaluate(test, "hybrid", alpha, beta, threshold, args.minor_dilation_kernel)}
    prediction_dir = args.output_dir / "selected_test_predictions"; prediction_dir.mkdir(parents=True, exist_ok=True)
    for sample in test:
        loc, damage = predict(sample, "hybrid", alpha, beta, threshold, args.minor_dilation_kernel)
        cv2.imwrite(str(prediction_dir / f"{sample['stem']}_localization.png"), loc)
        cv2.imwrite(str(prediction_dir / f"{sample['stem']}_damage.png"), damage)
    summary = {"experiment": args.experiment_label,
               "selection": args.selection_label,
               "selection_objective": objective,
               "selection_stride": args.selection_stride,
               "selected_parameters": {"swin_localization_weight": alpha, "third_place_localization_weight": 1-alpha,
                                       "swin_damage_weight": beta, "third_place_damage_weight": 1-beta,
                                       "localization_threshold": threshold,
                                       "minor_dilation_kernel": args.minor_dilation_kernel},
               "validation_samples": len(validation), "test_samples": len(test),
               "validation_third_place_reproduction": preflight, "test_metrics": metrics}
    (args.output_dir / "ensemble_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [summary["experiment"], summary["selection"], "", json.dumps(summary["selected_parameters"], indent=2), ""]
    for name, result in metrics.items():
        lines += [f"TEST - {name}", *(f"{key}: {value:.6f}" for key, value in result.items()), ""]
    report = "\n".join(lines) + "\n"
    (args.output_dir / "ensemble_metrics.txt").write_text(report); print(report, flush=True)


if __name__ == "__main__":
    main()
