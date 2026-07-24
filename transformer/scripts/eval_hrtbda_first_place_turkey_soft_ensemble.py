#!/usr/bin/env python3
"""Select HRTBDA/first-place soft-fusion parameters on val and test once."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


FIRST_DAMAGE_FOLDERS = [
    "dpn92cls_cce_0_tuned",
    "dpn92cls_cce_1_tuned",
    "dpn92cls_cce_2_tuned",
    "res34cls2_0_tuned",
    "res34cls2_1_tuned",
    "res34cls2_2_tuned",
    "res50cls_cce_0_tuned",
    "res50cls_cce_1_tuned",
    "res50cls_cce_2_tuned",
    "se154cls_0_tuned",
    "se154cls_1_tuned",
    "se154cls_2_tuned",
]
FIRST_LOC_FOLDERS = [
    "pred50_loc_tuned",
    "pred92_loc_tuned",
    "pred34_loc",
    "pred154_loc",
]
CLASS_NAMES = {1: "no_damage", 2: "minor_damage", 3: "major_damage", 4: "destroyed"}


def read_unchanged(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(path)
    return array


def first_place_probabilities(root: Path, stem: str) -> Tuple[np.ndarray, np.ndarray]:
    filename = f"{stem}_pre_disaster_part1.png"
    localization = np.mean(
        [
            read_unchanged(root / folder / filename).astype(np.float32) / 255.0
            for folder in FIRST_LOC_FOLDERS
        ],
        axis=0,
    )

    damage_members = []
    for folder in FIRST_DAMAGE_FOLDERS:
        part1 = read_unchanged(root / folder / filename)
        part2 = read_unchanged(
            root / folder / filename.replace("_part1.png", "_part2.png")
        )
        if part1.ndim != 3 or part2.ndim != 3:
            raise RuntimeError(f"Expected multichannel probabilities for {stem} in {folder}")
        probability5 = np.concatenate([part1, part2[..., 1:]], axis=2).astype(np.float32)
        damage_members.append(probability5[..., 1:5] / 255.0)

    damage = np.mean(damage_members, axis=0).transpose(2, 0, 1)
    damage /= np.maximum(damage.sum(axis=0, keepdims=True), 1e-7)
    return localization, damage


def load_split(hrtbda_root: Path, first_root: Path):
    samples = []
    files = sorted(hrtbda_root.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No HRTBDA probabilities found under {hrtbda_root}")
    for path in files:
        with np.load(path) as data:
            h_loc = data["loc_probability"].astype(np.float32)
            h_damage = data["damage_probability"].astype(np.float32)
            loc_true = data["loc_true"].astype(np.uint8)
            damage_true = data["damage_true"].astype(np.uint8)
            h_threshold = float(data["phase1_threshold"])
        f_loc, f_damage = first_place_probabilities(first_root, path.stem)
        if h_loc.shape != f_loc.shape or h_damage.shape[1:] != f_loc.shape:
            raise RuntimeError(f"Probability shape mismatch for {path.stem}")
        samples.append(
            {
                "stem": path.stem,
                "h_loc": h_loc,
                "h_damage": h_damage,
                "h_threshold": h_threshold,
                "f_loc": f_loc,
                "f_damage": f_damage,
                "loc_true": loc_true,
                "damage_true": damage_true,
            }
        )
    return samples


def dilate_minor(damage: np.ndarray, loc: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return damage
    output = damage.copy()
    minor = (damage == 2).astype(np.uint8)
    dilated = cv2.dilate(minor, np.ones((kernel_size, kernel_size), np.uint8)) > 0
    output[dilated & loc & (output == 1)] = 2
    return output


def predictions(sample, mode: str, alpha=0.5, beta=0.5, threshold=0.5):
    if mode == "hrtbda":
        loc = sample["h_loc"] > sample["h_threshold"]
        damage = sample["h_damage"].argmax(axis=0).astype(np.uint8) + 1
        damage = dilate_minor(damage, loc, 3)
    elif mode == "first_place":
        damage = sample["f_damage"].argmax(axis=0).astype(np.uint8) + 1
        probability = sample["f_loc"]
        loc = (
            (probability > 0.38)
            | ((probability > 0.13) & (damage > 1) & (damage < 4))
            | ((probability > 0.14) & (damage > 1))
        )
        damage = dilate_minor(damage, loc, 5)
    elif mode == "hybrid":
        loc_probability = alpha * sample["h_loc"] + (1.0 - alpha) * sample["f_loc"]
        damage_probability = (
            beta * sample["h_damage"] + (1.0 - beta) * sample["f_damage"]
        )
        loc = loc_probability > threshold
        damage = damage_probability.argmax(axis=0).astype(np.uint8) + 1
        damage = dilate_minor(damage, loc, 3)
    else:
        raise ValueError(mode)

    final_damage = np.zeros_like(damage, dtype=np.uint8)
    final_damage[loc] = damage[loc]
    return loc.astype(np.uint8), final_damage


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def evaluate(samples, mode: str, alpha=0.5, beta=0.5, threshold=0.5):
    loc_tp = loc_fp = loc_fn = 0
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(1, 5)}

    for sample in samples:
        loc_pred, damage_pred = predictions(sample, mode, alpha, beta, threshold)
        loc_true = sample["loc_true"] > 0
        damage_true = sample["damage_true"]

        loc_tp += int(((loc_pred == 1) & loc_true).sum())
        loc_fp += int(((loc_pred == 1) & ~loc_true).sum())
        loc_fn += int(((loc_pred == 0) & loc_true).sum())

        valid = (damage_true >= 1) & (damage_true <= 4)
        for class_id in range(1, 5):
            truth = (damage_true == class_id) & valid
            prediction = (damage_pred == class_id) & valid
            counts[class_id]["tp"] += int((truth & prediction).sum())
            counts[class_id]["fp"] += int((~truth & prediction & valid).sum())
            counts[class_id]["fn"] += int((truth & ~prediction).sum())

    localization_f1 = f1(loc_tp, loc_fp, loc_fn)
    class_f1 = {
        class_id: f1(value["tp"], value["fp"], value["fn"])
        for class_id, value in counts.items()
    }
    macro_damage_f1 = float(np.mean(list(class_f1.values())))
    return {
        "localization_f1": localization_f1,
        "no_damage_f1": class_f1[1],
        "minor_damage_f1": class_f1[2],
        "major_damage_f1": class_f1[3],
        "destroyed_f1": class_f1[4],
        "macro_damage_f1": macro_damage_f1,
        "overall_xview2_style_score": 0.3 * localization_f1 + 0.7 * macro_damage_f1,
    }


def grid_values(text: str) -> Iterable[float]:
    return [float(value) for value in text.split(",")]


def save_predictions(samples, output: Path, alpha: float, beta: float, threshold: float):
    output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        loc, damage = predictions(sample, "hybrid", alpha, beta, threshold)
        stem = sample["stem"]
        cv2.imwrite(
            str(output / f"{stem}_localization_disaster_prediction.png"),
            loc,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        cv2.imwrite(
            str(output / f"{stem}_damage_disaster_prediction.png"),
            damage,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )


def parse_args() -> argparse.Namespace:
    project = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    experiment = project / "output/hybrid_hrtbda_first_place_turkey"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrtbda-root", type=Path, default=experiment / "probabilities/hrtbda")
    parser.add_argument("--first-place-root", type=Path, default=experiment / "probabilities/first_place")
    parser.add_argument("--output-dir", type=Path, default=experiment / "results")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--expected-val-samples", type=int, default=93)
    parser.add_argument("--expected-test-samples", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading validation probabilities...", flush=True)
    validation = load_split(
        args.hrtbda_root / "val", args.first_place_root / "val"
    )
    print(f"Validation samples: {len(validation)}", flush=True)
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(
            f"Expected {args.expected_val_samples} common validation samples, got {len(validation)}"
        )

    rows = []
    for alpha in grid_values(args.alphas):
        for beta in grid_values(args.betas):
            for threshold in grid_values(args.thresholds):
                metrics = evaluate(
                    validation, "hybrid", alpha=alpha, beta=beta, threshold=threshold
                )
                rows.append(
                    {
                        "alpha_hrtbda_localization": alpha,
                        "beta_hrtbda_damage": beta,
                        "localization_threshold": threshold,
                        **metrics,
                    }
                )

    rows.sort(
        key=lambda row: (
            row["overall_xview2_style_score"],
            row["macro_damage_f1"],
            row["localization_f1"],
        ),
        reverse=True,
    )
    best = rows[0]
    grid_path = args.output_dir / "validation_fusion_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Selected only on validation:", json.dumps(best, indent=2), flush=True)
    validation_metrics = {
        "hrtbda": evaluate(validation, "hrtbda"),
        "first_place": evaluate(validation, "first_place"),
        "hybrid_selected": {
            key: best[key]
            for key in (
                "localization_f1",
                "no_damage_f1",
                "minor_damage_f1",
                "major_damage_f1",
                "destroyed_f1",
                "macro_damage_f1",
                "overall_xview2_style_score",
            )
        },
    }

    print("Loading test probabilities after selection...", flush=True)
    test = load_split(args.hrtbda_root / "test", args.first_place_root / "test")
    print(f"Test samples: {len(test)}", flush=True)
    if len(test) != args.expected_test_samples:
        raise RuntimeError(
            f"Expected {args.expected_test_samples} common test samples, got {len(test)}"
        )
    alpha = float(best["alpha_hrtbda_localization"])
    beta = float(best["beta_hrtbda_damage"])
    threshold = float(best["localization_threshold"])
    test_metrics = {
        "hrtbda": evaluate(test, "hrtbda"),
        "first_place": evaluate(test, "first_place"),
        "hybrid_selected": evaluate(
            test, "hybrid", alpha=alpha, beta=beta, threshold=threshold
        ),
    }
    save_predictions(
        test,
        args.output_dir / "hybrid_test_predictions",
        alpha,
        beta,
        threshold,
    )

    summary = {
        "experiment": "Validation-tuned soft ensemble: HRTBDA-v5 + first-place xView2, Earthquake Turkey",
        "selection_rule": "alpha, beta, and localization threshold selected only on Turkey validation; test evaluated once",
        "alpha_definition": "weight assigned to HRTBDA localization probability",
        "beta_definition": "weight assigned to HRTBDA damage probability",
        "selected_parameters": {
            "alpha_hrtbda_localization": alpha,
            "beta_hrtbda_damage": beta,
            "first_place_localization_weight": 1.0 - alpha,
            "first_place_damage_weight": 1.0 - beta,
            "localization_threshold": threshold,
            "minor_dilation_kernel": 3,
        },
        "validation_samples": len(validation),
        "test_samples": len(test),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    summary_path = args.output_dir / "hybrid_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    text_lines = [
        "HRTBDA-v5 + first-place xView2 soft ensemble on Earthquake Turkey",
        "Fusion parameters selected on validation only; test evaluated once.",
        "",
        f"Selected HRTBDA localization weight: {alpha:.2f}",
        f"Selected first-place localization weight: {1.0 - alpha:.2f}",
        f"Selected HRTBDA damage weight: {beta:.2f}",
        f"Selected first-place damage weight: {1.0 - beta:.2f}",
        f"Selected localization threshold: {threshold:.2f}",
        "",
    ]
    for model_name, metrics in test_metrics.items():
        text_lines.extend(
            [
                f"TEST — {model_name}",
                f"Localization F1: {metrics['localization_f1']:.6f}",
                f"No Damage F1: {metrics['no_damage_f1']:.6f}",
                f"Minor Damage F1: {metrics['minor_damage_f1']:.6f}",
                f"Major Damage F1: {metrics['major_damage_f1']:.6f}",
                f"Destroyed F1: {metrics['destroyed_f1']:.6f}",
                f"Macro Damage F1: {metrics['macro_damage_f1']:.6f}",
                f"Overall Score: {metrics['overall_xview2_style_score']:.6f}",
                "",
            ]
        )
    text_path = args.output_dir / "hybrid_metrics_summary.txt"
    text_path.write_text("\n".join(text_lines), encoding="utf-8")
    print("\n".join(text_lines), flush=True)
    print(f"Wrote validation grid: {grid_path}", flush=True)
    print(f"Wrote JSON summary: {summary_path}", flush=True)
    print(f"Wrote text summary: {text_path}", flush=True)


if __name__ == "__main__":
    main()
