#!/usr/bin/env python3
"""Evaluate an equal-weight Swin/PVTv2/Twins ensemble on Earthquake Turkey."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


MEMBER_NAMES = ("swin", "pvtv2", "twins")
DISPLAY_NAMES = {
    "swin": "Swin-T",
    "pvtv2": "PVTv2-B2",
    "twins": "Twins-SVT-S",
}


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",")]


def load_member_split(root: Path) -> dict[str, dict[str, np.ndarray | float]]:
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No probability files found under: {root}")
    output = {}
    for path in paths:
        with np.load(path) as data:
            output[path.stem] = {
                "loc_probability": data["loc_probability"].astype(np.float32),
                "damage_probability": data["damage_probability"].astype(np.float32),
                "loc_true": data["loc_true"].astype(np.uint8),
                "damage_true": data["damage_true"].astype(np.uint8),
                "threshold": float(data["phase1_threshold"]),
            }
    return output


def load_split(roots: dict[str, Path], split: str):
    members = {name: load_member_split(root / split) for name, root in roots.items()}
    reference_ids = set(members[MEMBER_NAMES[0]])
    for name in MEMBER_NAMES[1:]:
        ids = set(members[name])
        if ids != reference_ids:
            missing = sorted(reference_ids - ids)
            extra = sorted(ids - reference_ids)
            raise RuntimeError(
                f"{split}: {name} sample IDs differ; missing={missing[:10]}, extra={extra[:10]}"
            )

    samples = []
    for stem in sorted(reference_ids):
        reference = members["swin"][stem]
        shape = reference["loc_probability"].shape
        for name in MEMBER_NAMES[1:]:
            current = members[name][stem]
            if current["loc_probability"].shape != shape:
                raise RuntimeError(f"{split}/{stem}: localization shape differs for {name}")
            if current["damage_probability"].shape != reference["damage_probability"].shape:
                raise RuntimeError(f"{split}/{stem}: damage shape differs for {name}")
            if not np.array_equal(current["loc_true"], reference["loc_true"]):
                raise RuntimeError(f"{split}/{stem}: localization truth differs for {name}")
            if not np.array_equal(current["damage_true"], reference["damage_true"]):
                raise RuntimeError(f"{split}/{stem}: damage truth differs for {name}")
        samples.append(
            {
                "stem": stem,
                "members": {name: members[name][stem] for name in MEMBER_NAMES},
                "loc_true": reference["loc_true"],
                "damage_true": reference["damage_true"],
            }
        )
    return samples


def dilate_minor(damage: np.ndarray, loc: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    output = damage.copy()
    minor = (damage == 2).astype(np.uint8)
    dilated = cv2.dilate(minor, np.ones((kernel_size, kernel_size), np.uint8)) > 0
    output[dilated & loc & (output == 1)] = 2
    return output


def predict(sample, mode: str, ensemble_threshold: float = 0.5):
    if mode in MEMBER_NAMES:
        member = sample["members"][mode]
        loc_probability = member["loc_probability"]
        damage_probability = member["damage_probability"]
        threshold = member["threshold"]
    elif mode == "ensemble":
        loc_probability = np.mean(
            [sample["members"][name]["loc_probability"] for name in MEMBER_NAMES],
            axis=0,
        )
        damage_probability = np.mean(
            [sample["members"][name]["damage_probability"] for name in MEMBER_NAMES],
            axis=0,
        )
        threshold = ensemble_threshold
    else:
        raise ValueError(mode)

    loc = loc_probability > threshold
    damage = damage_probability.argmax(axis=0).astype(np.uint8) + 1
    damage = dilate_minor(damage, loc, 3)
    final_damage = np.zeros_like(damage, dtype=np.uint8)
    final_damage[loc] = damage[loc]
    return loc.astype(np.uint8), final_damage


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def evaluate(samples, mode: str, threshold: float = 0.5):
    loc_tp = loc_fp = loc_fn = 0
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(1, 5)}
    for sample in samples:
        loc_pred, damage_pred = predict(sample, mode, threshold)
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
    macro_damage = float(np.mean(list(class_f1.values())))
    return {
        "localization_f1": localization_f1,
        "no_damage_f1": class_f1[1],
        "minor_damage_f1": class_f1[2],
        "major_damage_f1": class_f1[3],
        "destroyed_f1": class_f1[4],
        "macro_damage_f1": macro_damage,
        "overall_xview2_style_score": 0.3 * localization_f1 + 0.7 * macro_damage,
    }


def save_predictions(samples, output: Path, threshold: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        loc, damage = predict(sample, "ensemble", threshold)
        cv2.imwrite(
            str(output / f"{sample['stem']}_localization_disaster_prediction.png"),
            loc,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        cv2.imwrite(
            str(output / f"{sample['stem']}_damage_disaster_prediction.png"),
            damage,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )


def parse_args() -> argparse.Namespace:
    project = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    experiment = project / "output/Swin-PVTv2-Twins-EqualEnsemble_EARTHQUAKE_TURKEY"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swin-root", type=Path, default=experiment / "probabilities/swin")
    parser.add_argument("--pvtv2-root", type=Path, default=experiment / "probabilities/pvtv2")
    parser.add_argument("--twins-root", type=Path, default=experiment / "probabilities/twins")
    parser.add_argument("--output-dir", type=Path, default=experiment / "results")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6,0.7")
    parser.add_argument("--expected-val-samples", type=int, default=93)
    parser.add_argument("--expected-test-samples", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "swin": args.swin_root,
        "pvtv2": args.pvtv2_root,
        "twins": args.twins_root,
    }

    validation = load_split(roots, "val")
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(
            f"Expected {args.expected_val_samples} validation samples, got {len(validation)}"
        )
    validation_grid = []
    for threshold in parse_floats(args.thresholds):
        validation_grid.append(
            {"localization_threshold": threshold, **evaluate(validation, "ensemble", threshold)}
        )
    validation_grid.sort(
        key=lambda row: (
            row["overall_xview2_style_score"],
            row["macro_damage_f1"],
            row["localization_f1"],
        ),
        reverse=True,
    )
    selected_threshold = float(validation_grid[0]["localization_threshold"])

    grid_path = args.output_dir / "validation_threshold_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_grid[0]))
        writer.writeheader()
        writer.writerows(validation_grid)

    test = load_split(roots, "test")
    if len(test) != args.expected_test_samples:
        raise RuntimeError(f"Expected {args.expected_test_samples} test samples, got {len(test)}")

    validation_metrics = {
        **{name: evaluate(validation, name) for name in MEMBER_NAMES},
        "equal_ensemble_fixed_threshold_0.5": evaluate(validation, "ensemble", 0.5),
        "equal_ensemble_val_selected_threshold": evaluate(
            validation, "ensemble", selected_threshold
        ),
    }
    test_metrics = {
        **{name: evaluate(test, name) for name in MEMBER_NAMES},
        "equal_ensemble_fixed_threshold_0.5": evaluate(test, "ensemble", 0.5),
        "equal_ensemble_val_selected_threshold": evaluate(test, "ensemble", selected_threshold),
    }
    save_predictions(
        test,
        args.output_dir / "equal_ensemble_test_predictions",
        selected_threshold,
    )

    summary = {
        "experiment": "Equal-weight three-Transformer ensemble on Earthquake Turkey",
        "members": [DISPLAY_NAMES[name] for name in MEMBER_NAMES],
        "weights": {DISPLAY_NAMES[name]: 1.0 / 3.0 for name in MEMBER_NAMES},
        "training": "Each ImageNet-pretrained member independently trained on the same Turkey train split",
        "selection": "Member checkpoints and thresholds selected on Turkey validation; ensemble weights fixed equally",
        "selected_ensemble_localization_threshold": selected_threshold,
        "validation_samples": len(validation),
        "test_samples": len(test),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    json_path = args.output_dir / "three_transformer_equal_ensemble_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Swin-T + PVTv2-B2 + Twins-SVT-S equal-weight ensemble",
        "Dataset: Earthquake Turkey official split",
        "Each probability-map weight: 1/3",
        f"Validation-selected ensemble localization threshold: {selected_threshold:.2f}",
        f"Validation samples: {len(validation)}",
        f"Test samples: {len(test)}",
        "",
    ]
    for name, metrics in test_metrics.items():
        display = DISPLAY_NAMES.get(name, name)
        lines.extend(
            [
                f"TEST — {display}",
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
    text_path = args.output_dir / "three_transformer_equal_ensemble_summary.txt"
    text_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote validation grid: {grid_path}", flush=True)
    print(f"Wrote JSON summary: {json_path}", flush=True)
    print(f"Wrote text summary: {text_path}", flush=True)


if __name__ == "__main__":
    main()
