#!/usr/bin/env python3
"""Calibrated class-wise xBD fusion for Swin-T and an NPZ winner ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_swin_first_place_xbd_calibrated_classwise as core


def canonical_damage(localization, damage):
    building = localization > 0
    output = np.zeros(damage.shape, dtype=np.uint8)
    for class_id in range(1, 5):
        output[(damage == class_id) & building] = class_id
    output[building & ~np.isin(damage, [1, 2, 3, 4])] = 255
    return output


def load_split(swin_root: Path, winner_root: Path):
    swin_files = sorted(swin_root.glob("*.npz"))
    winner_files = {path.stem: path for path in winner_root.glob("*.npz")}
    if not swin_files or {path.stem for path in swin_files} != set(winner_files):
        raise RuntimeError("Swin and winner probability IDs do not match")
    samples = []
    for swin_path in swin_files:
        with np.load(swin_path) as swin, np.load(winner_files[swin_path.stem]) as winner:
            swin_loc = (swin["loc_true"] > 0).astype(np.uint8)
            winner_loc = (winner["loc_true"] > 0).astype(np.uint8)
            if not np.array_equal(swin_loc, winner_loc):
                raise RuntimeError(f"FAIL-FAST localization truth mismatch: {swin_path.stem}")
            swin_damage = canonical_damage(swin_loc, swin["damage_true"])
            winner_damage = canonical_damage(winner_loc, winner["damage_true"])
            if not np.array_equal(swin_damage, winner_damage):
                raise RuntimeError(f"FAIL-FAST damage truth mismatch: {swin_path.stem}")
            samples.append({
                "stem": swin_path.stem,
                "h_loc": swin["loc_probability"].astype(np.float32),
                "h_damage": swin["damage_probability"].astype(np.float32),
                "h_threshold": float(swin["phase1_threshold"]),
                "f_loc": winner["loc_probability"].astype(np.float32),
                "f_damage": winner["damage_probability"].astype(np.float32),
                "winner_loc_prediction": winner["loc_prediction"].astype(np.uint8),
                "winner_damage_prediction": winner["damage_prediction"].astype(np.uint8),
                "loc_true": winner_loc,
                "damage_true": winner_damage,
            })
    print(f"PASS exact probability/truth alignment: samples={len(samples)}", flush=True)
    return samples


def winner_prediction(sample, mode, threshold):
    if mode == "stored":
        loc = sample["winner_loc_prediction"] > 0
    elif mode == "threshold":
        loc = sample["f_loc"] > threshold
    else:
        raise ValueError(mode)
    damage = sample["winner_damage_prediction"].astype(np.uint8)
    final = np.zeros_like(damage, dtype=np.uint8)
    final[loc] = damage[loc]
    return loc.astype(np.uint8), final


def select_winner_threshold(samples, mode, thresholds):
    if mode == "stored":
        metrics = core.score(samples, lambda sample: winner_prediction(sample, mode, 0.5))
        return 0.5, metrics
    rows = []
    for threshold in thresholds:
        metrics = core.score(
            samples, lambda sample, value=threshold: winner_prediction(sample, mode, value)
        )
        rows.append((metrics["official_xview2_score"], metrics["localization_f1"], threshold, metrics))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return rows[0][2], rows[0][3]


def numbers(text):
    return [float(value) for value in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swin-root", type=Path, required=True)
    parser.add_argument("--winner-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--winner-name", required=True)
    parser.add_argument("--winner-mode", choices=["stored", "threshold"], required=True)
    parser.add_argument("--expected-val-samples", type=int, default=933)
    parser.add_argument("--expected-test-samples", type=int, default=933)
    parser.add_argument("--selection-stride", type=int, default=8)
    parser.add_argument("--temperatures", default="0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--damage-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--localization-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--localization-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = load_split(args.swin_root / "val", args.winner_root / "val")
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(f"Expected {args.expected_val_samples} validation samples, found {len(validation)}")
    selection = core.stride_samples(validation, args.selection_stride)
    temperature_candidates = numbers(args.temperatures)
    swin_temperature, swin_temperature_rows = core.select_temperature(selection, "h_damage", temperature_candidates)
    winner_temperature, winner_temperature_rows = core.select_temperature(selection, "f_damage", temperature_candidates)
    for sample in selection:
        sample["_h_damage_calibrated"] = core.calibrate(sample["h_damage"], swin_temperature)
        sample["_f_damage_calibrated"] = core.calibrate(sample["f_damage"], winner_temperature)
    best_damage, damage_search = core.coordinate_search(
        selection, numbers(args.damage_weights), swin_temperature, winner_temperature
    )
    damage_weights = best_damage[2]
    best_loc, localization_search = core.select_localization(
        selection, numbers(args.localization_weights), numbers(args.localization_thresholds)
    )
    _, loc_alpha, loc_threshold = best_loc
    winner_threshold, winner_validation = select_winner_threshold(
        validation, args.winner_mode, numbers(args.localization_thresholds)
    )
    candidate_validation = core.score(
        validation,
        lambda sample: core.selected_prediction(
            sample, loc_alpha, loc_threshold, damage_weights,
            swin_temperature, winner_temperature,
        ),
    )
    use_fusion = candidate_validation["official_xview2_score"] > winner_validation["official_xview2_score"]
    selected_name = "calibrated_classwise_fusion" if use_fusion else f"{args.winner_name}_fallback"
    print("Validation-only selection:", selected_name, flush=True)
    print(json.dumps({args.winner_name: winner_validation, "candidate": candidate_validation}, indent=2), flush=True)

    test = load_split(args.swin_root / "test", args.winner_root / "test")
    if len(test) != args.expected_test_samples:
        raise RuntimeError(f"Expected {args.expected_test_samples} test samples, found {len(test)}")
    winner_test = core.score(
        test, lambda sample: winner_prediction(sample, args.winner_mode, winner_threshold)
    )
    candidate_test = core.score(
        test,
        lambda sample: core.selected_prediction(
            sample, loc_alpha, loc_threshold, damage_weights,
            swin_temperature, winner_temperature,
        ),
    )
    selected_test = candidate_test if use_fusion else winner_test
    summary = {
        "experiment": f"xBD Swin-T + released {args.winner_name} xView2 calibrated class-wise fusion",
        "selection_policy": "Calibration and every decision used xBD hold only; xBD test evaluated after selection; winner fallback enabled.",
        "selection_stride": args.selection_stride,
        "selected_system": selected_name,
        "parameters": {
            "swin_temperature": swin_temperature,
            "winner_temperature": winner_temperature,
            "winner_localization_threshold": winner_threshold,
            "swin_localization_weight": loc_alpha,
            "fusion_localization_threshold": loc_threshold,
            "swin_damage_weights": dict(zip(core.CLASS_NAMES, damage_weights)),
        },
        "validation": {args.winner_name: winner_validation, "candidate_fusion": candidate_validation},
        "test": {args.winner_name: winner_test, "candidate_fusion": candidate_test, "selected_system": selected_test},
        "diagnostics": {
            "swin_temperature_nll": swin_temperature_rows,
            "winner_temperature_nll": winner_temperature_rows,
            "localization_search": localization_search,
            "damage_coordinate_starts": [
                {"harmonic": row[0], "macro": row[1], "weights": row[2]} for row in damage_search
            ],
        },
    }
    json_path = args.output_dir / "calibrated_classwise_metrics.json"
    text_path = args.output_dir / "calibrated_classwise_metrics.txt"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [summary["experiment"], summary["selection_policy"], "",
             f"Selected system: {selected_name}", json.dumps(summary["parameters"], indent=2), ""]
    for label, metrics in ((f"TEST - {args.winner_name}", winner_test),
                           ("TEST - calibrated_classwise_fusion", candidate_test),
                           ("TEST - selected_system", selected_test)):
        lines.append(label)
        lines.extend(f"{key}: {value:.6f}" for key, value in metrics.items())
        lines.append("")
    report = "\n".join(lines) + "\n"
    text_path.write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
