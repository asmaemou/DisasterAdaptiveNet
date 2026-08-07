#!/usr/bin/env python3
"""Validation-calibrated class-wise fusion of xBD Swin-T and first-place xView2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_hrtbda_first_place_turkey_soft_ensemble as fusion


CLASS_NAMES = ("no_damage", "minor_damage", "major_damage", "destroyed")


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def harmonic(values, epsilon: float = 1e-6) -> float:
    return len(values) / sum(1.0 / max(float(value), epsilon) for value in values)


def calibrate(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probability.astype(np.float32), 1e-7, 1.0)) / temperature
    logits -= logits.max(axis=0, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / np.maximum(calibrated.sum(axis=0, keepdims=True), 1e-7)


def classwise_damage(sample, weights, swin_temperature, winner_temperature):
    # The strided hold-set search evaluates many candidate weights. Cache its
    # calibrated arrays once; full-resolution validation/test samples simply
    # compute them on demand for their single final evaluation.
    swin = sample.get("_h_damage_calibrated")
    winner = sample.get("_f_damage_calibrated")
    if swin is None:
        swin = calibrate(sample["h_damage"], swin_temperature)
    if winner is None:
        winner = calibrate(sample["f_damage"], winner_temperature)
    weight = np.asarray(weights, np.float32)[:, None, None]
    score = weight * swin + (1.0 - weight) * winner
    return score.argmax(axis=0).astype(np.uint8) + 1


def first_place_prediction(sample):
    return fusion.predictions(sample, "first_place")


def selected_prediction(sample, loc_alpha, loc_threshold, damage_weights,
                        swin_temperature, winner_temperature):
    loc_probability = loc_alpha * sample["h_loc"] + (1.0 - loc_alpha) * sample["f_loc"]
    loc = loc_probability > loc_threshold
    damage = classwise_damage(
        sample, damage_weights, swin_temperature, winner_temperature
    )
    final = np.zeros_like(damage, dtype=np.uint8)
    final[loc] = damage[loc]
    return loc.astype(np.uint8), final


def score(samples, predictor):
    loc_tp = loc_fp = loc_fn = 0
    damage_counts = {class_id: [0, 0, 0] for class_id in range(1, 5)}
    for sample in samples:
        loc, damage = predictor(sample)
        loc_true = sample["loc_true"] > 0
        damage_true = sample["damage_true"]
        loc_tp += int(((loc == 1) & loc_true).sum())
        loc_fp += int(((loc == 1) & ~loc_true).sum())
        loc_fn += int(((loc == 0) & loc_true).sum())
        valid = (damage_true >= 1) & (damage_true <= 4)
        for class_id in range(1, 5):
            truth = (damage_true == class_id) & valid
            prediction = (damage == class_id) & valid
            damage_counts[class_id][0] += int((truth & prediction).sum())
            damage_counts[class_id][1] += int((~truth & prediction & valid).sum())
            damage_counts[class_id][2] += int((truth & ~prediction).sum())
    localization = f1(loc_tp, loc_fp, loc_fn)
    classes = [f1(*damage_counts[class_id]) for class_id in range(1, 5)]
    macro = float(np.mean(classes))
    damage_harmonic = harmonic(classes)
    return {
        "localization_f1": localization,
        **{f"{name}_f1": value for name, value in zip(CLASS_NAMES, classes)},
        "macro_damage_f1": macro,
        "harmonic_damage_f1": damage_harmonic,
        "official_xview2_score": 0.3 * localization + 0.7 * damage_harmonic,
        "macro_composite_score": 0.3 * localization + 0.7 * macro,
    }


def stride_samples(samples, stride):
    if stride <= 1:
        return samples
    reduced = []
    for sample in samples:
        item = dict(sample)
        for key in ("h_loc", "f_loc", "loc_true", "damage_true"):
            item[key] = sample[key][::stride, ::stride]
        for key in ("h_damage", "f_damage"):
            item[key] = sample[key][:, ::stride, ::stride]
        reduced.append(item)
    return reduced


def temperature_nll(samples, model_key, temperature):
    total = 0.0
    pixels = 0
    for sample in samples:
        probability = calibrate(sample[model_key], temperature)
        truth = sample["damage_true"]
        valid = (truth >= 1) & (truth <= 4)
        if valid.any():
            labels = truth[valid].astype(np.int64) - 1
            chosen = probability[:, valid][labels, np.arange(labels.size)]
            total -= float(np.log(np.clip(chosen, 1e-7, 1.0)).sum())
            pixels += labels.size
    return total / max(pixels, 1)


def select_temperature(samples, model_key, candidates):
    rows = [(temperature_nll(samples, model_key, value), value) for value in candidates]
    rows.sort()
    return rows[0][1], rows


def damage_only_score(samples, weights, swin_temperature, winner_temperature):
    metrics = score(
        samples,
        lambda sample: (
            sample["loc_true"].astype(np.uint8),
            classwise_damage(sample, weights, swin_temperature, winner_temperature),
        ),
    )
    return metrics


def coordinate_search(samples, candidates, swin_temperature, winner_temperature):
    starts = ([0.0] * 4, [0.25] * 4, [0.5] * 4, [0.75] * 4, [1.0] * 4)
    results = []
    for start in starts:
        weights = list(start)
        for _ in range(3):
            changed = False
            for class_index in range(4):
                trials = []
                for candidate in candidates:
                    proposal = list(weights)
                    proposal[class_index] = candidate
                    metrics = damage_only_score(
                        samples, proposal, swin_temperature, winner_temperature
                    )
                    trials.append((metrics["harmonic_damage_f1"], metrics["macro_damage_f1"], proposal))
                trials.sort(key=lambda row: (row[0], row[1]), reverse=True)
                if trials[0][2] != weights:
                    changed = True
                weights = trials[0][2]
            if not changed:
                break
        metrics = damage_only_score(samples, weights, swin_temperature, winner_temperature)
        results.append((metrics["harmonic_damage_f1"], metrics["macro_damage_f1"], weights, metrics))
    results.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return results[0], results


def select_localization(samples, alphas, thresholds):
    rows = []
    for alpha in alphas:
        for threshold in thresholds:
            metrics = score(
                samples,
                lambda sample, a=alpha, t=threshold: (
                    (a * sample["h_loc"] + (1.0 - a) * sample["f_loc"] > t).astype(np.uint8),
                    np.zeros_like(sample["damage_true"], dtype=np.uint8),
                ),
            )
            rows.append((metrics["localization_f1"], alpha, threshold))
    rows.sort(reverse=True)
    return rows[0], rows


def numbers(text):
    return [float(value) for value in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swin-root", type=Path, required=True)
    parser.add_argument("--first-place-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-val-samples", type=int, default=933)
    parser.add_argument("--expected-test-samples", type=int, default=933)
    parser.add_argument("--selection-stride", type=int, default=8)
    parser.add_argument("--temperatures", default="0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--damage-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--localization-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--localization-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    args = parser.parse_args()
    fusion.MINOR_DILATION_KERNEL = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = fusion.load_split(args.swin_root / "val", args.first_place_root / "val")
    if len(validation) != args.expected_val_samples:
        raise RuntimeError(f"Expected {args.expected_val_samples} validation samples, found {len(validation)}")
    selection = stride_samples(validation, args.selection_stride)
    temperatures = numbers(args.temperatures)
    swin_temperature, swin_temperature_rows = select_temperature(selection, "h_damage", temperatures)
    winner_temperature, winner_temperature_rows = select_temperature(selection, "f_damage", temperatures)
    for sample in selection:
        sample["_h_damage_calibrated"] = calibrate(sample["h_damage"], swin_temperature)
        sample["_f_damage_calibrated"] = calibrate(sample["f_damage"], winner_temperature)
    best_damage, damage_search = coordinate_search(
        selection, numbers(args.damage_weights), swin_temperature, winner_temperature
    )
    _, _, damage_weights, _ = best_damage
    best_loc, localization_search = select_localization(
        selection, numbers(args.localization_weights), numbers(args.localization_thresholds)
    )
    _, loc_alpha, loc_threshold = best_loc

    first_validation = score(validation, first_place_prediction)
    calibrated_validation = score(
        validation,
        lambda sample: selected_prediction(
            sample, loc_alpha, loc_threshold, damage_weights,
            swin_temperature, winner_temperature,
        ),
    )
    use_fusion = calibrated_validation["official_xview2_score"] > first_validation["official_xview2_score"]
    selected_name = "calibrated_classwise_fusion" if use_fusion else "first_place_fallback"
    print("Validation-only selection:", selected_name, flush=True)
    print(json.dumps({"first_place": first_validation, "candidate": calibrated_validation}, indent=2), flush=True)

    test = fusion.load_split(args.swin_root / "test", args.first_place_root / "test")
    if len(test) != args.expected_test_samples:
        raise RuntimeError(f"Expected {args.expected_test_samples} test samples, found {len(test)}")
    first_test = score(test, first_place_prediction)
    candidate_test = score(
        test,
        lambda sample: selected_prediction(
            sample, loc_alpha, loc_threshold, damage_weights,
            swin_temperature, winner_temperature,
        ),
    )
    selected_test = candidate_test if use_fusion else first_test
    summary = {
        "experiment": "xBD Swin-T + released first-place xView2 calibrated class-wise fusion",
        "selection_policy": "Calibration and all parameters selected on xBD hold only; xBD test evaluated once; first-place fallback enabled.",
        "selection_stride": args.selection_stride,
        "selected_system": selected_name,
        "parameters": {
            "swin_temperature": swin_temperature,
            "first_place_temperature": winner_temperature,
            "swin_localization_weight": loc_alpha,
            "localization_threshold": loc_threshold,
            "swin_damage_weights": dict(zip(CLASS_NAMES, damage_weights)),
        },
        "validation": {"first_place": first_validation, "candidate_fusion": calibrated_validation},
        "test": {"first_place": first_test, "candidate_fusion": candidate_test, "selected_system": selected_test},
        "diagnostics": {
            "swin_temperature_nll": swin_temperature_rows,
            "first_place_temperature_nll": winner_temperature_rows,
            "localization_search": localization_search,
            "damage_coordinate_starts": [
                {"harmonic": row[0], "macro": row[1], "weights": row[2]} for row in damage_search
            ],
        },
    }
    (args.output_dir / "calibrated_classwise_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [summary["experiment"], summary["selection_policy"], "", f"Selected system: {selected_name}",
             json.dumps(summary["parameters"], indent=2), ""]
    for label, metrics in (("TEST - first_place", first_test),
                           ("TEST - calibrated_classwise_fusion", candidate_test),
                           ("TEST - selected_system", selected_test)):
        lines.append(label)
        lines.extend(f"{key}: {value:.6f}" for key, value in metrics.items())
        lines.append("")
    report = "\n".join(lines) + "\n"
    (args.output_dir / "calibrated_classwise_metrics.txt").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
