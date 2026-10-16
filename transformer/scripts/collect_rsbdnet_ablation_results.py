#!/usr/bin/env python3
"""Collect final RS-BDNet xBD ablation metrics into one comparison CSV."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


VARIANTS = (
    "resnet_only",
    "swin_only",
    "no_cross_attention",
    "no_building_guidance",
    "no_ordinal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to output/RSBDNet_ablation_xBD/ablation_comparison.csv",
    )
    return parser.parse_args()


def read_summary(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    text = path.read_text(encoding="utf-8")
    epoch = re.search(r"Best epoch selected on hold:\s*(\d+)", text)
    threshold = re.search(
        r"Best localization threshold selected on hold:\s*([0-9.]+)", text
    )
    return (
        epoch.group(1) if epoch else "",
        threshold.group(1) if threshold else "",
    )


def read_parameter_counts(log_dir: Path, job_id: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    pattern = (
        f"xbd_rsbd_ablation_{job_id}_*.out"
        if job_id
        else "xbd_rsbd_ablation_*.out"
    )
    for path in sorted(log_dir.glob(pattern), key=lambda item: item.stat().st_mtime):
        text = path.read_text(encoding="utf-8", errors="replace")
        variant = re.search(r"Ablation variant:\s*(\S+)", text)
        parameters = re.search(r"Parameters: total=([0-9,]+)", text)
        if variant and parameters:
            counts[variant.group(1)] = int(parameters.group(1).replace(",", ""))
    return counts


def metric_row(
    method: str,
    result_dir: Path,
    parameters: int | None,
) -> dict[str, str | float]:
    score_path = result_dir / "scores" / "scores_xbd_test.json"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing completed test results: {score_path}")
    result = json.loads(score_path.read_text(encoding="utf-8"))
    class_f1 = [
        float(result["damage_f1_no_damage"]),
        float(result["damage_f1_minor_damage"]),
        float(result["damage_f1_major_damage"]),
        float(result["damage_f1_destroyed"]),
    ]
    epoch, threshold = read_summary(result_dir / "scores" / "summary.txt")
    return {
        "variant": method,
        "best_epoch": epoch,
        "localization_threshold": threshold,
        "localization_f1": float(result["localization_f1"]),
        "no_damage_f1": class_f1[0],
        "minor_damage_f1": class_f1[1],
        "major_damage_f1": class_f1[2],
        "destroyed_f1": class_f1[3],
        "damage_macro_f1": sum(class_f1) / len(class_f1),
        "damage_harmonic_f1": float(result["damage_f1"]),
        "combined_score": float(result["score"]),
        "parameters_m": "" if parameters is None else parameters / 1_000_000,
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    ablation_root = project_root / "output" / "RSBDNet_ablation_xBD"
    output_csv = args.output_csv or ablation_root / "ablation_comparison.csv"
    log_dir = project_root / "transformer" / "sbatch" / "logs"
    parameter_counts = read_parameter_counts(log_dir, args.job_id)

    rows: list[dict[str, str | float]] = []
    baseline_dir = (
        project_root
        / "output"
        / "BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3"
    )
    if (baseline_dir / "scores" / "scores_xbd_test.json").exists():
        # This is the parameter count reported for the full RS-BDNet model.
        rows.append(metric_row("full", baseline_dir, 55_304_000))
    else:
        print(f"Warning: full-model result not found at {baseline_dir}")

    missing: list[str] = []
    for variant in VARIANTS:
        result_dir = ablation_root / variant
        try:
            rows.append(metric_row(variant, result_dir, parameter_counts.get(variant)))
        except FileNotFoundError as error:
            missing.append(str(error))

    if missing:
        raise SystemExit(
            "Not all ablation tasks produced final test results:\n- "
            + "\n- ".join(missing)
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote comparison CSV: {output_csv}")


if __name__ == "__main__":
    main()
