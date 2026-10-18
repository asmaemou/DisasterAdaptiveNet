#!/usr/bin/env python3
"""Collect five-dataset RS-BDNet ablation results into CSV tables."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DATASETS = (
    "earthquake_turkey",
    "mount_semeru_eruption",
    "texas_tornadoes",
    "hurricane_delta",
    "pakistan_flooding",
)
VARIANTS = (
    "resnet_only",
    "swin_only",
    "no_cross_attention",
    "no_building_guidance",
    "no_ordinal",
)
BASELINE_DIRS = {
    "earthquake_turkey": "HBG_CAON_EARTHQUAKE_TURKEY",
    "mount_semeru_eruption": "HBG_CAON_MOUNT_SEMERU_ERUPTION",
    "texas_tornadoes": "ResNet34SwinTinyCrossAttentionOrdinal_TEXAS_TORNADOES",
    "hurricane_delta": "ResNet34SwinTinyCrossAttentionOrdinal_HURRICANE_DELTA",
    "pakistan_flooding": "ResNet34SwinTinyCrossAttentionOrdinal_PAKISTAN_FLOODING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    text = path.read_text(encoding="utf-8", errors="replace")
    epoch = re.search(r"Best epoch selected on [^:]+:\s*(\d+)", text)
    threshold = re.search(
        r"Best localization threshold selected on [^:]+:\s*([0-9.]+)", text
    )
    if not epoch:
        epoch = re.search(r"Best epoch selected on hold:\s*(\d+)", text)
    if not threshold:
        threshold = re.search(
            r"Best localization threshold selected on hold:\s*([0-9.]+)", text
        )
    return (
        epoch.group(1) if epoch else "",
        threshold.group(1) if threshold else "",
    )


def read_log_metadata(log_dir: Path, job_id: str) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for path in sorted(log_dir.glob(f"event_rsbd_ablation_{job_id}_*.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        dataset = re.search(r"Dataset:\s*(\S+)", text)
        variant = re.search(r"Ablation variant:\s*(\S+)", text)
        parameters = re.search(r"Parameters: total=([0-9,]+)", text)
        if dataset and variant and parameters:
            counts[(dataset.group(1), variant.group(1))] = int(
                parameters.group(1).replace(",", "")
            )
    return counts


def make_row(
    dataset: str,
    variant: str,
    result_dir: Path,
    parameters: int | None,
) -> dict[str, str | float]:
    json_path = result_dir / "scores" / "scores_xbd_test.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing result: {json_path}")
    result = json.loads(json_path.read_text(encoding="utf-8"))
    class_f1 = [
        float(result["damage_f1_no_damage"]),
        float(result["damage_f1_minor_damage"]),
        float(result["damage_f1_major_damage"]),
        float(result["damage_f1_destroyed"]),
    ]
    epoch, threshold = read_summary(result_dir / "scores" / "summary.txt")
    return {
        "dataset": dataset,
        "variant": variant,
        "best_epoch": epoch,
        "localization_threshold": threshold,
        "localization_f1": float(result["localization_f1"]),
        "no_damage_f1": class_f1[0],
        "minor_damage_f1": class_f1[1],
        "major_damage_f1": class_f1[2],
        "destroyed_f1": class_f1[3],
        "damage_macro_f1": sum(class_f1) / 4.0,
        "damage_harmonic_f1": float(result["damage_f1"]),
        "combined_score": float(result["score"]),
        "parameters_m": "" if parameters is None else parameters / 1_000_000,
    }


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {path}")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_root = root / "output"
    ablation_root = output_root / "RSBDNet_ablation_event_datasets"
    counts = read_log_metadata(root / "transformer" / "sbatch" / "logs", args.job_id)

    all_rows: list[dict[str, str | float]] = []
    missing: list[str] = []
    for dataset in DATASETS:
        dataset_rows: list[dict[str, str | float]] = []
        baseline_dir = output_root / BASELINE_DIRS[dataset]
        try:
            dataset_rows.append(make_row(dataset, "full", baseline_dir, 55_304_000))
        except FileNotFoundError as error:
            missing.append(str(error))
        for variant in VARIANTS:
            try:
                dataset_rows.append(
                    make_row(
                        dataset,
                        variant,
                        ablation_root / dataset / variant,
                        counts.get((dataset, variant)),
                    )
                )
            except FileNotFoundError as error:
                missing.append(str(error))
        if dataset_rows:
            write_csv(ablation_root / f"{dataset}_ablation.csv", dataset_rows)
            all_rows.extend(dataset_rows)

    if missing:
        raise SystemExit("Incomplete results:\n- " + "\n- ".join(missing))
    write_csv(ablation_root / "all_event_datasets_ablation.csv", all_rows)


if __name__ == "__main__":
    main()
