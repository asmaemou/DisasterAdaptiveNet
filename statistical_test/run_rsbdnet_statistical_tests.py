#!/usr/bin/env python3
"""Statistical analysis of RS-BDNet ablation results across six datasets.

The dataset is the paired/blocking unit. The script performs:
  1. Friedman omnibus test across all model configurations.
  2. Kendall's W effect size derived from the Friedman statistic.
  3. Paired two-sided Wilcoxon signed-rank tests comparing full RS-BDNet
     with each ablation.
  4. Holm adjustment of the five Wilcoxon p-values.
  5. Paired percentile-bootstrap 95% confidence intervals for the mean
     Macro-F1 difference (full model minus ablation) across datasets.

This analysis does not measure variation from random initialization or other
training stochasticity. That requires repeated training with multiple seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
except ImportError as exc:
    raise SystemExit(
        "Missing statistical dependency. Activate disasteradaptivenet_cu128 "
        "and ensure numpy and scipy are installed."
    ) from exc


VARIANT_ORDER = (
    "full",
    "resnet_only",
    "swin_only",
    "no_cross_attention",
    "no_building_guidance",
    "no_ordinal",
)

VARIANT_LABELS = {
    "full": "Full RS-BDNet",
    "resnet_only": "ResNet only",
    "swin_only": "Swin only",
    "no_cross_attention": "No cross-attention",
    "no_building_guidance": "No building guidance",
    "no_ordinal": "No ordinal learning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xbd-csv",
        type=Path,
        default=Path("output/RSBDNet_ablation_xBD/ablation_comparison.csv"),
        help="xBD ablation comparison CSV.",
    )
    parser.add_argument(
        "--event-csv",
        type=Path,
        default=Path(
            "output/RSBDNet_ablation_event_datasets/"
            "all_event_datasets_ablation.csv"
        ),
        help="Combined five-event ablation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/RSBDNet_statistical_tests"),
    )
    parser.add_argument("--metric", default="damage_macro_f1")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def normalize_variant(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "full_rs_bdnet": "full",
        "full_rsbdnet": "full",
        "no_ordinal_learning": "no_ordinal",
    }
    return aliases.get(key, key)


def read_rows(path: Path, default_dataset: str | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    for row in rows:
        if default_dataset is not None and not row.get("dataset"):
            row["dataset"] = default_dataset
        row["variant"] = normalize_variant(row["variant"])
    return rows


def build_matrix(rows: Iterable[dict[str, str]], metric: str) -> tuple[list[str], np.ndarray]:
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        dataset = row["dataset"].strip()
        variant = row["variant"]
        if variant not in VARIANT_ORDER:
            continue
        if metric not in row or row[metric] == "":
            raise KeyError(f"Metric '{metric}' is missing for {dataset}/{variant}")
        key = (dataset, variant)
        if key in lookup:
            raise ValueError(f"Duplicate result for dataset={dataset}, variant={variant}")
        lookup[key] = float(row[metric])

    datasets = sorted({dataset for dataset, _ in lookup})
    expected = {(dataset, variant) for dataset in datasets for variant in VARIANT_ORDER}
    missing = sorted(expected.difference(lookup))
    if missing:
        details = "\n".join(f"  - {dataset}/{variant}" for dataset, variant in missing)
        raise ValueError(f"Incomplete paired result matrix:\n{details}")
    if len(datasets) < 2:
        raise ValueError("At least two paired datasets are required.")

    matrix = np.asarray(
        [[lookup[(dataset, variant)] for variant in VARIANT_ORDER] for dataset in datasets],
        dtype=np.float64,
    )
    return datasets, matrix


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted_sorted: list[float] = []
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(index)])
        running_max = max(running_max, candidate)
        adjusted_sorted.append(running_max)
    adjusted = [0.0] * count
    for index, value in zip(order, adjusted_sorted):
        adjusted[int(index)] = value
    return adjusted


def paired_bootstrap_ci(
    differences: np.ndarray,
    confidence_level: float,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if resamples < 1000:
        raise ValueError("Use at least 1,000 bootstrap resamples")
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.xbd_csv, default_dataset="xBD")
    rows.extend(read_rows(args.event_csv))
    datasets, matrix = build_matrix(rows, args.metric)

    n_datasets, n_variants = matrix.shape
    friedman = friedmanchisquare(*(matrix[:, column] for column in range(n_variants)))
    kendalls_w = float(friedman.statistic) / (n_datasets * (n_variants - 1))

    # Higher Macro-F1 is better, so descending ranks assign rank 1 to the best.
    ranks = np.vstack([rankdata(-row, method="average") for row in matrix])
    average_ranks = ranks.mean(axis=0)

    rng = np.random.default_rng(args.random_seed)
    full = matrix[:, 0]
    raw_results: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for column, variant in enumerate(VARIANT_ORDER[1:], start=1):
        ablation = matrix[:, column]
        differences = full - ablation
        if np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(
                full,
                ablation,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic, p_value = float(test.statistic), float(test.pvalue)
        ci_low, ci_high = paired_bootstrap_ci(
            differences,
            args.confidence_level,
            args.bootstrap_resamples,
            rng,
        )
        raw_p_values.append(p_value)
        raw_results.append(
            {
                "comparison": f"Full RS-BDNet vs {VARIANT_LABELS[variant]}",
                "n_paired_datasets": n_datasets,
                "full_mean_macro_f1": float(full.mean()),
                "ablation_mean_macro_f1": float(ablation.mean()),
                "mean_difference_full_minus_ablation": float(differences.mean()),
                "median_difference_full_minus_ablation": float(np.median(differences)),
                "bootstrap_ci_level": args.confidence_level,
                "bootstrap_ci_lower": ci_low,
                "bootstrap_ci_upper": ci_high,
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value,
            }
        )

    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(raw_results, adjusted):
        row["p_value_holm"] = adjusted_p
        row["significant_holm_0.05"] = adjusted_p < 0.05

    args.output_dir.mkdir(parents=True, exist_ok=True)
    omnibus = {
        "metric": args.metric,
        "n_datasets": n_datasets,
        "n_variants": n_variants,
        "friedman_chi_square": float(friedman.statistic),
        "friedman_degrees_of_freedom": n_variants - 1,
        "friedman_p_value": float(friedman.pvalue),
        "kendalls_w": kendalls_w,
        "alpha": 0.05,
        "friedman_significant_0.05": bool(friedman.pvalue < 0.05),
    }
    write_csv(
        args.output_dir / "friedman_kendall_results.csv",
        list(omnibus),
        [omnibus],
    )
    write_csv(
        args.output_dir / "wilcoxon_holm_bootstrap_results.csv",
        list(raw_results[0]),
        raw_results,
    )
    rank_rows = [
        {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "mean_macro_f1": float(matrix[:, column].mean()),
            "average_rank": float(average_ranks[column]),
        }
        for column, variant in enumerate(VARIANT_ORDER)
    ]
    write_csv(
        args.output_dir / "average_ranks.csv",
        list(rank_rows[0]),
        rank_rows,
    )

    report = {
        "datasets": datasets,
        "variants": list(VARIANT_ORDER),
        "omnibus": omnibus,
        "average_ranks": rank_rows,
        "post_hoc": raw_results,
        "bootstrap": {
            "unit": "dataset",
            "resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence_level,
            "random_seed": args.random_seed,
            "method": "paired percentile bootstrap of mean Macro-F1 differences",
        },
        "limitations": [
            "Only six paired dataset-level observations are available.",
            "The analysis does not measure variability across training seeds.",
            "Dataset-level bootstrap intervals do not replace image-level intervals.",
            "Individual pixels must not be treated as independent observations.",
        ],
    }
    with (args.output_dir / "statistical_test_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("RS-BDNet statistical analysis completed")
    print(f"Datasets: {', '.join(datasets)}")
    print(
        f"Friedman chi-square({n_variants - 1}) = {friedman.statistic:.4f}, "
        f"p = {friedman.pvalue:.6g}"
    )
    print(f"Kendall's W = {kendalls_w:.4f}")
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
