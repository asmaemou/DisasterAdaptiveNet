#!/usr/bin/env python3
"""Plot per-dataset building-instance distributions for damage classes.

For each post-disaster target mask, the script counts 8-connected components
independently for labels 1--4 (no damage, minor, major, destroyed).  It scans
all requested splits, writes the exact counts to CSV, and creates a simple
publication-ready stacked-bar figure of the absolute building counts.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASS_LABELS = ("No damage", "Minor damage", "Major damage", "Destroyed")
CLASS_VALUES = (1, 2, 3, 4)
# A purpose-built palette distinct from the EBD reference figure.  The colors
# remain distinguishable when printed and follow increasing visual severity.
COLORS = ("#4C78A8", "#72B7B2", "#F2A541", "#C44E52")
MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    splits: Sequence[str]


DEFAULT_PROJECT_PARENT = Path("/homes/j244s673/documents/wsu/phd")


def default_specs(parent: Path) -> List[DatasetSpec]:
    return [
        DatasetSpec("xBD", parent / "xview2", ("train", "tier3", "hold", "test")),
        DatasetSpec(
            "Earthquake Turkey",
            parent / "earthquake_turkey_preprocessed",
            ("train", "val", "test"),
        ),
        DatasetSpec(
            "Mount Semeru Eruption",
            parent / "mount_semeru_eruption_preprocessed",
            ("train", "val", "test"),
        ),
        DatasetSpec(
            "Texas Tornadoes",
            parent / "texas_tornadoes_preprocessed",
            ("train", "val", "test"),
        ),
        DatasetSpec(
            "Hurricane Delta",
            parent / "hurricane_delta_preprocessed",
            ("train", "val", "test"),
        ),
        DatasetSpec(
            "Pakistan Flooding",
            parent / "pakistan_flooding_preprocessed",
            ("train", "val", "test"),
        ),
    ]


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Unable to read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def target_files(split_root: Path) -> List[Path]:
    """Locate post-disaster targets without counting duplicate mask copies."""
    targets = split_root / "targets"
    if targets.is_dir():
        files = [
            path
            for path in targets.iterdir()
            if path.is_file()
            and path.suffix.lower() in MASK_EXTENSIONS
            and "_post_disaster_target" in path.stem
        ]
        if files:
            return sorted(files)

    # Fallback for released EBD-style folders without a targets directory.
    masks = split_root / "masks"
    if masks.is_dir():
        return sorted(
            path
            for path in masks.iterdir()
            if path.is_file()
            and path.suffix.lower() in MASK_EXTENSIONS
            and "_post_disaster" in path.stem
        )
    return []


def count_components(mask: np.ndarray, class_value: int, minimum_pixels: int) -> int:
    binary = (mask == class_value).astype(np.uint8)
    number, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if number <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.count_nonzero(areas >= minimum_pixels))


def count_dataset(spec: DatasetSpec, minimum_pixels: int) -> Dict[str, object]:
    counts = np.zeros(4, dtype=np.int64)
    samples = 0
    split_samples: Dict[str, int] = {}

    if not spec.root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {spec.root}")

    for split in spec.splits:
        files = target_files(spec.root / split)
        if not files:
            raise RuntimeError(
                f"No post-disaster target masks found for {spec.name}, split={split}, "
                f"under {spec.root / split}"
            )
        split_samples[split] = len(files)
        samples += len(files)

        for index, path in enumerate(files, start=1):
            mask = read_mask(path)
            unique = set(int(value) for value in np.unique(mask))
            unexpected = unique.difference({0, 1, 2, 3, 4, 255})
            if unexpected:
                raise ValueError(f"Unexpected labels {sorted(unexpected)} in {path}")
            for class_index, class_value in enumerate(CLASS_VALUES):
                counts[class_index] += count_components(
                    mask, class_value=class_value, minimum_pixels=minimum_pixels
                )
            if index % 250 == 0 or index == len(files):
                print(
                    f"{spec.name} | {split}: {index}/{len(files)} masks processed",
                    flush=True,
                )

    total = int(counts.sum())
    percentages = counts.astype(np.float64) / total * 100.0 if total else np.zeros(4)
    print(
        f"{spec.name}: samples={samples:,}, components={total:,}, "
        + ", ".join(
            f"{label}={int(value):,}" for label, value in zip(CLASS_LABELS, counts)
        ),
        flush=True,
    )
    return {
        "name": spec.name,
        "root": str(spec.root),
        "samples": samples,
        "split_samples": split_samples,
        "counts": counts,
        "percentages": percentages,
        "total": total,
    }


def write_csv(results: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "paired_samples",
        "no_damage",
        "minor_damage",
        "major_damage",
        "destroyed",
        "total_components",
        "no_damage_percent",
        "minor_damage_percent",
        "major_damage_percent",
        "destroyed_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            counts = result["counts"]
            percentages = result["percentages"]
            writer.writerow(
                {
                    "dataset": result["name"],
                    "paired_samples": result["samples"],
                    "no_damage": int(counts[0]),
                    "minor_damage": int(counts[1]),
                    "major_damage": int(counts[2]),
                    "destroyed": int(counts[3]),
                    "total_components": result["total"],
                    "no_damage_percent": f"{percentages[0]:.4f}",
                    "minor_damage_percent": f"{percentages[1]:.4f}",
                    "major_damage_percent": f"{percentages[2]:.4f}",
                    "destroyed_percent": f"{percentages[3]:.4f}",
                }
            )


def plot_distribution(results: Sequence[Dict[str, object]], output_stem: Path) -> None:
    # Place the largest datasets first to make their relative scale intuitive.
    ordered = sorted(results, key=lambda item: int(item["total"]), reverse=True)
    names = [str(item["name"]) for item in ordered]
    counts = np.stack([item["counts"] for item in ordered]).astype(np.float64)
    totals = counts.sum(axis=1).astype(np.int64)
    overall_counts = counts.sum(axis=0)
    overall_percentages = overall_counts / overall_counts.sum() * 100.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # xBD is more than an order of magnitude larger than every event dataset.
    # Give it a compact full-scale panel and enlarge the remaining five below.
    # This avoids a broken axis and keeps every absolute class segment visible.
    fig, (xbd_ax, events_ax) = plt.subplots(
        2, 1,
        figsize=(11.8, 6.6),
        gridspec_kw={"height_ratios": (1.0, 4.2), "hspace": 0.22},
    )

    def draw_stacked_bars(axis, panel_counts, panel_names, panel_totals, x_limit):
        positions = np.arange(len(panel_names))
        cumulative = np.zeros(len(panel_names), dtype=np.float64)
        for class_index, (label, color) in enumerate(zip(CLASS_LABELS, COLORS)):
            axis.barh(
                positions, panel_counts[:, class_index], left=cumulative,
                color=color, height=0.62, label=label,
                edgecolor="white", linewidth=0.55,
            )
            cumulative += panel_counts[:, class_index]
        axis.set_yticks(positions, panel_names)
        axis.invert_yaxis()
        axis.set_xlim(0, x_limit)
        axis.grid(axis="x", color="#D8DEE6", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        for row, total in enumerate(panel_totals):
            axis.annotate(
                f"{int(total):,}", xy=(total, row), xytext=(7, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color="#333333", weight="bold", clip_on=False,
            )

    xbd_limit = int(np.ceil(totals[0] * 1.10 / 10000.0) * 10000)
    event_limit = int(np.ceil(totals[1:].max() * 1.16 / 1000.0) * 1000)
    draw_stacked_bars(xbd_ax, counts[:1], names[:1], totals[:1], xbd_limit)
    draw_stacked_bars(events_ax, counts[1:], names[1:], totals[1:], event_limit)
    xbd_ax.set_xlabel("xBD benchmark — full count scale", fontsize=9, color="#555555")
    events_ax.set_xlabel("Event datasets — enlarged count scale", fontsize=9, color="#555555")

    handles, _ = xbd_ax.get_legend_handles_labels()
    legend_labels = [
        f"{label}: {percentage:.2f}%"
        for label, percentage in zip(CLASS_LABELS, overall_percentages)
    ]
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        title="Overall distribution across all six datasets",
        title_fontsize=9,
    )
    fig.suptitle(
        "Building Damage-Class Distribution Across the Evaluated Datasets",
        y=1.015, fontsize=14, weight="bold", color="#20242A",
    )
    fig.supxlabel("Number of connected building components", y=0.025, fontsize=10.5)
    fig.subplots_adjust(top=0.79, bottom=0.12, left=0.20, right=0.94)

    for suffix, kwargs in (
        (".png", {"dpi": 400}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Wrote figure: {path}", flush=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-parent",
        type=Path,
        default=DEFAULT_PROJECT_PARENT,
        help="Directory containing xview2 and the five preprocessed event datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/RSBDNet_dataset_statistics"),
    )
    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=1,
        help="Ignore connected components smaller than this many pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_component_pixels < 1:
        raise ValueError("--minimum-component-pixels must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = default_specs(args.data_parent)
    results = [
        count_dataset(spec, minimum_pixels=args.minimum_component_pixels)
        for spec in specs
    ]
    csv_path = args.output_dir / "damage_class_connected_components.csv"
    write_csv(results, csv_path)
    print(f"Wrote counts: {csv_path}", flush=True)
    # Use a new filename so an earlier two-panel figure is preserved.
    plot_distribution(results, args.output_dir / "building_damage_statistics")


if __name__ == "__main__":
    main()
