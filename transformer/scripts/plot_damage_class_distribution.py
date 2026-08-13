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
COLORS = ("#20B2C1", "#18A64A", "#225EA8", "#EF2B2D")
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
    """Draw absolute class-wise building counts as grouped horizontal bars."""
    ordered = sorted(results, key=lambda item: int(item["total"]), reverse=True)
    names = [str(item["name"]) for item in ordered]
    counts = np.stack([item["counts"] for item in ordered]).astype(np.float64)

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
    fig, axis = plt.subplots(figsize=(10.8, 7.2))
    group_centers = np.arange(len(names), dtype=np.float64)
    bar_height = 0.16
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_height
    positive_counts = counts[counts > 0]
    x_min = max(1.0, 10.0 ** np.floor(np.log10(positive_counts.min())))
    x_max = 10.0 ** np.ceil(np.log10(positive_counts.max()))

    for class_index, (label, color) in enumerate(zip(CLASS_LABELS, COLORS)):
        values = counts[:, class_index]
        positions = group_centers + offsets[class_index]
        visible_values = np.maximum(values, x_min)
        axis.barh(
            positions,
            np.maximum(visible_values - x_min, 0.0),
            left=x_min,
            color=color,
            height=bar_height * 0.86,
            label=label,
            edgecolor="white",
            linewidth=0.4,
        )
        for row, value in enumerate(values):
            axis.annotate(
                f"{int(value):,}",
                xy=(visible_values[row], positions[row]),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.6,
                color="#30343B",
            )

    axis.set_yticks(group_centers, names)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlim(x_min, x_max * 1.55)
    axis.set_xlabel("Number of labeled building components (log scale)")
    axis.set_title(
        "Class-Wise Distribution of Building Damage Labels Across Datasets",
        loc="left",
        weight="bold",
        pad=18,
    )
    axis.grid(axis="x", which="major", color="#D9D9D9", linewidth=0.75)
    axis.grid(axis="x", which="minor", color="#ECEFF2", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.24, right=0.94)

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
