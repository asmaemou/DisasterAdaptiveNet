#!/usr/bin/env python3
"""Generate paper-ready qualitative results for RS-BDNet on six test sets.

Every model is loaded from its validation-selected ``best.pt`` checkpoint.
Examples are selected from ground truth only (never prediction quality) to
represent minor, major, destroyed, and mixed-severity building damage.  The
script writes per-dataset panels, a six-dataset overview, and an audit CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable
from transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal import (
    BuildingGuidedCrossAttentionOrdinalNet,
)


CLASS_NAMES = ("No damage", "Minor damage", "Major damage", "Destroyed")
CLASS_COLORS = np.array(
    [
        [45, 156, 219],   # blue: no damage
        [242, 201, 76],   # yellow: minor damage
        [242, 153, 74],   # orange: major damage
        [214, 40, 40],    # red: destroyed
    ],
    dtype=np.uint8,
)
ERROR_NAMES = ("Correct", "Wrong severity", "Missed building", "False building")
ERROR_COLORS = np.array(
    [
        [0, 168, 120],    # teal: correct
        [111, 45, 189],   # purple: wrong severity
        [245, 245, 245],  # white: missed building
        [255, 78, 174],   # pink: false building
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    name: str
    data_root: Path
    split: str
    checkpoint: Path
    xbd_loader: bool = False


@dataclass
class Example:
    dataset: str
    stem: str
    reason: str
    pre: np.ndarray
    post: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    correctness: np.ndarray
    threshold: float
    checkpoint_epoch: int


def dataset_specs(parent: Path, project: Path) -> List[DatasetSpec]:
    return [
        DatasetSpec(
            "xbd", "xBD", parent / "xview2", "test",
            project / "output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt",
            True,
        ),
        DatasetSpec(
            "earthquake_turkey", "Earthquake Turkey",
            parent / "earthquake_turkey_preprocessed", "test",
            project / "output/HBG_CAON_EARTHQUAKE_TURKEY/checkpoints/best.pt",
        ),
        DatasetSpec(
            "mount_semeru", "Mount Semeru Eruption",
            parent / "mount_semeru_eruption_preprocessed", "test",
            project / "output/HBG_CAON_MOUNT_SEMERU_ERUPTION/checkpoints/best.pt",
        ),
        DatasetSpec(
            "texas_tornadoes", "Texas Tornadoes",
            parent / "texas_tornadoes_preprocessed", "test",
            project / "output/ResNet34SwinTinyCrossAttentionOrdinal_TEXAS_TORNADOES/checkpoints/best.pt",
        ),
        DatasetSpec(
            "hurricane_delta", "Hurricane Delta",
            parent / "hurricane_delta_preprocessed", "test",
            project / "output/ResNet34SwinTinyCrossAttentionOrdinal_HURRICANE_DELTA/checkpoints/best.pt",
        ),
        DatasetSpec(
            "pakistan_flooding", "Pakistan Flooding",
            parent / "pakistan_flooding_preprocessed", "test",
            project / "output/ResNet34SwinTinyCrossAttentionOrdinal_PAKISTAN_FLOODING/checkpoints/best.pt",
        ),
    ]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Unable to read target mask: {path}")
    return mask[..., 0] if mask.ndim == 3 else mask


def standard_truth(sample) -> np.ndarray:
    loc = read_mask(sample.pre_target_path) > 0
    damage = read_mask(sample.post_target_path)
    truth = np.zeros(damage.shape, dtype=np.uint8)
    for label in range(1, 5):
        truth[(damage == label) & loc] = label
    return truth


def selection_statistics(dataset) -> List[Dict[str, object]]:
    statistics = []
    for index, sample in enumerate(dataset.samples):
        truth = standard_truth(sample)
        counts = np.array([(truth == label).sum() for label in range(1, 5)], dtype=np.int64)
        statistics.append(
            {
                "index": index,
                "stem": sample.stem,
                "counts": counts,
                "classes": int(np.count_nonzero(counts)),
                "building_pixels": int(counts.sum()),
            }
        )
    return statistics


def choose_examples(statistics: Sequence[Dict[str, object]], number: int) -> List[Tuple[int, str]]:
    """Select from ground truth only, targeting rare severities then diversity."""
    selected: List[Tuple[int, str]] = []
    used = set()
    targets = ((2, "minor-damage example"), (3, "major-damage example"), (4, "destroyed example"))
    for label, reason in targets:
        candidates = [row for row in statistics if int(row["counts"][label - 1]) > 0]
        candidates.sort(
            key=lambda row: (
                int(row["counts"][label - 1]), int(row["classes"]),
                int(row["building_pixels"]), str(row["stem"]),
            ),
            reverse=True,
        )
        candidate = next((row for row in candidates if row["index"] not in used), None)
        if candidate is not None and len(selected) < number:
            selected.append((int(candidate["index"]), reason))
            used.add(candidate["index"])

    remaining = sorted(
        (row for row in statistics if row["index"] not in used),
        key=lambda row: (
            int(row["classes"]), int(row["counts"][1:].sum()),
            int(row["building_pixels"]), str(row["stem"]),
        ),
        reverse=True,
    )
    for row in remaining:
        if len(selected) >= number:
            break
        selected.append((int(row["index"]), "multi-class representative example"))
        used.add(row["index"])
    if not selected:
        raise RuntimeError("No test examples were available for qualitative visualization")
    return selected


def load_checkpoint(model, path: Path, device: torch.device) -> Dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"Checkpoint has no model/state_dict entry: {path}")
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def color_overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for label, color in enumerate(CLASS_COLORS, start=1):
        mask = labels == label
        output[mask] = (1.0 - alpha) * output[mask] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)


def correctness_map(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    # Black is reserved exclusively for pixels where neither truth nor the
    # model identifies a building.
    result = np.zeros((*truth.shape, 3), dtype=np.uint8)
    true_building = truth > 0
    pred_building = prediction > 0
    result[true_building & pred_building & (truth == prediction)] = ERROR_COLORS[0]
    result[true_building & pred_building & (truth != prediction)] = ERROR_COLORS[1]
    result[true_building & ~pred_building] = ERROR_COLORS[2]
    result[~true_building & pred_building] = ERROR_COLORS[3]
    return result


@torch.inference_mode()
def infer_selected(spec: DatasetSpec, args, device: torch.device):
    dataset_class = stable.MultiSplitHazardDataset if spec.xbd_loader else runner.XBDOriginalDataset
    dataset = dataset_class(spec.data_root, spec.split, args.image_size, False, 0)
    selected = choose_examples(selection_statistics(dataset), args.examples_per_dataset)

    model = BuildingGuidedCrossAttentionOrdinalNet(image_size=args.image_size, width=96).to(device)
    checkpoint = load_checkpoint(model, spec.checkpoint, device)
    model.eval()
    threshold = float(checkpoint.get("best_threshold", 0.5))
    epoch = int(checkpoint.get("epoch", -1))
    print(
        f"{spec.name}: test={len(dataset)}, checkpoint epoch={epoch}, "
        f"localization threshold={threshold:.2f}", flush=True,
    )

    examples = []
    for index, reason in selected:
        sample = dataset.samples[index]
        item = dataset[index]
        image = item["img"].unsqueeze(0).to(device)
        condition = item["cond_id"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(image, condition)
        loc = torch.sigmoid(logits[0, 0]) > threshold
        damage = torch.argmax(logits[0, 1:5], dim=0).to(torch.uint8) + 1
        prediction = (damage * loc.to(torch.uint8)).cpu().numpy()

        pre = read_rgb(sample.pre_image_path)
        post = read_rgb(sample.post_image_path)
        truth = standard_truth(sample)
        prediction = cv2.resize(
            prediction, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        examples.append(
            Example(
                spec.name, sample.stem, reason, pre, post, truth, prediction,
                correctness_map(truth, prediction), threshold, epoch,
            )
        )
        print(f"  selected {sample.stem}: {reason}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return examples


def damage_legend():
    return [Patch(facecolor=color / 255.0, label=name) for name, color in zip(CLASS_NAMES, CLASS_COLORS)]


def error_legend():
    handles = [
        Patch(facecolor=color / 255.0, label=name, edgecolor="#9CA3AF" if name == "Missed building" else "none")
        for name, color in zip(ERROR_NAMES, ERROR_COLORS)
    ]
    handles.append(Patch(facecolor="black", label="No-building area"))
    return handles


def save_figure(fig, stem: Path) -> None:
    for suffix, options in ((".png", {"dpi": 350}), (".pdf", {}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        print(f"Wrote: {path}", flush=True)
    plt.close(fig)


def plot_dataset(examples: Sequence[Example], output_stem: Path) -> None:
    columns = ("Pre-disaster", "Post-disaster", "Ground truth", "Prediction", "Error analysis")
    fig, axes = plt.subplots(len(examples), 5, figsize=(15.2, 3.0 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        images = (
            example.pre, example.post, color_overlay(example.post, example.truth),
            color_overlay(example.post, example.prediction), example.correctness,
        )
        for column, image in enumerate(images):
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(columns[column], fontsize=11, weight="bold")
        axes[row, 0].set_ylabel(
            f"Test example {row + 1}\n{example.stem}", fontsize=8.2, rotation=0,
            ha="right", va="center", labelpad=8,
        )
    fig.suptitle(f"{examples[0].dataset} Test-Set Qualitative Results", fontsize=15, weight="bold", y=0.995)
    fig.legend(handles=damage_legend(), loc="lower center", bbox_to_anchor=(0.39, -0.006), ncol=4, frameon=False)
    fig.legend(handles=error_legend(), loc="lower center", bbox_to_anchor=(0.80, -0.006), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.09, right=0.995, wspace=0.025, hspace=0.10)
    save_figure(fig, output_stem)


def plot_overview(all_examples: Dict[str, Sequence[Example]], output_stem: Path) -> None:
    selected = [max(items, key=lambda item: len(np.unique(item.truth[item.truth > 0]))) for items in all_examples.values()]
    columns = ("Pre-disaster", "Post-disaster", "Ground truth", "Prediction", "Error analysis")
    fig, axes = plt.subplots(len(selected), 5, figsize=(15.0, 2.65 * len(selected)), squeeze=False)
    for row, example in enumerate(selected):
        images = (
            example.pre, example.post, color_overlay(example.post, example.truth),
            color_overlay(example.post, example.prediction), example.correctness,
        )
        for column, image in enumerate(images):
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(columns[column], fontsize=11, weight="bold")
        axes[row, 0].set_ylabel(example.dataset, fontsize=10, weight="bold", rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("RS-BDNet Qualitative Results Across Six Disaster Datasets", fontsize=15, weight="bold", y=0.995)
    fig.legend(handles=damage_legend(), loc="lower center", bbox_to_anchor=(0.39, -0.002), ncol=4, frameon=False)
    fig.legend(handles=error_legend(), loc="lower center", bbox_to_anchor=(0.80, -0.002), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.95, bottom=0.045, left=0.12, right=0.995, wspace=0.025, hspace=0.09)
    save_figure(fig, output_stem)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-parent", type=Path, default=Path("/homes/j244s673/documents/wsu/phd"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/RSBDNet_qualitative_results"))
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else args.project_root / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs(args.data_parent, args.project_root)
    missing = [str(spec.checkpoint) for spec in specs if not spec.checkpoint.is_file()]
    if missing:
        raise FileNotFoundError("Missing validation-selected checkpoint(s):\n- " + "\n- ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    all_examples: Dict[str, List[Example]] = {}
    manifest_rows = []
    for spec in specs:
        examples = infer_selected(spec, args, device)
        all_examples[spec.name] = examples
        plot_dataset(examples, args.output_dir / f"qualitative_{spec.slug}")
        for example in examples:
            manifest_rows.append(
                {
                    "dataset": example.dataset, "test_stem": example.stem,
                    "selection_reason": example.reason, "checkpoint": str(spec.checkpoint),
                    "checkpoint_epoch": example.checkpoint_epoch,
                    "validation_selected_localization_threshold": example.threshold,
                }
            )

    plot_overview(all_examples, args.output_dir / "qualitative_overview_six_datasets")
    manifest = args.output_dir / "qualitative_selection_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "selection_policy": "Ground-truth severity coverage only; prediction quality was not used.",
        "test_sets_only": True,
        "examples_per_dataset": args.examples_per_dataset,
        "damage_classes": list(CLASS_NAMES),
        "outputs": "Per-dataset and combined PNG/PDF/SVG panels",
    }
    (args.output_dir / "qualitative_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote audit manifest: {manifest}", flush=True)
    print("DONE: six-dataset RS-BDNet qualitative evaluation", flush=True)


if __name__ == "__main__":
    main()
