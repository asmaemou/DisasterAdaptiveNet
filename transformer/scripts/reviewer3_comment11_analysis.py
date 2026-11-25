#!/usr/bin/env python3
"""Reviewer #3, Comment (11): xBD minor-damage pixel-level error breakdown.

Re-evaluate the original epoch-35 RS-BDNet checkpoint on the held-out xBD
Test partition, using the original categorical prediction pathway and
validation-selected localization threshold (0.50).

Report how minor-damage ground-truth pixels are distributed among:
  - missed localization,
  - predicted no damage,
  - correctly predicted minor damage,
  - predicted major damage,
  - predicted destroyed.

This is a *pixel-level* analysis, not a count of individual building polygons.
It never modifies training scripts, original checkpoints, or original scores.
Run it with an allocated CUDA GPU, not on the HPC login/head node.
"""

from pathlib import Path
import csv
import json
import torch

from transformer.scripts import (
    train_xbd_supervised_disasteradaptivenet as runner,
    train_xbd_bitemporal_building_crossattention_ordinal as architecture,
)

checkpoint_path = Path(
    "output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/"
    "checkpoints/best.pt"
)
xbd_root = Path("/homes/j244s673/documents/wsu/phd/xview2")
output_dir = Path("output/Reviewer3_Comment11")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run this analysis on an allocated GPU compute node.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output folder already exists: {output_dir}. "
            "Refusing to overwrite an earlier analysis."
        )

    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    threshold = float(checkpoint["best_threshold"])
    if int(checkpoint["epoch"]) != 35 or abs(threshold - 0.5) >= 1e-8:
        raise RuntimeError("Unexpected checkpoint epoch or localization threshold.")

    model = architecture.make_model(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()

    dataset = runner.XBDOriginalDataset(
        root=xbd_root,
        split="test",
        image_size=896,
        training=False,
        conditioning_id=0,
    )
    loader = runner.make_eval_loader(dataset, 2, 0)

    # Rows (ground truth): no damage, minor, major, destroyed (0..3).
    # Columns (predictions): missed localization, no damage,
    #                        minor, major, destroyed (0..4).
    confusion = torch.zeros((4, 5), dtype=torch.int64)
    processed = 0

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            images = batch["img"].to(device, non_blocking=True)
            gt_loc = batch["loc"].to(device, non_blocking=True).long()
            gt_dmg = batch["dmg"].to(device, non_blocking=True).long()
            cond = batch["cond_id"].to(device, non_blocking=True)

            logits = model(images, cond)
            pred_loc = (torch.sigmoid(logits[:, 0]) > threshold).long()
            pred_dmg = (torch.argmax(logits[:, 1:5], dim=1) + 1) * pred_loc

            valid = (gt_loc == 1) & (gt_dmg != 255)
            if bool(valid.any()):
                counts = torch.bincount(
                    gt_dmg[valid] * 5 + pred_dmg[valid],
                    minlength=20,
                ).reshape(4, 5)
                confusion += counts.cpu()

            processed += images.shape[0]
            if (index + 1) % 50 == 0 or processed == len(dataset):
                print(f"Processed {processed}/{len(dataset)} test pairs", flush=True)

    if processed != len(dataset):
        raise RuntimeError(f"Incomplete test evaluation: {processed} != {len(dataset)}")

    # Reproduce exact minor-damage pixel counts from scores_xbd_test.json.
    tp = int(confusion[1, 2])
    fp = int(confusion[:, 2].sum()) - tp
    fn = int(confusion[1, :].sum()) - tp
    expected = (1956658, 1340213, 1675510)
    if (tp, fp, fn) != expected:
        raise RuntimeError(
            "Original minor-damage evaluation was not reproduced: "
            f"got TP/FP/FN={(tp, fp, fn)}, expected={expected}. "
            "Check dataset, checkpoint and inference configuration."
        )

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    prediction_names = [
        "missed_localization",
        "predicted_no_damage",
        "correct_minor_damage",
        "predicted_major_damage",
        "predicted_destroyed",
    ]
    minor_counts = {
        name: int(confusion[1, i]) for i, name in enumerate(prediction_names)
    }

    # Only create a separate, new output folder after the metric check passes.
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "ground_truth",
            "missed_localization",
            "no_damage",
            "minor_damage",
            "major_damage",
            "destroyed",
        ])
        for name, row in zip(
            ["no_damage", "minor_damage", "major_damage", "destroyed"],
            confusion.tolist(),
        ):
            writer.writerow([name, *row])

    results = {
        "level": "pixel",
        "checkpoint_epoch": 35,
        "localization_threshold": threshold,
        "test_pairs": len(dataset),
        "minor_precision": precision,
        "minor_recall": recall,
        "minor_f1": f1,
        "minor_true_positive_pixels": tp,
        "minor_false_positive_pixels": fp,
        "minor_false_negative_pixels": fn,
        "minor_ground_truth_pixels": int(confusion[1].sum()),
        "minor_error_breakdown_pixels": minor_counts,
    }
    with (output_dir / "minor_damage_analysis.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nMINOR-DAMAGE ANALYSIS (pixel-level)")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")
    total = int(confusion[1].sum())
    for name, count in minor_counts.items():
        print(f"{name}: {count:,} ({100 * count / total:.2f}%)")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
