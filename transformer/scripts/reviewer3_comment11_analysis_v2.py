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
This v2 saves full confusion results even when a reference check fails.
Tiny numerical discrepancies are reported as near-matches, NOT exact matches.
It never modifies training scripts, original checkpoints, or original scores.
Run it with an allocated CUDA GPU, not on the HPC login/head node.
"""

from pathlib import Path
import csv
import json
import os
import torch

# Match the successful 933-pair inference run on the Blackwell GPU.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from transformer.scripts import (
    train_xbd_supervised_disasteradaptivenet as runner,
    train_xbd_bitemporal_building_crossattention_ordinal as architecture,
)

checkpoint_path = Path(
    "output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/"
    "checkpoints/best.pt"
)
xbd_root = Path("/homes/j244s673/documents/wsu/phd/xview2")
output_dir = Path(os.environ.get(
    "REVIEWER3_COMMENT11_OUTPUT_DIR", "output/Reviewer3_Comment11_v2"
))


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

    # Verify against the frozen original scores, without silently treating a
    # near-match as an exact reproduction. A very small number of boundary
    # pixels can change when GPU inference kernels/configurations differ.
    tp = int(confusion[1, 2])
    fp = int(confusion[:, 2].sum()) - tp
    fn = int(confusion[1, :].sum()) - tp
    expected = (1956658, 1340213, 1675510)
    observed = (tp, fp, fn)
    deltas = tuple(abs(a - b) for a, b in zip(observed, expected))
    relative_differences = tuple(d / b for d, b in zip(deltas, expected))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    reference_tp, reference_fp, reference_fn = expected
    reference_precision = reference_tp / (reference_tp + reference_fp)
    reference_recall = reference_tp / (reference_tp + reference_fn)
    reference_f1 = 2 * reference_precision * reference_recall / (reference_precision + reference_recall)
    metrics_match_at_four_decimals = all(
        round(a, 4) == round(b, 4)
        for a, b in zip(
            (precision, recall, f1),
            (reference_precision, reference_recall, reference_f1),
        )
    )
    exact_match = observed == expected
    tolerance = 0.0002  # 0.02% of each original TP/FP/FN count
    near_match = (
        metrics_match_at_four_decimals
        and max(relative_differences) <= tolerance
    )
    qa_status = (
        "exact_match" if exact_match
        else "near_match_not_exact" if near_match
        else "mismatch_requires_investigation"
    )
    print("\nREFERENCE METRIC REPRODUCTION CHECK", flush=True)
    print(f"Original TP/FP/FN: {expected}", flush=True)
    print(f"Current  TP/FP/FN: {observed}", flush=True)
    print(f"Absolute differences: {deltas}", flush=True)
    print(f"Relative differences (%): {[round(x * 100, 6) for x in relative_differences]}", flush=True)
    print(f"Quality assurance: {qa_status}", flush=True)
    print("Saving the raw confusion matrix and calculated metrics regardless of QA status.", flush=True)

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

    # Save the actual 933-pair results even if a validation check fails;
    # never discard a finished evaluation's raw confusion counts.
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
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "reference_reproduction_check": {
            "status": qa_status,
            "original_tp_fp_fn": list(expected),
            "observed_tp_fp_fn": list(observed),
            "absolute_differences": list(deltas),
            "relative_differences_percent": [x * 100 for x in relative_differences],
            "tolerance_percent": tolerance * 100,
            "precision_recall_f1_agree_at_four_decimals": metrics_match_at_four_decimals,
            "original_precision": reference_precision,
            "original_recall": reference_recall,
            "original_f1": reference_f1,
        },
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
    if not near_match:
        raise RuntimeError(
            "Reference metric check requires investigation (see JSON and CSV "
            f"in {output_dir}); raw results were preserved."
        )


if __name__ == "__main__":
    main()
