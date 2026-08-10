#!/usr/bin/env python3
"""Train the ResNet34-Swin-Tiny cross-attention model on Hurricane Delta."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable
import transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal as architecture


BASE_DATASET = runner.XBDOriginalDataset
BASE_WRITE_OUTPUTS = runner.write_final_outputs


class HurricaneDeltaDataset(BASE_DATASET):
    """Standard paired loader for Hurricane Delta train/val/test data."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        # Required by the common runner but ignored by this hazard-agnostic
        # architecture. This does not introduce xBD information.
        item["cond_id"] = torch.tensor([0], dtype=torch.long)
        return item


def write_hurricane_delta_outputs(
    results, output_dir: Path, best_epoch: int, best_threshold: float
):
    """Write compatibility files and clearly named Hurricane Delta results."""
    BASE_WRITE_OUTPUTS(results, output_dir, best_epoch, best_threshold)
    scores_dir = output_dir / "scores"
    json_path = scores_dir / "scores_hurricane_delta_test.json"
    text_path = scores_dir / "scores_hurricane_delta_test.txt"
    summary_path = scores_dir / "summary_hurricane_delta.txt"

    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)

    lines = [
        "Experiment: ImageNet-pretrained Siamese ResNet34-Swin-Tiny with "
        "building-guided cross-attention and ordinal damage learning",
        "Protocol: Hurricane Delta train -> validation selection -> untouched test",
        "xBD data or xBD-trained checkpoints used: No",
        f"Best epoch selected on Hurricane Delta validation: {best_epoch}",
        f"Best localization threshold selected on Hurricane Delta validation: {best_threshold:.4f}",
        f"Localization F1: {results['localization_f1']:.6f}",
        f"No Damage F1:    {results['damage_f1_no_damage']:.6f}",
        f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}",
        f"Major Damage F1: {results['damage_f1_major_damage']:.6f}",
        f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}",
        f"Damage F1:       {results['damage_f1']:.6f}",
        f"Overall Score:   {results['score']:.6f}",
    ]
    content = "\n".join(lines) + "\n"
    text_path.write_text(content, encoding="utf-8")
    summary_path.write_text(content, encoding="utf-8")
    print(f"Wrote Hurricane Delta JSON:    {json_path}", flush=True)
    print(f"Wrote Hurricane Delta summary: {summary_path}", flush=True)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        print("AMP autocast dtype: bfloat16", flush=True)
    runner.XBDOriginalDataset = HurricaneDeltaDataset
    runner.make_model = architecture.make_model
    runner.compute_supervised_losses = architecture.compute_losses
    runner.aggregate_counts = stable.stable_aggregate_counts
    runner.torch.optim.AdamW = stable.ClippedAdamW
    runner.write_final_outputs = write_hurricane_delta_outputs
    runner.main()
