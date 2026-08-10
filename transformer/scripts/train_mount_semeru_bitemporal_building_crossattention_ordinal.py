#!/usr/bin/env python3
"""Train the exact HBG-CAON architecture directly on Mount Semeru eruption."""
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


class MountSemeruDataset(BASE_DATASET):
    """Standard paired loader with explicit volcanic-event metadata."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        # ID 0 is the documented volcanic condition. HBG-CAON itself remains
        # hazard-agnostic, but the common training interface requires this.
        item["cond_id"] = torch.tensor([0], dtype=torch.long)
        return item


def write_semeru_outputs(results, output_dir: Path, best_epoch: int, best_threshold: float):
    BASE_WRITE_OUTPUTS(results, output_dir, best_epoch, best_threshold)
    scores_dir = output_dir / "scores"
    json_path = scores_dir / "scores_mount_semeru_test.json"
    text_path = scores_dir / "scores_mount_semeru_test.txt"
    summary_path = scores_dir / "summary_mount_semeru.txt"

    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)

    lines = [
        "Experiment: HBG-CAON ResNet34-Swin-T cross-attention + ordinal learning",
        "Protocol: Mount Semeru train -> val selection -> untouched test",
        f"Best epoch selected on Mount Semeru validation: {best_epoch}",
        f"Best localization threshold selected on Mount Semeru validation: {best_threshold:.4f}",
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
    print(f"Wrote Mount Semeru JSON:    {json_path}", flush=True)
    print(f"Wrote Mount Semeru summary: {summary_path}", flush=True)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        print("AMP autocast dtype: bfloat16", flush=True)
    runner.XBDOriginalDataset = MountSemeruDataset
    runner.make_model = architecture.make_model
    runner.compute_supervised_losses = architecture.compute_losses
    runner.aggregate_counts = stable.stable_aggregate_counts
    runner.torch.optim.AdamW = stable.ClippedAdamW
    runner.write_final_outputs = write_semeru_outputs
    runner.main()
