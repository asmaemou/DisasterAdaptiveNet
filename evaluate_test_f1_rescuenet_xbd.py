from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_rescuenet_xbd import RescueNetXBDDataset
from utils.models import DisasterAdaptiveNet


def parse_args():
    parser = argparse.ArgumentParser("Evaluate trained DisasterAdaptiveNet on RescueNet-xBD test set")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/rescuenet_xbd_disasteradaptivenet/checkpoints/best.pt",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--conditioning-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loc-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/rescuenet_xbd_disasteradaptivenet/scores",
    )
    return parser.parse_args()


class F1Recorder:
    def __init__(self, tp: int, fp: int, fn: int, name: str):
        self.tp = int(tp)
        self.fp = int(fp)
        self.fn = int(fn)
        self.name = name

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 0.0 if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 0.0 if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 0.0 if (p == 0.0 or r == 0.0) else (2.0 * p * r) / (p + r)

    def as_dict(self):
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def harmonic_mean(xs):
    xs = [float(x) for x in xs]
    return len(xs) / sum((x + 1e-6) ** -1 for x in xs)


def make_model(device: torch.device) -> nn.Module:
    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(OUT_CHANNELS=5),
        DATASET=SimpleNamespace(CONDITIONING_KEY={"generic": 0}),
    )
    model = DisasterAdaptiveNet(cfg)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    model.to(device)
    return model


def load_checkpoint_safely(model: nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    stripped = {}
    for k, v in state.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v
    try:
        model.load_state_dict(stripped, strict=True)
        return
    except RuntimeError:
        pass

    added = {}
    for k, v in state.items():
        if not k.startswith("module."):
            added[f"module.{k}"] = v
        else:
            added[k] = v
    model.load_state_dict(added, strict=True)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, loc_threshold: float):
    model.eval()

    # Localization positive class = building
    loc_tp = 0
    loc_fp = 0
    loc_fn = 0

    # Damage classes in xBD-style numbering:
    # 1 = no damage, 2 = minor, 3 = major, 4 = destroyed
    dmg_counts = {
        1: {"tp": 0, "fp": 0, "fn": 0, "name": "no_damage"},
        2: {"tp": 0, "fp": 0, "fn": 0, "name": "minor_damage"},
        3: {"tp": 0, "fp": 0, "fn": 0, "name": "major_damage"},
        4: {"tp": 0, "fp": 0, "fn": 0, "name": "destroyed"},
    }

    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        dmg_true_raw = batch["dmg"].to(device, non_blocking=True).long()
        cond_id = batch["cond_id"].to(device, non_blocking=True)

        logits = model(img, cond_id)
        loc_logits = logits[:, 0]
        dmg_logits = logits[:, 1:5]

        # Localization prediction
        loc_pred = (torch.sigmoid(loc_logits) > loc_threshold).long()

        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())

        # Damage prediction: model outputs 0..3, convert to 1..4
        dmg_pred = torch.argmax(dmg_logits, dim=1) + 1

        # Only give damage credit where buildings are predicted
        dmg_pred = dmg_pred * loc_pred

        # Valid GT damage pixels: building pixels with non-ignore damage label
        valid_gt = (loc_true == 1) & (dmg_true_raw != 255)

        # Convert GT damage from 0..3 to 1..4
        dmg_true = torch.zeros_like(dmg_true_raw)
        dmg_true[valid_gt] = dmg_true_raw[valid_gt] + 1

        dp = dmg_pred[valid_gt]
        dt = dmg_true[valid_gt]

        for c in [1, 2, 3, 4]:
            tp = ((dp == c) & (dt == c)).sum()
            fp = ((dp == c) & (dt != c)).sum()
            fn = ((dp != c) & (dt == c)).sum()

            dmg_counts[c]["tp"] += int(tp.item())
            dmg_counts[c]["fp"] += int(fp.item())
            dmg_counts[c]["fn"] += int(fn.item())

    loc_f1 = F1Recorder(loc_tp, loc_fp, loc_fn, "localization")

    no_damage_f1 = F1Recorder(
        dmg_counts[1]["tp"], dmg_counts[1]["fp"], dmg_counts[1]["fn"], "no_damage"
    )
    minor_damage_f1 = F1Recorder(
        dmg_counts[2]["tp"], dmg_counts[2]["fp"], dmg_counts[2]["fn"], "minor_damage"
    )
    major_damage_f1 = F1Recorder(
        dmg_counts[3]["tp"], dmg_counts[3]["fp"], dmg_counts[3]["fn"], "major_damage"
    )
    destroyed_f1 = F1Recorder(
        dmg_counts[4]["tp"], dmg_counts[4]["fp"], dmg_counts[4]["fn"], "destroyed"
    )

    damage_f1s = [
        no_damage_f1.f1,
        minor_damage_f1.f1,
        major_damage_f1.f1,
        destroyed_f1.f1,
    ]
    damage_f1 = harmonic_mean(damage_f1s)
    score = 0.3 * loc_f1.f1 + 0.7 * damage_f1

    results = {
        "score": score,
        "localization_f1": loc_f1.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no_damage_f1.f1,
        "damage_f1_minor_damage": minor_damage_f1.f1,
        "damage_f1_major_damage": major_damage_f1.f1,
        "damage_f1_destroyed": destroyed_f1.f1,
        "details": {
            "localization": loc_f1.as_dict(),
            "no_damage": no_damage_f1.as_dict(),
            "minor_damage": minor_damage_f1.as_dict(),
            "major_damage": major_damage_f1.as_dict(),
            "destroyed": destroyed_f1.as_dict(),
        },
    }
    return results


def write_outputs(results: dict, output_dir: Path, split: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"scores_rescuenet_xbd_{split}.json"
    txt_path = output_dir / f"scores_rescuenet_xbd_{split}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Localization F1: {results['localization_f1']:.6f}\n")
        f.write(f"No Damage F1:    {results['damage_f1_no_damage']:.6f}\n")
        f.write(f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}\n")
        f.write(f"Major Damage F1: {results['damage_f1_major_damage']:.6f}\n")
        f.write(f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}\n")
        f.write(f"Damage F1:       {results['damage_f1']:.6f}\n")
        f.write(f"Overall Score:   {results['score']:.6f}\n")

    return json_path, txt_path


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== EVAL INFO =====", flush=True)
    print(f"device:        {device}", flush=True)
    print(f"dataset_root:  {args.dataset_root}", flush=True)
    print(f"checkpoint:    {args.checkpoint}", flush=True)
    print(f"split:         {args.split}", flush=True)
    print(f"batch_size:    {args.batch_size}", flush=True)
    print(f"num_workers:   {args.num_workers}", flush=True)
    print(f"img_size:      {args.img_size}", flush=True)
    print(f"loc_threshold: {args.loc_threshold}", flush=True)
    print("=====================", flush=True)

    dataset = RescueNetXBDDataset(
        root=args.dataset_root,
        split=args.split,
        image_size=args.img_size,
        training=False,
        conditioning_id=args.conditioning_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Loaded {len(dataset)} samples", flush=True)

    model = make_model(device)
    load_checkpoint_safely(model, args.checkpoint, device)

    results = evaluate(model, loader, device, args.loc_threshold)

    print("\n===== TEST F1 SCORES =====", flush=True)
    print(f"Localization F1: {results['localization_f1']:.6f}", flush=True)
    print(f"No Damage F1:    {results['damage_f1_no_damage']:.6f}", flush=True)
    print(f"Minor Damage F1: {results['damage_f1_minor_damage']:.6f}", flush=True)
    print(f"Major Damage F1: {results['damage_f1_major_damage']:.6f}", flush=True)
    print(f"Destroyed F1:    {results['damage_f1_destroyed']:.6f}", flush=True)
    print(f"Damage F1:       {results['damage_f1']:.6f}", flush=True)
    print(f"Overall Score:   {results['score']:.6f}", flush=True)

    output_dir = Path(args.output_dir)
    json_path, txt_path = write_outputs(results, output_dir, args.split)

    print(f"\nSaved JSON to: {json_path}", flush=True)
    print(f"Saved TXT to:  {txt_path}", flush=True)


if __name__ == "__main__":
    main()