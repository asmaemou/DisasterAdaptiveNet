from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_rescuenet_xbd import RescueNetXBDDataset
from utils.models import DisasterAdaptiveNet


try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser("Evaluate DisasterAdaptiveNet on RescueNet-xBD")
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
    parser.add_argument("--output-json", type=str, default="")
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


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    try:
        model.load_state_dict(state, strict=True)
        return ckpt
    except RuntimeError:
        pass

    # strip "module." if checkpoint was saved from DataParallel
    stripped = {}
    for k, v in state.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v
    try:
        model.load_state_dict(stripped, strict=True)
        return ckpt
    except RuntimeError:
        pass

    # add "module." if current model is DataParallel but checkpoint is not
    added = {}
    for k, v in state.items():
        if not k.startswith("module."):
            added[f"module.{k}"] = v
        else:
            added[k] = v
    model.load_state_dict(added, strict=True)
    return ckpt


@torch.no_grad()
def evaluate(model, loader, device: torch.device, loc_threshold: float):
    model.eval()

    # Localization: positive class is building
    loc_tp = 0
    loc_fp = 0
    loc_fn = 0

    # Damage classes in xBD-style numbering:
    # 1=no damage, 2=minor, 3=major, 4=destroyed
    dmg_stats = {
        1: {"tp": 0, "fp": 0, "fn": 0, "name": "no_damage"},
        2: {"tp": 0, "fp": 0, "fn": 0, "name": "minor_damage"},
        3: {"tp": 0, "fp": 0, "fn": 0, "name": "major_damage"},
        4: {"tp": 0, "fp": 0, "fn": 0, "name": "destroyed"},
    }

    use_tqdm = tqdm is not None and sys.stderr.isatty()
    iterator = tqdm(loader, desc="eval", leave=False) if use_tqdm else loader

    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        dmg_true_raw = batch["dmg"].to(device, non_blocking=True).long()
        cond_id = batch["cond_id"].to(device, non_blocking=True)

        logits = model(img, cond_id)
        logit_loc = logits[:, 0]
        logit_dmg = logits[:, 1:5]

        # Localization prediction
        loc_pred = (torch.sigmoid(logit_loc) > loc_threshold).long()

        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true != 1)).sum().item())
        loc_fn += int(((loc_pred != 1) & (loc_true == 1)).sum().item())

        # Damage prediction:
        # model outputs classes 0..3 -> convert to xBD-style 1..4
        dmg_pred = torch.argmax(logit_dmg, dim=1) + 1

        # only give damage credit where buildings are predicted
        dmg_pred = dmg_pred * loc_pred

        # ground truth damage:
        # valid building pixels are where damage label is not ignore and localization gt says building
        valid_gt = (dmg_true_raw != 255) & (loc_true == 1)

        # convert gt from 0..3 to 1..4 on valid pixels
        dmg_true = torch.zeros_like(dmg_true_raw)
        dmg_true[valid_gt] = dmg_true_raw[valid_gt] + 1

        dp = dmg_pred[valid_gt]
        dt = dmg_true[valid_gt]

        for cls_idx in [1, 2, 3, 4]:
            tp = ((dp == cls_idx) & (dt == cls_idx)).sum()
            fn = ((dp != cls_idx) & (dt == cls_idx)).sum()
            fp = ((dp == cls_idx) & (dt != cls_idx)).sum()

            dmg_stats[cls_idx]["tp"] += int(tp.item())
            dmg_stats[cls_idx]["fn"] += int(fn.item())
            dmg_stats[cls_idx]["fp"] += int(fp.item())

    loc_f1r = F1Recorder(loc_tp, loc_fp, loc_fn, "building")

    no_damage_f1r = F1Recorder(
        dmg_stats[1]["tp"], dmg_stats[1]["fp"], dmg_stats[1]["fn"], "no_damage"
    )
    minor_damage_f1r = F1Recorder(
        dmg_stats[2]["tp"], dmg_stats[2]["fp"], dmg_stats[2]["fn"], "minor_damage"
    )
    major_damage_f1r = F1Recorder(
        dmg_stats[3]["tp"], dmg_stats[3]["fp"], dmg_stats[3]["fn"], "major_damage"
    )
    destroyed_f1r = F1Recorder(
        dmg_stats[4]["tp"], dmg_stats[4]["fp"], dmg_stats[4]["fn"], "destroyed"
    )

    damage_f1s = [
        no_damage_f1r.f1,
        minor_damage_f1r.f1,
        major_damage_f1r.f1,
        destroyed_f1r.f1,
    ]
    damage_f1 = harmonic_mean(damage_f1s)
    score = 0.3 * loc_f1r.f1 + 0.7 * damage_f1

    results = {
        "score": score,
        "localization_f1": loc_f1r.f1,
        "damage_f1": damage_f1,
        "damage_f1_no_damage": no_damage_f1r.f1,
        "damage_f1_minor_damage": minor_damage_f1r.f1,
        "damage_f1_major_damage": major_damage_f1r.f1,
        "damage_f1_destroyed": destroyed_f1r.f1,
        "details": {
            "localization_building": loc_f1r.as_dict(),
            "no_damage": no_damage_f1r.as_dict(),
            "minor_damage": minor_damage_f1r.as_dict(),
            "major_damage": major_damage_f1r.as_dict(),
            "destroyed": destroyed_f1r.as_dict(),
        },
    }
    return results


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("===== EVAL INFO =====", flush=True)
    print(f"device:       {device}", flush=True)
    print(f"dataset_root: {args.dataset_root}", flush=True)
    print(f"checkpoint:   {args.checkpoint}", flush=True)
    print(f"split:        {args.split}", flush=True)
    print(f"batch_size:   {args.batch_size}", flush=True)
    print(f"num_workers:  {args.num_workers}", flush=True)
    print(f"img_size:     {args.img_size}", flush=True)
    print(f"loc_threshold:{args.loc_threshold}", flush=True)
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

    print(f"Loaded {len(dataset)} samples for split='{args.split}'", flush=True)

    model = make_model(device)
    _ = load_model_checkpoint(model, args.checkpoint, device)

    results = evaluate(model, loader, device, args.loc_threshold)

    print(json.dumps(results, indent=2), flush=True)

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        ckpt_path = Path(args.checkpoint).resolve()
        out_dir = ckpt_path.parent.parent / "scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"scores_rescuenet_xbd_{args.split}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to: {out_path}", flush=True)


if __name__ == "__main__":
    main()