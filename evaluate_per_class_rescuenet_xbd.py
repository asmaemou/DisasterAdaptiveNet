from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_rescuenet_xbd import RescueNetXBDDataset
from utils.models import DisasterAdaptiveNet


def parse_args():
    parser = argparse.ArgumentParser("Evaluate DisasterAdaptiveNet on RescueNet-xBD with per-class F1")
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--conditioning-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


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


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.view(-1).cpu()
        y_pred = y_pred.view(-1).cpu()
        valid = (y_true >= 0) & (y_true < self.num_classes)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if y_true.numel() == 0:
            return
        idx = self.num_classes * y_true + y_pred
        bins = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.matrix += bins.reshape(self.num_classes, self.num_classes)

    def per_class_metrics(self, class_names):
        cm = self.matrix.float()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)

        rows = {}
        for i, name in enumerate(class_names):
            rows[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "iou": float(iou[i]),
                "support": int(cm[i].sum().item()),
            }
        return rows

    def macro_f1(self):
        cm = self.matrix.float()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return float(torch.nanmean(f1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    # Localization confusion: binary, class 1 = building
    loc_conf = ConfusionMatrix(num_classes=2)

    # Damage confusion: 4 classes
    dmg_conf = ConfusionMatrix(num_classes=4)

    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        loc = batch["loc"].to(device, non_blocking=True)
        dmg = batch["dmg"].to(device, non_blocking=True)
        cond_id = batch["cond_id"].to(device, non_blocking=True)

        logits = model(img, cond_id)
        logit_loc = logits[:, 0]
        logit_dmg = logits[:, 1:5]

        # Localization
        loc_pred = (torch.sigmoid(logit_loc) > 0.5).long()
        loc_true = loc.long()
        loc_conf.update(loc_true, loc_pred)

        # Damage
        dmg_pred = torch.argmax(logit_dmg, dim=1)
        valid = dmg != 255
        if valid.any():
            dmg_conf.update(dmg[valid], dmg_pred[valid])

    loc_metrics = loc_conf.per_class_metrics(["background", "building"])
    dmg_metrics = dmg_conf.per_class_metrics(["no_damage", "minor_damage", "major_damage", "destroyed"])

    results = {
        "localization": {
            "background": loc_metrics["background"],
            "building": loc_metrics["building"],
            "macro_f1": loc_conf.macro_f1(),
            "building_f1": loc_metrics["building"]["f1"],
        },
        "damage": {
            "no_damage": dmg_metrics["no_damage"],
            "minor_damage": dmg_metrics["minor_damage"],
            "major_damage": dmg_metrics["major_damage"],
            "destroyed": dmg_metrics["destroyed"],
            "macro_f1": dmg_conf.macro_f1(),
        },
    }
    return results


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    model = make_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    results = evaluate(model, loader, device)

    print(json.dumps(results, indent=2))

    out_path = Path(args.checkpoint).resolve().parent.parent / f"{args.split}_per_class_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()