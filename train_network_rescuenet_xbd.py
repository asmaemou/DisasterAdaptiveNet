from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset_rescuenet_xbd import RescueNetXBDDataset
from utils.models import DisasterAdaptiveNet


try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train DisasterAdaptiveNet on RescueNet-xBD")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/homes/j244s673/documents/wsu/phd/uda_two_stage/rescuenet_xbd",
        help="Root directory of rescuenet_xbd.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/rescuenet_xbd_disasteradaptivenet",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--conditioning-id", type=int, default=0, help="Dummy condition id used for every sample.")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loc-bce-weight", type=float, default=1.0)
    parser.add_argument("--loc-dice-weight", type=float, default=1.0)
    parser.add_argument("--dmg-ce-weight", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += int(n)


class RunningConfusionMatrix:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
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

    def macro_f1(self) -> float:
        cm = self.matrix.float()
        tp = torch.diag(cm)
        precision = tp / (cm.sum(dim=0) + 1e-7)
        recall = tp / (cm.sum(dim=1) + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return float(torch.nanmean(f1))

    def accuracy(self) -> float:
        cm = self.matrix.float()
        total = cm.sum()
        return float(torch.diag(cm).sum() / (total + 1e-7))


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + 1e-7) / (union + 1e-7)).mean()
        return bce, dice


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


def make_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    train_ds = RescueNetXBDDataset(
        root=args.dataset_root,
        split="train",
        image_size=args.img_size,
        training=True,
        conditioning_id=args.conditioning_id,
    )
    val_ds = RescueNetXBDDataset(
        root=args.dataset_root,
        split="val",
        image_size=args.img_size,
        training=False,
        conditioning_id=args.conditioning_id,
    )
    test_ds = RescueNetXBDDataset(
        root=args.dataset_root,
        split="test",
        image_size=args.img_size,
        training=False,
        conditioning_id=args.conditioning_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    loc_pos, loc_neg = train_ds.get_localization_pixel_counts()
    loc_pos_weight = torch.tensor([max(1.0, loc_neg / max(loc_pos, 1))], dtype=torch.float32)

    dmg_counts = train_ds.get_damage_class_counts().astype(np.float64)
    dmg_counts[dmg_counts == 0] = 1.0
    inv = dmg_counts.sum() / dmg_counts
    dmg_class_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")
    print(f"Localization pixel counts (pos, neg): {(loc_pos, loc_neg)}")
    print(f"Damage class counts [no, minor, major, destroyed]: {train_ds.get_damage_class_counts().tolist()}")
    print(f"Damage CE class weights: {dmg_class_weights.tolist()}")

    return train_loader, val_loader, test_loader, loc_pos_weight, dmg_class_weights


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loc_criterion: BCEDiceLoss,
    dmg_criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    loc_bce_meter = AverageMeter()
    loc_dice_loss_meter = AverageMeter()
    loc_dice_meter = AverageMeter()
    dmg_ce_meter = AverageMeter()
    dmg_acc_meter = AverageMeter()
    conf = RunningConfusionMatrix(num_classes=4)

    iterator = tqdm(loader, desc="eval", leave=False) if tqdm is not None else loader
    for batch in iterator:
        img = batch["img"].to(device, non_blocking=True)
        loc = batch["loc"].to(device, non_blocking=True)
        dmg = batch["dmg"].to(device, non_blocking=True)
        cond_id = batch["cond_id"].to(device, non_blocking=True)

        logits = model(img, cond_id)
        logit_loc = logits[:, 0]
        logit_dmg = logits[:, 1:5]

        loc_bce, loc_dice_loss = loc_criterion(logit_loc, loc)
        dmg_ce = dmg_criterion(logit_dmg, dmg)
        loss = (
            args.loc_bce_weight * loc_bce
            + args.loc_dice_weight * loc_dice_loss
            + args.dmg_ce_weight * dmg_ce
        )

        loc_pred = (torch.sigmoid(logit_loc) > 0.5).float()
        loc_intersection = (loc_pred * loc).sum(dim=(1, 2))
        loc_union = loc_pred.sum(dim=(1, 2)) + loc.sum(dim=(1, 2))
        loc_dice = ((2.0 * loc_intersection + 1e-7) / (loc_union + 1e-7)).mean().item()

        dmg_pred = torch.argmax(logit_dmg, dim=1)
        valid = dmg != 255
        if valid.any():
            dmg_acc = (dmg_pred[valid] == dmg[valid]).float().mean().item()
            conf.update(dmg[valid], dmg_pred[valid])
        else:
            dmg_acc = 0.0

        bs = img.size(0)
        loss_meter.update(loss.item(), bs)
        loc_bce_meter.update(loc_bce.item(), bs)
        loc_dice_loss_meter.update(loc_dice_loss.item(), bs)
        loc_dice_meter.update(loc_dice, bs)
        dmg_ce_meter.update(dmg_ce.item(), bs)
        dmg_acc_meter.update(dmg_acc, bs)

    return {
        "loss": loss_meter.avg,
        "loc_bce": loc_bce_meter.avg,
        "loc_dice_loss": loc_dice_loss_meter.avg,
        "loc_dice": loc_dice_meter.avg,
        "dmg_ce": dmg_ce_meter.avg,
        "dmg_acc": dmg_acc_meter.avg,
        "dmg_macro_f1": conf.macro_f1(),
    }


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "args": vars(args),
    }
    torch.save(state, save_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, loc_pos_weight, dmg_class_weights = make_dataloaders(args)

    model = make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5, 11, 17, 23, 29, 33],
        gamma=0.5,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    loc_pos_weight = loc_pos_weight.to(device)
    dmg_class_weights = dmg_class_weights.to(device)
    loc_criterion = BCEDiceLoss(pos_weight=loc_pos_weight).to(device)
    dmg_criterion = nn.CrossEntropyLoss(weight=dmg_class_weights, ignore_index=255).to(device)

    start_epoch = 1
    best_score = -1.0
    history = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        loss_meter = AverageMeter()
        loc_bce_meter = AverageMeter()
        loc_dice_loss_meter = AverageMeter()
        dmg_ce_meter = AverageMeter()

        iterator = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}") if tqdm is not None else train_loader
        for batch in iterator:
            img = batch["img"].to(device, non_blocking=True)
            loc = batch["loc"].to(device, non_blocking=True)
            dmg = batch["dmg"].to(device, non_blocking=True)
            cond_id = batch["cond_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(img, cond_id)
                logit_loc = logits[:, 0]
                logit_dmg = logits[:, 1:5]

                loc_bce, loc_dice_loss = loc_criterion(logit_loc, loc)
                dmg_ce = dmg_criterion(logit_dmg, dmg)
                loss = (
                    args.loc_bce_weight * loc_bce
                    + args.loc_dice_weight * loc_dice_loss
                    + args.dmg_ce_weight * dmg_ce
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = img.size(0)
            loss_meter.update(loss.item(), bs)
            loc_bce_meter.update(loc_bce.item(), bs)
            loc_dice_loss_meter.update(loc_dice_loss.item(), bs)
            dmg_ce_meter.update(dmg_ce.item(), bs)

            if tqdm is not None:
                iterator.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    loc_bce=f"{loc_bce_meter.avg:.4f}",
                    loc_dice=f"{loc_dice_loss_meter.avg:.4f}",
                    dmg_ce=f"{dmg_ce_meter.avg:.4f}",
                )

        scheduler.step()

        val_metrics = evaluate(model, val_loader, loc_criterion, dmg_criterion, device, args)
        score = val_metrics["loc_dice"] + val_metrics["dmg_macro_f1"]

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_meter.avg,
            "train_loc_bce": loc_bce_meter.avg,
            "train_loc_dice_loss": loc_dice_loss_meter.avg,
            "train_dmg_ce": dmg_ce_meter.avg,
            "val_loss": val_metrics["loss"],
            "val_loc_dice": val_metrics["loc_dice"],
            "val_dmg_acc": val_metrics["dmg_acc"],
            "val_dmg_macro_f1": val_metrics["dmg_macro_f1"],
            "val_score": score,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_loc_dice={row['val_loc_dice']:.4f} | "
            f"val_dmg_acc={row['val_dmg_acc']:.4f} | val_dmg_macro_f1={row['val_dmg_macro_f1']:.4f}"
        )

        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_score,
            args,
        )

        if score > best_score:
            best_score = score
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                args,
            )
            print(f"Saved new best checkpoint with val_score={best_score:.4f}")

        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                args,
            )

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Evaluating best checkpoint on test split...")
    best_ckpt = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_metrics = evaluate(model, test_loader, loc_criterion, dmg_criterion, device, args)
    print(json.dumps({"best_val_score": best_score, "test": test_metrics}, indent=2))

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_score": best_score, "test": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()