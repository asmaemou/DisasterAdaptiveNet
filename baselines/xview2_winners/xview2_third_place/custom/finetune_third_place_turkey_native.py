#!/usr/bin/env python3
"""Fine-tune one released third-place xView2 checkpoint without Catalyst."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from xview.dataset import INPUT_IMAGE_KEY, INPUT_MASK_KEY, OUTPUT_MASK_KEY, get_datasets
from xview.models import get_model


IGNORE_INDEX = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--criterion", choices=["weighted_ce", "ohem_ce"], required=True)
    parser.add_argument("--extra-criterion", choices=["none", "focal"], default="none")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--augmentations", default="medium")
    parser.add_argument("--post-transform", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        losses = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = target != self.ignore_index
        positive = (target > 0) & valid
        negative = (target == 0) & valid
        selected = []

        for index in range(target.size(0)):
            positive_losses = losses[index][positive[index]]
            negative_losses = losses[index][negative[index]]
            positive_count = int(positive_losses.numel())
            negative_count = min(
                int(negative_losses.numel()),
                max(5, 2 * positive_count, int(negative_losses.numel()) // 4),
            )
            if negative_count:
                negative_losses = torch.topk(negative_losses, negative_count).values
            selected.append(torch.cat([positive_losses, negative_losses]))

        selected = [values for values in selected if values.numel()]
        if not selected:
            return logits.sum() * 0.0
        return torch.cat(selected).mean()


class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        losses = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        losses = losses[valid]
        return (((1.0 - torch.exp(-losses)) ** self.gamma) * losses).mean()


def build_losses(criterion_name, extra_criterion, device):
    weights = torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0], device=device)
    if criterion_name == "weighted_ce":
        losses = [nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)]
    else:
        losses = [OHEMCrossEntropyLoss(weights)]
    if extra_criterion == "focal":
        losses.append(MulticlassFocalLoss())
    return losses


def f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else float(2 * tp / denominator)


def validation_metrics(rows):
    localization = np.zeros(3, dtype=np.int64)
    damage = np.zeros((4, 3), dtype=np.int64)

    for prediction, target in rows:
        valid = target != IGNORE_INDEX
        prediction = prediction[valid]
        target = target[valid]

        pred_building = prediction > 0
        true_building = target > 0
        localization += (
            np.logical_and(pred_building, true_building).sum(),
            np.logical_and(pred_building, ~true_building).sum(),
            np.logical_and(~pred_building, true_building).sum(),
        )

        prediction = prediction[true_building]
        target = target[true_building]
        for class_index in range(1, 5):
            pred_class = prediction == class_index
            true_class = target == class_index
            damage[class_index - 1] += (
                np.logical_and(pred_class, true_class).sum(),
                np.logical_and(pred_class, ~true_class).sum(),
                np.logical_and(~pred_class, true_class).sum(),
            )

    localization_f1 = f1_from_counts(*localization)
    damage_f1s = [f1_from_counts(*counts) for counts in damage]
    damage_f1 = len(damage_f1s) / sum(1.0 / (score + 1e-6) for score in damage_f1s)
    weighted_f1 = 0.3 * localization_f1 + 0.7 * damage_f1
    return weighted_f1, localization_f1, damage_f1, damage_f1s


@torch.no_grad()
def validate(model, loader, losses, device, amp_enabled):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    rows = []

    for batch in loader:
        images = batch[INPUT_IMAGE_KEY].to(device, non_blocking=True)
        targets = batch[INPUT_MASK_KEY].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)[OUTPUT_MASK_KEY]
            loss = sum(loss_fn(logits, targets) for loss_fn in losses)
        total_loss += float(loss.detach())
        total_batches += 1
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        rows.extend(zip(predictions, targets))

    metrics = validation_metrics(rows)
    return total_loss / max(total_batches, 1), metrics


def cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_checkpoint(path, model, args, epoch, train_loss, valid_loss, metrics):
    weighted_f1, localization_f1, damage_f1, damage_f1s = metrics
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": cpu_state_dict(model),
        "epoch_metrics": {
            "train": {"loss": train_loss},
            "valid": {
                "loss": valid_loss,
                "weighted_f1": weighted_f1,
                "weighted_f1/localization_f1": localization_f1,
                "weighted_f1/damage_f1": damage_f1,
                "weighted_f1/no_damage": damage_f1s[0],
                "weighted_f1/minor_damage": damage_f1s[1],
                "weighted_f1/major_damage": damage_f1s[2],
                "weighted_f1/destroyed": damage_f1s[3],
            },
        },
        "checkpoint_data": {
            "cmd_args": {
                "model": args.model,
                "fold": args.fold,
                "source_checkpoint": str(args.source_checkpoint),
                "finetuned_on": "Earthquake Turkey official train/validation split",
            }
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for third-place fine-tuning")

    seed_everything(args.seed)
    device = torch.device("cuda")
    amp_enabled = args.amp
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Source checkpoint:", args.source_checkpoint)
    print("Output checkpoint:", args.output_checkpoint)
    print("Criterion:", args.criterion, "+", args.extra_criterion)

    train_dataset, valid_dataset, train_sampler = get_datasets(
        data_dir=args.data_dir,
        image_size=(args.size, args.size),
        augmentation=args.augmentations,
        fold=args.fold,
        enable_post_image_transform=args.post_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(valid_dataset))

    model = get_model(args.model, pretrained=False).to(device)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    load_result = model.load_state_dict(source["model_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print("Missing checkpoint keys:", load_result.missing_keys)
        print("Unexpected checkpoint keys:", load_result.unexpected_keys)
    print("Loaded released checkpoint at epoch:", source.get("epoch", "unknown"))
    del source

    losses = build_losses(args.criterion, args.extra_criterion, device)
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            images = batch[INPUT_IMAGE_KEY].to(device, non_blocking=True)
            targets = batch[INPUT_MASK_KEY].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)[OUTPUT_MASK_KEY]
                loss = sum(loss_fn(logits, targets) for loss_fn in losses)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            batches += 1

        scheduler.step()
        train_loss = running_loss / max(batches, 1)
        valid_loss, metrics = validate(model, valid_loader, losses, device, amp_enabled)
        score, localization_f1, damage_f1, damage_f1s = metrics
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"score={score:.6f} localization_f1={localization_f1:.6f} "
            f"damage_f1={damage_f1:.6f} classes={damage_f1s}"
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                args.output_checkpoint,
                model,
                args,
                epoch,
                train_loss,
                valid_loss,
                metrics,
            )
            print("Saved new best checkpoint:", args.output_checkpoint)

    print("Best validation weighted F1:", best_score)


if __name__ == "__main__":
    main()
