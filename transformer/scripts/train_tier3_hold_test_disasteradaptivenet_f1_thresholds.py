#!/usr/bin/env python3
import os
import sys
import csv
import json
import timeit
import argparse
import random
from typing import Dict, Any, List, Tuple

# Add project root to Python path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Compatibility patch for older scikit-learn with NumPy 2.x.
try:
    import numpy as np
    import numpy.core.numeric as np_numeric

    if not hasattr(np_numeric, "ComplexWarning"):
        from numpy.exceptions import ComplexWarning
        np_numeric.ComplexWarning = ComplexWarning
except Exception as exc:
    print(f"Warning: NumPy/sklearn compatibility patch failed: {exc}")

import torch
from torch import optim
from torch.utils import data as torch_data

try:
    import wandb
except Exception as exc:
    print(f"Warning: wandb import failed, using dummy wandb. Error: {exc}")

    class DummyWandb:
        def init(self, *args, **kwargs):
            print("wandb.init skipped")

        def log(self, *args, **kwargs):
            pass

    wandb = DummyWandb()

from utils import datasets, losses, measurers, metrics
from utils.models import DisasterAdaptiveNet


CLASS_NAMES = [
    "Localization F1",
    "No damage F1",
    "Minor damage F1",
    "Major damage F1",
    "Destroyed F1",
]


class Cfg(dict):
    """
    Small config object with cfg.A.B access.
    This avoids loading YAML config files.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, Cfg):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def dump(self):
        return json.dumps(self.to_dict(), indent=2)


def node(**kwargs):
    c = Cfg()
    for k, v in kwargs.items():
        if isinstance(v, dict):
            c[k] = node(**v)
        else:
            c[k] = v
    return c


def build_cfg(args) -> Cfg:
    cfg = node(
        NAME=args.experiment_name,
        OUTPUT_DIR=args.output_dir,
        SEED=args.seed,
        RANDOM_SEED=True,
        DEBUG=False,
        LOG_FREQ=args.log_freq,

        DATALOADER=dict(
            NUM_WORKER=args.num_workers,
            SHUFFLE=True,
        ),

        TRAINER=dict(
            LEARNING_RATE=args.lr,
            BATCH_SIZE=args.batch_size,
            EPOCHS=args.epochs,
            LOSS=dict(
                WEIGHTS=args.loss_weights,
                CLASS_WEIGHTS=args.class_weights,
            ),
        ),

        DATASET=dict(
            ROOT=args.data_root,
            ROOT_DIR=args.data_root,
            DATA_ROOT=args.data_root,
            DATA_DIR=args.data_root,
            PATH=args.data_root,
            XBD_ROOT=args.data_root,
            INCLUDE_CONDITIONING_INFORMATION=True,
        ),

        MODEL=dict(
            TYPE="disasteradaptivenet",
            NAME="DisasterAdaptiveNet",
            NUM_CLASSES=5,
            OUT_CHANNELS=5,
            IN_CHANNELS=6,
        ),

        PATHS=dict(
            DATA=args.data_root,
            DATASET=args.data_root,
            OUTPUT=args.output_dir,
        ),
    )

    return cfg


def create_model(cfg: Cfg) -> torch.nn.Module:
    """
    Explicitly uses DisasterAdaptiveNet.
    """

    try:
        return DisasterAdaptiveNet(cfg)
    except TypeError as exc:
        print("ERROR: Could not create DisasterAdaptiveNet(cfg).")
        print("Open utils/models.py and check class DisasterAdaptiveNet.__init__.")
        print("Original error:")
        print(exc)
        raise


def f1_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def compute_f1_metrics(
    net: torch.nn.Module,
    cfg: Cfg,
    device: torch.device,
    run_type: str,
    threshold: float,
) -> Dict[str, Any]:

    dataset = datasets.xBDDataset(cfg, run_type=run_type)

    loader = torch_data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    print(f"[{run_type}] dataset:")
    print(dataset)
    print(f"[{run_type}] dataset size: {len(dataset)}")
    print(f"[{run_type}] threshold: {threshold}")

    counts = {
        class_name: {"tp": 0, "fp": 0, "fn": 0}
        for class_name in CLASS_NAMES
    }

    net.eval()

    for batch_idx, batch in enumerate(loader):
        x = batch["img"].to(device)
        msk = batch["msk"].to(device)

        x_cond = batch["cond_id"].to(device)
        logits = net(x, x_cond)

        num_channels = min(logits.size(1), msk.size(1), len(CLASS_NAMES))

        if batch_idx == 0:
            print(f"[{run_type}] logits shape: {tuple(logits.shape)}")
            print(f"[{run_type}] mask shape: {tuple(msk.shape)}")
            print(f"[{run_type}] using {num_channels} channels for F1")

        for ch in range(num_channels):
            pred = torch.sigmoid(logits[:, ch]) > threshold
            true = msk[:, ch] > 0

            tp = torch.logical_and(pred, true).sum().item()
            fp = torch.logical_and(pred, torch.logical_not(true)).sum().item()
            fn = torch.logical_and(torch.logical_not(pred), true).sum().item()

            class_name = CLASS_NAMES[ch]
            counts[class_name]["tp"] += int(tp)
            counts[class_name]["fp"] += int(fp)
            counts[class_name]["fn"] += int(fn)

    results: Dict[str, Any] = {}
    f1_values = []

    for class_name in CLASS_NAMES:
        tp = counts[class_name]["tp"]
        fp = counts[class_name]["fp"]
        fn = counts[class_name]["fn"]

        scores = f1_from_counts(tp, fp, fn)

        results[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
        }

        f1_values.append(scores["f1"])

    results["Macro F1"] = float(np.mean(f1_values))

    return results


def print_f1_results(title: str, results: Dict[str, Any]) -> None:
    print("")
    print("=" * 90)
    print(title)
    print("=" * 90)

    for class_name in CLASS_NAMES:
        row = results[class_name]
        print(
            f"{class_name}: "
            f"F1={row['f1']:.6f}, "
            f"Precision={row['precision']:.6f}, "
            f"Recall={row['recall']:.6f}, "
            f"TP={row['tp']}, "
            f"FP={row['fp']}, "
            f"FN={row['fn']}"
        )

    print(f"Macro F1: {results['Macro F1']:.6f}")
    print("=" * 90)
    print("")


def save_f1_results(path_prefix: str, split_name: str, results: Dict[str, Any]) -> None:
    csv_path = f"{path_prefix}_{split_name}_f1.csv"
    txt_path = f"{path_prefix}_{split_name}_f1.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "f1", "precision", "recall", "tp", "fp", "fn"])

        for class_name in CLASS_NAMES:
            row = results[class_name]
            writer.writerow([
                class_name,
                row["f1"],
                row["precision"],
                row["recall"],
                row["tp"],
                row["fp"],
                row["fn"],
            ])

        writer.writerow(["Macro F1", results["Macro F1"], "", "", "", "", ""])

    with open(txt_path, "w") as f:
        f.write(f"{split_name.upper()} F1 RESULTS\n")
        f.write("=" * 90 + "\n")

        for class_name in CLASS_NAMES:
            row = results[class_name]
            f.write(
                f"{class_name}: "
                f"F1={row['f1']:.6f}, "
                f"Precision={row['precision']:.6f}, "
                f"Recall={row['recall']:.6f}, "
                f"TP={row['tp']}, "
                f"FP={row['fp']}, "
                f"FN={row['fn']}\n"
            )

        f.write(f"Macro F1: {results['Macro F1']:.6f}\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved TXT: {txt_path}")


def scan_thresholds_on_hold(
    net: torch.nn.Module,
    cfg: Cfg,
    device: torch.device,
    thresholds: List[float],
    epoch: int,
    scores_dir: str,
) -> Tuple[float, Dict[str, Any], float]:

    scan_path = os.path.join(scores_dir, f"epoch_{epoch:03d}_threshold_scan_hold.csv")

    best_threshold = None
    best_results = None
    best_macro_f1 = -1.0

    with open(scan_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold",
            "macro_f1",
            "localization_f1",
            "no_damage_f1",
            "minor_damage_f1",
            "major_damage_f1",
            "destroyed_f1",
        ])

        for threshold in thresholds:
            print("")
            print(f"===== HOLD THRESHOLD SCAN | epoch={epoch} | threshold={threshold} =====")

            results = compute_f1_metrics(
                net=net,
                cfg=cfg,
                device=device,
                run_type="val",
                threshold=threshold,
            )

            macro_f1 = results["Macro F1"]

            writer.writerow([
                threshold,
                macro_f1,
                results["Localization F1"]["f1"],
                results["No damage F1"]["f1"],
                results["Minor damage F1"]["f1"],
                results["Major damage F1"]["f1"],
                results["Destroyed F1"]["f1"],
            ])

            print_f1_results(
                f"HOLD RESULTS | epoch={epoch} | threshold={threshold}",
                results,
            )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_threshold = threshold
                best_results = results

    print(f"Saved threshold scan: {scan_path}")
    print(f"Best threshold on HOLD for epoch {epoch}: {best_threshold}")
    print(f"Best HOLD Macro F1 for epoch {epoch}: {best_macro_f1:.6f}")

    return best_threshold, best_results, best_macro_f1


def run_training(cfg: Cfg, device: torch.device, thresholds: List[float]) -> None:
    run_dir = cfg.OUTPUT_DIR
    scores_dir = os.path.join(run_dir, "scores")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    print("===== CONFIG USED, NO YAML FILE =====")
    print(cfg.dump())
    print("=====================================")

    print("===== SPLIT PLAN =====")
    print("Training:   run_type='train' -> /homes/j244s673/documents/wsu/phd/xview2/tier3")
    print("Validation: run_type='val'   -> /homes/j244s673/documents/wsu/phd/xview2/hold")
    print("Testing:    run_type='test'  -> /homes/j244s673/documents/wsu/phd/xview2/test")
    print("======================")

    print("===== THRESHOLDS =====")
    print(thresholds)
    print("======================")

    net = create_model(cfg)
    net.to(device)

    print("===== MODEL CLASS =====")
    print(type(net))
    print("===== MODEL ARCHITECTURE =====")
    print(net)
    print("=======================")

    optimizer = optim.AdamW(
        net.parameters(),
        lr=cfg.TRAINER.LEARNING_RATE,
        weight_decay=0.01,
    )

    criterion = losses.ComboLoss(weights=cfg.TRAINER.LOSS.WEIGHTS)

    m_total = measurers.AverageMeter()
    m_loc = measurers.AverageMeter()
    m_dmg = measurers.AverageMeter()
    m_dice = measurers.AverageMeter()
    measurers_list = [m_total, m_loc, m_dmg, m_dice]

    train_dataset = datasets.xBDDataset(cfg, run_type="train")

    print("[train] dataset:")
    print(train_dataset)
    print(f"[train] dataset size: {len(train_dataset)}")

    try:
        class_weights = losses.loss_class_weights(
            cfg.TRAINER.LOSS.CLASS_WEIGHTS,
            train_dataset.get_class_counts(),
        )
        print("Using class weights from losses.loss_class_weights:")
        print(class_weights)
    except Exception as exc:
        print(f"Warning: could not compute class weights. Using 1.0 for every output channel. Error: {exc}")
        class_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAINER.BATCH_SIZE,
        num_workers=0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        shuffle=cfg.DATALOADER.SHUFFLE,
        drop_last=True,
        pin_memory=True,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training loader has 0 batches. Check tier3 path and dataset structure.")

    global_step = 0
    best_val_macro_f1 = -1.0
    best_checkpoint_path = os.path.join(run_dir, "best_disasteradaptivenet_by_hold_macro_f1.pth")

    for epoch in range(1, cfg.TRAINER.EPOCHS + 1):
        print("")
        print(f"Starting epoch {epoch}/{cfg.TRAINER.EPOCHS}")
        start = timeit.default_timer()

        net.train()

        for batch in train_loader:
            optimizer.zero_grad()

            x = batch["img"].to(device)
            msk = batch["msk"].to(device)

            x_cond = batch["cond_id"].to(device)
            logits = net(x, x_cond)

            loss_loc = criterion(logits[:, 0], msk[:, 0].long()) * class_weights[0]

            loss_dmg = torch.tensor([0.0], device=device)
            for c in range(1, logits.size(1)):
                weight = class_weights[c] if c < len(class_weights) else 1.0
                loss_dmg = loss_dmg + criterion(logits[:, c], msk[:, c].long()) * weight

            loss = loss_loc + loss_dmg

            with torch.no_grad():
                y_hat = torch.sigmoid(logits[:, 0])
                dice_sc = 1 - metrics.dice_round(y_hat, msk[:, 0])

            m_loc.update(loss_loc.item(), x.size(0))
            m_dmg.update(loss_dmg.item(), x.size(0))
            m_total.update(loss.item(), x.size(0))
            m_dice.update(dice_sc, x.size(0))

            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % cfg.LOG_FREQ == 0:
                elapsed = timeit.default_timer() - start

                print(
                    f"step={global_step} "
                    f"epoch={epoch} "
                    f"train_loss={m_total.avg:.6f} "
                    f"train_loc_loss={m_loc.avg:.6f} "
                    f"train_dmg_loss={m_dmg.avg:.6f} "
                    f"train_dice={m_dice.avg:.6f}"
                )

                wandb.log({
                    "train_loss_loc": m_loc.avg,
                    "train_loss_dmg": m_dmg.avg,
                    "train_loss": m_total.avg,
                    "train_dice": m_dice.avg,
                    "time": elapsed,
                    "step": global_step,
                    "epoch": epoch,
                })

                start = timeit.default_timer()

                for measurer in measurers_list:
                    measurer.reset()

        print(f"Finished epoch {epoch}/{cfg.TRAINER.EPOCHS}")
        print("Scanning thresholds on HOLD validation split.")

        best_epoch_threshold, val_results, val_macro_f1 = scan_thresholds_on_hold(
            net=net,
            cfg=cfg,
            device=device,
            thresholds=thresholds,
            epoch=epoch,
            scores_dir=scores_dir,
        )

        print_f1_results(
            f"BEST HOLD VALIDATION RESULTS | epoch={epoch} | threshold={best_epoch_threshold}",
            val_results,
        )

        save_f1_results(
            path_prefix=os.path.join(scores_dir, f"epoch_{epoch:03d}_best_threshold_{best_epoch_threshold}"),
            split_name="hold_val",
            results=val_results,
        )

        wandb.log({
            "val_hold_best_threshold": best_epoch_threshold,
            "val_hold_macro_f1": val_macro_f1,
            "val_hold_localization_f1": val_results["Localization F1"]["f1"],
            "val_hold_no_damage_f1": val_results["No damage F1"]["f1"],
            "val_hold_minor_damage_f1": val_results["Minor damage F1"]["f1"],
            "val_hold_major_damage_f1": val_results["Major damage F1"]["f1"],
            "val_hold_destroyed_f1": val_results["Destroyed F1"]["f1"],
            "step": global_step,
            "epoch": epoch,
        })

        if val_macro_f1 > best_val_macro_f1:
            print(f"New best HOLD Macro F1: {val_macro_f1:.6f}")
            print(f"New best threshold: {best_epoch_threshold}")

            best_val_macro_f1 = val_macro_f1

            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "best_threshold": best_epoch_threshold,
                "val_results": val_results,
                "experiment_name": cfg.NAME,
                "model_class": str(type(net)),
                "cfg": cfg.to_dict(),
                "thresholds_tried": thresholds,
            }, best_checkpoint_path)

            print(f"Saved best checkpoint: {best_checkpoint_path}")

    print("")
    print("Training complete.")
    print(f"Best HOLD Macro F1: {best_val_macro_f1:.6f}")
    print(f"Loading best checkpoint: {best_checkpoint_path}")

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])

    best_threshold = float(checkpoint["best_threshold"])
    best_epoch = checkpoint["epoch"]

    print(f"Best checkpoint epoch: {best_epoch}")
    print(f"Best threshold selected on HOLD: {best_threshold}")
    print("Running final test on XVIEW2 TEST split using selected threshold.")

    test_results = compute_f1_metrics(
        net=net,
        cfg=cfg,
        device=device,
        run_type="test",
        threshold=best_threshold,
    )

    print_f1_results(
        f"FINAL TEST RESULTS ON XVIEW2 TEST | threshold={best_threshold}",
        test_results,
    )

    save_f1_results(
        path_prefix=os.path.join(scores_dir, "final"),
        split_name="xview2_test",
        results=test_results,
    )

    summary_path = os.path.join(scores_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"experiment_name: {cfg.NAME}\n")
        f.write("config_file_used: NONE\n")
        f.write("model_import: from utils.models import DisasterAdaptiveNet\n")
        f.write("model_created_by: DisasterAdaptiveNet(cfg)\n")
        f.write(f"model_type: {cfg.MODEL.TYPE}\n")
        f.write(f"model_class: {str(type(net))}\n")
        f.write("conditioning_information: True\n")
        f.write("train_split: /homes/j244s673/documents/wsu/phd/xview2/tier3\n")
        f.write("val_split: /homes/j244s673/documents/wsu/phd/xview2/hold\n")
        f.write("test_split: /homes/j244s673/documents/wsu/phd/xview2/test\n")
        f.write(f"epochs: {cfg.TRAINER.EPOCHS}\n")
        f.write(f"batch_size: {cfg.TRAINER.BATCH_SIZE}\n")
        f.write(f"learning_rate: {cfg.TRAINER.LEARNING_RATE}\n")
        f.write(f"thresholds_tried: {thresholds}\n")
        f.write(f"best_epoch_selected_on_hold: {best_epoch}\n")
        f.write(f"best_threshold_selected_on_hold: {best_threshold}\n")
        f.write(f"best_hold_macro_f1: {best_val_macro_f1:.6f}\n")
        f.write(f"final_test_macro_f1: {test_results['Macro F1']:.6f}\n")
        f.write(f"final_test_localization_f1: {test_results['Localization F1']['f1']:.6f}\n")
        f.write(f"final_test_no_damage_f1: {test_results['No damage F1']['f1']:.6f}\n")
        f.write(f"final_test_minor_damage_f1: {test_results['Minor damage F1']['f1']:.6f}\n")
        f.write(f"final_test_major_damage_f1: {test_results['Major damage F1']['f1']:.6f}\n")
        f.write(f"final_test_destroyed_f1: {test_results['Destroyed F1']['f1']:.6f}\n")

    print(f"Saved summary: {summary_path}")
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--log-freq", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        required=True,
        help="Threshold values to scan on HOLD validation. Best threshold is used once on TEST.",
    )

    parser.add_argument(
        "--loss-weights",
        type=float,
        nargs="+",
        required=True,
        help="Weights passed to losses.ComboLoss.",
    )

    parser.add_argument(
        "--class-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional class weights. If omitted, script tries project default; otherwise falls back to 1.0.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = build_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== EXPERIMENT =====")
    print(cfg.NAME)
    print("======================")
    print("=== Running on device:", device)

    wandb.init(
        name=cfg.NAME,
        config=cfg.to_dict(),
        project="disasteradaptivenet",
        tags=[
            "DisasterAdaptiveNet",
            "xBD",
            "tier3_train",
            "hold_val",
            "test_xview2",
            "classwise_f1",
            "threshold_scan",
            "no_yaml_config",
        ],
        mode="disabled",
    )

    try:
        run_training(cfg, device, thresholds=args.thresholds)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)