#!/usr/bin/env python3
"""
Gated ensemble for IDA-BD:
  base prediction = best IDA-BD v2 cascaded model
  optional overwrite = DANN/v5 model changes base no-damage -> minor only when confident

This is intentionally conservative.  It does not retrain anything.  It selects
minor-confidence threshold on IDA-BD validation and then evaluates once on the
real 8-image IDA-BD test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_xbd_hrtbda_v2_cascaded_phase1mask as v2
import train_xbd_hrtbda_v5_multilabel_crop_cascade as v5
import train_idabd_xbdv5_supervised_finetune as idft


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def harmonic(vals: List[float]) -> float:
    vals = [max(float(v), 1e-6) for v in vals]
    return len(vals) / sum(1.0 / v for v in vals)


def load_v2_models(args, device):
    p1 = v2.HRTBDAPhase1(args.base_channels, args.decoder_channels, args.window_size).to(device)
    ck1 = v2.load_model_weights(p1, Path(args.base_phase1_checkpoint), device)
    p1.eval()
    p2 = v2.HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size, num_classes=4).to(device)
    ck2 = v2.load_model_weights(p2, Path(args.base_phase2_checkpoint), device)
    p2.eval()
    return p1, p2, ck1, ck2


def load_v5_models(args, device):
    p1 = v5.HRTBDAPhase1(args.base_channels, args.decoder_channels, args.window_size).to(device)
    ck1 = v5.load_model_weights(p1, Path(args.dann_phase1_checkpoint), device)
    p1.eval()
    p2 = v5.HRTBDAPhase2(args.base_channels, args.decoder_channels, args.window_size, num_classes=4).to(device)
    ck2 = v5.load_model_weights(p2, Path(args.dann_phase2_checkpoint), device)
    p2.eval()
    return p1, p2, ck1, ck2


@torch.no_grad()
def evaluate_gated(args, loader, device, models, base_threshold: float, dann_threshold: float, minor_conf: float) -> Dict[str, float]:
    base_p1, base_p2, _, _ = models["base"]
    dann_p1, dann_p2, _, _ = models["dann"]

    loc_tp = loc_fp = loc_fn = 0
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in [1, 2, 3, 4]}
    overwrite_minor_pixels = 0

    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        loc_true = batch["loc"].to(device, non_blocking=True).long()
        target = batch["target5"].to(device, non_blocking=True).long()

        # Base v2 cascade prediction.
        base_loc = (torch.sigmoid(base_p1(pre)) > base_threshold).long()
        base_logits4 = base_p2(pre, post)
        base_damage = torch.argmax(base_logits4, dim=1).long() + 1
        final_pred = torch.zeros_like(base_damage)
        final_pred[base_loc.bool()] = base_damage[base_loc.bool()]

        # DANN/v5 confidence. Use DANN damage head only as a conservative minor-recall booster.
        dann_out = dann_p2(pre, post)
        dann_logits = v5.get_damage_logits(dann_out)
        dann_prob = torch.sigmoid(dann_logits)
        dann_damage = torch.argmax(dann_prob, dim=1).long() + 1
        minor_prob = dann_prob[:, 1]

        if args.use_dann_loc_mask:
            dann_loc = (torch.sigmoid(dann_p1(pre)) > dann_threshold).long()
            gate_loc = (base_loc.bool() | dann_loc.bool())
        else:
            gate_loc = base_loc.bool()

        overwrite_minor = (
            gate_loc
            & (final_pred == 1)
            & (dann_damage == 2)
            & (minor_prob > minor_conf)
        )
        overwrite_minor_pixels += int(overwrite_minor.sum().item())
        final_pred[overwrite_minor] = 2

        loc_pred = base_loc
        loc_tp += int(((loc_pred == 1) & (loc_true == 1)).sum().item())
        loc_fp += int(((loc_pred == 1) & (loc_true == 0)).sum().item())
        loc_fn += int(((loc_pred == 0) & (loc_true == 1)).sum().item())

        valid = (target >= 1) & (target <= 4)
        pred_valid = final_pred[valid]
        true_valid = target[valid]
        for c in [1, 2, 3, 4]:
            counts[c]["tp"] += int(((pred_valid == c) & (true_valid == c)).sum().item())
            counts[c]["fp"] += int(((pred_valid == c) & (true_valid != c)).sum().item())
            counts[c]["fn"] += int(((pred_valid != c) & (true_valid == c)).sum().item())

    loc = f1(loc_tp, loc_fp, loc_fn)
    no = f1(**counts[1])
    minor = f1(**counts[2])
    major = f1(**counts[3])
    destroyed = f1(**counts[4])
    dmg = harmonic([no, minor, major, destroyed])
    overall = 0.3 * loc + 0.7 * dmg
    return {
        "base_threshold": float(base_threshold),
        "dann_threshold": float(dann_threshold),
        "minor_conf": float(minor_conf),
        "localization_f1": loc,
        "no_damage_f1": no,
        "minor_damage_f1": minor,
        "major_damage_f1": major,
        "destroyed_f1": destroyed,
        "damage_f1": dmg,
        "overall_score": overall,
        "overwrite_minor_pixels": overwrite_minor_pixels,
    }


def make_eval_loaders(args):
    # Fill attributes required by idft.make_loaders.
    args.phase1_batch_size = 1
    args.phase2_batch_size = 1
    args.eval_batch_size = args.eval_batch_size
    args.phase2_crop_size = 0
    args.crop_candidate_count = 1
    args.crop_weight_no_damage = 1.0
    args.crop_weight_minor = 1.0
    args.crop_weight_major = 1.0
    args.crop_weight_destroyed = 1.0
    args.extra_photometric_aug = False
    _train, val_loader, test_loader, _train_ds = idft.make_loaders(args, phase2_training=False)
    return val_loader, test_loader


def main():
    p = argparse.ArgumentParser("IDA-BD v2 + DANN gated ensemble evaluation")
    p.add_argument("--idabd-root", required=True)
    p.add_argument("--split-file", default="")
    p.add_argument("--force-resplit", action="store_true")
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--base-phase1-checkpoint", required=True)
    p.add_argument("--base-phase2-checkpoint", required=True)
    p.add_argument("--dann-phase1-checkpoint", required=True)
    p.add_argument("--dann-phase2-checkpoint", required=True)

    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-dann-loc-mask", action="store_true")

    p.add_argument("--base-thresholds", type=float, nargs="+", default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    p.add_argument("--dann-thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.50])
    p.add_argument("--minor-conf-thresholds", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    v2.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    val_loader, test_loader = make_eval_loaders(args)
    models = {
        "base": load_v2_models(args, device),
        "dann": load_v5_models(args, device),
    }

    rows = []
    for bt in args.base_thresholds:
        for dt in args.dann_thresholds:
            for mc in args.minor_conf_thresholds:
                res = evaluate_gated(args, val_loader, device, models, bt, dt, mc)
                rows.append(res)
                print(
                    f"VAL gated | base_th={bt:.2f} dann_th={dt:.2f} minor_conf={mc:.2f} "
                    f"| loc={res['localization_f1']:.6f} no={res['no_damage_f1']:.6f} "
                    f"minor={res['minor_damage_f1']:.6f} major={res['major_damage_f1']:.6f} "
                    f"destroyed={res['destroyed_f1']:.6f} damage={res['damage_f1']:.6f} "
                    f"overall={res['overall_score']:.6f} overwrites={res['overwrite_minor_pixels']}",
                    flush=True,
                )

    rows_sorted = sorted(rows, key=lambda r: r["overall_score"], reverse=True)
    best = rows_sorted[0]

    csv_path = scores_dir / "validation_gated_ensemble_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    json_path = scores_dir / "validation_gated_ensemble_ablation.json"
    json_path.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")

    print("\n===== BEST VALIDATION GATED SETTING =====")
    print(json.dumps(best, indent=2))
    print("=========================================")

    test_res = evaluate_gated(
        args,
        test_loader,
        device,
        models,
        base_threshold=float(best["base_threshold"]),
        dann_threshold=float(best["dann_threshold"]),
        minor_conf=float(best["minor_conf"]),
    )

    summary = {
        "experiment": "IDA-BD v2 cascade plus DANN minor-confidence gated ensemble",
        "base_phase1_checkpoint": args.base_phase1_checkpoint,
        "base_phase2_checkpoint": args.base_phase2_checkpoint,
        "dann_phase1_checkpoint": args.dann_phase1_checkpoint,
        "dann_phase2_checkpoint": args.dann_phase2_checkpoint,
        "validation_selected_setting": best,
        "test": test_res,
    }
    summary_json = scores_dir / "summary_gated_ensemble_selected_test.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Experiment: IDA-BD v2 cascade + DANN minor-confidence gated ensemble",
        f"Selected base Phase-I threshold: {best['base_threshold']:.2f}",
        f"Selected DANN Phase-I threshold: {best['dann_threshold']:.2f}",
        f"Selected minor confidence threshold: {best['minor_conf']:.2f}",
        f"Validation overall: {best['overall_score']:.6f}",
        "",
        "Final test result:",
        f"Test Localization F1 from base Phase-I mask: {test_res['localization_f1']:.6f}",
        f"No Damage F1:    {test_res['no_damage_f1']:.6f}",
        f"Minor Damage F1: {test_res['minor_damage_f1']:.6f}",
        f"Major Damage F1: {test_res['major_damage_f1']:.6f}",
        f"Destroyed F1:    {test_res['destroyed_f1']:.6f}",
        f"Damage F1:       {test_res['damage_f1']:.6f}",
        f"Overall Score:   {test_res['overall_score']:.6f}",
        f"Minor overwrite pixels: {test_res['overwrite_minor_pixels']}",
    ]
    summary_txt = scores_dir / "summary_gated_ensemble_selected_test.txt"
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote: {summary_txt}", flush=True)


if __name__ == "__main__":
    main()
