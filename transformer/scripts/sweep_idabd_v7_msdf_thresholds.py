#!/usr/bin/env python3
"""
Threshold/dilation sweep for IDA-BD evaluation.

Works for:
  - direct transfer checkpoint
  - DANN checkpoint
  - ST-UDA checkpoint

This does NOT retrain. It only reloads the checkpoint and evaluates with:
  - different Phase-I localization thresholds
  - different damage post-processing dilation modes
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


def load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_v7_args(args):
    return SimpleNamespace(
        phase="test",
        resume_phase1=None,
        phase1_checkpoint=str(args.phase1_checkpoint),
        phase2_checkpoint=str(args.phase2_checkpoint),
        xbd_root=str(args.source_xbd_root),
        train_split=["train", "tier3"],
        val_split="hold",
        test_split="test",
        output_dir=str(args.output_dir),
        phase1_epochs=150,
        phase2_epochs=60,
        phase1_batch_size=1,
        phase2_batch_size=2,
        batch_size=2,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=4,
        num_workers=args.num_workers,
        img_size=args.img_size,
        phase2_crop_size=608,
        crop_candidate_count=8,
        lr=1e-4,
        weight_decay=1e-4,
        seed=args.seed,
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        max_grad_norm=1.0,
        warmup_epochs=5,
        save_every=1,
        early_stopping_patience=999,
        focal_gamma=2.0,
        loc_loss_weight=1.0,
        cls_loss_weight=1.0,
        aux_loc_weight=0.2,
        minor_damage_boost=1.5,
        major_damage_boost=1.5,
        max_damage_class_weight=10.0,
        crop_weight_no_damage=1.0,
        crop_weight_minor=12.0,
        crop_weight_major=12.0,
        crop_weight_destroyed=4.0,
        finetune_epochs=3,
        finetune_lr=5e-5,
        postprocess_dilation="minor",
        dilation_kernel=args.dilation_kernel,
        phase1_threshold=0.50,
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        amp=False,
        extra_photometric_aug=False,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--direct-script", type=Path, default=Path("transformer/scripts/eval_idabd_v7_msdf_direct_transfer.py"))
    p.add_argument("--v7-script", type=Path, default=Path("transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py"))

    p.add_argument("--phase1-checkpoint", type=Path, required=True)
    p.add_argument("--phase2-checkpoint", type=Path, required=True)

    p.add_argument("--idabd-root", type=Path, required=True)
    p.add_argument("--source-xbd-root", type=Path, default=Path("/homes/j244s673/documents/wsu/phd/xview2"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)

    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)

    p.add_argument("--thresholds", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    p.add_argument("--dilations", nargs="+", default=["none", "minor", "minor_major"])
    p.add_argument("--dilation-kernel", type=int, default=3)

    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = args.output_dir / "threshold_sweep"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if not args.direct_script.exists():
        raise FileNotFoundError(f"Missing direct helper script: {args.direct_script}")
    if not args.v7_script.exists():
        raise FileNotFoundError(f"Missing v7 script: {args.v7_script}")
    if not args.phase1_checkpoint.exists():
        raise FileNotFoundError(f"Missing Phase-I checkpoint: {args.phase1_checkpoint}")
    if not args.phase2_checkpoint.exists():
        raise FileNotFoundError(f"Missing Phase-II checkpoint: {args.phase2_checkpoint}")
    if not args.split_json.exists():
        raise FileNotFoundError(f"Missing split JSON: {args.split_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    direct = load_module(args.direct_script, "idabd_direct_sweep")
    v7 = load_module(args.v7_script, "hrtbda_v7_sweep")
    v7_args = make_v7_args(args)

    print("Loading IDA-BD split...", flush=True)
    samples = direct.discover_idabd_samples(args.idabd_root, require_mask=True)
    split = direct.get_or_create_split(samples, args.split_json, args.seed)

    print("===== IDA-BD SPLIT SUMMARY =====", flush=True)
    print(f"Train: {len(split['train'])}", flush=True)
    print(f"Val:   {len(split['val'])}", flush=True)
    print(f"Test:  {len(split['test'])}", flush=True)
    print("=================================", flush=True)

    ds = direct.IDABDDataset(samples, split[args.eval_split], img_size=args.img_size, require_mask=True)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Loading Phase-I model...", flush=True)
    phase1_model, stored_threshold, phase1_meta = v7.load_phase1_model_for_cascade(
        args=v7_args,
        device=device,
        phase1_ckpt=args.phase1_checkpoint,
    )
    print(f"Stored Phase-I threshold in checkpoint: {stored_threshold}", flush=True)
    print(f"Phase-I meta: {phase1_meta}", flush=True)

    print("Loading Phase-II model...", flush=True)
    phase2_model = v7.HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=4,
    ).to(device)
    ckpt = v7.load_model_weights(phase2_model, args.phase2_checkpoint, device)
    print(f"Loaded Phase-II checkpoint epoch: {ckpt.get('epoch', 'unknown')}", flush=True)

    results = []

    for th in args.thresholds:
        for dil in args.dilations:
            print()
            print("=" * 80, flush=True)
            print(f"Evaluating threshold={th:.2f}, dilation={dil}", flush=True)
            print("=" * 80, flush=True)

            metrics = direct.evaluate_idabd(
                v7=v7,
                phase1_model=phase1_model,
                phase2_model=phase2_model,
                loader=loader,
                device=device,
                phase1_threshold=float(th),
                dilation=dil,
                dilation_kernel=args.dilation_kernel,
            )

            row = {
                "threshold": float(th),
                "dilation": dil,
                "dilation_kernel": int(args.dilation_kernel),
                **metrics,
            }
            results.append(row)

            print(
                f"TH={th:.2f} DIL={dil:11s} "
                f"loc={metrics['loc_f1']:.6f} "
                f"no={metrics['no_damage_f1']:.6f} "
                f"minor={metrics['minor_damage_f1']:.6f} "
                f"major={metrics['major_damage_f1']:.6f} "
                f"destroyed={metrics['destroyed_f1']:.6f} "
                f"damage_h={metrics['damage_f1_hmean']:.6f} "
                f"damage_macro={metrics['damage_f1_macro']:.6f} "
                f"overall={metrics['overall_score']:.6f}",
                flush=True,
            )

    results_sorted = sorted(results, key=lambda x: x["overall_score"], reverse=True)

    json_path = scores_dir / f"idabd_{args.eval_split}_threshold_sweep_results.json"
    txt_path = scores_dir / f"summary_idabd_{args.eval_split}_threshold_sweep.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase1_checkpoint": str(args.phase1_checkpoint),
                "phase2_checkpoint": str(args.phase2_checkpoint),
                "idabd_root": str(args.idabd_root),
                "split_json": str(args.split_json),
                "eval_split": args.eval_split,
                "stored_phase1_threshold": float(stored_threshold),
                "phase1_meta": phase1_meta,
                "results_sorted": results_sorted,
            },
            f,
            indent=2,
        )

    lines = []
    lines.append("===== IDA-BD V7-MSDF THRESHOLD / DILATION SWEEP =====")
    lines.append(f"Phase-I checkpoint: {args.phase1_checkpoint}")
    lines.append(f"Phase-II checkpoint: {args.phase2_checkpoint}")
    lines.append(f"Eval split: {args.eval_split}")
    lines.append("")
    lines.append("Rank | Threshold | Dilation | Loc F1 | No F1 | Minor F1 | Major F1 | Destroyed F1 | Damage H | Damage Macro | Overall")

    for i, r in enumerate(results_sorted, start=1):
        lines.append(
            f"{i:4d} | {r['threshold']:.2f} | {r['dilation']:11s} | "
            f"{r['loc_f1']:.6f} | {r['no_damage_f1']:.6f} | "
            f"{r['minor_damage_f1']:.6f} | {r['major_damage_f1']:.6f} | "
            f"{r['destroyed_f1']:.6f} | {r['damage_f1_hmean']:.6f} | "
            f"{r['damage_f1_macro']:.6f} | {r['overall_score']:.6f}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines), flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)


if __name__ == "__main__":
    main()