#!/usr/bin/env python3
"""Source-selected ordinal/categorical fusion with a frozen RS-BDNet checkpoint.

TRAINED WEIGHTS NEVER CHANGE.  The only checkpoint load comes from a verified,
independent copy inside the NEW output directory.  All candidate inference
rules are compared on xBD HOLD only, before touching any EBD test set.

Source candidates (predeclared, no EBD-based optimization):
  categorical : argmax(z_damage[4]) (original inference)
  ordinal     : convert cumulative probabilities to a four-class distribution
  fusion50    : equally weighted categorical and ordinal probabilities

An xBD-Hold winner, selected by arithmetic damage Macro-F1 (ties favor
categorical, then fusion50, then ordinal), is frozen for five EBD test sets.

The original checkpoint directory and original zero-shot results directory are
READ ONLY.  The new output directory must be fresh or contain only an already
verified independent checkpoint copy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from transformer.scripts import evaluate_rsbdnet_xbd_zeroshot_ebd as base
from transformer.scripts import train_xbd_supervised_disasteradaptivenet as runner
from transformer.scripts import train_xbd_bitemporal_building_crossattention_ordinal as architecture

METHODS = ("categorical", "fusion50", "ordinal")  # stable source-only tie-break
ORIGINAL_SHA256 = "c781c1d9539618e2d5c67bc6365daa24557e9c01e64f87fc7d92282fe90aea77"
EXPECTED_XBD_MACRO = 0.7756
CLASS_NAMES = base.CLASS_NAMES
PREDICTION_NAMES = base.PREDICTION_NAMES
EBD_DATASETS = base.EBD_DATASETS
EXPECTED_ORIGINAL = PROJECT_ROOT / "output/BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3/checkpoints/best.pt"
EXPECTED_OLD_OUTPUT = PROJECT_ROOT / "output/RSBDNet_xBD_ZeroShot_EBD"
EXPECTED_NEW_OUTPUT = PROJECT_ROOT / "output/RSBDNet_xBD_ZeroShot_OrdinalFusion"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    old_checkpoint = args.source_checkpoint.resolve(strict=True)
    original_dir = EXPECTED_ORIGINAL.parent.parent.resolve(strict=True)
    old_output = EXPECTED_OLD_OUTPUT.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    allowed = EXPECTED_NEW_OUTPUT.resolve(strict=False)
    if old_checkpoint != EXPECTED_ORIGINAL.resolve(strict=True):
        raise RuntimeError("Source checkpoint must be the original known epoch-35 best.pt")
    if output != allowed:
        raise RuntimeError(f"For safety, --output-dir must be exactly {allowed}; received {output}")
    if args.output_dir.is_symlink() or (args.output_dir / "input_checkpoint").is_symlink():
        raise RuntimeError("Symlink output directory/checkpoint directory not permitted")
    for forbidden in (original_dir, old_output):
        if output == forbidden or forbidden in output.parents or output in forbidden.parents:
            raise RuntimeError("Output path overlaps a protected historical results directory")
    if not args.xbd_root.is_dir() or not (args.xbd_root / "hold").is_dir():
        raise FileNotFoundError("xBD Hold directory is required")
    if not args.ebd_base.is_dir():
        raise FileNotFoundError("EBD data base is missing")
    if base.sha256(old_checkpoint) != ORIGINAL_SHA256:
        raise RuntimeError("Original checkpoint SHA256 differs from verified published checkpoint")
    return old_checkpoint, output


def copy_input_checkpoint(source: Path, out_dir: Path) -> Path:
    """Copy via exclusive-create only. Never overwrite or alter original source."""
    copy_dir = out_dir / "input_checkpoint"
    copy_path = copy_dir / "best_xbd_epoch35.pt"
    out_dir.mkdir(parents=False, exist_ok=True)
    allowed_existing = {copy_path}
    existing = [p for p in out_dir.rglob("*") if p.is_file() or p.is_symlink()]
    if any(p != copy_path for p in existing):
        raise FileExistsError(f"New output directory is not empty: {out_dir}. Refusing overwrite.")
    if copy_path.is_symlink() or copy_dir.is_symlink():
        raise RuntimeError("Refusing symlink checkpoint copy")
    copy_dir.mkdir(parents=True, exist_ok=True)
    if not copy_path.exists():
        fd = os.open(str(copy_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            with os.fdopen(fd, "wb") as dst, source.open("rb") as src:
                shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
        except BaseException:
            copy_path.unlink(missing_ok=True)  # only our NEW copy; never source
            raise
    if base.sha256(copy_path) != ORIGINAL_SHA256:
        raise RuntimeError("Independent copy SHA256 mismatch; aborting without changing any source files")
    copy_path.chmod(0o444)  # only copied file; source mode remains unchanged
    print(f"[PROTECTED] Original checkpoint read-only: {source}", flush=True)
    print(f"[VERIFIED] Copied checkpoint, SHA256={ORIGINAL_SHA256}: {copy_path}", flush=True)
    return copy_path


def check_original_baseline(old_output: Path) -> dict:
    """Read original experiment results; DO NOT use its EBD scores for selection."""
    manifest_path = old_output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reproduction = manifest.get("xbd_test_reproduction", {})
    if manifest.get("checkpoint_sha256") != ORIGINAL_SHA256:
        raise RuntimeError("Original baseline is based on a different checkpoint")
    if reproduction.get("status") != "PASS":
        raise RuntimeError("Original baseline does not report an xBD reproduction PASS")
    if abs(float(reproduction.get("damage_macro_f1", 0)) - EXPECTED_XBD_MACRO) > 0.00015:
        raise RuntimeError("Original baseline xBD reproduction does not match published Macro-F1")
    return {
        "baseline_manifest_read_only": str(manifest_path),
        "original_evaluation_completed_utc": manifest.get("completed_utc"),
        "original_xbd_test_reproduction": reproduction,
    }


def class_probabilities(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert 8 logits to probability maps; ensure valid ordinal distributions.

    The ordinal training targets are (damage>0, damage>1, damage>2), with
    damage labels 0..3.  Pointwise cumulative minimum enforces q0>=q1>=q2.
    """
    if logits.ndim != 4 or logits.shape[1] != 8:
        raise RuntimeError(f"Expected logits [B,8,H,W]; got {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Non-finite network logits")
    cat = torch.softmax(logits[:, 1:5].float(), dim=1)
    q = torch.sigmoid(logits[:, 5:8].float())
    q = torch.cummin(q, dim=1).values  # q0 >= q1 >= q2
    ordinal = torch.stack((1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1] - q[:, 2], q[:, 2]), dim=1)
    # Subtraction is theoretically nonnegative; remove only rounding noise.
    ordinal = ordinal.clamp_min(0.0)
    ordinal = ordinal / ordinal.sum(dim=1, keepdim=True).clamp_min(1e-12)
    fused = 0.5 * cat + 0.5 * ordinal
    return {"categorical": cat, "fusion50": fused, "ordinal": ordinal}


def confusion_for_batch(pred_damage: torch.Tensor, gt_loc: torch.Tensor,
                        gt_damage: torch.Tensor) -> np.ndarray:
    valid = (gt_loc == 1) & (gt_damage != 255)
    if not bool(valid.any()):
        return np.zeros((4, 5), dtype=np.int64)
    actual, predicted = gt_damage[valid], pred_damage[valid]
    if not bool(((actual >= 0) & (actual <= 3)).all()):
        raise RuntimeError("Unexpected EBD/xBD damage labels; expected 0..3 or 255")
    if not bool(((predicted >= 0) & (predicted <= 4)).all()):
        raise RuntimeError("Unexpected prediction IDs")
    return torch.bincount(actual * 5 + predicted, minlength=20).reshape(4, 5).cpu().numpy().astype(np.int64)


def localization_counts(pred: torch.Tensor, actual: torch.Tensor) -> np.ndarray:
    return np.array([
        ((pred == 1) & (actual == 1)).sum().item(),
        ((pred == 1) & (actual == 0)).sum().item(),
        ((pred == 0) & (actual == 1)).sum().item(),
        ((pred == 0) & (actual == 0)).sum().item(),
    ], dtype=np.int64)


@torch.inference_mode()
def evaluate_methods(model: torch.nn.Module, dataset, threshold: float, device: torch.device,
                     batch_size: int, num_workers: int, methods: tuple[str, ...],
                     display_name: str, write_per_image: bool = False) -> tuple[dict[str, dict], dict[str, list]]:
    loader = runner.make_eval_loader(dataset, batch_size, num_workers)
    model.eval()
    counts = {name: np.zeros((4, 5), dtype=np.int64) for name in methods}
    loc_totals = np.zeros(4, dtype=np.int64)
    individual_rows: dict[str, list] = {name: [] for name in methods}
    n_seen = 0
    for batch_index, batch in enumerate(loader):
        image = batch["img"].to(device, non_blocking=True)
        gt_loc = batch["loc"].to(device, non_blocking=True).long()
        gt_dmg = batch["dmg"].to(device, non_blocking=True).long()
        condition = batch["cond_id"].to(device, non_blocking=True)
        logits = model(image, condition)
        if logits.shape[-2:] != image.shape[-2:]:
            raise RuntimeError("Unexpected inference image/logit size mismatch")
        prob_map = class_probabilities(logits)
        pred_loc = (torch.sigmoid(logits[:, 0]) > threshold).long()
        loc_totals += localization_counts(pred_loc, gt_loc)
        for method in methods:
            pred_damage = (prob_map[method].argmax(dim=1) + 1) * pred_loc
            counts[method] += confusion_for_batch(pred_damage, gt_loc, gt_dmg)
            if write_per_image:
                for j, stem in enumerate(batch["stem"]):
                    l = localization_counts(pred_loc[j], gt_loc[j])
                    c = confusion_for_batch(pred_damage[j], gt_loc[j], gt_dmg[j])
                    metrics = base.metrics_from_counts(l, c, 1)
                    individual_rows[method].append([
                        str(stem), float(metrics["localization"]["f1"]),
                        float(metrics["damage_macro"]["f1"]),
                        int(c.sum()), int(c[:, 0].sum()),
                    ])
        n_seen += image.shape[0]
        if batch_index % 20 == 0 or n_seen == len(dataset):
            print(f"  [{display_name}] {n_seen}/{len(dataset)} pairs", flush=True)
    if n_seen != len(dataset):
        raise RuntimeError(f"Incomplete evaluation: {n_seen} != {len(dataset)}")
    return ({name: base.metrics_from_counts(loc_totals, counts[name], n_seen)
             for name in methods}, individual_rows)


def write_dataset_metrics(folder: Path, metrics: dict, per_image_rows: list) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    base.write_json(folder / "metrics.json", metrics)
    base.write_csv(folder / "damage_confusion_matrix.csv",
                   ["ground_truth\\prediction", *PREDICTION_NAMES],
                   [[name, *metrics["damage_confusion_matrix"]["counts"][i]]
                    for i, name in enumerate(CLASS_NAMES)])
    base.write_csv(folder / "class_metrics.csv",
                   ["category", "precision", "recall", "f1", "tp", "fp", "fn", "support_pixels"],
                   [["localization", *[metrics["localization"][k]
                                      for k in ("precision", "recall", "f1", "tp", "fp", "fn")],
                     metrics["localization"]["tp"] + metrics["localization"]["fn"]]] +
                   [[name, *[metrics["damage_classes"][name][k] for k in
                             ("precision", "recall", "f1", "tp", "fp", "fn", "support_valid_gt_pixels")]]
                    for name in CLASS_NAMES] +
                   [["damage_macro", *[metrics["damage_macro"][k] for k in ("precision", "recall", "f1")],
                     "", "", "", metrics["valid_ground_truth_damage_pixels"]]])
    base.write_csv(folder / "per_image_metrics.csv",
                   ["stem", "localization_f1", "damage_macro_f1",
                    "valid_gt_damage_pixels", "missed_gt_damage_pixels"], per_image_rows)


def read_existing_baseline_f1(old_output: Path) -> dict[str, float]:
    # Only called after the frozen method has been selected AND evaluated on EBD.
    with (old_output / "cross_dataset_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["dataset"]: float(row["damage_macro_f1"]) for row in rows}


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-checkpoint", type=Path, default=EXPECTED_ORIGINAL)
    p.add_argument("--output-dir", type=Path, default=EXPECTED_NEW_OUTPUT)
    p.add_argument("--baseline-output", type=Path, default=EXPECTED_OLD_OUTPUT)
    p.add_argument("--xbd-root", type=Path, default=Path("/homes/j244s673/documents/wsu/phd/xview2"))
    p.add_argument("--ebd-base", type=Path, default=Path("/homes/j244s673/documents/wsu/phd"))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=896)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-only", action="store_true",
                   help="Run xBD Hold comparison and stop (no EBD data opened)")
    p.add_argument("--hash-ebd-inputs", action="store_true",
                   help="SHA256 individual EBD input files (slower)")
    return p


def main() -> None:
    args = args_parser().parse_args()
    if args.image_size != 896 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Requires image-size 896, batch size>=1, num-workers>=0")
    if args.baseline_output.resolve() != EXPECTED_OLD_OUTPUT.resolve(strict=True):
        raise RuntimeError("Original baseline folder path mismatch")
    old_ckpt, output = check_paths(args)
    baseline_info = check_original_baseline(args.baseline_output)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an allocated CUDA GPU")
    copy_path = copy_input_checkpoint(old_ckpt, output)
    runner.set_seed(args.seed)
    device = torch.device("cuda:0")

    checkpoint = torch.load(copy_path, map_location="cpu", weights_only=True)
    if int(checkpoint["epoch"]) != 35 or abs(float(checkpoint["best_threshold"]) - 0.5) > 1e-8:
        raise RuntimeError("Checkpoint epoch/threshold differs from the validated original")
    if int(checkpoint.get("args", {}).get("img_size", 896)) != 896:
        raise RuntimeError("Checkpoint image-size metadata differs from 896")
    model = architecture.make_model(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    print(f"[MODEL] Strict 8-channel RS-BDNet load complete; source epoch=35, loc threshold=0.5", flush=True)
    print("[METHODS] Fixed candidates: categorical, fusion50 (equal weight), ordinal", flush=True)
    print("[METHODS] Select using xBD Hold arithmetic damage Macro-F1 only", flush=True)
    print("[SAFETY] Only independent new output directory is writable by this script", flush=True)

    manifest = {
        "protocol": "Source-only selected inference rule (xBD Hold); frozen xBD checkpoint -> five EBD TEST datasets",
        "began_utc": now_utc(), "completed_utc": None,
        "protected_original_training_dir": str(EXPECTED_ORIGINAL.parent.parent),
        "protected_previous_zero_shot_results": str(EXPECTED_OLD_OUTPUT),
        "original_checkpoint_sha256_pre": base.sha256(old_ckpt),
        "independent_checkpoint_copy": str(copy_path),
        "checkpoint_copy_sha256_pre": base.sha256(copy_path),
        "original_epoch": 35, "localization_threshold_source_selected": 0.5,
        "source_method_candidates": list(METHODS),
        "ordinal_conversion": "q=sigmoid(ordinal logits); q=cumulative-min(q) to ensure q0>=q1>=q2; [1-q0,q0-q1,q1-q2,q2]",
        "fusion_weight": 0.5,
        "source_selection": "maximize arithmetic damage Macro-F1 on xBD Hold; strict tie precedence categorical > fusion50 > ordinal",
        "source_checkpoint_selection_caveat": "Original checkpoint was also selected on xBD Hold; this is source-only post-hoc inference selection, not a new independent validation set.",
        "ebd_results_reviewed_before_method_design": True,
        "interpretation": "Exploratory zero-shot follow-up; baseline remains separately published/archived; do not describe EBD-selected improvements as unbiased.",
        "seed": args.seed, "image_size": args.image_size, "batch_size": args.batch_size,
        "num_workers": args.num_workers, "amp": False,
        "original_baseline": baseline_info,
        "environment": None,
        "selected_method": None,
        "selected_on": None,
        "source_validation_results": None,
        "ebd_results": {},
    }
    base.write_json(output / "experiment_manifest.json", manifest)
    manifest["environment"] = base.dependencies_and_sources(output)
    manifest["environment"]["source_sha256"][str(Path(__file__).resolve())] = base.sha256(Path(__file__).resolve())
    base.write_json(output / "environment.json", manifest["environment"])
    base.write_json(output / "experiment_manifest.json", manifest)

    print("\n[SOURCE] Evaluating all three methods on xBD HOLD (no EBD data accessed)", flush=True)
    hold = runner.XBDOriginalDataset(root=args.xbd_root, split="hold", image_size=896,
                                     training=False, conditioning_id=0)
    hold_metrics, _ = evaluate_methods(model, hold, 0.5, device, args.batch_size, args.num_workers,
                                       METHODS, "xbd_hold")
    hold_rows = []
    for name in METHODS:
        result = hold_metrics[name]
        hold_rows.append([name, len(hold), result["localization"]["f1"],
                          *[result["damage_classes"][c]["f1"] for c in CLASS_NAMES],
                          result["damage_macro"]["precision"], result["damage_macro"]["recall"],
                          result["damage_macro"]["f1"]])
        base.write_json(output / f"xbd_hold_{name}_metrics.json", result)
        print(f"[SOURCE] {name:12s} Hold Macro-F1={result['damage_macro']['f1']:.6f}", flush=True)
    base.write_csv(output / "xbd_hold_method_comparison.csv",
                   ["method", "n_hold_pairs", "localization_f1", "no_damage_f1", "minor_damage_f1",
                    "major_damage_f1", "destroyed_f1", "macro_precision", "macro_recall", "macro_f1"],
                   hold_rows)

    winner = max(METHODS, key=lambda name: hold_metrics[name]["damage_macro"]["f1"])
    baseline_hold_f1 = hold_metrics["categorical"]["damage_macro"]["f1"]
    manifest["selected_method"] = winner
    manifest["selected_on"] = "xBD Hold only; before EBD was evaluated"
    manifest["source_validation_results"] = {
        name: {
            "damage_macro_f1": hold_metrics[name]["damage_macro"]["f1"],
            "damage_class_f1": {c: hold_metrics[name]["damage_classes"][c]["f1"] for c in CLASS_NAMES},
        } for name in METHODS
    }
    manifest["source_selection_improved_hold_vs_categorical"] = (
        hold_metrics[winner]["damage_macro"]["f1"] > baseline_hold_f1)
    base.write_json(output / "experiment_manifest.json", manifest)
    print(f"\n[SELECTED ON xBD ONLY] {winner} (Hold Macro-F1={hold_metrics[winner]['damage_macro']['f1']:.6f})", flush=True)

    if args.source_only:
        manifest["completed_utc"] = now_utc()
        manifest["mode"] = "xBD Hold only; no EBD data accessed"
        base.write_json(output / "experiment_manifest.json", manifest)
        print("[DONE] --source-only requested. No EBD sets read.", flush=True)
    else:
        print("\n[TARGET] Applying fixed source-selected method. No EBD tuning or retraining.", flush=True)
        new_results = OrderedDict()
        for dataset_name, (folder_name, expected_pairs) in EBD_DATASETS.items():
            root = args.ebd_base / folder_name
            output_folder = output / dataset_name
            # Preflight writes only into NEW output directory.
            preflight = base.dataset_preflight(root, expected_pairs,
                                                output / f"{dataset_name}_input_manifest.csv",
                                                args.hash_ebd_inputs)
            ds = runner.XBDOriginalDataset(root=root, split="test", image_size=896,
                                           training=False, conditioning_id=0)
            if len(ds) != expected_pairs:
                raise RuntimeError(f"{dataset_name}: expected {expected_pairs}, got {len(ds)} paired samples")
            result_dict, per_image_rows = evaluate_methods(
                model, ds, 0.5, device, args.batch_size, args.num_workers,
                (winner,), dataset_name, write_per_image=True)
            result = result_dict[winner]
            result["reproducibility"] = {
                "method_frozen_on_xbd_hold": winner, "selected_xbd_hold_macro_f1": hold_metrics[winner]["damage_macro"]["f1"],
                "checkpoint_sha256": ORIGINAL_SHA256, "checkpoint_epoch": 35,
                "localization_threshold": 0.5, "no_ebd_train_or_val_used": True,
                "dataset_preflight": preflight,
            }
            write_dataset_metrics(output_folder, result, per_image_rows[winner])
            new_results[dataset_name] = result
            manifest["ebd_results"][dataset_name] = {
                "n_test_pairs": len(ds),
                "damage_macro_f1": result["damage_macro"]["f1"],
                "localization_f1": result["localization"]["f1"],
                "input_manifest_sha256": preflight["input_manifest_sha256"],
            }
            base.write_json(output / "experiment_manifest.json", manifest)
            print(f"[EBD] {dataset_name}: Macro-F1={result['damage_macro']['f1']:.6f}", flush=True)

        # The old baseline is read only AFTER EBD selection/evaluation is complete.
        prior = read_existing_baseline_f1(EXPECTED_OLD_OUTPUT)
        comparison_rows = []
        for name, result in new_results.items():
            f1 = result["damage_macro"]["f1"]
            previous = prior[name]
            comparison_rows.append([name, result["n_image_pairs"], winner,
                                    result["localization"]["precision"],
                                    result["localization"]["recall"],
                                    result["localization"]["f1"],
                                    *[result["damage_classes"][cls]["f1"] for cls in CLASS_NAMES],
                                    result["damage_macro"]["precision"],
                                    result["damage_macro"]["recall"], f1, previous, f1-previous])
        base.write_csv(output / "cross_dataset_comparison.csv",
                       ["dataset", "n_test_pairs", "source_selected_method", "localization_precision", "localization_recall",
                        "localization_f1", "no_damage_f1", "minor_damage_f1", "major_damage_f1", "destroyed_f1",
                        "damage_macro_precision", "damage_macro_recall", "damage_macro_f1",
                        "original_categorical_macro_f1_read_only", "macro_f1_delta_exploratory"],
                       comparison_rows)
        mean_new = sum(r["damage_macro"]["f1"] for r in new_results.values()) / len(new_results)
        mean_original = sum(prior[name] for name in new_results) / len(new_results)
        manifest["five_ebd_macro_f1_mean"] = mean_new
        manifest["five_ebd_original_baseline_macro_f1_mean"] = mean_original
        manifest["five_ebd_macro_f1_difference_exploratory"] = mean_new - mean_original
        print(f"[SUMMARY] EBD five-dataset mean damage Macro-F1: "
              f"{mean_new:.6f} vs original {mean_original:.6f} "
              f"(exploratory delta {mean_new - mean_original:+.6f})", flush=True)
        manifest["completed_utc"] = now_utc()
        manifest["mode"] = "xBD Hold method selection then five EBD Test sets"
        base.write_json(output / "experiment_manifest.json", manifest)
        print(f"\n[DONE] New results: {output / 'cross_dataset_comparison.csv'}", flush=True)
    final_source_sha = base.sha256(old_ckpt)
    final_copy_sha = base.sha256(copy_path)
    if final_source_sha != ORIGINAL_SHA256 or final_copy_sha != ORIGINAL_SHA256:
        raise RuntimeError("Checkpoint SHA256 changed during evaluation! Investigate integrity.")
    print(f"[SAFETY] Original checkpoint and independent copy verified unchanged: {final_source_sha}", flush=True)
    print(f"[SAFETY] Both protected original output directories were never opened for writing", flush=True)


if __name__ == "__main__":
    main()
