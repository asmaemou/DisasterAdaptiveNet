#!/usr/bin/env python3
"""Strict inference-only xBD -> five EBD zero-shot evaluation for RS-BDNet.

Reuses the project's *original* dataset and model implementations:
- runner.XBDOriginalDataset (pre_target is localization; post_target is damage),
- architecture.make_model (identical HBG-CAON 8-channel network),
- runner.evaluate_f1's pixel-level localization/damage counting semantics.

The model is selected ONLY on xBD Hold.  No EBD train/val data are loaded,
no thresholds are scanned, and no optimizer or train() calls occur.

Usage is provided in accompanying SLURM submission file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# When copied to transformer/scripts/, parents[2] is project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
import transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal as architecture
from utils.models import Res34_Unet_Loc

EBD_DATASETS = OrderedDict([
    ("earthquake_turkey", ("earthquake_turkey_preprocessed", 95)),
    ("mount_semeru_eruption", ("mount_semeru_eruption_preprocessed", 60)),
    ("texas_tornadoes", ("texas_tornadoes_preprocessed", 46)),
    ("hurricane_delta", ("hurricane_delta_preprocessed", 137)),
    ("pakistan_flooding", ("pakistan_flooding_preprocessed", 354)),
])
CLASS_NAMES = ("no_damage", "minor_damage", "major_damage", "destroyed")
PREDICTION_NAMES = ("missed_localization",) + CLASS_NAMES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def record_prf(tp: int, fp: int, fn: int, tn: int | None = None) -> dict[str, float | int]:
    """Matches runner.F1Recorder: zero denominator gives a score of zero."""
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision and recall else 0.0
    result: dict[str, float | int] = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
    if tn is not None:
        result["tn"] = int(tn)
    return result


def dataset_preflight(root: Path, expected_pairs: int, output_path: Path, hash_inputs: bool) -> dict:
    """Fail rather than silently skip samples, as the original loader would."""
    split = root / "test"
    images = split / "images"
    targets = split / "targets"
    masks = split / "masks"
    for directory in (images, targets, masks):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required EBD test directory does not exist: {directory}")

    observed_post = sorted(images.glob("*_post_disaster.png"))
    observed_pre = set(images.glob("*_pre_disaster.png"))
    if len(observed_post) != expected_pairs or len(observed_pre) != expected_pairs:
        raise RuntimeError(f"{root.name}: expected {expected_pairs} image pairs, "
                           f"found {len(observed_pre)} pre / {len(observed_post)} post")

    all_paths = []
    expected_image_names = set()
    expected_target_names = set()
    for post in observed_post:
        stem = post.name[:-len("_post_disaster.png")]
        pre = images / f"{stem}_pre_disaster.png"
        pre_tgt = targets / f"{stem}_pre_disaster_target.png"
        post_tgt = targets / f"{stem}_post_disaster_target.png"
        pre_mask = masks / f"{stem}_pre_disaster.png"
        post_mask = masks / f"{stem}_post_disaster.png"
        for p in (pre, post, pre_tgt, post_tgt, pre_mask, post_mask):
            if not p.is_file():
                raise FileNotFoundError(f"Missing required paired file: {p}")
            all_paths.append(p)
        expected_image_names.update((pre.name, post.name))
        expected_target_names.update((pre_tgt.name, post_tgt.name))

    for directory, wanted in ((images, expected_image_names),
                              (masks, expected_image_names),
                              (targets, expected_target_names)):
        present = {p.name for p in directory.glob("*.png")}
        if present != wanted:
            raise RuntimeError(f"Unexpected or missing files in {directory}: "
                               f"extra={sorted(present - wanted)[:8]}, "
                               f"missing={sorted(wanted - present)[:8]}")

    manifest_rows = []
    manifest_digest = hashlib.sha256()
    for p in sorted(all_paths):
        rel = str(p.relative_to(root))
        stat = p.stat()
        digest = sha256(p) if hash_inputs else ""
        row = [rel, stat.st_size, digest]
        manifest_rows.append(row)
        manifest_digest.update(("\t".join(map(str, row)) + "\n").encode("utf-8"))
    write_csv(output_path, ["relative_path", "size_bytes", "sha256_if_enabled"], manifest_rows)
    return {
        "dataset_root": str(root.resolve()),
        "split": "test",
        "expected_pairs": expected_pairs,
        "verified_pairs": len(observed_post),
        "file_records": len(manifest_rows),
        "input_manifest_sha256": manifest_digest.hexdigest(),
        "individual_file_sha256_in_manifest": hash_inputs,
        "input_manifest_csv": str(output_path),
    }


def metrics_from_counts(loc_counts: np.ndarray, damage_conf: np.ndarray, n_samples: int) -> dict:
    """Confusion matrix rows: GT classes 0..3; columns: missed + predicted 0..3.

    This replicates runner.evaluate_f1's damage semantics exactly: valid_gt =
    (loc_true == 1) & (dmg_target != 255), with predictions set to category 0
    (missed) wherever predicted localization == 0.  Predicted positive areas
    outside the GT building mask do NOT count as damage false positives.
    """
    lp = record_prf(int(loc_counts[0]), int(loc_counts[1]),
                    int(loc_counts[2]), int(loc_counts[3]))
    per_class = OrderedDict()
    for idx, label in enumerate(CLASS_NAMES):
        tp = int(damage_conf[idx, idx + 1])
        fp = int(damage_conf[:, idx + 1].sum()) - tp
        fn = int(damage_conf[idx, :].sum()) - tp
        metrics = record_prf(tp, fp, fn)
        metrics["support_valid_gt_pixels"] = int(damage_conf[idx, :].sum())
        per_class[label] = metrics

    ps = [per_class[k]["precision"] for k in CLASS_NAMES]
    rs = [per_class[k]["recall"] for k in CLASS_NAMES]
    fs = [per_class[k]["f1"] for k in CLASS_NAMES]
    macro_p = sum(ps) / 4.0
    macro_r = sum(rs) / 4.0
    macro_f = sum(fs) / 4.0
    # The original runner uses 1e-6 smoothing in the harmonic mean.
    harmonic_original = runner.harmonic_mean(fs)
    original_score = 0.3 * lp["f1"] + 0.7 * harmonic_original

    return {
        "n_image_pairs": n_samples,
        "localization": lp,
        "damage_classes": per_class,
        "damage_macro": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f,
        },
        "damage_harmonic_f1_original_runner": harmonic_original,
        "overall_score_original_runner": original_score,
        "valid_ground_truth_damage_pixels": int(damage_conf.sum()),
        "missed_localization_on_valid_damage_pixels": int(damage_conf[:, 0].sum()),
        "damage_confusion_matrix": {
            "row_labels_ground_truth": list(CLASS_NAMES),
            "column_labels_prediction": list(PREDICTION_NAMES),
            "counts": damage_conf.astype(int).tolist(),
        },
        "evaluation_definition": (
            "Pixel-level. Localization TP/FP/FN from all pixels; damage class "
            "TP/FP/FN restricted to pixels with loc_true=1 and damage_target!=255. "
            "Predicted damage is gated by predicted localization; category 0 "
            "is a missed building and is counted as FN for its true class. "
            "Damage Macro-F1 is the unweighted arithmetic mean of four F1 scores. "
            "The original-runner harmonic damage F1 is reported separately."
        ),
    }


@torch.inference_mode()
def evaluate(model: torch.nn.Module, dataset, batch_size: int,
             num_workers: int, threshold: float, device: torch.device,
             output_dir: Path, dataset_name: str) -> dict:
    loader = runner.make_eval_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    loc_counts = np.zeros(4, dtype=np.int64)  # TP FP FN TN
    confusion = np.zeros((4, 5), dtype=np.int64)
    per_image_rows: list[list[Any]] = []
    n_seen = 0
    model.eval()

    for batch_i, batch in enumerate(loader):
        imgs = batch["img"].to(device, non_blocking=True)
        gt_loc = batch["loc"].to(device, non_blocking=True).long()
        gt_dmg = batch["dmg"].to(device, non_blocking=True).long()
        cond = batch["cond_id"].to(device, non_blocking=True)
        logits = model(imgs, cond)
        if logits.ndim != 4 or logits.shape[1] != 8 or logits.shape[-2:] != imgs.shape[-2:]:
            raise RuntimeError(f"Unexpected output shape {tuple(logits.shape)}; expected [B,8,H,W]")
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite network outputs: dataset {dataset_name}, batch {batch_i}")
        loc_pred = (torch.sigmoid(logits[:, 0]) > threshold).long()
        damage_pred = (torch.argmax(logits[:, 1:5], dim=1) + 1) * loc_pred

        for j in range(imgs.shape[0]):
            tl, pl = gt_loc[j], loc_pred[j]
            td, pd = gt_dmg[j], damage_pred[j]
            loc_i = np.array([
                ((pl == 1) & (tl == 1)).sum().item(),
                ((pl == 1) & (tl == 0)).sum().item(),
                ((pl == 0) & (tl == 1)).sum().item(),
                ((pl == 0) & (tl == 0)).sum().item(),
            ], dtype=np.int64)
            if (tl.min() < 0) or (tl.max() > 1):
                raise RuntimeError(f"Invalid localization labels for image {batch['stem'][j]}")
            valid = (tl == 1) & (td != 255)
            if bool(valid.any()):
                actual = td[valid]
                predicted = pd[valid]
                if not bool(((actual >= 0) & (actual <= 3)).all()):
                    raise RuntimeError(f"Invalid damage target values for {batch['stem'][j]}")
                cm_i = torch.bincount(actual * 5 + predicted, minlength=20)
                cm_i = cm_i.reshape(4, 5).cpu().numpy().astype(np.int64)
            else:
                cm_i = np.zeros((4, 5), dtype=np.int64)
            loc_counts += loc_i
            confusion += cm_i
            n_seen += 1
            im = metrics_from_counts(loc_i, cm_i, 1)
            per_image_rows.append([
                str(batch["stem"][j]),
                int(loc_i[0]), int(loc_i[1]), int(loc_i[2]), int(loc_i[3]),
                im["localization"]["f1"], im["damage_macro"]["f1"],
                int(cm_i.sum()), int(cm_i[:, 0].sum()),
            ])
        if batch_i % 10 == 0:
            print(f"  {dataset_name}: {n_seen}/{len(dataset)} image pairs", flush=True)
    if n_seen != len(dataset):
        raise RuntimeError(f"Dataset {dataset_name}: evaluated {n_seen} != loader's {len(dataset)}")

    out = metrics_from_counts(loc_counts, confusion, n_seen)
    write_json(output_dir / "metrics.json", out)
    write_csv(output_dir / "damage_confusion_matrix.csv",
              ["ground_truth\\prediction", *PREDICTION_NAMES],
              [[name, *[int(x) for x in confusion[i]]] for i, name in enumerate(CLASS_NAMES)])
    write_csv(output_dir / "per_image_metrics.csv",
              ["stem", "loc_tp", "loc_fp", "loc_fn", "loc_tn", "loc_f1",
               "damage_macro_f1", "valid_gt_damage_pixels", "missed_damage_pixels"],
              per_image_rows)
    write_csv(output_dir / "class_metrics.csv",
              ["category", "precision", "recall", "f1", "tp", "fp", "fn", "support_pixels"],
              [["localization", out["localization"]["precision"], out["localization"]["recall"],
                out["localization"]["f1"], out["localization"]["tp"], out["localization"]["fp"],
                out["localization"]["fn"], int(loc_counts[0] + loc_counts[2])]] +
              [[name, *[out["damage_classes"][name][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "support_valid_gt_pixels")]]
               for name in CLASS_NAMES] +
              [["damage_macro", out["damage_macro"]["precision"], out["damage_macro"]["recall"],
                out["damage_macro"]["f1"], "", "", "", int(confusion.sum())]])
    return out


def get_git_info() -> dict:
    def run(*args: str) -> str | None:
        p = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                           capture_output=True, text=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def dependencies_and_sources(output_dir: Path) -> dict:
    paths = [
        Path(__file__),
        PROJECT_ROOT / "transformer/scripts/train_xbd_bitemporal_building_crossattention_ordinal.py",
        PROJECT_ROOT / "transformer/scripts/train_xbd_supervised_disasteradaptivenet.py",
        PROJECT_ROOT / "transformer/scripts/train_xbd_resnet34_swin_film_gated.py",
        PROJECT_ROOT / "transformer/scripts/train_xbd_hrtbda_v5_swin_pretrained_cascade.py",
        Path(inspect.getfile(Res34_Unet_Loc)),
    ]
    source_hashes = {str(p): sha256(p) if p.is_file() else None for p in paths}
    env = {
        "timestamp_utc": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "git": get_git_info(),
        "source_sha256": source_hashes,
    }
    write_json(output_dir / "environment.json", env)
    p = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                       capture_output=True, text=True, check=False)
    if p.returncode == 0:
        (output_dir / "pip_freeze.txt").write_text(p.stdout, encoding="utf-8")
    else:
        (output_dir / "pip_freeze_error.txt").write_text(p.stderr, encoding="utf-8")
    return env


def summary_row(name: str, result: dict) -> dict[str, Any]:
    r = {"dataset": name, "n_test_pairs": result["n_image_pairs"],
         "localization_precision": result["localization"]["precision"],
         "localization_recall": result["localization"]["recall"],
         "localization_f1": result["localization"]["f1"]}
    for cls in CLASS_NAMES:
        for metric in ("precision", "recall", "f1"):
            r[f"{cls}_{metric}"] = result["damage_classes"][cls][metric]
    for metric in ("precision", "recall", "f1"):
        r[f"damage_macro_{metric}"] = result["damage_macro"][metric]
    r["damage_harmonic_f1_existing_runner"] = result["damage_harmonic_f1_original_runner"]
    return r


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ebd-base", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--xbd-root", type=Path)
    p.add_argument("--verify-xbd-test", action="store_true",
                   help="Evaluate xBD Test first; fail if published F1 scores cannot be reproduced")
    p.add_argument("--expected-xbd-macro-f1", type=float, default=0.7756)
    p.add_argument("--expected-xbd-localization-f1", type=float, default=0.8447)
    p.add_argument("--xbd-reproduction-tolerance", type=float, default=0.00015)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=896)
    p.add_argument("--expected-epoch", type=int, default=35)
    p.add_argument("--expected-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hash-input-files", action="store_true",
                   help="SHA256 each image, mask and target in test sets (slower but archival-grade)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers nonnegative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.ebd_base.is_dir():
        raise FileNotFoundError(args.ebd_base)
    if args.verify_xbd_test and (args.xbd_root is None or not (args.xbd_root / "test").is_dir()):
        raise ValueError("--verify-xbd-test requires --xbd-root with an existing test directory")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; refusing unplanned CPU-only evaluation")

    out_dir = args.output_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use a new directory "
                              "so previous evaluation results cannot be overwritten.")
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.set_seed(args.seed)
    device = torch.device("cuda:0")
    print(f"[START] {utc_now()}  checkpoint={args.checkpoint}  output={out_dir}", flush=True)

    # weights_only=True avoids deserializing arbitrary executable objects.
    # Original runner.save_checkpoint stores a dict of tensors and simple values.
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError as e:
        raise RuntimeError("This experiment requires PyTorch with weights_only=True support") from e
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("Checkpoint is missing the original 'model' state dictionary")
    epoch = int(checkpoint["epoch"])
    threshold = float(checkpoint["best_threshold"])
    if epoch != args.expected_epoch or abs(threshold - args.expected_threshold) > 1e-8:
        raise RuntimeError(f"Selected checkpoint metadata unexpected: epoch={epoch}, "
                           f"threshold={threshold}; expected epoch={args.expected_epoch}, "
                           f"threshold={args.expected_threshold}")
    ckpt_args = checkpoint.get("args") or {}
    if not isinstance(ckpt_args, dict):
        raise RuntimeError("Checkpoint 'args' metadata is not a dictionary")
    ckpt_img_size = ckpt_args.get("img_size")
    if ckpt_img_size is not None and int(ckpt_img_size) != args.image_size:
        raise RuntimeError(f"Checkpoint image size {ckpt_img_size} != requested {args.image_size}")
    if args.image_size != 896:
        raise ValueError("The original architecture.make_model() fixes Swin at 896; do not alter it")

    # Architecture import uses the same make_model as the original xBD run.
    model = architecture.make_model(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        model.requires_grad_(False)
    print(f"[MODEL] strict state loaded; epoch {epoch}, xBD-Hold threshold {threshold:.4f}", flush=True)
    env = dependencies_and_sources(out_dir)
    manifest = {
        "protocol": "xBD Train+Tier3 -> xBD Hold checkpoint+threshold -> five EBD Test sets; no adaptation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch_1_based": epoch,
        "checkpoint_xbd_hold_score": checkpoint.get("best_score"),
        "checkpoint_xbd_hold_threshold": threshold,
        "checkpoint_saved_args": ckpt_args,
        "seed": args.seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "inference_autocast": False,
        "data_source": "runner.XBDOriginalDataset; EBD test/images and test/targets",
        "damage_label_mapping": "pre_target >0 => building; post_target codes 1,2,3,4 => no,minor,major,destroyed; ignored=255",
        "xBD_test_reproduction_required": args.verify_xbd_test,
        "environment_file": str(out_dir / "environment.json"),
        "environment_source_sha256": env["source_sha256"],
        "dataset_preflights": {},
        "completed_utc": None,
    }
    write_json(out_dir / "experiment_manifest.json", manifest)

    if args.verify_xbd_test:
        print("\n[REPRODUCE] Original xBD Test evaluation, before EBD transfer", flush=True)
        xbd_dataset = runner.XBDOriginalDataset(
            root=args.xbd_root, split="test", image_size=args.image_size,
            training=False, conditioning_id=0)
        if len(xbd_dataset) != 933:
            raise RuntimeError(f"Expected 933 xBD Test image pairs; got {len(xbd_dataset)}")
        xbd_result = evaluate(model, xbd_dataset, args.batch_size, args.num_workers,
                              threshold, device, out_dir / "xbd_test_reproduction", "xbd_test")
        macro = xbd_result["damage_macro"]["f1"]
        loc = xbd_result["localization"]["f1"]
        print(f"[REPRODUCE] xBD: loc F1={loc:.6f}, macro F1={macro:.6f}", flush=True)
        expected_xbd_class_f1 = {
            "no_damage": 0.9356,
            "minor_damage": 0.5648,
            "major_damage": 0.7520,
            "destroyed": 0.8499,
        }
        mismatched_classes = {
            name: (xbd_result["damage_classes"][name]["f1"], expected)
            for name, expected in expected_xbd_class_f1.items()
            if abs(xbd_result["damage_classes"][name]["f1"] - expected)
            > args.xbd_reproduction_tolerance
        }
        if (abs(macro - args.expected_xbd_macro_f1) > args.xbd_reproduction_tolerance or
                abs(loc - args.expected_xbd_localization_f1) > args.xbd_reproduction_tolerance or
                mismatched_classes):
            raise RuntimeError("xBD benchmark reproduction FAILED; do NOT report EBD scores "
                               f"until resolved. loc={loc:.6f} macro={macro:.6f}, "
                               f"per-class mismatches={mismatched_classes}")
        manifest["xbd_test_reproduction"] = {
            "localization_f1": loc, "damage_macro_f1": macro,
            "status": "PASS", "expected_macro": args.expected_xbd_macro_f1,
            "expected_localization": args.expected_xbd_localization_f1,
            "tolerance": args.xbd_reproduction_tolerance,
        }
        write_json(out_dir / "experiment_manifest.json", manifest)

    summary_rows: list[dict[str, Any]] = []
    for name, (directory, expected_pairs) in EBD_DATASETS.items():
        print(f"\n[DATASET] {name}  expected={expected_pairs} pairs", flush=True)
        ds_dir = out_dir / name
        ds_dir.mkdir(parents=True, exist_ok=False)
        data_root = args.ebd_base / directory
        preflight = dataset_preflight(data_root, expected_pairs,
                                      ds_dir / "input_manifest.csv", args.hash_input_files)
        manifest["dataset_preflights"][name] = preflight
        write_json(out_dir / "experiment_manifest.json", manifest)
        ds = runner.XBDOriginalDataset(
            root=data_root, split="test", image_size=args.image_size,
            training=False, conditioning_id=0)
        if len(ds) != expected_pairs:
            raise RuntimeError(f"Original data loader included {len(ds)} pairs for {name}; "
                               f"expected {expected_pairs}")
        result = evaluate(model, ds, args.batch_size, args.num_workers,
                          threshold, device, ds_dir, name)
        result["reproducibility"] = {
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "checkpoint_epoch": epoch, "threshold_selected_on_xbd_hold": threshold,
            "input_manifest_sha256": preflight["input_manifest_sha256"],
            "seed": args.seed, "test_split_only": True,
            "no_target_finetuning_or_threshold_tuning": True,
        }
        write_json(ds_dir / "metrics.json", result)
        row = summary_row(name, result)
        summary_rows.append(row)
        headers = list(summary_rows[0].keys())
        write_csv(out_dir / "cross_dataset_summary.csv", headers,
                  [[r.get(key, "") for key in headers] for r in summary_rows])
        print(f"[RESULT] {name}: loc F1={result['localization']['f1']:.6f}; "
              f"macro P/R/F1={result['damage_macro']['precision']:.6f}/"
              f"{result['damage_macro']['recall']:.6f}/"
              f"{result['damage_macro']['f1']:.6f}", flush=True)

    manifest["completed_utc"] = utc_now()
    write_json(out_dir / "experiment_manifest.json", manifest)
    print("\n[DONE] Zero-shot EBD evaluation completed successfully.", flush=True)
    print(f"Results: {out_dir / 'cross_dataset_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
