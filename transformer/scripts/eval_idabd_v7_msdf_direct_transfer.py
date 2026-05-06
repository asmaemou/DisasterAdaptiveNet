#!/usr/bin/env python3
"""
Direct transfer baseline: evaluate an xBD-trained HRTBDA v7-MSDF model on IDA-BD.

No IDA-BD images or labels are used for training. IDA-BD labels are used only for final evaluation.

Expected IDA-BD root structure:
  idabd/
    images/
    masks/

The loader is intentionally flexible about filenames. It looks for pre/post image pairs in images/
and a matching mask in masks/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

PALETTE = np.array([
    [0, 0, 90],        # 0 background
    [0, 210, 0],       # 1 no damage
    [255, 255, 0],     # 2 minor
    [255, 150, 0],     # 3 major
    [255, 0, 0],       # 4 destroyed
], dtype=np.float32)

COMMON_RGB_LABELS = np.array([
    [0, 0, 0],
    [0, 0, 90],
    [0, 255, 0],
    [0, 210, 0],
    [255, 255, 0],
    [255, 150, 0],
    [255, 165, 0],
    [255, 0, 0],
], dtype=np.float32)


def load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("hrtbda_v7_module", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hrtbda_v7_module"] = module
    spec.loader.exec_module(module)
    return module


def make_v7_args(args: argparse.Namespace) -> SimpleNamespace:
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
        postprocess_dilation=args.postprocess_dilation,
        dilation_kernel=args.dilation_kernel,
        phase1_threshold=args.phase1_threshold,
        thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        amp=False,
        extra_photometric_aug=False,
    )


def normalize_image(arr: np.ndarray) -> torch.Tensor:
    arr = arr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean[None, None, :]) / std[None, None, :]
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def read_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return normalize_image(np.asarray(img))


def map_rgb_mask_to_ids(rgb: np.ndarray) -> np.ndarray:
    rgbf = rgb.astype(np.float32)
    flat = rgbf.reshape(-1, 3)
    # map to the five canonical classes using nearest palette, but also handle pure black as background.
    dist = ((flat[:, None, :] - PALETTE[None, :, :]) ** 2).sum(axis=2)
    labels = dist.argmin(axis=1).reshape(rgb.shape[:2]).astype(np.uint8)
    black = (rgb[..., 0] < 10) & (rgb[..., 1] < 10) & (rgb[..., 2] < 10)
    labels[black] = 0
    return labels


def read_mask(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path)
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    arr = np.asarray(img)

    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            labels = map_rgb_mask_to_ids(arr[..., :3])
        else:
            labels = arr[..., 0].astype(np.int64)
    else:
        labels = arr.astype(np.int64)
        # Common case: mask values are already 0..4.
        if labels.max() > 4:
            # Try to compress unique values by sorted order, keeping 0 as background.
            uniq = sorted(int(x) for x in np.unique(labels))
            if len(uniq) <= 5:
                mapping = {u: i for i, u in enumerate(uniq)}
                if 0 in mapping:
                    mapping[0] = 0
                labels = np.vectorize(lambda x: mapping.get(int(x), 0))(labels).astype(np.int64)
            else:
                # Fallback: anything nonzero becomes no-damage building.
                labels = (labels > 0).astype(np.int64)

    labels = np.where((labels >= 0) & (labels <= 4), labels, 0).astype(np.int64)
    return torch.from_numpy(labels).long()


def canonical_base_from_pre(pre_path: Path) -> str:
    stem = pre_path.stem
    for token in ["_pre_disaster", "_pre", "pre_disaster", "pre"]:
        if token in stem:
            stem = stem.replace(token, "")
    return stem.strip("_- ")


def find_corresponding_post(pre_path: Path) -> Optional[Path]:
    candidates = []
    s = str(pre_path)
    replacements = [
        ("pre_disaster", "post_disaster"),
        ("PRE_DISASTER", "POST_DISASTER"),
        ("_pre", "_post"),
        ("_PRE", "_POST"),
        ("pre", "post"),
        ("PRE", "POST"),
    ]
    for a, b in replacements:
        if a in s:
            candidates.append(Path(s.replace(a, b)))
    for c in candidates:
        if c.exists():
            return c
    return None


def find_mask_for_sample(masks_dir: Path, pre_path: Path, post_path: Path, base: str) -> Optional[Path]:
    names = [
        f"{base}.png",
        f"{base}_mask.png",
        f"{base}_damage.png",
        f"{base}_target.png",
        f"{base}_label.png",
        f"{base}_labels.png",
        post_path.name,
        post_path.stem + ".png",
        pre_path.name,
        pre_path.stem + ".png",
    ]
    # Also try removing common suffix from post stem.
    post_base = canonical_base_from_pre(Path(post_path.name.replace("post", "pre")))
    names += [f"{post_base}.png", f"{post_base}_mask.png", f"{post_base}_damage.png"]

    for n in names:
        p = masks_dir / n
        if p.exists():
            return p

    # Fallback: recursive fuzzy match by base.
    hits = []
    for p in masks_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and base in p.stem:
            hits.append(p)
    if hits:
        return sorted(hits)[0]
    return None


@dataclass
class IDABDSample:
    stem: str
    pre: str
    post: str
    mask: Optional[str]


def discover_idabd_samples(root: Path, require_mask: bool = False) -> List[IDABDSample]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists():
        raise FileNotFoundError(f"Could not find IDA-BD images directory: {images_dir}")

    pre_files = []
    for p in images_dir.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        st = p.stem.lower()
        if "pre" in st:
            pre_files.append(p)

    samples: List[IDABDSample] = []
    for pre in sorted(pre_files):
        post = find_corresponding_post(pre)
        if post is None:
            continue
        base = canonical_base_from_pre(pre)
        mask = find_mask_for_sample(masks_dir, pre, post, base) if masks_dir.exists() else None
        if require_mask and mask is None:
            continue
        samples.append(IDABDSample(stem=base, pre=str(pre), post=str(post), mask=str(mask) if mask else None))

    if not samples:
        raise RuntimeError(
            f"No IDA-BD samples found under {root}. Expected paired pre/post images in {images_dir}."
        )
    return samples


def get_or_create_split(samples: List[IDABDSample], split_json: Path, seed: int) -> Dict[str, List[str]]:
    if split_json.exists():
        with open(split_json, "r", encoding="utf-8") as f:
            return json.load(f)

    stems = [s.stem for s in samples]
    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(round(0.80 * n))
    n_val = int(round(0.10 * n))
    train = stems[:n_train]
    val = stems[n_train:n_train + n_val]
    test = stems[n_train + n_val:]
    if len(test) == 0 and len(val) > 1:
        test = [val.pop()]

    split = {"train": train, "val": val, "test": test, "seed": seed, "ratio": "80/10/10"}
    split_json.parent.mkdir(parents=True, exist_ok=True)
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


class IDABDDataset(Dataset):
    def __init__(self, samples: Sequence[IDABDSample], stems: Sequence[str], img_size: int, require_mask: bool = True):
        by_stem = {s.stem: s for s in samples}
        self.samples = [by_stem[s] for s in stems if s in by_stem]
        if require_mask:
            self.samples = [s for s in self.samples if s.mask is not None]
        self.img_size = img_size
        self.require_mask = require_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        item = {
            "stem": s.stem,
            "pre": read_rgb(Path(s.pre), self.img_size),
            "post": read_rgb(Path(s.post), self.img_size),
        }
        if s.mask is not None:
            item["target5"] = read_mask(Path(s.mask), self.img_size)
        elif self.require_mask:
            raise FileNotFoundError(f"Mask missing for {s.stem}")
        return item


def phase1_forward_logits(phase1_model: torch.nn.Module, pre: torch.Tensor) -> torch.Tensor:
    out = phase1_model(pre)
    if isinstance(out, dict):
        for key in ["loc_logits", "logits", "out", "mask_logits"]:
            if key in out:
                out = out[key]
                break
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    elif out.ndim == 4 and out.shape[1] > 1:
        out = out[:, 0]
    return out


def get_damage_logits(v7, out):
    if hasattr(v7, "get_damage_logits"):
        return v7.get_damage_logits(out)
    if isinstance(out, dict):
        for key in ["damage_logits", "logits", "out"]:
            if key in out:
                return out[key]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def damage_logits_to_pred(v7, damage_logits: torch.Tensor) -> torch.Tensor:
    if hasattr(v7, "damage_logits_to_pred"):
        return v7.damage_logits_to_pred(damage_logits)
    return torch.argmax(damage_logits, dim=1).long() + 1


def apply_damage_dilation(v7, damage_pred: torch.Tensor, loc_pred: torch.Tensor, mode: str, kernel_size: int) -> torch.Tensor:
    if hasattr(v7, "apply_damage_dilation"):
        return v7.apply_damage_dilation(damage_pred, loc_pred, mode=mode, kernel_size=kernel_size)
    if mode == "none":
        return damage_pred
    out = damage_pred.clone()
    classes = [2] if mode == "minor" else [2, 3]
    pad = kernel_size // 2
    for cls in classes:
        m = (damage_pred == cls).float().unsqueeze(1)
        md = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad).squeeze(1).bool()
        out[md & loc_pred.bool()] = cls
    return out


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def harmonic(values: List[float], eps: float = 1e-8) -> float:
    vals = [max(float(v), eps) for v in values]
    return float(len(vals) / sum(1.0 / v for v in vals))


def evaluate_idabd(v7, phase1_model, phase2_model, loader, device, phase1_threshold: float, dilation: str, dilation_kernel: int):
    phase1_model.eval()
    phase2_model.eval()
    loc_tp = loc_fp = loc_fn = 0
    cls_counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in [1, 2, 3, 4]}

    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            target = batch["target5"].to(device).long()

            loc_logits = phase1_forward_logits(phase1_model, pre)
            loc_pred = (torch.sigmoid(loc_logits) > phase1_threshold).long()

            out = phase2_model(pre, post)
            damage_logits = get_damage_logits(v7, out)
            damage_pred = damage_logits_to_pred(v7, damage_logits)
            damage_pred = apply_damage_dilation(v7, damage_pred, loc_pred, dilation, dilation_kernel)

            final_pred = torch.zeros_like(damage_pred)
            final_pred[loc_pred.bool()] = damage_pred[loc_pred.bool()]

            gt_loc = target > 0
            pr_loc = final_pred > 0
            loc_tp += int((gt_loc & pr_loc).sum().item())
            loc_fp += int((~gt_loc & pr_loc).sum().item())
            loc_fn += int((gt_loc & ~pr_loc).sum().item())

            for c in [1, 2, 3, 4]:
                gt = target == c
                pr = final_pred == c
                cls_counts[c]["tp"] += int((gt & pr).sum().item())
                cls_counts[c]["fp"] += int((~gt & pr).sum().item())
                cls_counts[c]["fn"] += int((gt & ~pr).sum().item())

    loc_f1 = f1_from_counts(loc_tp, loc_fp, loc_fn)
    class_f1 = {c: f1_from_counts(v["tp"], v["fp"], v["fn"]) for c, v in cls_counts.items()}
    damage_f1_hmean = harmonic([class_f1[c] for c in [1, 2, 3, 4]])
    damage_f1_macro = float(np.mean([class_f1[c] for c in [1, 2, 3, 4]]))
    overall = 0.3 * loc_f1 + 0.7 * damage_f1_hmean
    return {
        "loc_f1": loc_f1,
        "no_damage_f1": class_f1[1],
        "minor_damage_f1": class_f1[2],
        "major_damage_f1": class_f1[3],
        "destroyed_f1": class_f1[4],
        "damage_f1_hmean": damage_f1_hmean,
        "damage_f1_macro": damage_f1_macro,
        "overall_score": overall,
        "loc_counts": {"tp": loc_tp, "fp": loc_fp, "fn": loc_fn},
        "class_counts": cls_counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v7-script", type=Path, default=Path("transformer/scripts/train_xbd_hrtbda_v7_msdf_full_two_stage.py"))
    p.add_argument("--phase1-checkpoint", type=Path, required=True)
    p.add_argument("--phase2-checkpoint", type=Path, required=True)
    p.add_argument("--idabd-root", type=Path, required=True)
    p.add_argument("--source-xbd-root", type=Path, default=Path("/homes/j244s673/documents/wsu/phd/xview2"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, default=None)
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=128)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--phase1-threshold", type=float, default=0.50)
    p.add_argument("--postprocess-dilation", choices=["none", "minor", "minor_major"], default="minor")
    p.add_argument("--dilation-kernel", type=int, default=3)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.split_json is None:
        args.split_json = args.output_dir / "idabd_splits_seed42_80_10_10.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    v7 = load_module(args.v7_script)
    v7_args = make_v7_args(args)

    samples = discover_idabd_samples(args.idabd_root, require_mask=True)
    split = get_or_create_split(samples, args.split_json, args.seed)
    print("===== IDA-BD SPLIT SUMMARY =====", flush=True)
    print(f"Train: {len(split['train'])}", flush=True)
    print(f"Val:   {len(split['val'])}", flush=True)
    print(f"Test:  {len(split['test'])}", flush=True)
    print("=================================", flush=True)

    ds = IDABDDataset(samples, split[args.eval_split], img_size=args.img_size, require_mask=True)
    if len(ds) == 0:
        raise RuntimeError(f"No labeled samples found for split {args.eval_split}")
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    phase1_model, phase1_threshold, phase1_meta = v7.load_phase1_model_for_cascade(
        args=v7_args, device=device, phase1_ckpt=args.phase1_checkpoint
    )
    print(f"Loaded Phase-I threshold from checkpoint: {phase1_threshold}", flush=True)
    print(f"Phase-I meta: {phase1_meta}", flush=True)

    phase2_model = v7.HRTBDAPhase2(
        base_channels=args.base_channels,
        decoder_channels=args.decoder_channels,
        window_size=args.window_size,
        num_classes=4,
    ).to(device)
    ckpt = v7.load_model_weights(phase2_model, args.phase2_checkpoint, device)
    print(f"Loaded Phase-II checkpoint epoch: {ckpt.get('epoch', 'unknown')}", flush=True)

    metrics = evaluate_idabd(
        v7=v7,
        phase1_model=phase1_model,
        phase2_model=phase2_model,
        loader=loader,
        device=device,
        phase1_threshold=phase1_threshold,
        dilation=args.postprocess_dilation,
        dilation_kernel=args.dilation_kernel,
    )

    result = {
        "experiment": "HRTBDA v7-MSDF Direct Transfer xBD -> IDA-BD",
        "phase1_checkpoint": str(args.phase1_checkpoint),
        "phase2_checkpoint": str(args.phase2_checkpoint),
        "idabd_root": str(args.idabd_root),
        "split_json": str(args.split_json),
        "eval_split": args.eval_split,
        "phase1_threshold": float(phase1_threshold),
        "postprocess_dilation": args.postprocess_dilation,
        "dilation_kernel": args.dilation_kernel,
        "metrics": metrics,
    }

    scores_dir = args.output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    json_path = scores_dir / f"idabd_direct_transfer_{args.eval_split}_scores.json"
    txt_path = scores_dir / f"summary_idabd_direct_transfer_{args.eval_split}.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = [
        "Experiment: HRTBDA v7-MSDF Direct Transfer xBD -> IDA-BD",
        f"IDA-BD root: {args.idabd_root}",
        f"Eval split: {args.eval_split}",
        f"Phase I checkpoint: {args.phase1_checkpoint}",
        f"Phase II checkpoint: {args.phase2_checkpoint}",
        f"Phase I threshold used for mask: {phase1_threshold:.2f}",
        f"Damage post-processing dilation: {args.postprocess_dilation}",
        f"Localization F1: {metrics['loc_f1']:.6f}",
        f"No Damage F1:    {metrics['no_damage_f1']:.6f}",
        f"Minor Damage F1: {metrics['minor_damage_f1']:.6f}",
        f"Major Damage F1: {metrics['major_damage_f1']:.6f}",
        f"Destroyed F1:    {metrics['destroyed_f1']:.6f}",
        f"Damage F1 harmonic: {metrics['damage_f1_hmean']:.6f}",
        f"Damage F1 macro:    {metrics['damage_f1_macro']:.6f}",
        f"Overall Score:      {metrics['overall_score']:.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)


if __name__ == "__main__":
    main()
