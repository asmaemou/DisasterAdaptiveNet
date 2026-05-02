#!/usr/bin/env python3
"""
IDA-BD adapter for the HRTBDA v2 cascaded Phase-I-mask pipeline.

This keeps the same core logic from train_xbd_hrtbda_v2_cascaded_phase1mask.py:
  - Phase I: pre-disaster image -> binary building localization mask
  - Phase II: pre/post images -> foreground-only 4-class damage severity
  - Final inference: outside Phase-I mask = background; inside Phase-I mask = Phase-II damage

The only dataset-specific change is that IDA-BD is read from a flat structure:
  ROOT/images/
  ROOT/masks/

and a deterministic 80/10/10 train/validation/test split is created automatically.

Important:
  Keep train_xbd_hrtbda_v2_cascaded_phase1mask.py in the same transformer/scripts
  directory because this script imports and reuses its model, losses, training loop,
  checkpointing, and cascaded evaluation functions.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Reuse the exact v2 architecture/training/evaluation code.
import train_xbd_hrtbda_v2_cascaded_phase1mask as v2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class IDABDSample:
    stem: str
    pre_image_path: Path
    post_image_path: Path
    post_mask_path: Path
    pre_mask_path: Optional[Path] = None


def tile_id_from_name(path_or_name: str | Path) -> str:
    base = Path(path_or_name).stem
    for suffix in [
        "_pre_disaster_target",
        "_post_disaster_target",
        "_pre_disaster_mask",
        "_post_disaster_mask",
        "_pre_disaster",
        "_post_disaster",
        "_target",
        "_mask",
    ]:
        base = base.replace(suffix, "")
    return base


def list_images_by_split(images_dir: Path, split: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in IMG_EXTS:
        for p in images_dir.glob(f"*_{split}_disaster{ext}"):
            out[tile_id_from_name(p)] = p
    return dict(sorted(out.items()))


def find_mask(masks_dir: Path, stem: str, split: str = "post") -> Optional[Path]:
    """Find IDA-BD mask using several common naming conventions."""
    candidate_bases = [
        f"{stem}_{split}_disaster_target",
        f"{stem}_{split}_disaster_mask",
        f"{stem}_{split}_disaster",
    ]
    if split == "post":
        candidate_bases += [f"{stem}_target", f"{stem}_mask", stem]

    for base in candidate_bases:
        for ext in IMG_EXTS:
            p = masks_dir / f"{base}{ext}"
            if p.exists():
                return p
    return None


def collect_idabd_samples(root: str | Path) -> List[IDABDSample]:
    root = Path(root)
    images_dir = root / "images"
    masks_dir = root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Expected IDA-BD images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Expected IDA-BD masks directory not found: {masks_dir}")

    pre_map = list_images_by_split(images_dir, "pre")
    post_map = list_images_by_split(images_dir, "post")
    stems = sorted(set(pre_map.keys()) & set(post_map.keys()))

    samples: List[IDABDSample] = []
    missing_masks: List[str] = []

    for stem in stems:
        post_mask = find_mask(masks_dir, stem, split="post")
        if post_mask is None:
            missing_masks.append(stem)
            continue
        pre_mask = find_mask(masks_dir, stem, split="pre")
        samples.append(
            IDABDSample(
                stem=stem,
                pre_image_path=pre_map[stem],
                post_image_path=post_map[stem],
                post_mask_path=post_mask,
                pre_mask_path=pre_mask,
            )
        )

    if not samples:
        raise RuntimeError(
            f"No valid IDA-BD paired samples found under {root}. "
            "Expected images/*_pre_disaster.*, images/*_post_disaster.*, and matching masks."
        )

    if missing_masks:
        print(f"WARNING: skipped {len(missing_masks)} sample(s) with missing post masks.", flush=True)

    return samples


def make_idabd_splits(
    samples: List[IDABDSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    stems = [s.stem for s in samples]
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    if n < 3:
        raise RuntimeError(f"Need at least 3 samples for train/val/test split. Found {n}.")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Guarantee non-empty validation and test when possible.
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    train = sorted(stems[:n_train])
    val = sorted(stems[n_train:n_train + n_val])
    test = sorted(stems[n_train + n_val:])

    return {"train": train, "val": val, "test": test}


def prepare_or_load_splits(args: argparse.Namespace, samples: List[IDABDSample]) -> Dict[str, List[str]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file = Path(args.split_file) if args.split_file else output_dir / f"idabd_splits_seed{args.seed}_80_10_10.json"

    if split_file.exists() and not args.force_resplit:
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        print(f"Loaded existing IDA-BD split file: {split_file}", flush=True)
    else:
        splits = make_idabd_splits(
            samples=samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"Wrote IDA-BD split file: {split_file}", flush=True)

    all_stems = {s.stem for s in samples}
    for key in ["train", "val", "test"]:
        splits[key] = [s for s in splits.get(key, []) if s in all_stems]
        if not splits[key]:
            raise RuntimeError(f"Split '{key}' is empty. Check split file or dataset discovery.")

    print("===== IDA-BD SPLIT SUMMARY =====", flush=True)
    print(f"Train: {len(splits['train'])}", flush=True)
    print(f"Val:   {len(splits['val'])}", flush=True)
    print(f"Test:  {len(splits['test'])}", flush=True)
    print("=================================", flush=True)

    return splits


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        # The IDA-BD masks used here are expected to be single-channel class IDs.
        # If they are saved as 3-channel grayscale, taking channel 0 is correct.
        m = m[..., 0]
    m = m.astype(np.int64)
    legal = (m == 0) | (m == 1) | (m == 2) | (m == 3) | (m == 4) | (m == 255)
    m = np.where(legal, m, 255).astype(np.uint8)
    return m


class IDABDHRTBDADataset(Dataset):
    """IDA-BD dataset exposing the same keys as XBDHRTBDADataset."""

    def __init__(
        self,
        root: str | Path,
        samples_by_stem: Dict[str, IDABDSample],
        stems: List[str],
        image_size: int,
        training: bool,
    ):
        self.root = Path(root)
        self.samples_by_stem = samples_by_stem
        self.stems = list(stems)
        self.image_size = int(image_size)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.stems)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def _loc_from_sample(self, s: IDABDSample) -> np.ndarray:
        if s.pre_mask_path is not None:
            pre_mask = read_mask(s.pre_mask_path)
            return (pre_mask > 0).astype(np.uint8)
        post_mask = read_mask(s.post_mask_path)
        return np.isin(post_mask, [1, 2, 3, 4]).astype(np.uint8)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        stem = self.stems[index]
        s = self.samples_by_stem[stem]

        pre = read_rgb(s.pre_image_path)
        post = read_rgb(s.post_image_path)
        loc_raw = self._loc_from_sample(s)
        target5 = read_mask(s.post_mask_path)

        [pre, post], [loc_raw, target5] = v2.resize_rgb_and_masks(
            image_list=[pre, post],
            mask_list=[loc_raw, target5],
            image_size=self.image_size,
        )

        [pre, post], [loc_raw, target5] = v2.apply_shared_augmentations(
            image_list=[pre, post],
            mask_list=[loc_raw, target5],
            training=self.training,
            image_size=self.image_size,
        )

        loc = (loc_raw > 0).astype(np.float32)

        return {
            "pre": torch.from_numpy(self._normalize(pre)).float(),
            "post": torch.from_numpy(self._normalize(post)).float(),
            "loc": torch.from_numpy(loc).float(),
            "target5": torch.from_numpy(target5.astype(np.int64)).long(),
            "stem": stem,
            "split": "idabd",
        }

    def localization_counts(self) -> Tuple[int, int]:
        pos = 0
        neg = 0
        for stem in self.stems:
            loc = self._loc_from_sample(self.samples_by_stem[stem]) > 0
            pos += int(loc.sum())
            neg += int((~loc).sum())
        return pos, neg

    def class5_counts(self) -> np.ndarray:
        counts = np.zeros(5, dtype=np.int64)
        for stem in self.stems:
            m = read_mask(self.samples_by_stem[stem].post_mask_path)
            valid = m != 255
            vals, freqs = np.unique(m[valid], return_counts=True)
            for v, f in zip(vals.tolist(), freqs.tolist()):
                if 0 <= int(v) <= 4:
                    counts[int(v)] += int(f)
        counts[counts == 0] = 1
        return counts


def make_loaders(args: argparse.Namespace):
    """Replacement for v2.make_loaders, using IDA-BD 80/10/10 splits."""
    samples = collect_idabd_samples(args.idabd_root)
    sample_map = {s.stem: s for s in samples}
    splits = prepare_or_load_splits(args, samples)

    train_ds = IDABDHRTBDADataset(args.idabd_root, sample_map, splits["train"], args.img_size, training=True)
    val_ds = IDABDHRTBDADataset(args.idabd_root, sample_map, splits["val"], args.img_size, training=False)
    test_ds = IDABDHRTBDADataset(args.idabd_root, sample_map, splits["test"], args.img_size, training=False)

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

    return train_loader, val_loader, test_loader, train_ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("IDA-BD HRTBDA v2 cascaded Phase-I-mask pipeline")

    parser.add_argument("--phase", type=str, default="both", choices=["both", "phase1", "phase2", "phase2_test", "test", "inspect_phase1"])

    parser.add_argument("--idabd-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-file", type=str, default="")
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)

    # Kept for compatibility with the original v2 summary strings.
    parser.add_argument("--train-split", type=str, nargs="+", default=["train80"])
    parser.add_argument("--val-split", type=str, default="val10")
    parser.add_argument("--test-split", type=str, default="test10")

    parser.add_argument("--resume-phase1-from", type=str, default="")
    parser.add_argument("--phase1-checkpoint", type=str, default="")
    parser.add_argument("--phase2-checkpoint", type=str, default="")
    parser.add_argument("--phase1-threshold", type=float, default=0.5)

    parser.add_argument("--phase1-epochs", type=int, default=150)
    parser.add_argument("--phase2-epochs", type=int, default=30)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)

    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--loc-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The imported v2 code expects args.xbd_root for printing only.
    args.xbd_root = args.idabd_root

    v2.set_seed(args.seed)
    # Monkey-patch only the data loader. Everything else is the original v2 cascade logic.
    v2.make_loaders = make_loaders

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "scores").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints_dir = output_dir / "checkpoints"

    phase1_ckpt = Path(args.phase1_checkpoint) if args.phase1_checkpoint else checkpoints_dir / "phase1_best.pt"
    phase2_ckpt = Path(args.phase2_checkpoint) if args.phase2_checkpoint else checkpoints_dir / "phase2_best.pt"

    print("===== IDA-BD HRTBDA V2 CASCADED PHASE-I MASK PIPELINE =====", flush=True)
    print(f"Phase: {args.phase}", flush=True)
    print(f"IDA-BD root: {args.idabd_root}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print("Split: 80% train / 10% validation / 10% test", flush=True)
    print("Architecture/training logic: imported from train_xbd_hrtbda_v2_cascaded_phase1mask.py", flush=True)
    print("Phase I: trained from scratch unless --phase1-checkpoint is provided", flush=True)
    print("Phase II: foreground-only 4-class softmax damage classification", flush=True)
    print("Final inference: Phase I mask gives localization; Phase II predicts damage inside mask", flush=True)
    print(f"Device: {device}", flush=True)
    print("===========================================================", flush=True)

    if args.phase == "inspect_phase1":
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Phase I checkpoint not found: {phase1_ckpt}")
        v2.print_phase1_checkpoint_summary(phase1_ckpt, device)
        return

    if args.phase == "phase1":
        v2.train_phase1(args, device)

    elif args.phase == "phase2":
        v2.train_phase2(args, device, phase1_ckpt)

    elif args.phase == "phase2_test":
        phase2_ckpt = v2.train_phase2(args, device, phase1_ckpt)
        v2.test_phase2(args, device, phase2_ckpt, phase1_ckpt)

    elif args.phase == "test":
        v2.test_phase2(args, device, phase2_ckpt, phase1_ckpt)

    elif args.phase == "both":
        phase1_ckpt = v2.train_phase1(args, device)
        phase2_ckpt = v2.train_phase2(args, device, phase1_ckpt)
        v2.test_phase2(args, device, phase2_ckpt, phase1_ckpt)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
