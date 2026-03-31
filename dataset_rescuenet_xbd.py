from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    loc_path: Path
    dmg_path: Path


class RescueNetXBDDataset(Dataset):
    """
    Dataset wrapper for RescueNet-xBD arranged as:

        rescuenet_xbd/
          train/
            images/
            masks/localization/
            masks/damage/
          val/
            images/
            masks/localization/
            masks/damage/
          test/
            images/
            masks/localization/
            masks/damage/

    This dataset only exposes a single image per sample.
    DisasterAdaptiveNet expects a 6-channel tensor made from a pre/post pair,
    so we duplicate the same RGB image and concatenate it with itself.

    Returned item fields:
      - img: float tensor [6, H, W]
      - loc: float tensor [H, W] with values {0, 1}
      - dmg: long tensor [H, W] with values {0,1,2,3,255}
             where:
               0 = no-damage
               1 = minor
               2 = major
               3 = destroyed
               255 = ignore
      - cond_id: long tensor [1]
      - stem: file stem
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 512,
        training: bool = False,
        conditioning_id: int = 0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.training = bool(training)
        self.conditioning_id = int(conditioning_id)
        self.normalize = bool(normalize)

        split_root = self.root / split
        self.images_dir = split_root / "images"
        self.loc_dir = split_root / "masks" / "localization"
        self.dmg_dir = split_root / "masks" / "damage"

        for p in (self.images_dir, self.loc_dir, self.dmg_dir):
            if not p.exists():
                raise FileNotFoundError(f"Expected path not found: {p}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No matched image/mask samples found under {split_root}")

        self._mean = np.array([0.485, 0.456, 0.406] * 2, dtype=np.float32)[:, None, None]
        self._std = np.array([0.229, 0.224, 0.225] * 2, dtype=np.float32)[:, None, None]

    def _collect_samples(self) -> List[Sample]:
        image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
            key=lambda p: p.stem,
        )

        samples: List[Sample] = []
        for image_path in image_files:
            stem = image_path.stem
            loc_path = self.loc_dir / f"{stem}.png"
            dmg_path = self.dmg_dir / f"{stem}.png"
            if not loc_path.exists() or not dmg_path.exists():
                continue
            samples.append(Sample(stem=stem, image_path=image_path, loc_path=loc_path, dmg_path=dmg_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image_rgb(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    def _build_damage_target(self, loc: np.ndarray, dmg: np.ndarray) -> np.ndarray:
        loc_bin = (loc > 0)

        # Damage head predicts only building damage classes.
        # 0=no damage, 1=minor, 2=major, 3=destroyed, 255=ignore.
        target = np.full(loc.shape, 255, dtype=np.uint8)

        # Only assign labels on building pixels.
        target[(dmg == 1) & loc_bin] = 0
        target[(dmg == 2) & loc_bin] = 1
        target[(dmg == 3) & loc_bin] = 2
        target[(dmg == 4) & loc_bin] = 3

        # Keep void / outside-building pixels as ignore.
        return target

    def _resize(
        self,
        image: np.ndarray,
        loc: np.ndarray,
        dmg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if image.shape[0] == self.image_size and image.shape[1] == self.image_size:
            return image, loc, dmg
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        loc = cv2.resize(loc, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        dmg = cv2.resize(dmg, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return image, loc, dmg

    def _augment(
        self,
        image: np.ndarray,
        loc: np.ndarray,
        dmg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.training:
            return image, loc, dmg

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            loc = np.flip(loc, axis=1).copy()
            dmg = np.flip(dmg, axis=1).copy()

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
            loc = np.flip(loc, axis=0).copy()
            dmg = np.flip(dmg, axis=0).copy()

        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k=k).copy()
            loc = np.rot90(loc, k=k).copy()
            dmg = np.rot90(dmg, k=k).copy()

        return image, loc, dmg

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        image = self._read_image_rgb(sample.image_path)
        loc = self._read_mask(sample.loc_path)
        dmg = self._read_mask(sample.dmg_path)

        image, loc, dmg = self._resize(image, loc, dmg)
        image, loc, dmg = self._augment(image, loc, dmg)

        loc = (loc > 0).astype(np.float32)
        dmg = self._build_damage_target(loc=loc, dmg=dmg)

        # Duplicate RGB image to create a fake [pre, post] 6-channel tensor.
        image = image.astype(np.float32) / 255.0
        image_6ch = np.concatenate([image, image], axis=2)  # [H, W, 6]
        image_6ch = image_6ch.transpose(2, 0, 1)            # [6, H, W]

        if self.normalize:
            image_6ch = (image_6ch - self._mean) / self._std

        item: Dict[str, torch.Tensor | str] = {
            "img": torch.from_numpy(image_6ch).float(),
            "loc": torch.from_numpy(loc).float(),
            "dmg": torch.from_numpy(dmg).long(),
            "cond_id": torch.tensor([self.conditioning_id], dtype=torch.long),
            "stem": sample.stem,
        }
        return item

    def get_damage_class_counts(self) -> np.ndarray:
        counts = np.zeros(4, dtype=np.int64)
        for sample in self.samples:
            loc = self._read_mask(sample.loc_path)
            dmg = self._read_mask(sample.dmg_path)
            if loc.shape != dmg.shape:
                loc = cv2.resize(loc, (dmg.shape[1], dmg.shape[0]), interpolation=cv2.INTER_NEAREST)
            target = self._build_damage_target((loc > 0).astype(np.float32), dmg)
            valid = target != 255
            if valid.any():
                vals, freqs = np.unique(target[valid], return_counts=True)
                for v, f in zip(vals.tolist(), freqs.tolist()):
                    counts[int(v)] += int(f)
        return counts

    def get_localization_pixel_counts(self) -> Tuple[int, int]:
        pos = 0
        neg = 0
        for sample in self.samples:
            loc = self._read_mask(sample.loc_path)
            loc_bin = (loc > 0)
            pos += int(loc_bin.sum())
            neg += int((~loc_bin).sum())
        return pos, neg


__all__ = ["RescueNetXBDDataset"]