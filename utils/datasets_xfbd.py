from typing import Sequence, Dict, Any, Union
from pathlib import Path

import cv2
import numpy as np
import torch

from utils import augmentations, helpers
from utils.experiment_manager import CfgNode


class xFBDDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode, run_type: str, disable_augmentations: bool = False):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.metadata = helpers.load_json(self.root_path / "metadata.json")
        self.run_type = run_type

        augs = True if (run_type == "train" and not disable_augmentations) else False
        self.transforms = augmentations.compose_transformations(cfg, augs_enabled=augs)

        self.samples = list(self.metadata[run_type]["patches"])
        self.length = len(self.samples)
        self.n_dmg_classes = 4

    def load_images(self, subset: str, event: str, patch_id: str) -> Sequence[np.ndarray]:
        img_pre_file = self.root_path / subset / "images" / f"{event}_{patch_id}_pre_disaster.png"
        img_post_file = self.root_path / subset / "images" / f"{event}_{patch_id}_post_disaster.png"

        img_pre = cv2.imread(str(img_pre_file), cv2.IMREAD_COLOR)
        img_post = cv2.imread(str(img_post_file), cv2.IMREAD_COLOR)

        if img_pre is None:
            raise FileNotFoundError(f"Could not read pre-image: {img_pre_file}")
        if img_post is None:
            raise FileNotFoundError(f"Could not read post-image: {img_post_file}")

        return img_pre, img_post

    def load_masks(self, subset: str, event: str, patch_id: str) -> Sequence[np.ndarray]:
        msk_pre_file = self.root_path / subset / "masks" / f"{event}_{patch_id}_pre_disaster.png"
        msk_post_file = self.root_path / subset / "masks" / f"{event}_{patch_id}_post_disaster.png"

        msk_pre = cv2.imread(str(msk_pre_file), cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(str(msk_post_file), cv2.IMREAD_UNCHANGED)

        if msk_pre is None:
            raise FileNotFoundError(f"Could not read pre-mask: {msk_pre_file}")
        if msk_post is None:
            raise FileNotFoundError(f"Could not read post-mask: {msk_post_file}")

        msk_pre = msk_pre.astype(np.float32) / 255
        return msk_pre, msk_post

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Any, str]]:
        sample = self.samples[index]
        event, patch_id, subset = sample["event"], sample["patch_id"], sample["subset"]

        img_pre, img_post = self.load_images(subset, event, patch_id)
        img = np.concatenate([img_pre, img_post], axis=2)

        msk_loc, msk_dmg = self.load_masks(subset, event, patch_id)
        msk = np.stack((msk_loc, msk_dmg), axis=-1)

        img, msk = self.transforms((img, msk))

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).bool()

        item = {
            "img": img,
            "msk": msk,
            "event": event,
            "patch_id": patch_id,
            "subset": subset,
        }

        if self.cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION:
            cond_attr = str(self.cfg.DATASET.EVENT_CONDITIONING[event]).lower()
            cond_key = {str(k).lower(): v for k, v in self.cfg.DATASET.CONDITIONING_KEY.items()}
            cond_id = int(cond_key[cond_attr])
            item["cond_id"] = torch.tensor([cond_id]).long()

        return item

    def get_class_counts(self) -> Sequence[int]:
        class_counts = [0, 0, 0, 0, 0]
        for sample in self.samples:
            class_counts[0] += sample["loc"]
            for i in range(1, 5):
                class_counts[i] += sample[f"cls_{i}"]
        return class_counts

    def __len__(self):
        return self.length

    def __str__(self):
        return f"xFBDDataset with {self.length} samples."