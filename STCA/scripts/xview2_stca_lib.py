from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
DAMAGE_MAP = {
    'no-damage': 1,
    'no_damage': 1,
    'undamaged': 1,
    'un-classified': 1,
    'minor-damage': 2,
    'minor_damage': 2,
    'major-damage': 3,
    'major_damage': 3,
    'destroyed': 4,
    'destruction': 4,
}


@dataclass
class XView2Sample:
    sample_id: str
    pre_image: Path
    post_image: Path
    pre_label: Optional[Path]
    post_label: Optional[Path]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


class MetricAccumulator:
    def __init__(self) -> None:
        self.loc_tp = 0
        self.loc_fp = 0
        self.loc_fn = 0
        self.cls_tp = {c: 0 for c in range(1, 5)}
        self.cls_fp = {c: 0 for c in range(1, 5)}
        self.cls_fn = {c: 0 for c in range(1, 5)}

    @staticmethod
    def _f1(tp: int, fp: int, fn: int) -> float:
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else 2.0 * tp / denom

    def update(self, loc_pred: torch.Tensor, loc_true: torch.Tensor, dam_pred: torch.Tensor, dam_true: torch.Tensor) -> None:
        loc_pred = loc_pred.bool()
        loc_true = loc_true.bool()
        self.loc_tp += int((loc_pred & loc_true).sum().item())
        self.loc_fp += int((loc_pred & ~loc_true).sum().item())
        self.loc_fn += int((~loc_pred & loc_true).sum().item())

        for c in range(1, 5):
            pred_c = dam_pred == c
            true_c = dam_true == c
            self.cls_tp[c] += int((pred_c & true_c).sum().item())
            self.cls_fp[c] += int((pred_c & ~true_c).sum().item())
            self.cls_fn[c] += int((~pred_c & true_c).sum().item())

    def summary(self) -> Dict[str, float]:
        f_loc = self._f1(self.loc_tp, self.loc_fp, self.loc_fn)
        cls_f1 = {c: self._f1(self.cls_tp[c], self.cls_fp[c], self.cls_fn[c]) for c in range(1, 5)}
        vals = [max(v, 1e-8) for v in cls_f1.values()]
        f_dam = len(vals) / sum(1.0 / v for v in vals)
        f_avg = 0.3 * f_loc + 0.7 * f_dam
        return {
            'f1_avg': 100.0 * f_avg,
            'f1_loc': 100.0 * f_loc,
            'f1_dam': 100.0 * f_dam,
            'f1_regular': 100.0 * cls_f1[1],
            'f1_minor': 100.0 * cls_f1[2],
            'f1_major': 100.0 * cls_f1[3],
            'f1_destroyed': 100.0 * cls_f1[4],
        }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"Favg={metrics['f1_avg']:.2f} | Floc={metrics['f1_loc']:.2f} | Fdam={metrics['f1_dam']:.2f} | "
        f"reg={metrics['f1_regular']:.2f} | minor={metrics['f1_minor']:.2f} | "
        f"major={metrics['f1_major']:.2f} | dest={metrics['f1_destroyed']:.2f}"
    )


def _normalize_image(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1))


def discover_xview2_samples(root: Path, split: str) -> List[XView2Sample]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f'Split directory does not exist: {split_dir}')

    pre_images: Dict[str, Path] = {}
    post_images: Dict[str, Path] = {}
    pre_labels: Dict[str, Path] = {}
    post_labels: Dict[str, Path] = {}

    for path in split_dir.rglob('*'):
        if not path.is_file():
            continue
        stem = path.stem
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            if stem.endswith('_pre_disaster'):
                pre_images[stem[:-13]] = path
            elif stem.endswith('_post_disaster'):
                post_images[stem[:-14]] = path
        elif suffix == '.json':
            if stem.endswith('_pre_disaster'):
                pre_labels[stem[:-13]] = path
            elif stem.endswith('_post_disaster'):
                post_labels[stem[:-14]] = path

    sample_ids = sorted(set(pre_images) & set(post_images))
    return [
        XView2Sample(
            sample_id=sid,
            pre_image=pre_images[sid],
            post_image=post_images[sid],
            pre_label=pre_labels.get(sid),
            post_label=post_labels.get(sid),
        )
        for sid in sample_ids
    ]


def _parse_wkt_polygon_fallback(wkt_text: str) -> List[List[Tuple[float, float]]]:
    text = wkt_text.strip()
    polygons: List[List[Tuple[float, float]]] = []

    def ring_to_points(ring_text: str) -> List[Tuple[float, float]]:
        pts = []
        for pair in ring_text.split(','):
            parts = pair.strip().split()
            if len(parts) >= 2:
                pts.append((float(parts[0]), float(parts[1])))
        return pts

    if text.upper().startswith('POLYGON'):
        matches = re.findall(r'\(\((.*?)\)\)', text)
        if matches:
            polygons.append(ring_to_points(matches[0]))
    elif text.upper().startswith('MULTIPOLYGON'):
        matches = re.findall(r'\(\((.*?)\)\)', text)
        for match in matches:
            first_ring = match.split('), (')[0]
            polygons.append(ring_to_points(first_ring))
    return polygons


def _wkt_to_polygons(wkt_text: str) -> List[List[Tuple[float, float]]]:
    try:
        from shapely import wkt as shapely_wkt  # type: ignore
        geom = shapely_wkt.loads(wkt_text)
        if geom.geom_type == 'Polygon':
            return [list(geom.exterior.coords)]
        if geom.geom_type == 'MultiPolygon':
            return [list(poly.exterior.coords) for poly in geom.geoms]
    except Exception:
        pass
    return _parse_wkt_polygon_fallback(wkt_text)


def _extract_objects(label_path: Optional[Path]) -> List[dict]:
    if label_path is None or not label_path.exists():
        return []
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        if isinstance(data.get('features'), dict) and isinstance(data['features'].get('xy'), list):
            return data['features']['xy']
        if isinstance(data.get('features'), list):
            return data['features']
        if isinstance(data.get('labels'), list):
            return data['labels']
    return []


def rasterize_damage_and_building_masks(pre_label: Optional[Path], post_label: Optional[Path], image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    loc_img = Image.new('L', (width, height), 0)
    dam_img = Image.new('L', (width, height), 0)
    draw_loc = ImageDraw.Draw(loc_img)
    draw_dam = ImageDraw.Draw(dam_img)

    for obj in _extract_objects(pre_label):
        wkt_text = obj.get('wkt') or obj.get('geometry') or ''
        if not wkt_text:
            continue
        for poly in _wkt_to_polygons(wkt_text):
            if len(poly) >= 3:
                draw_loc.polygon(poly, fill=1)

    for obj in _extract_objects(post_label):
        props = obj.get('properties', {}) if isinstance(obj, dict) else {}
        subtype = str(props.get('subtype', props.get('damage', props.get('type', '')))).lower().strip()
        damage_val = DAMAGE_MAP.get(subtype)
        if damage_val is None:
            continue
        wkt_text = obj.get('wkt') or obj.get('geometry') or ''
        if not wkt_text:
            continue
        for poly in _wkt_to_polygons(wkt_text):
            if len(poly) >= 3:
                draw_dam.polygon(poly, fill=int(damage_val))

    return np.array(loc_img, dtype=np.uint8), np.array(dam_img, dtype=np.uint8)


class XView2Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        mode: str,
        crop_size: int = 512,
        augment: bool = False,
        cache_dir: Optional[str | Path] = None,
        max_items: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.crop_size = crop_size
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.samples = discover_xview2_samples(self.root, split)
        if max_items is not None:
            self.samples = self.samples[:max_items]
        if not self.samples:
            raise RuntimeError(f'No samples found for split={split} under {self.root}')

    def __len__(self) -> int:
        return len(self.samples)

    def _read_rgb(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img.convert('RGB'))

    def _cached_masks(self, sample: XView2Sample, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_dir is None:
            return rasterize_damage_and_building_masks(sample.pre_label, sample.post_label, image_size)
        split_cache = self.cache_dir / self.split
        split_cache.mkdir(parents=True, exist_ok=True)
        cache_file = split_cache / f'{sample.sample_id}.npz'
        if cache_file.exists():
            data = np.load(cache_file)
            return data['loc'], data['dam']
        loc, dam = rasterize_damage_and_building_masks(sample.pre_label, sample.post_label, image_size)
        np.savez_compressed(cache_file, loc=loc.astype(np.uint8), dam=dam.astype(np.uint8))
        return loc, dam

    def _crop_or_pad(self, arr: np.ndarray, y: int, x: int, size: int, fill: int = 0) -> np.ndarray:
        if arr.ndim == 3:
            h, w, c = arr.shape
            out = np.full((size, size, c), fill_value=fill, dtype=arr.dtype)
        else:
            h, w = arr.shape
            out = np.full((size, size), fill_value=fill, dtype=arr.dtype)
        y2, x2 = min(y + size, h), min(x + size, w)
        out[:y2-y, :x2-x] = arr[y:y2, x:x2]
        return out

    def _joint_transform(self, pre_img: np.ndarray, post_img: Optional[np.ndarray], loc_mask: Optional[np.ndarray], dam_mask: Optional[np.ndarray]):
        h, w = pre_img.shape[:2]
        size = self.crop_size
        if self.augment:
            y = random.randint(0, max(h - size, 0))
            x = random.randint(0, max(w - size, 0))
        else:
            y = max((h - size) // 2, 0)
            x = max((w - size) // 2, 0)

        pre_img = self._crop_or_pad(pre_img, y, x, size, fill=0)
        if post_img is not None:
            post_img = self._crop_or_pad(post_img, y, x, size, fill=0)
        if loc_mask is not None:
            loc_mask = self._crop_or_pad(loc_mask, y, x, size, fill=0)
        if dam_mask is not None:
            dam_mask = self._crop_or_pad(dam_mask, y, x, size, fill=0)

        if self.augment:
            if random.random() < 0.5:
                pre_img = np.flip(pre_img, axis=1).copy()
                if post_img is not None: post_img = np.flip(post_img, axis=1).copy()
                if loc_mask is not None: loc_mask = np.flip(loc_mask, axis=1).copy()
                if dam_mask is not None: dam_mask = np.flip(dam_mask, axis=1).copy()
            if random.random() < 0.5:
                pre_img = np.flip(pre_img, axis=0).copy()
                if post_img is not None: post_img = np.flip(post_img, axis=0).copy()
                if loc_mask is not None: loc_mask = np.flip(loc_mask, axis=0).copy()
                if dam_mask is not None: dam_mask = np.flip(dam_mask, axis=0).copy()
            k = random.randint(0, 3)
            if k:
                pre_img = np.rot90(pre_img, k).copy()
                if post_img is not None: post_img = np.rot90(post_img, k).copy()
                if loc_mask is not None: loc_mask = np.rot90(loc_mask, k).copy()
                if dam_mask is not None: dam_mask = np.rot90(dam_mask, k).copy()
        return pre_img, post_img, loc_mask, dam_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        pre_img = self._read_rgb(sample.pre_image)
        h, w = pre_img.shape[:2]
        if self.mode == 'target_pre':
            pre_img, _, _, _ = self._joint_transform(pre_img, None, None, None)
            return {'pre': _normalize_image(pre_img), 'id': sample.sample_id}

        post_img = self._read_rgb(sample.post_image)
        loc_mask, dam_mask = self._cached_masks(sample, (w, h))
        pre_img, post_img, loc_mask, dam_mask = self._joint_transform(pre_img, post_img, loc_mask, dam_mask)
        base = {
            'pre': _normalize_image(pre_img),
            'post': _normalize_image(post_img),
            'loc': torch.from_numpy(loc_mask.astype(np.float32)),
            'dam': torch.from_numpy(dam_mask.astype(np.int64)),
            'id': sample.sample_id,
        }
        if self.mode == 'source_post':
            return {'post': base['post'], 'dam': base['dam'], 'id': base['id']}
        return base


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class TinyResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicResidualBlock(64, 64), BasicResidualBlock(64, 64))
        self.layer2 = nn.Sequential(BasicResidualBlock(64, 128, stride=2), BasicResidualBlock(128, 128))
        self.layer3 = nn.Sequential(BasicResidualBlock(128, 256, stride=2), BasicResidualBlock(256, 256))
        self.layer4 = nn.Sequential(BasicResidualBlock(256, 512, stride=2), BasicResidualBlock(512, 512))
        self.out_channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FeatureDecoder(nn.Module):
    def __init__(self, channels: Sequence[int], feat_dim: int = 64):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.d3 = DecoderBlock(c4, c3, 256)
        self.d2 = DecoderBlock(256, c2, 128)
        self.d1 = DecoderBlock(128, c1, 64)
        self.proj = nn.Sequential(ConvBNReLU(64, feat_dim), nn.Conv2d(feat_dim, feat_dim, 1))

    def forward(self, feats):
        x0, x1, x2, x3, x4 = feats
        y = self.d3(x4, x3)
        y = self.d2(y, x2)
        y = self.d1(y, x1)
        return self.proj(y)


class PairMLPClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleSTCANet(nn.Module):
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.encoder = TinyResNet18Encoder()
        self.decoder = FeatureDecoder(self.encoder.out_channels, feat_dim=feat_dim)
        self.loc_head = nn.Conv2d(feat_dim, 1, 1)
        self.dam_head = nn.Sequential(ConvBNReLU(feat_dim * 3, feat_dim), nn.Conv2d(feat_dim, 5, 1))
        self.pair_classifier = PairMLPClassifier(feat_dim=feat_dim, num_classes=5)

    def encode_pre(self, pre: torch.Tensor):
        feats = self.encoder(pre)
        pre_feat = self.decoder(feats)
        loc_low = self.loc_head(pre_feat)
        return pre_feat, loc_low

    def encode_post(self, post: torch.Tensor):
        feats = self.encoder(post)
        return self.decoder(feats)

    def dense_damage_logits(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        pair_feat = torch.cat([pre_feat, post_feat, torch.abs(pre_feat - post_feat)], dim=1)
        return self.dam_head(pair_feat)

    def classify_pair_features(self, src_post_feats: torch.Tensor, tgt_pre_feats: torch.Tensor) -> torch.Tensor:
        ks, c = src_post_feats.shape
        kt, _ = tgt_pre_feats.shape
        src_e = src_post_feats[:, None, :].expand(ks, kt, c)
        tgt_e = tgt_pre_feats[None, :, :].expand(ks, kt, c)
        pair = torch.cat([tgt_e, src_e, torch.abs(tgt_e - src_e)], dim=-1).reshape(ks * kt, 3 * c)
        return self.pair_classifier(pair)

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        pre_feat, loc_low = self.encode_pre(pre)
        post_feat = self.encode_post(post)
        dam_low = self.dense_damage_logits(pre_feat, post_feat)
        return {
            'loc_logits': F.interpolate(loc_low, size=pre.shape[-2:], mode='bilinear', align_corners=False),
            'dam_logits': F.interpolate(dam_low, size=pre.shape[-2:], mode='bilinear', align_corners=False),
            'loc_low': loc_low,
            'pre_feat': pre_feat,
            'post_feat': post_feat,
        }


def build_model(device: torch.device, feat_dim: int = 64) -> SimpleSTCANet:
    return SimpleSTCANet(feat_dim=feat_dim).to(device)


def create_optimizer(model: nn.Module, lr: float, weight_decay: float = 1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt


def evaluate_model(model: nn.Module, loader, device: torch.device):
    model.eval()
    meter = MetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            pre = batch['pre'].to(device)
            post = batch['post'].to(device)
            loc_true = batch['loc'].to(device)
            dam_true = batch['dam'].to(device)
            out = model(pre, post)
            loc_pred = torch.sigmoid(out['loc_logits']).squeeze(1) > 0.5
            dam_pred = out['dam_logits'].argmax(dim=1)
            meter.update(loc_pred.cpu(), loc_true.cpu() > 0.5, dam_pred.cpu(), dam_true.cpu())
    return meter.summary()


def sample_target_building_features(pre_feat: torch.Tensor, loc_low_logits: torch.Tensor, threshold: float, max_samples: int):
    prob = torch.sigmoid(loc_low_logits).squeeze(1)
    feats = []
    for b in range(pre_feat.size(0)):
        idx = torch.nonzero(prob[b] > threshold, as_tuple=False)
        if idx.numel() == 0:
            continue
        if idx.size(0) > max_samples:
            idx = idx[torch.randperm(idx.size(0), device=idx.device)[:max_samples]]
        feats.append(pre_feat[b, :, idx[:, 0], idx[:, 1]].T)
    if not feats:
        return None
    return torch.cat(feats, dim=0)


def sample_source_damage_features(post_feat: torch.Tensor, dam_mask: torch.Tensor, max_samples: int, include_classes: Sequence[int] = (2, 3, 4)):
    low_mask = F.interpolate(dam_mask.unsqueeze(1).float(), size=post_feat.shape[-2:], mode='nearest').squeeze(1).long()
    feat_list, label_list = [], []
    quota = max(1, max_samples // max(len(include_classes), 1))
    for b in range(post_feat.size(0)):
        for cls in include_classes:
            idx = torch.nonzero(low_mask[b] == cls, as_tuple=False)
            if idx.numel() == 0:
                continue
            if idx.size(0) > quota:
                idx = idx[torch.randperm(idx.size(0), device=idx.device)[:quota]]
            feat_list.append(post_feat[b, :, idx[:, 0], idx[:, 1]].T)
            label_list.append(torch.full((idx.size(0),), cls, device=idx.device, dtype=torch.long))
    if not feat_list:
        return None, None
    return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0)


class BCEPlusCELoss(nn.Module):
    def __init__(self, loc_weight: float = 1.0, dam_weight: float = 1.0):
        super().__init__()
        self.loc_weight = loc_weight
        self.dam_weight = dam_weight
        self.loc_loss = nn.BCEWithLogitsLoss()
        self.dam_loss = nn.CrossEntropyLoss()

    def forward(self, loc_logits: torch.Tensor, loc_true: torch.Tensor, dam_logits: torch.Tensor, dam_true: torch.Tensor):
        loss_loc = self.loc_loss(loc_logits.squeeze(1), loc_true)
        loss_dam = self.dam_loss(dam_logits, dam_true)
        return self.loc_weight * loss_loc + self.dam_weight * loss_dam
