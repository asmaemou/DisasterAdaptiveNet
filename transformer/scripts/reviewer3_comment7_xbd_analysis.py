#!/usr/bin/env python3
"""Reviewer #3 Comment 7: xBD acquisition-change and damage false-positive audit.

Evaluate two ORIGINAL trained xBD checkpoints, without retraining, with identical
xBD preprocessing and checkpoint-specific localization thresholds. Produce:
  * all-undamaged BUILDING-level false-positive damage rate per model;
  * all-undamaged PIXEL-level false-positive damage rate (separately labeled);
  * a paired comparison and predictions for a previously prepared manual-review set;
  * illustrative pre/post and model-output crops;
  * screening metadata and a detailed reproducibility report.

IMPORTANT: No algorithm can establish that illumination caused a false positive.
Manual inspection is required for significant visual change and attribution.
Use on a SLURM GPU compute node. Requires torch, cv2, numpy, Pillow and the
original DisasterAdaptiveNet repository. Never edits checkpoint files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DEFAULT = Path('/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet')
DATA_DEFAULT = Path('/homes/j244s673/documents/wsu/phd/xview2')
MODEL_DIR_FULL = 'BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3'
MODEL_DIR_NOCA = 'RSBDNet_ablation_xBD/no_cross_attention'
SIZE = 896
EXPECTED_PAIRS = 933
EXPECTED_UNDAMAGED_POLYGONS = 41427
# Original xBD test score JSON values as supplied, used ONLY for reproduction QA.
REFERENCE_TP = {'full': 30778610, 'no_cross_attention': 30741278}
REFERENCE_FN = {'full': 2806590, 'no_cross_attention': 2843922}


def say(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S]'), message, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, columns):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta_angle(v1, v2, circular=False):
    a, b = numeric(v1), numeric(v2)
    if a is None or b is None:
        return None
    diff = abs(a - b)
    return min(diff % 360.0, 360.0 - diff % 360.0) if circular else diff


def is_yes(value):
    return str(value).strip().lower() in {'yes', 'y', '1', 'true'}


def review_status(value):
    v = str(value).strip().lower()
    if v in {'yes', 'y', '1', 'true'}:
        return 'yes'
    if v in {'no', 'n', '0', 'false'}:
        return 'no'
    return 'pending'


def read_manual_sheet(path):
    if not path.is_file():
        say(f'No manual worksheet at {path}; visual-change count remains PENDING')
        return {}
    results = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            key = (str(row['image_id']).strip(), int(row['feature_index']))
            if key in results:
                raise ValueError(f'Duplicate manual worksheet case: {key}')
            results[key] = row
    say(f'Manual worksheet loaded: {len(results)} candidate cases')
    return results


# xBD WKT geometries are normally simple POLYGONs.  For geometries with
# multiple exterior/interior rings, use shapely if installed; fail explicitly
# rather than knowingly accepting incorrect polygon regions.
NUMBER = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')


def wkt_outer_ring(wkt):
    text = str(wkt).strip()
    if not text.upper().startswith('POLYGON'):
        raise ValueError(f'Unsupported xBD geometry (needs POLYGON): {text[:100]}')
    start = text.find('((')
    if start < 0:
        raise ValueError(f'Invalid WKT: {text[:100]}')
    # Standard xBD simple POLYGON((x y, ...)); reject holes for correctness.
    end = text.find('))', start)
    if end < 0:
        raise ValueError(f'Invalid polygon WKT: {text[:100]}')
    inside = text[start + 2:end]
    if '), (' in inside or '),(' in inside:
        raise ValueError('Interior rings require geometry handling; no silent omission')
    coords = []
    for point in inside.split(','):
        xy = NUMBER.findall(point)
        if len(xy) != 2:
            raise ValueError(f'Unsupported WKT coordinate: {point!r}')
        coords.append((float(xy[0]), float(xy[1])))
    if len(coords) < 3:
        raise ValueError('WKT polygon has fewer than 3 vertices')
    return coords


def polygon_mask(coords, width, height, target, cv2, np):
    # Rasterize at the annotation's ORIGINAL resolution, then resize nearest,
    # as the original xBD masks are also resized by the project dataset loader.
    raster = np.zeros((height, width), dtype=np.uint8)
    vertices = np.rint(np.asarray(coords)).astype(np.int32)
    cv2.fillPoly(raster, [vertices], color=1)
    if (width, height) != (target, target):
        raster = cv2.resize(raster, (target, target), interpolation=cv2.INTER_NEAREST)
    return raster > 0


def load_model(kind, checkpoint_path, device, torch):
    from transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal import (
        BuildingGuidedCrossAttentionOrdinalNet,
    )

    class TemporalIdentity(torch.nn.Module):
        def forward(self, pre, post):
            return pre, post

    say(f'Loading {kind} checkpoint from {checkpoint_path}')
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError('Expected checkpoint dict with a model state_dict')
    if 'best_threshold' not in checkpoint:
        raise ValueError('Missing validation-selected best_threshold in checkpoint')
    threshold = float(checkpoint['best_threshold'])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f'Invalid best_threshold: {threshold}')
    checkpoint_size = (checkpoint.get('args') or {}).get('img_size')
    if checkpoint_size is not None and int(checkpoint_size) != SIZE:
        raise ValueError(f'Checkpoint image size {checkpoint_size} != {SIZE}')

    model = BuildingGuidedCrossAttentionOrdinalNet(image_size=SIZE, width=96)
    if kind == 'no_cross_attention':
        # Source architecture calls these modules with TWO features and expects
        # TWO outputs.  An ordinary nn.Identity would be wrong here.
        model.res_cross_attention = TemporalIdentity()
        model.swin_cross_attention = TemporalIdentity()
        if any('cross_attention.' in key for key in checkpoint['model']):
            raise RuntimeError('Ablated checkpoint contains cross-attention weights. '
                               'Its architecture differs from the expected identity-pair ablation.')
    else:
        if not any('res_cross_attention.' in key for key in checkpoint['model']):
            raise RuntimeError('Full-model checkpoint lacks ResNet cross-attention weights')

    actual_keys = set(model.state_dict())
    ckpt_keys = set(checkpoint['model'])
    missing = sorted(actual_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - actual_keys)
    if missing or unexpected:
        raise RuntimeError(
            f'{kind} checkpoint is NOT compatible with inferred architecture. '
            f'First missing keys: {missing[:12]}; unexpected keys: {unexpected[:12]}. '
            'STOP. Do not use any metrics until the training ablation class is inspected.'
        )
    model.load_state_dict(checkpoint['model'], strict=True)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    expected_m = 55.304 if kind == 'full' else 55.154
    if abs(params - expected_m) > 0.020:
        raise RuntimeError(f'{kind}: model parameter count {params:.3f}M does not '
                           f'match expected ~{expected_m:.3f}M; wrong ablation architecture')
    model = model.to(device).eval().requires_grad_(False)
    say(f'{kind}: strict weight load OK; {params:.3f}M params; threshold={threshold:.4f}')
    metadata = {
        'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
        'checkpoint_threshold': threshold, 'parameters_million': round(params, 6),
        'strict_state_dict_load': True,
    }
    del checkpoint
    return model, threshold, metadata


def mask_to_pred(logits, threshold, torch):
    if logits.ndim != 4 or tuple(logits.shape[1:]) != (8, SIZE, SIZE):
        raise RuntimeError(f'Unexpected RS-BDNet logits shape {tuple(logits.shape)}')
    building = torch.sigmoid(logits[:, 0]) > threshold
    # 0 = background / no predicted building, 1 = undamaged, 2..4 = damaged.
    return ((logits[:, 1:5].argmax(dim=1) + 1) * building.long())[0].to(torch.uint8).cpu().numpy()


def make_example_figures(out, reviewed, data_samples, pred_paths, PIL_Image, PIL_Draw, np):
    # Case figures show ground-truth footprint location and both model outcomes.
    from PIL import Image, ImageDraw
    figures = out / 'representative_cases'
    figures.mkdir(parents=True, exist_ok=True)
    colors = np.asarray([
        (32, 36, 42),  # not predicted as a building
        (65, 165, 93),  # no damage
        (240, 202, 74), # minor
        (238, 128, 42), # major
        (203, 62, 64),  # destroyed
    ], dtype=np.uint8)
    produced = 0
    for key, row in sorted(reviewed.items(), key=lambda kv: int(kv[1].get('case_number', '9999') or '9999')):
        stem, idx = key
        sample = data_samples.get(stem)
        if sample is None:
            continue
        full_path = pred_paths['full'].get(stem)
        no_path = pred_paths['no_cross_attention'].get(stem)
        if full_path is None or no_path is None:
            continue
        annotation_path = Path(sample['annotation'])
        features = json.loads(annotation_path.read_text())['features']['xy']
        if idx >= len(features):
            continue
        pts = wkt_outer_ring(features[idx]['wkt'])
        with Image.open(sample['pre']) as im:
            pre = im.convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        with Image.open(sample['post']) as im:
            post = im.convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        with Image.open(full_path) as im:
            full = np.asarray(im, dtype=np.uint8)
        with Image.open(no_path) as im:
            no = np.asarray(im, dtype=np.uint8)
        rgb_full = Image.fromarray(colors[full], 'RGB')
        rgb_no = Image.fromarray(colors[no], 'RGB')
        src_w, src_h = sample['native_size']
        xs = [x * SIZE / src_w for x, _ in pts]
        ys = [y * SIZE / src_h for _, y in pts]
        cx = sum([min(xs), max(xs)]) / 2
        cy = sum([min(ys), max(ys)]) / 2
        span = int(max(160, min(440, max(max(xs)-min(xs), max(ys)-min(ys)) + 90)))
        left, top = int(cx - span / 2), int(cy - span / 2)
        region = (left, top, left + span, top + span)
        tile = 240
        margin = 15
        width = 4 * (tile + margin) + margin
        sheet = Image.new('RGB', (width, tile + 75), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, (img, title) in enumerate(zip(
            [pre, post, rgb_full, rgb_no],
            ['PRE', 'POST', 'FULL MODEL', 'NO CROSS-ATTN'])):
            crop = img.crop(region).resize((tile, tile), Image.Resampling.BILINEAR if j < 2 else Image.Resampling.NEAREST)
            x = margin + j * (tile + margin)
            sheet.paste(crop, (x, 35))
            cd = ImageDraw.Draw(sheet)
            cd.text((x, 12), title, fill='black')
            polygon = [
                (x + (p_x - left) * tile / span, 35 + (p_y - top) * tile / span)
                for p_x, p_y in zip(xs, ys)
            ]
            if len(polygon) >= 3:
                cd.line(polygon + [polygon[0]], fill=(250, 0, 250), width=3)
        case_no = int(row.get('case_number') or 0)
        draw.text((margin, tile + 49), f'Case {case_no} | {stem} | WKT feature {idx} | visual review: {review_status(row.get("significant_visual_change", ""))}', fill='black')
        path = figures / f'case_{case_no:03d}_{stem}_polygon_{idx}.png'
        sheet.save(path)
        produced += 1
    say(f'Saved {produced} 4-panel case images to {figures}')
    return produced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_DEFAULT)
    parser.add_argument('--xbd-root', type=Path, default=DATA_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DEFAULT / 'output/Reviewer3_Comment7')
    parser.add_argument('--manual-sheet', type=Path, default=None,
                        help='Default: OUTPUT_DIR/manual_review_worksheet.csv')
    parser.add_argument('--full-checkpoint', type=Path, default=None)
    parser.add_argument('--no-cross-checkpoint', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=0,
                        help='0=entire 933-pair test set; positive integer=SMOKE TEST, not publishable')
    parser.add_argument('--disable-cudnn', action='store_true', help='For debugging prior CUDA stalls')
    parser.add_argument('--no-reproduction-check', action='store_true',
                        help='Diagnostics ONLY. Do not report unvalidated numbers in manuscript.')
    args = parser.parse_args()

    project = args.project_root.resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    if not (project / 'transformer').is_dir():
        raise RuntimeError(f'Not a DisasterAdaptiveNet project root: {project}')
    args.xbd_root = args.xbd_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    review_path = args.manual_sheet or out / 'manual_review_worksheet.csv'
    review = read_manual_sheet(review_path)

    import numpy as np
    import cv2
    import torch
    from PIL import Image, ImageDraw
    from transformer.scripts.train_xbd_supervised_disasteradaptivenet import XBDOriginalDataset

    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. Run via the supplied SLURM sbatch, NOT on the head node.')
    cv2.setNumThreads(0)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    say(f'GPU={torch.cuda.get_device_name(0)}; torch={torch.__version__}; cv2={cv2.__version__}')
    say(f'cuDNN enabled={torch.backends.cudnn.enabled}; output={out}')

    dataset = XBDOriginalDataset(root=args.xbd_root, split='test', image_size=SIZE,
                                 training=False, conditioning_id=0)
    if len(dataset) != EXPECTED_PAIRS:
        raise RuntimeError(f'Expected {EXPECTED_PAIRS} test pairs; found {len(dataset)}. Abort.')
    n = min(len(dataset), args.limit) if args.limit > 0 else len(dataset)
    if args.limit > 0:
        say('SMOKE TEST only: --limit supplied; statistics are NOT FOR PUBLICATION')
    say(f'Found {len(dataset)} xBD test pairs; processing {n} for EACH model')

    checkpoints = {
        'full': args.full_checkpoint or project / 'output' / MODEL_DIR_FULL / 'checkpoints/best.pt',
        'no_cross_attention': args.no_cross_checkpoint or project / 'output' / MODEL_DIR_NOCA / 'checkpoints/best.pt',
    }
    for name, path in checkpoints.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f'{name} checkpoint not found: {path}')

    # Index 0..n-1 first, then ensure every annotation file is available.
    samples = {}
    for item in dataset.samples[:n]:
        stem = item.stem
        ann = args.xbd_root / 'test/labels' / f'{stem}_post_disaster.json'
        pre_ann = args.xbd_root / 'test/labels' / f'{stem}_pre_disaster.json'
        if not (ann.is_file() and pre_ann.is_file()):
            raise RuntimeError(f'Missing paired xBD metadata: {stem}')
        with Image.open(item.post_image_path) as im:
            size = im.size
        samples[stem] = {
            'pre': str(item.pre_image_path), 'post': str(item.post_image_path),
            'annotation': str(ann), 'pre_annotation': str(pre_ann),
            'native_size': size,
        }
    if len(samples) != n:
        raise RuntimeError('Duplicate xBD sample stems; cannot match predictions to polygons')
    if any(stem not in samples for stem, _ in review):
        say('NOTE: some manual-review examples lie outside --limit; they will be skipped')

    # Record one row per no-damage polygon, and preserve exact case index from
    # the previous manually generated worksheet.
    building_rows = {}
    predictions_for_examples = {'full': {}, 'no_cross_attention': {}}
    export_maps_dir = out / 'example_prediction_masks'
    model_stats = {}
    model_meta = {}
    no_damage_polygon_counts = {}
    pair_screen = []
    for kind in ('full', 'no_cross_attention'):
        model, threshold, meta = load_model(kind, Path(checkpoints[kind]), device, torch)
        model_meta[kind] = meta
        stats = Counter()
        start = time.monotonic()
        folder = export_maps_dir / kind
        folder.mkdir(parents=True, exist_ok=True)
        try:
            for index in range(n):
                sample = dataset[index]
                stem = str(sample['stem'])
                if stem not in samples:
                    raise RuntimeError(f'Unknown dataset stem at {index}: {stem}')
                x = sample['img'].unsqueeze(0).to(device, non_blocking=True)
                cond = sample['cond_id'].unsqueeze(0).to(device, non_blocking=True)
                with torch.inference_mode():
                    logits = model(x, cond)
                    pred = mask_to_pred(logits, threshold, torch)
                del logits, x, cond
                gt = sample['dmg'].numpy()  # 0=no damage; 1-3=damage; 255=ignore
                if pred.shape != gt.shape:
                    raise RuntimeError(f'Output/target shape mismatch for {stem}')
                undamaged_pixels = gt == 0
                # Simple ranking score (mean absolute RGB difference, 0..255),
                # NOT a validated imaging artifact or an FPR. Compute for every
                # no-damage polygon in the entire test set, not only 24 examples.
                rgb = sample['img'].numpy()
                channel_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
                absolute_rgb_difference = (
                    np.abs(rgb[:3] - rgb[3:6]) * channel_std * 255.0
                ).mean(axis=0)
                stats['gt_undamaged_pixels'] += int(undamaged_pixels.sum())
                stats['no_damage_tp'] += int(((pred == 1) & undamaged_pixels).sum())
                stats['no_damage_fn'] += int(((pred != 1) & undamaged_pixels).sum())
                stats['pixel_damage_fp'] += int(((pred >= 2) & undamaged_pixels).sum())
                stats['pixel_missed_localization'] += int(((pred == 0) & undamaged_pixels).sum())

                post_json = json.loads(Path(samples[stem]['annotation']).read_text())
                pre_json = json.loads(Path(samples[stem]['pre_annotation']).read_text())
                post_meta = post_json.get('metadata', {})
                pre_meta = pre_json.get('metadata', {})
                az = delta_angle(pre_meta.get('sun_azimuth'), post_meta.get('sun_azimuth'), True)
                el = delta_angle(pre_meta.get('sun_elevation'), post_meta.get('sun_elevation'))
                nad = delta_angle(pre_meta.get('off_nadir_angle'), post_meta.get('off_nadir_angle'))
                if kind == 'full':
                    pair_screen.append({
                        'image_id': stem, 'disaster': post_meta.get('disaster', ''),
                        'delta_sun_azimuth_deg': az, 'delta_sun_elevation_deg': el,
                        'delta_off_nadir_deg': nad,
                    })
                feats = post_json.get('features', {}).get('xy', [])
                no_damage = [
                    (i, f) for i, f in enumerate(feats)
                    if str(f.get('properties', {}).get('subtype', '')).lower().replace('_', '-') == 'no-damage'
                ]
                stats['undamaged_polygons_total'] += len(no_damage)
                w, h = samples[stem]['native_size']
                for idx, feature in no_damage:
                    key = (stem, idx)
                    poly = polygon_mask(
                        wkt_outer_ring(feature['wkt']), w, h, SIZE, cv2, np
                    )
                    valid_poly = poly & undamaged_pixels
                    pixel_count = int(valid_poly.sum())
                    if kind == 'full':
                        props = feature.get('properties', {})
                        row = {
                            'image_id': stem, 'disaster': post_meta.get('disaster', ''),
                            'feature_index': idx, 'building_uid': props.get('uid', ''),
                            'gt_label': 'no-damage', 'gt_valid_polygon_pixels': pixel_count,
                            'mean_abs_rgb_difference_0_255': (
                                round(float(absolute_rgb_difference[valid_poly].mean()), 3)
                                if pixel_count else ''
                            ),
                            'delta_sun_azimuth_deg': az,
                            'delta_sun_elevation_deg': el,
                            'delta_off_nadir_deg': nad,
                        }
                        manual = review.get(key)
                        if manual:
                            row.update({
                                'manual_case': manual.get('case_number', ''),
                                'manual_visual_change': review_status(manual.get('significant_visual_change', '')),
                                'manual_possible_cause': manual.get('possible_cause', ''),
                                'manual_notes': manual.get('review_notes', ''),
                                'rgb_change_score_from_shortlist': manual.get('rgb_change_score', ''),
                            })
                        else:
                            row.update({'manual_case': '', 'manual_visual_change': 'not_selected',
                                        'manual_possible_cause': '', 'manual_notes': '',
                                        'rgb_change_score_from_shortlist': ''})
                        building_rows[key] = row
                    else:
                        if key not in building_rows:
                            raise RuntimeError(f'Ablation changed test polygon ordering: {key}')
                        if building_rows[key]['gt_valid_polygon_pixels'] != pixel_count:
                            raise RuntimeError(f'Ground truth changed between model runs: {key}')
                    result = building_rows[key]
                    prefix = 'full' if kind == 'full' else 'no_ca'
                    if pixel_count == 0:
                        stats['polygons_no_valid_gt_pixels'] += 1
                        result.update({prefix+'_missed': '', prefix+'_damage_fp': '',
                                       prefix+'_damage_fraction_detected': '',
                                       prefix+'_detected_fraction_of_polygon': '',
                                       prefix+'_majority_class': 'unscorable'})
                        continue
                    stats['eligible_polygons'] += 1
                    values = pred[valid_poly]
                    detected = values != 0
                    n_detected = int(detected.sum())
                    if n_detected == 0:
                        stats['missed_buildings'] += 1
                        missed = True
                        fp = False
                        frac_damaged = None
                        cls = 'missed'
                    else:
                        stats['detected_buildings'] += 1
                        missed = False
                        frac_damaged = float((values[detected] >= 2).mean())
                        fp = frac_damaged >= 0.5
                        most_common = np.bincount(values[detected], minlength=5).argmax()
                        cls = ['background', 'no_damage', 'minor_damage', 'major_damage', 'destroyed'][most_common]
                        if fp:
                            stats['building_damage_fp'] += 1
                    result.update({
                        prefix+'_missed': int(missed), prefix+'_damage_fp': int(fp),
                        prefix+'_damage_fraction_detected': (round(frac_damaged, 6) if frac_damaged is not None else ''),
                        prefix+'_detected_fraction_of_polygon': round(n_detected / pixel_count, 6),
                        prefix+'_majority_class': cls,
                    })

                if any(k[0] == stem for k in review):
                    out_mask = folder / f'{stem}.png'
                    Image.fromarray(pred).save(out_mask)
                    predictions_for_examples[kind][stem] = out_mask
                if (index + 1) % 25 == 0 or index + 1 == n:
                    elapsed = time.monotonic() - start
                    say(f'{kind}: {index+1}/{n} pairs; {elapsed/60:.1f} min elapsed')
            model_stats[kind] = dict(stats)
            no_damage_polygon_counts[kind] = stats['undamaged_polygons_total']
            say(f'{kind}: completed {n} pairs; found {stats["undamaged_polygons_total"]} undamaged polygons')
        finally:
            del model
            torch.cuda.empty_cache()

    if no_damage_polygon_counts['full'] != no_damage_polygon_counts['no_cross_attention']:
        raise RuntimeError('Ground-truth polygon counts are inconsistent between model runs')
    if not args.limit and no_damage_polygon_counts['full'] != EXPECTED_UNDAMAGED_POLYGONS:
        raise RuntimeError(f'Expected {EXPECTED_UNDAMAGED_POLYGONS} no-damage annotations, got '
                           f'{no_damage_polygon_counts["full"]}. Check JSON/partition identity.')

    # Reproduction check against published original test metrics; reject reports
    # from an unverified loading/processing convention.
    qa = {}
    for kind, s in model_stats.items():
        qa[kind] = {
            'tp_reference': REFERENCE_TP[kind], 'tp_computed': s['no_damage_tp'],
            'fn_reference': REFERENCE_FN[kind], 'fn_computed': s['no_damage_fn'],
            'reproduction_pass': None,
        }
        if not args.limit:
            original_gt = REFERENCE_TP[kind] + REFERENCE_FN[kind]
            current_gt = s['gt_undamaged_pixels']
            if original_gt != current_gt:
                raise RuntimeError(
                    f'{kind}: GT undamaged pixel count mismatch: original {original_gt}, '
                    f'computed {current_gt}. Dataset preprocessing differs!'
                )
            recall_orig = REFERENCE_TP[kind] / original_gt
            recall_new = s['no_damage_tp'] / current_gt
            # within 0.1 percentage point tolerates small numerical kernel differences
            passed = abs(recall_orig - recall_new) <= 0.001
            qa[kind].update({'original_recall': recall_orig, 'computed_recall': recall_new,
                             'reproduction_pass': passed})
            if not passed and not args.no_reproduction_check:
                raise RuntimeError(
                    f'{kind}: original no-damage recall={recall_orig:.6f}, '
                    f'computed={recall_new:.6f}. Abort: model or preprocessing mismatch. '
                    'DO NOT REPORT THESE RESULTS until investigated.'
                )
    validated = not args.limit and not args.no_reproduction_check and all(
        x['reproduction_pass'] for x in qa.values()
    )
    if not validated:
        say('WARNING: QA NOT VALIDATED; results must not be used as publishable evidence')

    all_rows = list(building_rows.values())
    selected = [row for row in all_rows if row['manual_case'] != '']
    selected.sort(key=lambda row: int(row['manual_case']))
    chosen_yes = [r for r in selected if r['manual_visual_change'] == 'yes']
    reviewed = [r for r in selected if r['manual_visual_change'] in ('yes', 'no')]
    columns = [
        'image_id','disaster','feature_index','building_uid','gt_label','gt_valid_polygon_pixels','mean_abs_rgb_difference_0_255',
        'delta_sun_azimuth_deg','delta_sun_elevation_deg','delta_off_nadir_deg',
        'manual_case','manual_visual_change','manual_possible_cause','manual_notes',
        'rgb_change_score_from_shortlist',
        'full_missed','full_damage_fp','full_damage_fraction_detected',
        'full_detected_fraction_of_polygon','full_majority_class',
        'no_ca_missed','no_ca_damage_fp','no_ca_damage_fraction_detected',
        'no_ca_detected_fraction_of_polygon','no_ca_majority_class',
    ]
    write_csv(out / 'all_undamaged_building_predictions.csv', all_rows, columns)
    write_csv(out / 'manual_review_cases_with_predictions.csv', selected, columns)
    # Reproducible full-test screening queue. Raw difference alone DOES NOT
    # establish an illumination effect or actual damage.
    ranked = sorted(
        [r for r in all_rows if r['mean_abs_rgb_difference_0_255'] != ''],
        key=lambda r: float(r['mean_abs_rgb_difference_0_255']), reverse=True
    )
    write_csv(out / 'full_test_visual_change_screening_ranked.csv', ranked, columns)
    write_csv(out / 'top100_visual_change_candidates_to_review.csv', ranked[:100], columns)
    write_csv(out / 'xbd_test_acquisition_differences.csv', pair_screen,
              ['image_id','disaster','delta_sun_azimuth_deg',
               'delta_sun_elevation_deg','delta_off_nadir_deg'])

    report = {
        'analysis': 'Reviewer #3 Comment 7 xBD false-positive damage audit',
        'source_split': 'xBD original TEST only', 'test_pairs': n,
        'full_dataset': not bool(args.limit), 'validated_for_reporting': validated,
        'rgb_screening_method': 'Mean absolute RGB difference on the original-model-resized images (0..255), ranked across all undamaged reference polygons; human verification still required',
        'scope': 'Building-level 50% damage-majority among detected footprint pixels; missed buildings reported separately',
        'pixel_fpr_definition': 'Ground-truth no-damage pixels predicted as minor/major/destroyed divided by all ground-truth no-damage pixels',
        'building_fpr_definition': 'Ground-truth undamaged polygons with >=50% of their predicted-building pixels classified damaged divided by all scorable ground-truth undamaged polygons; missed localizations separately counted',
        'models': model_meta, 'qa_against_saved_xbd_scores': qa, 'stats': {},
        'manual_review': {
            'candidate_cases_from_worksheet': len(review),
            'in_evaluated_subset': len(selected),
            'human_reviewed_yes_or_no': len(reviewed),
            'visually_changed_confirmed_yes': len(chosen_yes),
            'not_yet_reviewed': len(selected) - len(reviewed),
            'exhaustive_test_set_visual_change_count': 'NOT MEASURED: manual sample is selected, not exhaustive',
        },
        'caveats': [
            'Acquisition angles flag candidates; they do not prove shadow or view-angle causation.',
            '24 manual cases were preselected using large raw RGB differences and angle changes; not a random or exhaustive sample.',
            'The image-pair metadata and original annotations should be rechecked for unusual cases.',
            'This protocol uses no-damage polygon boundaries intersected with original xBD test damage masks.',
            'Any missed building is separate from a positive damage classification.',
        ],
    }
    for kind, s in model_stats.items():
        den_b = s['eligible_polygons']
        den_p = s['gt_undamaged_pixels']
        report['stats'][kind] = {
            **s,
            'building_fp_rate_percent': 100.0 * s['building_damage_fp'] / den_b if den_b else None,
            'pixel_damage_fp_rate_percent': 100.0 * s['pixel_damage_fp'] / den_p if den_p else None,
            'building_miss_rate_percent': 100.0 * s['missed_buildings'] / den_b if den_b else None,
            'visual_yes_total': len(chosen_yes),
            'visual_yes_scorable': sum(r['full_damage_fp' if kind == 'full' else 'no_ca_damage_fp'] != '' for r in chosen_yes),
            'visual_yes_false_positive': sum(int(r['full_damage_fp' if kind == 'full' else 'no_ca_damage_fp']) for r in chosen_yes if r['full_damage_fp' if kind == 'full' else 'no_ca_damage_fp'] != ''),
        }
        sc = report['stats'][kind]['visual_yes_scorable']
        report['stats'][kind]['visual_yes_fp_rate_percent'] = (
            100.0 * report['stats'][kind]['visual_yes_false_positive'] / sc if sc else None
        )
    full_fpr = report['stats']['full']['building_fp_rate_percent']
    no_fpr = report['stats']['no_cross_attention']['building_fp_rate_percent']
    report['difference_no_ca_minus_full_percentage_points'] = no_fpr - full_fpr if full_fpr is not None and no_fpr is not None else None
    save_json(out / 'comment7_results_summary.json', report)

    md = ['# Reviewer #3 Comment 7 | xBD false-positive damage evaluation', '',
          f'- Test image pairs: **{n}**',
          f'- Ground-truth undamaged polygons: **{no_damage_polygon_counts["full"]}**',
          f'- Reproduction validation: **{"PASS" if validated else "NOT VALIDATED"}**',
          '- False-positive damage means a ground-truth undamaged building is predicted damaged; '
          '>=50% damaged pixels among the pixels classified as building in its footprint.',
          '- Completely missed buildings are counted separately (not damage false positives).', '',
          '| Model | Undamaged polygons eligible | Damage false positives | Building-level FPR | '
          'Undamaged pixels | Pixel-level damage FPR |',
          '|---|---:|---:|---:|---:|---:|']
    for kind in ('full', 'no_cross_attention'):
        s = report['stats'][kind]
        md.append(f'| {kind} | {s["eligible_polygons"]:,} | {s["building_damage_fp"]:,} | '
                  f'{s["building_fp_rate_percent"]:.3f}% | {s["gt_undamaged_pixels"]:,} | '
                  f'{s["pixel_damage_fp_rate_percent"]:.3f}% |')
    md.extend(['', f'**No-cross minus full FPR:** {report["difference_no_ca_minus_full_percentage_points"]:+.3f} percentage points.', '',
               '## Manual visual review',
               f'- Candidate cases in worksheet: **{len(selected)}**',
               f'- Cases marked Yes/No by human reviewer: **{len(reviewed)}**',
               f'- Confirmed visually changed (Yes): **{len(chosen_yes)}**',
               f'- Pending manual classification: **{len(selected)-len(reviewed)}**',
               '- The visually changed count is from the **selected review sample only**, not the full xBD test set.'])
    if chosen_yes:
        for kind in ('full', 'no_cross_attention'):
            s = report['stats'][kind]
            md.append(f'- {kind}: **{s["visual_yes_false_positive"]}/{s["visual_yes_scorable"]}** '
                      f'false positives among verified visually changed, scorable cases '
                      f'({s["visual_yes_fp_rate_percent"]:.2f}%).')
    else:
        md.append('- **Visual-change false-positive rates PENDING** until the worksheet Yes/No fields are completed.')
    md.extend(['', '## Important qualifications',
               '- Acquisition metadata differences do not prove the cause of a false positive.',
               '- Do not claim exhaustive counting of visually changed buildings based on a 24-case targeted sample.',
               '- Illustrations require visual human confirmation and can show ambiguous/non-imaging changes.',
               '- See comment7_results_summary.json and all_undamaged_building_predictions.csv for full definitions and data.'])
    (out / 'comment7_results_summary.md').write_text('\n'.join(md) + '\n')

    make_example_figures(out, review, samples, predictions_for_examples, Image, ImageDraw, np)
    say(f'FINAL: full building FPR={full_fpr:.3f}% ; no-cross FPR={no_fpr:.3f}%')
    say(f'OUTPUT: {out / "comment7_results_summary.md"}')
    say(f'OUTPUT: {out / "all_undamaged_building_predictions.csv"}')
    say(f'OUTPUT: {out / "manual_review_cases_with_predictions.csv"}')
    say(f'OUTPUT: {out / "top100_visual_change_candidates_to_review.csv"}')
    if not validated:
        say('NOT VALIDATED - diagnostics only, do not cite results')


if __name__ == '__main__':
    main()
