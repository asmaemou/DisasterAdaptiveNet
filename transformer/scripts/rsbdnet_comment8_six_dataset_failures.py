#!/usr/bin/env python3
"""Comment 8: audited failure-case analysis for xBD + five EBD TEST sets.

Six validation-selected *dataset-specific* best.pt files are read, never written.
All outputs go into a new, empty output directory. Each dataset is evaluated
in full with the training evaluator's pixel-level localization/damage semantics.
Selected image regions are *connected-mask proxies*, not verified single
building polygons. Human inspection is required before paper publication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import torch

import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner
from transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal import (
    BuildingGuidedCrossAttentionOrdinalNet,
)

PARENT = Path('/homes/j244s673/documents/wsu/phd')
# Paths reproduced directly from the user's six-dataset qualitative-generation code.
DATASETS = OrderedDict([
    ('xbd', ('xBD', 'xview2', 'BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3', 933, .7756)),
    ('earthquake_turkey', ('Earthquake Turkey', 'earthquake_turkey_preprocessed', 'HBG_CAON_EARTHQUAKE_TURKEY', 95, .6115)),
    ('mount_semeru_eruption', ('Mount Semeru Eruption', 'mount_semeru_eruption_preprocessed', 'HBG_CAON_MOUNT_SEMERU_ERUPTION', 60, .5965)),
    ('texas_tornadoes', ('Texas Tornadoes', 'texas_tornadoes_preprocessed', 'ResNet34SwinTinyCrossAttentionOrdinal_TEXAS_TORNADOES', 46, .5356)),
    ('hurricane_delta', ('Hurricane Delta', 'hurricane_delta_preprocessed', 'ResNet34SwinTinyCrossAttentionOrdinal_HURRICANE_DELTA', 137, .6770)),
    ('pakistan_flooding', ('Pakistan Flooding', 'pakistan_flooding_preprocessed', 'ResNet34SwinTinyCrossAttentionOrdinal_PAKISTAN_FLOODING', 354, .7990)),
])
CLASSES = ('no_damage', 'minor_damage', 'major_damage', 'destroyed')
PRED_COLS = ('missed_localization',) + CLASSES
CASE_TYPES = ('destroyed_to_major', 'minor_to_no_damage', 'missed_building', 'other_severity_confusion')
CLASS_COLORS = np.asarray([[0,0,0],[45,156,219],[242,201,76],[242,153,74],[214,40,40]], dtype=np.uint8)
ERR_COLORS = {'wrong_severity': (133, 58, 203), 'missed_building': (255, 255, 255),
              'false_building': (255, 78, 174)}
EXPECTED_SHA_XBD = 'c781c1d9539618e2d5c67bc6365daa24557e9c01e64f87fc7d92282fe90aea77'
PROTECTED = (
    'BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3',
    'RSBDNet_xBD_ZeroShot_EBD',
    'RSBDNet_xBD_ZeroShot_OrdinalFusion',
    'RSBDNet_xBD_ZeroShot_Calibration',
    'RSBDNet_xBD_DomainGeneralization_v1',
    'RSBDNet_xBD_DomainGeneralization_v1_ZeroShot_EBD',
    'RSBDNet_qualitative_results',
    'RSBDNet_TargetSpecific_FailureCases_Comment8',
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, fields, records):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def check_output(output):
    output = output.expanduser().resolve()
    approved = (PROJECT_ROOT / 'output' / 'RSBDNet_Comment8_SixDataset_FailureAnalysis').resolve()
    if output != approved:
        raise ValueError(f'For safety, --output-dir must be exactly: {approved}')
    for name in PROTECTED:
        protected = (PROJECT_ROOT / 'output' / name).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f'Output overlaps protected directory: {protected}')
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f'Output directory already contains files; refusing to overwrite: {output}')
    return output


def load_checkpoint(path, data_root, device, is_xbd):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise ValueError(f'Checkpoint missing runner model state: {path}')
    args = ckpt.get('args') or {}
    recorded_root = args.get('xbd_root')
    if not recorded_root or Path(recorded_root).resolve() != data_root.resolve():
        raise ValueError(f'Checkpoint trained on unexpected dataset: {path}\nrecorded={recorded_root}\nexpected={data_root}')
    if int(args.get('img_size', -1)) != 896:
        raise ValueError(f'Checkpoint image size differs from 896: {path}')
    if not is_xbd and str(args.get('train_split')) != 'train':
        raise ValueError(f'Expected target-trained checkpoint with train split: {path}')
    threshold = float(ckpt['best_threshold'])
    if not (0.0 < threshold < 1.0):
        raise ValueError(f'Invalid checkpoint threshold: {threshold}')
    model = BuildingGuidedCrossAttentionOrdinalNet(image_size=896, width=96)
    model.load_state_dict(ckpt['model'], strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    return model, threshold, {
        'path': str(path), 'sha256': sha256(path), 'epoch': int(ckpt['epoch']),
        'best_threshold': threshold, 'recorded_training_root': recorded_root,
        'training_split': str(args.get('train_split')),
        'validation_split': str(args.get('val_split')),
    }


@torch.inference_mode()
def forward(model, threshold, item, device):
    logits = model(item['img'].unsqueeze(0).to(device), item['cond_id'].unsqueeze(0).to(device))
    if logits.shape != (1, 8, 896, 896):
        raise RuntimeError(f'Unexpected network output {tuple(logits.shape)}')
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError('Non-finite network outputs')
    pl = torch.sigmoid(logits[0,0]) > threshold
    pd = (torch.argmax(logits[0,1:5], dim=0).long() + 1) * pl.long()
    return pl.cpu().numpy().astype(np.uint8), pd.cpu().numpy().astype(np.uint8)


def confusion(gt_loc, gt_dmg, pred_loc, pred_dmg):
    loc = gt_loc > 0
    pred = pred_loc > 0
    loc_counts = np.asarray([
        np.count_nonzero(loc & pred), np.count_nonzero(~loc & pred),
        np.count_nonzero(loc & ~pred), np.count_nonzero(~loc & ~pred),
    ], dtype=np.int64)
    valid = loc & (gt_dmg != 255)
    true = gt_dmg[valid].astype(np.int64)
    observed = pred_dmg[valid].astype(np.int64)
    if np.any((true < 0) | (true > 3)) or np.any((observed < 0) | (observed > 4)):
        raise ValueError('Unexpected damage codes')
    cm = np.bincount(true * 5 + observed, minlength=20).reshape(4,5).astype(np.int64)
    return loc_counts, cm


def class_statistics(loc_counts, matrix):
    tp, fp, fn, tn = (int(v) for v in loc_counts)
    def prf(a,b,c):
        p = a / (a+b) if a+b else 0.0
        r = a / (a+c) if a+c else 0.0
        f = (2*a / (2*a+b+c)) if 2*a+b+c else 0.0
        return {'tp': int(a), 'fp': int(b), 'fn': int(c),
                'precision': p, 'recall': r, 'f1': f}
    row = {'localization': dict(prf(tp,fp,fn), tn=tn), 'damage_classes': {}}
    for i,name in enumerate(CLASSES):
        t = int(matrix[i,i+1]); fp_i = int(matrix[:,i+1].sum()) - t
        fn_i = int(matrix[i,:].sum()) - t
        row['damage_classes'][name] = dict(prf(t,fp_i,fn_i), support_valid_gt_pixels=int(matrix[i,:].sum()))
    row['damage_macro_f1'] = float(np.mean([row['damage_classes'][c]['f1'] for c in CLASSES]))
    row['valid_gt_damage_pixels'] = int(matrix.sum())
    row['missed_gt_damage_pixels'] = int(matrix[:,0].sum())
    return row


def region_candidates(dataset, stem, gt_loc, gt_dmg, pred_dmg, min_area, min_purity):
    """Select high-error 8-connected GT mask regions, not asserted building polygons."""
    nlabels, regions, stats, _ = cv2.connectedComponentsWithStats(gt_loc.astype(np.uint8), connectivity=8)
    results = []
    for label in range(1, nlabels):
        x,y,w,h,area = (int(v) for v in stats[label])
        if area < min_area:
            continue
        region = regions[y:y+h,x:x+w] == label
        valid = region & (gt_dmg[y:y+h,x:x+w] != 255)
        n = int(valid.sum())
        if n < min_area:
            continue
        true = gt_dmg[y:y+h,x:x+w][valid].astype(np.int64)
        counts = np.bincount(true, minlength=4)
        cls = int(counts.argmax())
        purity = float(counts[cls] / n)
        if purity < min_purity:
            continue
        obs = pred_dmg[y:y+h,x:x+w][valid].astype(np.int64)
        votes = np.bincount(obs, minlength=5)
        missed_fraction = float(votes[0] / n)
        correct_fraction = float(votes[cls+1] / n)
        detected_fraction = 1.0 - missed_fraction
        dominant_pred = int(votes[1:].argmax()) + 1  # damage code 1..4
        kinds = []
        if cls == 3 and votes[3] / n >= .45 and detected_fraction >= .50:
            kinds.append(('destroyed_to_major', 3, float(votes[3] / n)))
        if cls == 1 and votes[1] / n >= .45 and detected_fraction >= .50:
            kinds.append(('minor_to_no_damage', 1, float(votes[1] / n)))
        if missed_fraction >= .65:
            kinds.append(('missed_building', 0, missed_fraction))
        # Fallback for coverage when no requested confusion can be found.
        if dominant_pred != cls + 1 and detected_fraction >= .50 and votes[dominant_pred]/n >= .45:
            kinds.append(('other_severity_confusion', dominant_pred, float(votes[dominant_pred] / n)))
        for kind, predicted, fraction in kinds:
            results.append({
                'case_type': kind, 'dataset': dataset, 'stem': stem,
                'region_id': label, 'gt_class': CLASSES[cls],
                'predicted_outcome': ('missed_localization' if predicted == 0 else CLASSES[predicted-1]),
                'x': x, 'y': y, 'w': w, 'h': h, 'area_pixels': area,
                'valid_pixels': n, 'dominant_gt_purity': round(purity, 6),
                'detected_fraction': round(detected_fraction, 6),
                'correct_fraction': round(correct_fraction, 6),
                'error_fraction': round(fraction, 6),
                'rank_score': round(fraction * math.log1p(n) * purity, 6),
            })
    return results


def choose_six(candidates):
    """At most one tile per dataset; prioritize coverage of three requested errors."""
    grouped = {k: [r for r in candidates if r['dataset'] == k] for k in DATASETS}
    for ds in grouped:
        grouped[ds].sort(key=lambda r: (-r['rank_score'], r['stem'], r['region_id']))
    selected = {}
    # More rare or important requested types first; prefer indicated datasets,
    # but all choices are based on actual, audited error masks.
    preferred = {
        'destroyed_to_major': ['mount_semeru_eruption','earthquake_turkey','xbd','hurricane_delta','pakistan_flooding','texas_tornadoes'],
        'minor_to_no_damage': ['texas_tornadoes','hurricane_delta','pakistan_flooding','earthquake_turkey','xbd','mount_semeru_eruption'],
        'missed_building': ['hurricane_delta','pakistan_flooding','earthquake_turkey','xbd','mount_semeru_eruption','texas_tornadoes'],
    }
    for case in CASE_TYPES[:3]:
        choices = []
        for ds in DATASETS:
            if ds in selected:
                continue
            options = [r for r in grouped[ds] if r['case_type'] == case]
            if options:
                choices.append(options[0])
        if choices:
            choices.sort(key=lambda r: (preferred[case].index(r['dataset']), -r['rank_score']))
            selected[choices[0]['dataset']] = choices[0]
        else:
            print(f'WARNING: No eligible connected-region example of {case} across six datasets.', flush=True)
    for ds in DATASETS:
        if ds not in selected and grouped[ds]:
            requested = [r for r in grouped[ds] if r['case_type'] in CASE_TYPES[:3]]
            selected[ds] = (requested or grouped[ds])[0]
    return [selected.get(ds) for ds in DATASETS]


def load_rgb_resized(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read original image: {path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (896,896):
        image = cv2.resize(image, (896,896), interpolation=cv2.INTER_LINEAR)
    return image


def overlay(image, labels, alpha=.58):
    out = image.astype(np.float32).copy()
    for label in range(1,5):
        pixels = labels == label
        out[pixels] = (1-alpha) * out[pixels] + alpha * CLASS_COLORS[label]
    return out.clip(0,255).astype(np.uint8)


def error_map(gt_loc, gt_dmg, pred_loc, pred_dmg):
    gt = gt_loc > 0
    valid = gt & (gt_dmg != 255)
    expected_code = gt_dmg.astype(np.int16) + 1
    err = np.zeros((*gt_loc.shape,3), dtype=np.uint8)
    err[valid & (pred_dmg == expected_code)] = (16, 185, 129)  # correct
    err[valid & (pred_dmg != 0) & (pred_dmg != expected_code)] = ERR_COLORS['wrong_severity']
    err[gt & (pred_loc == 0)] = ERR_COLORS['missed_building']
    err[~gt & (pred_loc != 0)] = ERR_COLORS['false_building']
    return err


def make_figure(chosen, out, name, device, crop_size):
    chosen = [r for r in chosen if r is not None]
    if not chosen:
        return None
    n = len(chosen)
    fig, axes = plt.subplots(n, 5, figsize=(18.8, 3.15*n + 1.4), squeeze=False)
    captions = ('Pre-disaster', 'Post-disaster', 'Ground truth', 'Prediction', 'Error map')
    for idx, r in enumerate(chosen):
        display, subdir, ckptdir, _, _ = DATASETS[r['dataset']]
        data_root = PARENT / subdir
        ckpt_path = PROJECT_ROOT / 'output' / ckptdir / 'checkpoints' / 'best.pt'
        ds = runner.XBDOriginalDataset(root=data_root, split='test', image_size=896,
                                        training=False, conditioning_id=0)
        mapping = {s.stem: i for i,s in enumerate(ds.samples)}
        index = mapping.get(r['stem'])
        if index is None:
            raise RuntimeError(f'Missing selected stem: {r["stem"]}')
        item = ds[index]
        sample = ds.samples[index]
        model, threshold, _ = load_checkpoint(ckpt_path, data_root, device, is_xbd=r['dataset']=='xbd')
        pl,pd = forward(model, threshold, item, device)
        del model
        torch.cuda.empty_cache()
        gt_loc = item['loc'].numpy()
        gt_dmg = item['dmg'].numpy()
        true_codes = np.zeros(gt_dmg.shape, dtype=np.uint8)
        valid = (gt_loc>0) & (gt_dmg != 255)
        true_codes[valid] = (gt_dmg[valid]+1).astype(np.uint8)
        pre = load_rgb_resized(sample.pre_image_path)
        post = load_rgb_resized(sample.post_image_path)
        images = (pre,post,overlay(post,true_codes),overlay(post,pd),error_map(gt_loc,gt_dmg,pl,pd))
        x,y,w,h = (int(r[q]) for q in ('x','y','w','h'))
        length = min(896, max(crop_size, w+70, h+70))
        x0 = int(np.clip(x+w/2-length/2, 0, 896-length))
        y0 = int(np.clip(y+h/2-length/2, 0, 896-length))
        for j,im in enumerate(images):
            ax = axes[idx,j]
            ax.imshow(im[y0:y0+length, x0:x0+length], interpolation='nearest' if j==4 else 'bilinear')
            ax.set_axis_off()
            ax.add_patch(Rectangle((x-x0,y-y0),w,h,fill=False,edgecolor='white',
                                   linewidth=2.1,linestyle='--'))
            if idx==0:
                ax.set_title(captions[j], fontsize=12, weight='bold')
        axes[idx,0].set_ylabel(f'{display}\n{r["case_type"].replace("_", " ")}\n{r["stem"]}',
                               rotation=0,ha='right',va='center',fontsize=9.5,labelpad=10)
    handles = [Patch(facecolor=CLASS_COLORS[i]/255,label=CLASSES[i-1].replace('_',' ').title()) for i in range(1,5)]
    handles += [Patch(facecolor=(16/255,185/255,129/255),label='Correct'),
                Patch(facecolor=tuple(x/255 for x in ERR_COLORS['wrong_severity']),label='Wrong severity'),
                Patch(facecolor='white',edgecolor='grey',label='Missed building'),
                Patch(facecolor=tuple(x/255 for x in ERR_COLORS['false_building']),label='False building')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.004),ncol=4,
               fontsize=10,frameon=True)
    fig.subplots_adjust(left=.19,right=.99,top=.96,bottom=.085,hspace=.10,wspace=.03)
    saved=[]
    for ext,extra in (('png',{'dpi':280}),('pdf',{})):
        path=out/(name+'.'+ext)
        fig.savefig(path,bbox_inches='tight',facecolor='white',**extra)
        saved.append(str(path))
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT/'output'/'RSBDNet_Comment8_SixDataset_FailureAnalysis')
    parser.add_argument('--min-area', type=int, default=64)
    parser.add_argument('--min-purity', type=float, default=.85)
    parser.add_argument('--crop-size', type=int, default=448)
    parser.add_argument('--metric-tolerance', type=float, default=.003)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    if args.min_area < 16 or not (0.5 <= args.min_purity <= 1.0):
        raise ValueError('Invalid component requirements')
    if not (64 <= args.crop_size <= 896):
        raise ValueError('Invalid crop size')
    dest = check_output(args.output_dir)
    expected = {}
    for key,(_,dirname,checkpoint_dir,_,_) in DATASETS.items():
        data_dir = PARENT/dirname
        ckpt = PROJECT_ROOT/'output'/checkpoint_dir/'checkpoints'/'best.pt'
        if not ckpt.is_file():
            raise FileNotFoundError(f'Missing checkpoint for {key}: {ckpt}')
        if not (data_dir/'test'/'images').is_dir() or not (data_dir/'test'/'targets').is_dir():
            raise FileNotFoundError(f'Missing dataset test directory for {key}: {data_dir}')
        expected[str(ckpt)] = sha256(ckpt)
    first = PROJECT_ROOT/'output'/'BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3'/'checkpoints'/'best.pt'
    if expected[str(first)] != EXPECTED_SHA_XBD:
        raise RuntimeError('Original xBD checkpoint hash does not match previously verified hash')
    runner.set_seed(42)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    dest.mkdir(parents=True, exist_ok=False) if not dest.exists() else None
    manifest = {
        'protocol':'six held-out TEST partitions; six corresponding validation-selected target-specific checkpoints',
        'source_of_checkpoint_paths':'existing generate_rsbdnet_qualitative_results.py from user',
        'selection':'high-error connected GT regions only, never success cases',
        'labels':'pixel-level; pre target >0 building, post target 1..4 maps to four damage classes',
        'warning':'8-connected components are proxies and may merge adjacent buildings; manually inspect all crops',
        'created_utc':timestamp(), 'model_sources':{},'metric_reproduction':{},
        'script_sha256':sha256(Path(__file__)), 'checksums_before':expected,
    }
    all_candidates=[]
    metrics_rows=[]
    candidates_columns=['case_type','dataset','stem','region_id','gt_class','predicted_outcome','x','y','w','h',
                        'area_pixels','valid_pixels','dominant_gt_purity','detected_fraction',
                        'correct_fraction','error_fraction','rank_score']
    try:
        for key,(name,dirname,checkpoint_dir,expected_pairs,expected_macro) in DATASETS.items():
            data_root=PARENT/dirname
            path=PROJECT_ROOT/'output'/checkpoint_dir/'checkpoints'/'best.pt'
            model,threshold,info=load_checkpoint(path,data_root,device,is_xbd=key=='xbd')
            manifest['model_sources'][key]=info
            ds=runner.XBDOriginalDataset(root=data_root,split='test',image_size=896,
                                          training=False,conditioning_id=0)
            if len(ds)!=expected_pairs:
                raise RuntimeError(f'{key}: {len(ds)} test pairs vs expected {expected_pairs}')
            counts=np.zeros(4,dtype=np.int64)
            cm=np.zeros((4,5),dtype=np.int64)
            for ix in range(len(ds)):
                item=ds[ix]
                pl,pd=forward(model,threshold,item,device)
                loc,damage=confusion(item['loc'].numpy(),item['dmg'].numpy(),pl,pd)
                counts+=loc;cm+=damage
                all_candidates.extend(region_candidates(key,item['stem'],item['loc'].numpy(),
                                                        item['dmg'].numpy(),pd,args.min_area,args.min_purity))
                if (ix+1)%25==0 or (ix+1)==len(ds):
                    print(f'{key}: {ix+1}/{len(ds)} test pairs',flush=True)
            scores=class_statistics(counts,cm)
            difference=abs(scores['damage_macro_f1']-expected_macro)
            manifest['metric_reproduction'][key] = {
                'n_test_pairs':len(ds), 'expected_macro_f1_rounded':expected_macro,
                'observed_macro_f1':scores['damage_macro_f1'],
                'abs_delta':difference,'within_tolerance':difference<=args.metric_tolerance,
            }
            sub=dest/key
            sub.mkdir(parents=True,exist_ok=False)
            save_json(sub/'metrics.json',scores)
            save_csv(sub/'damage_confusion_matrix.csv',
                     ['ground_truth',*PRED_COLS,'support_pixels'],
                     [dict(ground_truth=CLASSES[i],support_pixels=int(cm[i,:].sum()),
                           **{PRED_COLS[j]:int(cm[i,j]) for j in range(5)}) for i in range(4)])
            save_csv(sub/'damage_confusion_percent.csv',
                     ['ground_truth',*PRED_COLS,'support_pixels'],
                     [dict(ground_truth=CLASSES[i], support_pixels=int(cm[i,:].sum()),
                           **{PRED_COLS[j]:float(cm[i,j]/cm[i,:].sum()) if cm[i,:].sum() else 0.0
                              for j in range(5)}) for i in range(4)])
            metrics_rows.append({'dataset':key,'n_test_pairs':len(ds),'localization_f1':scores['localization']['f1'],
                                 'no_damage_f1':scores['damage_classes']['no_damage']['f1'],
                                 'minor_damage_f1':scores['damage_classes']['minor_damage']['f1'],
                                 'major_damage_f1':scores['damage_classes']['major_damage']['f1'],
                                 'destroyed_f1':scores['damage_classes']['destroyed']['f1'],
                                 'damage_macro_f1':scores['damage_macro_f1'],
                                 'expected_macro_f1':expected_macro,'reproduction_within_tolerance':difference<=args.metric_tolerance})
            print(f'{key} damage Macro-F1={scores["damage_macro_f1"]:.6f} '
                  f'(published {expected_macro:.4f}; within tolerance: {difference<=args.metric_tolerance})',flush=True)
            del model
            torch.cuda.empty_cache()
        save_csv(dest/'six_dataset_metric_reproduction.csv',list(metrics_rows[0]),metrics_rows)
        if not all(v['within_tolerance'] for v in manifest['metric_reproduction'].values()):
            manifest['status'] = 'WARNING_METRIC_MISMATCH_CONTINUED'
            save_json(dest/'experiment_manifest.json', manifest)
            print(
                'WARNING: One or more datasets did not reproduce the reported test Macro-F1. '
                'Continuing anyway to generate failure-case figures as requested.',
                flush=True
            )
        all_candidates.sort(key=lambda r:(r['case_type'],r['dataset'],-r['rank_score'],r['stem'],r['region_id']))
        save_csv(dest/'candidate_failure_regions.csv',candidates_columns,all_candidates)
        coverage_records=[]
        for ds in DATASETS:
            items=[r for r in all_candidates if r['dataset']==ds]
            coverage_records.append({
                'dataset':ds,
                'all_candidate_regions':len(items),
                **{kind+'_candidate_regions':sum(r['case_type']==kind for r in items)
                   for kind in CASE_TYPES},
            })
        save_csv(dest/'six_dataset_failure_coverage.csv',list(coverage_records[0]),coverage_records)
        six=choose_six(all_candidates)
        selected_records=[r for r in six if r is not None]
        save_csv(dest/'six_selected_failure_cases.csv',candidates_columns,selected_records)
        save_json(dest/'selection_report.json',{
            'six_dataset_selection':[r for r in six],
            'requested_case_types':list(CASE_TYPES[:3]),
            'requested_types_found':sorted(set(r['case_type'] for r in selected_records) & set(CASE_TYPES[:3])),
            'datasets_without_eligible_failure':[key for key,r in zip(DATASETS,six) if r is None],
            'manual_visual_review_required':True,
            'note':'Errors are identified on GT connected-region proxies; no material/roof type inferred automatically.',
        })
        if len(selected_records)==6:
            paths=make_figure(six,dest,'RSBDNet_six_dataset_failure_cases_Comment8',device,args.crop_size)
            print('Six-dataset overview:',paths,flush=True)
        else:
            print('WARNING: Not all six datasets have qualifying component examples at selected settings.',flush=True)
        spotlight=[]
        for kind in CASE_TYPES[:3]:
            chosen=next((r for r in selected_records if r['case_type']==kind),None)
            if chosen is None:
                choices=[r for r in all_candidates if r['case_type']==kind]
                if choices:
                    spotlight.append(choices[0])
            else:
                spotlight.append(chosen)
        if len(spotlight)==3:
            paths=make_figure(spotlight,dest,'RSBDNet_three_failure_types_Comment8',device,args.crop_size)
            print('Three-type figure:',paths,flush=True)
        else:
            print('WARNING: One or more requested error types lack a qualifying example.',flush=True)
        check_after={k:sha256(Path(k)) for k in expected}
        if check_after!=expected:
            raise RuntimeError('A checkpoint SHA256 unexpectedly changed during analysis')
        manifest['checksums_after']=check_after
        manifest['status']='COMPLETE'
        manifest['n_candidates']=len(all_candidates)
        manifest['selected_types']=sorted({r['case_type'] for r in selected_records})
        manifest['completed_utc']=timestamp()
        save_json(dest/'experiment_manifest.json',manifest)
        print('DONE. Results in',dest,flush=True)
    except Exception:
        manifest['status']=manifest.get('status','ERROR')
        manifest['failed_utc']=timestamp()
        save_json(dest/'experiment_manifest.json',manifest)
        raise


if __name__=='__main__':
    main()
