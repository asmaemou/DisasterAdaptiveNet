#!/usr/bin/env python3
"""Reviewer #3 Comment (13): frozen-checkpoint test-only error analysis.

Designed for the user's existing DisasterAdaptiveNet checkout. Uses the exact
public model/dataset interfaces established by their qualitative-results
script and original event-specific wrappers. DOES NOT TRAIN OR MODIFY CHECKPOINTS.

Reports:
- Number of held-out test image pairs containing each damage class
- Pixel-level ground-truth support at the evaluation resolution
- Four-by-five confusion matrix, columns: missed localization, no/minor/major/destroyed
- Error attribution and selected qualitative failure examples
- A reproducibility check against the original saved per-class TP/FP/FN

Connected components are optional region proxies, NOT verified building polygon
counts; the study's original source polygons should be used for building counts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal import (
    BuildingGuidedCrossAttentionOrdinalNet,
)
import transformer.scripts.train_xbd_resnet34_swin_film_gated as stable
import transformer.scripts.train_xbd_supervised_disasteradaptivenet as runner

NAMES = ('No damage', 'Minor damage', 'Major damage', 'Destroyed')
LABELS = ('Missed localization', *NAMES)
COLORS = np.array([[0,0,0], [45,156,219], [242,201,76], [242,153,74], [214,40,40]], dtype=np.uint8)
ERROR_COLORS = np.array([[0,0,0], [0,167,120], [111,45,189], [247,247,247], [255,78,174]], dtype=np.uint8)

SPECS = {
    'texas': {
        'label': 'Texas Tornadoes', 'slug': 'texas_tornadoes', 'expected_pairs': 46,
        'data': '/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed',
        'checkpoint': 'output/ResNet34SwinTinyCrossAttentionOrdinal_TEXAS_TORNADOES/checkpoints/best.pt',
        'scores': 'output/ResNet34SwinTinyCrossAttentionOrdinal_TEXAS_TORNADOES/scores/scores_texas_tornadoes_test.json',
        'wrapper': 'transformer.scripts.train_texas_bitemporal_building_crossattention_ordinal',
        'threshold': 0.75, 'focus_class': 2,
    },
    'semeru': {
        'label': 'Mount Semeru Eruption', 'slug': 'mount_semeru', 'expected_pairs': 60,
        'data': '/homes/j244s673/documents/wsu/phd/mount_semeru_eruption_preprocessed',
        'checkpoint': 'output/HBG_CAON_MOUNT_SEMERU_ERUPTION/checkpoints/best.pt',
        'scores': 'output/HBG_CAON_MOUNT_SEMERU_ERUPTION/scores/scores_mount_semeru_test.json',
        'wrapper': 'transformer.scripts.train_mount_semeru_bitemporal_building_crossattention_ordinal',
        'threshold': 0.55, 'focus_class': 4,
    },
}


def say(s):
    print(s, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def write_csv(path, headings, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headings)
        writer.writerows(rows)


def sample_stem(dataset, index):
    s = dataset.samples[index]
    return getattr(s, 'stem', f'test_index_{index:04d}')


def component_proxy(label_mask):
    # 8-connected class-mask regions; these can merge adjacent building polygons.
    n, _ = cv2.connectedComponents(label_mask.astype(np.uint8), connectivity=8)
    return max(0, n - 1)


def overlay(rgb, mask):
    out = rgb.astype(np.float32).copy()
    for label in range(1, 5):
        where = mask == label
        out[where] = out[where] * .38 + COLORS[label] * .62
    return out.astype(np.uint8)


def rgba_errors(gt, pred):
    # Black elsewhere; green correct, purple class confusion, white missed, pink false building.
    z = np.zeros((*gt.shape, 3), dtype=np.uint8)
    bg, pg = gt > 0, pred > 0
    z[bg & pg & (gt == pred)] = ERROR_COLORS[1]
    z[bg & pg & (gt != pred)] = ERROR_COLORS[2]
    z[bg & ~pg] = ERROR_COLORS[3]
    z[~bg & pg] = ERROR_COLORS[4]
    return z


def load_checkpoint(path, model):
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location='cpu')
    if not isinstance(ck, dict):
        raise TypeError('Expected checkpoint dictionary')
    state = ck.get('model', ck.get('state_dict'))
    if state is None:
        raise KeyError('checkpoint lacks model/state_dict')
    if state and next(iter(state)).startswith('module.'):
        state = {k.removeprefix('module.'): v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    return ck


def plot_example(folder, item):
    # Arrays already at the model/evaluation resolution (896x896).
    img, truth, pred = item['post'], item['gt'], item['pred']
    fig, ax = plt.subplots(1, 4, figsize=(17.6, 4.9), constrained_layout=True)
    panes = (img, overlay(img, truth), overlay(img, pred), rgba_errors(truth, pred))
    for a, ar, title in zip(ax, panes, ('Post-disaster', 'Ground truth', 'RS-BDNet', 'Spatial errors')):
        a.imshow(ar); a.set_title(title); a.axis('off')
    fig.suptitle(item['title'], fontsize=13)
    for ext in ('png','pdf'):
        fig.savefig(folder / f"{item['name']}.{ext}", dpi=190 if ext=='png' else None, bbox_inches='tight')
    plt.close(fig)


def process(name, config, output, amp, disable_cudnn):
    import importlib
    dataset_type = getattr(importlib.import_module(config['wrapper']),
                           'TexasTornadoesDataset' if name=='texas' else 'MountSemeruDataset')
    data = Path(config['data']); cp = ROOT/config['checkpoint']; score_path=ROOT/config['scores']
    for p in (data, cp, score_path):
        if not p.exists(): raise FileNotFoundError(p)
    orig = json.loads(score_path.read_text())
    ds = dataset_type(root=data, split='test', image_size=896, training=False, conditioning_id=0)
    if len(ds) != config['expected_pairs']:
        raise RuntimeError(f"{name}: wrong test split length {len(ds)} != {config['expected_pairs']}")

    gpu = torch.device('cuda:0')
    model = BuildingGuidedCrossAttentionOrdinalNet(image_size=896,width=96).to(gpu)
    ck = load_checkpoint(cp, model)
    model.eval()
    threshold = float(ck.get('best_threshold',config['threshold']))
    if abs(threshold - config['threshold']) > 1e-6:
        raise RuntimeError(f'{name}: checkpoint threshold {threshold} differs from documented {config["threshold"]}')
    say(f'{name}: test pairs={len(ds)}, best epoch={ck.get("epoch", "unknown")}, threshold={threshold}, checkpoint={cp}')
    confusion = np.zeros((4, 5), dtype=np.int64)
    gt_support_per_image = []
    cc_proxy = np.zeros(4, dtype=np.int64)
    per_image = []
    selected = {'severity': None, 'localization': None}
    selected_n = {'severity':-1, 'localization':-1}
    fp_loc = 0
    fn_loc = 0
    tp_loc = 0
    pred_class_pixels = np.zeros(4,dtype=np.int64)

    for i in range(len(ds)):
        item = ds[i]
        inp = item['img'].unsqueeze(0).to(gpu)
        condition = item['cond_id'].unsqueeze(0).to(gpu)
        with torch.inference_mode():
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp):
                logits = model(inp, condition)
            ploc = torch.sigmoid(logits[0,0]) > threshold
            pdmg = logits[0,1:5].argmax(dim=0).long() + 1
            p = (pdmg * ploc.long()).cpu().numpy().astype(np.uint8)
        gtloc = item['loc'].cpu().numpy()
        gtdmg = item['dmg'].cpu().numpy()
        gtloc = np.squeeze(gtloc)
        gtdmg = np.squeeze(gtdmg)
        if gtloc.shape != p.shape or gtdmg.shape != p.shape:
            raise ValueError(f'{name} sample {i}: GT shape {gtloc.shape} dmg {gtdmg.shape} pred {p.shape}')
        # Valid damage pixels are annotated GT building pixels with labels 1..4.
        valid = (gtloc > 0) & (gtdmg>=1) & (gtdmg<=4)
        gt = np.where(valid,gtdmg,0).astype(np.uint8)
        # Count connected class regions as a PROXY only (not polygon counts).
        support = [int(np.count_nonzero(gt==k)) for k in range(1,5)]
        for k in range(1,5):
            cc_proxy[k-1] += component_proxy(gt==k)
            pred_class_pixels[k-1] += int(np.count_nonzero((p==k)&valid))
        gt_support_per_image.append(support)
        y=gt[valid].astype(np.int64)-1
        pred=p[valid].astype(np.int64)
        np.add.at(confusion,(y,pred),1)
        l_true = gtloc > 0
        l_pred = p > 0
        tp_loc += int(np.count_nonzero(l_true & l_pred))
        fp_loc += int(np.count_nonzero(~l_true & l_pred))
        fn_loc += int(np.count_nonzero(l_true & ~l_pred))
        focus = config['focus_class']
        focus_missed = int(np.count_nonzero((gt==focus)&(p==0)))
        focus_confused = int(np.count_nonzero((gt==focus)&(p>0)&(p!=focus)))
        stem = sample_stem(ds,i)
        per_image.append([i,stem,*support,focus_missed,focus_confused])
        for kind, amount in [('severity',focus_confused),('localization',focus_missed)]:
            if amount>selected_n[kind]:
                selected_n[kind]=amount
                # Create post RGB at the model/evaluation resolution (not native dimensions).
                image_path = ds.samples[i].post_image_path
                raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if raw is None: raise IOError(image_path)
                raw = cv2.cvtColor(cv2.resize(raw,(p.shape[1],p.shape[0])),cv2.COLOR_BGR2RGB)
                selected[kind]={'post':raw, 'gt':gt.copy(), 'pred':p.copy(), 'stem':stem,
                                'title':f"{config['label']} | {kind} | {stem}",
                                'name':f"{config['slug']}_{kind}_example"}
        if (i+1)%10==0 or i+1==len(ds): say(f'{name}: processed {i+1}/{len(ds)} pairs')

    support = confusion.sum(axis=1)
    ref_keys=('no_damage','minor_damage','major_damage','destroyed')
    verify=[]
    for k, key in enumerate(ref_keys):
        tp=int(confusion[k,k+1])
        fn=int(support[k]-tp)
        fp=int(confusion[:,k+1].sum()-tp)
        d=orig['details'][key]
        verify.append({'class':key,'reference_tp':int(d['tp']),'measured_tp':tp,'reference_fp':int(d['fp']),
                       'measured_fp':fp,'reference_fn':int(d['fn']),'measured_fn':fn,
                       'reference_f1':float(d['f1']),
                       'measured_f1':float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0,
                       'exact_count_match':(tp,fp,fn)==(int(d['tp']),int(d['fp']),int(d['fn']))})

    sub=output/config['slug']
    sub.mkdir(exist_ok=False)
    write_csv(sub/'confusion_matrix.csv',['GT class',*LABELS],[[NAMES[i],*map(int,confusion[i])] for i in range(4)])
    write_csv(sub/'per_test_image.csv',['test_index','stem',*[f'{v} GT pixels' for v in NAMES],
                                      'focus_missed_pixels','focus_severity_confusion_pixels'],per_image)
    write_csv(sub/'score_reproduction_check.csv',list(verify[0].keys()),[list(v.values()) for v in verify])
    write_csv(sub/'test_class_support.csv',
              ['class','images_with_class','ground_truth_pixels_896','connected_regions_proxy_not_polygon_count'],
              [[NAMES[k],int(np.count_nonzero(np.asarray(gt_support_per_image)[:,k])),
                int(support[k]),int(cc_proxy[k])] for k in range(4)])
    focus=config['focus_class']-1
    item={
        'dataset':config['label'], 'checkpoint':str(cp), 'checkpoint_sha256':sha(cp),
        'checkpoint_epoch':ck.get('epoch'),'localization_threshold':threshold,
        'test_image_pairs':len(ds),'model_input_size':896,'amp_bfloat16':amp,
        'cuDNN_enabled':not disable_cudnn,
        'confusion_rows':list(NAMES),'confusion_columns':list(LABELS),
        'confusion_counts':confusion.tolist(),
        'focus_class':NAMES[focus],
        'focus_support_gt_pixels':int(support[focus]),
        'focus_missed_localization_pixels':int(confusion[focus,0]),
        'focus_severity_confusion_pixels':int(confusion[focus,1:].sum()-confusion[focus,focus+1]),
        'focus_correct_pixels':int(confusion[focus,focus+1]),
        'score_reproduction_check':verify,
        'note':'Pixel-level error attribution, not count of wholly missed buildings. Connected regions are proxies, not verified original polygon counts. Original class F1 rule must be verified from training evaluator source.'
    }
    dump_json(sub/'analysis.json',item)
    for k, example in selected.items():
        if example is not None:
            plot_example(sub,example)
    say(f'{name}: wrote {sub} | matching original count rows = {sum(v["exact_count_match"] for v in verify)}/4')
    if not all(v['exact_count_match'] for v in verify):
        say(f'WARNING {name}: observed differences vs original evaluation. Inspect score_reproduction_check.csv; do NOT claim exact reproduction.')
    del model
    torch.cuda.empty_cache()
    return item


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--amp',action='store_true',help='Use bfloat16 autocast (training wrapper uses bfloat16).')
    ap.add_argument('--disable-cudnn', action='store_true',help='Avoid cuDNN Blackwell hangs.')
    ap.add_argument('--datasets',nargs='+',choices=list(SPECS),default=list(SPECS))
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; submit through Slurm, not on a login node')
    torch.backends.cudnn.enabled = not args.disable_cudnn
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    target=args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=False)
    say(f'Output: {target}; GPU: {torch.cuda.get_device_name(0)}')
    outcomes={}
    for name in args.datasets:
        outcomes[name]=process(name,SPECS[name],target,args.amp,args.disable_cudnn)
        dump_json(target/'summary_partial.json',outcomes)
    dump_json(target/'summary.json',outcomes)
    say('SUCCESS. All outputs written, original checkpoints and scores unchanged.')

if __name__=='__main__': main()
