#!/usr/bin/env python3
"""Reviewer #3, Comment (16): matched-area xBD qualitative comparison.

Runs complete xView2 winner ensemble GPU adapters from the EXISTING Comment 7
benchmark, and reproduces RS-BDNet's existing qualitative inference code.

This does NOT use the official CPU watershed step of the second-place solution,
or final CPU formatting/image export of the other baselines. Thus figures are
labeled as GPU-stage prediction comparisons (not native exported submissions).

Work in distinct stages/processes to avoid retaining all ensemble models on GPU.
Never overwrites an existing prediction or plot file.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEMS_EXPECTED = (
    'hurricane-harvey_00000479',
    'hurricane-harvey_00000096',
    'palu-tsunami_00000181',
    'hurricane-matthew_00000345',
)
METHODS = ('rsbdnet', 'first', 'second', 'third')
METHOD_LABELS = {
    'rsbdnet':'RS-BDNet', 'first':'xView2 1st',
    'second':'xView2 2nd', 'third':'xView2 3rd',
}
COLORS = np.array([
    [45,156,219], [242,201,76], [242,153,74], [214,40,40]
], dtype=np.uint8)


def say(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S]'), message, flush=True)


def save_unique(path: Path, *, data=None, **arrays):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        with path.open('wb') as stream:
            np.save(stream, data)
    else:
        with path.open('wb') as stream:
            np.savez_compressed(stream, **arrays)


def load_stems(output_dir):
    manifest = output_dir / 'selection_manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'Generate the rsbdnet stage first: {manifest}')
    with manifest.open(newline='', encoding='utf-8') as stream:
        names = [row['test_stem'] for row in csv.DictReader(stream)]
    if len(names) != 4 or set(names) != set(STEMS_EXPECTED):
        raise RuntimeError(f'Unexpected selected xBD examples: {names}')
    return names


def run_rsbdnet(out, xbd_root):
    import torch
    from transformer.scripts.generate_rsbdnet_qualitative_results import (
        dataset_specs, infer_selected
    )

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('RS-BDNet comparison needs a CUDA GPU')
    device = torch.device('cuda:0')
    specs = dataset_specs(xbd_root.parent, ROOT)
    # Dataset specs expect the sibling folder 'xview2' under data-parent.
    spec = next(spec for spec in specs if spec.slug == 'xbd')
    if spec.data_root.resolve() != xbd_root.resolve():
        raise RuntimeError(f'xBD root mismatch: spec={spec.data_root} vs {xbd_root}')
    settings = argparse.Namespace(image_size=896, examples_per_dataset=4, amp=True)
    examples = infer_selected(spec, settings, device)
    actual = [example.stem for example in examples]
    if len(actual) != 4 or set(actual) != set(STEMS_EXPECTED):
        raise RuntimeError(f'Existing selection algorithm no longer matches manifest: {actual}')
    rows=[]
    for e in examples:
        if not (e.truth.shape == e.prediction.shape == e.post.shape[:2]):
            raise RuntimeError(f'Spatial alignment failed on {e.stem}')
        if np.any(e.truth > 4) or np.any(e.prediction > 4):
            raise RuntimeError(f'Damage labels outside [0,4] for {e.stem}')
        save_unique(out / 'cases' / f'{e.stem}.npz', pre=e.pre, post=e.post,
                    truth=e.truth, rsbdnet=e.prediction,
                    threshold=np.array(e.threshold, dtype=np.float32))
        rows.append({'test_stem':e.stem, 'selection_reason':e.reason,
                     'checkpoint_epoch':e.checkpoint_epoch,
                     'localization_threshold':e.threshold})
        say(f'Saved RS-BDNet / ground truth: {e.stem}')
    p=out/'selection_manifest.csv'
    if p.exists():
        raise FileExistsError(f'Refusing to overwrite {p}')
    with p.open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    say(f'Wrote {p}')


def run_baseline(out, xbd_root, backend_name):
    import torch
    from transformer.scripts.benchmark_xview2_full_ensembles_comment7 import (
        build_backend, preprocess_pair
    )
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Ensemble inference requires CUDA')
    device = torch.device('cuda:0')
    names = load_stems(out)
    say(f'Loading ALL checkpoints for {backend_name}-place xView2 ensemble')
    backend = build_backend(backend_name, device, True, time.perf_counter())
    expected={'first':24,'second':21,'third':13}[backend_name]
    if backend.n_models != expected:
        raise RuntimeError(f'Incomplete baseline: {backend.n_models} of {expected} checkpoints')
    if backend.input_resolution != 1024:
        raise RuntimeError('Unexpected baseline input resolution')
    for i, stem in enumerate(names, 1):
        pre=xbd_root/'test'/'images'/f'{stem}_pre_disaster.png'
        post=xbd_root/'test'/'images'/f'{stem}_post_disaster.png'
        if not pre.is_file() or not post.is_file():
            raise FileNotFoundError(f'Missing input pair for {stem}')
        with np.load(out/'cases'/f'{stem}.npz') as stored:
            expected_shape=stored['truth'].shape
        inputs=preprocess_pair(pre, post, backend_name)
        with torch.inference_mode():
            _, label_tensor = backend.predict(inputs)
            torch.cuda.synchronize()
        mask = label_tensor[0].to(torch.uint8).cpu().numpy()
        if mask.shape != expected_shape:
            mask = cv2.resize(mask, (expected_shape[1],expected_shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        if np.any(mask>4):
            raise RuntimeError(f'{backend_name} yielded invalid labels: {np.unique(mask)}')
        dest=out/'predictions'/f'{backend_name}_{stem}.npy'
        save_unique(dest,data=mask)
        say(f'{backend_name} {i}/{len(names)}: {stem}, classes={np.unique(mask).tolist()}, saved={dest}')
    say(f'Finished complete {backend_name}-place GPU-stage predictions')


def overlay(post, labels, alpha=0.58):
    result=post.astype(np.float32).copy()
    for label,color in enumerate(COLORS,start=1):
        matched=(labels==label)
        result[matched]=result[matched]*(1-alpha)+color*alpha
    return np.clip(result,0,255).astype(np.uint8)


def error_mask(truth, prediction, kind):
    t=truth>0; p=prediction>0
    if kind=='missed':
        return t & ~p
    if kind=='severity':
        return t & p & (truth!=prediction)
    raise ValueError(kind)


def candidate_crops(truth, predictions, kind, exclude=None, size=224, step=56):
    h,w=truth.shape
    size=min(size,h,w)
    if size<32:
        return None
    ys=sorted(set(list(range(0,h-size+1,step))+[h-size]))
    xs=sorted(set(list(range(0,w-size+1,step))+[w-size]))
    # Prefer evidence from RS-BDNet, but fall back to baseline errors where needed.
    options=[]
    for model in METHODS:
        err=error_mask(truth,predictions[model],kind).astype(np.uint8)
        for y in ys:
            for x in xs:
                support=int(np.count_nonzero(truth[y:y+size,x:x+size]>0))
                if support<100:
                    continue
                count=int(err[y:y+size,x:x+size].sum())
                if count<8:
                    continue
                rect=(int(x),int(y),int(size),int(size))
                if exclude is not None:
                    x1,y1,s1,_=exclude
                    intersection=max(0,min(x+size,x1+s1)-max(x,x1))*max(0,min(y+size,y1+s1)-max(y,y1))
                    if intersection>.30*size*size:
                        continue
                # Stable selection: prefer RSBDNet errors, then largest pixel count.
                preferred=int(model=='rsbdnet')
                options.append((preferred,count,support,model,rect))
    if not options:
        return None
    selected=max(options,key=lambda row:(row[0],row[1],row[2]))
    return {'reference_model':selected[3], 'error_pixels_in_crop':selected[1],
            'ground_truth_building_pixels_in_crop':selected[2],
            'rect':selected[4], 'kind':kind}


def write_figure(out,stem,raw,predictions,areas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    truth=raw['truth']; post=raw['post']
    headers=('Post-disaster','Ground truth','RS-BDNet',
             'xView2 1st','xView2 2nd','xView2 3rd')
    labels=[None,truth,predictions['rsbdnet'],predictions['first'],
            predictions['second'],predictions['third']]
    full=[post]+[overlay(post,p) for p in labels[1:]]
    if any(a is None for a in areas):
        say(f'WARNING {stem}: at least one requested error category was not found; panel will show that explicitly')
    fig,axes=plt.subplots(3,6,figsize=(20,10),squeeze=False)
    for row in range(3):
        selected=areas[row-1] if row>0 else None
        if row==0:
            images=full
        elif selected is None:
            images=[np.full((224,224,3),247,dtype=np.uint8)]*6
        else:
            x,y,s,_=selected['rect']
            images=[im[y:y+s,x:x+s] for im in full]
        for col,image in enumerate(images):
            ax=axes[row,col]
            ax.imshow(image,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:
                ax.set_title(headers[col],fontsize=11,fontweight='bold')
                for area,color in zip(areas,('#23A867','#9D4EDD')):
                    if area is None:continue
                    x,y,s,_=area['rect']
                    ax.add_patch(Rectangle((x,y),s,s,fill=False,edgecolor=color,linewidth=3))
            elif selected is None:
                ax.text(.5,.5,'No qualifying\nerror crop',ha='center',va='center',
                        transform=ax.transAxes,fontsize=10)
            elif col>=2:
                method=METHODS[col-2]
                x,y,s,_=selected['rect']
                count=int(error_mask(truth,predictions[method],selected['kind'])[y:y+s,x:x+s].sum())
                ax.set_xlabel(f'{count} error pixels',fontsize=8)
        if row==0:axes[row,0].set_ylabel('Full scene',fontsize=11,fontweight='bold')
        elif row==1:axes[row,0].set_ylabel('Missed detections\nzoom',fontsize=10,fontweight='bold')
        else:axes[row,0].set_ylabel('Severity confusion\nzoom',fontsize=10,fontweight='bold')
    legend=[Patch(facecolor=tuple(c/255.0), label=n) for c,n in zip(
        COLORS,('No damage','Minor damage','Major damage','Destroyed'))]
    legend.extend([Patch(facecolor='none',edgecolor='#23A867',label='Missed crop'),
                   Patch(facecolor='none',edgecolor='#9D4EDD',label='Severity crop')])
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.012),ncol=6,fontsize=10)
    fig.suptitle(f'Same-area xBD test predictions: {stem}',fontsize=16,fontweight='bold',y=.995)
    fig.subplots_adjust(left=.045,right=.995,top=.925,bottom=.075,hspace=.18,wspace=.035)
    for suffix in ('.png','.pdf'):
        dest=out/'figures'/f'comparison_{stem}{suffix}'
        if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
        fig.savefig(dest,dpi=300 if suffix=='.png' else None,bbox_inches='tight',facecolor='white')
        say(f'Wrote {dest}')
    plt.close(fig)


def run_assemble(out):
    names=load_stems(out)
    report=[]
    for stem in names:
        with np.load(out/'cases'/f'{stem}.npz') as data:
            raw={key:data[key] for key in ('pre','post','truth','rsbdnet')}
        preds={'rsbdnet':raw['rsbdnet']}
        for name in ('first','second','third'):
            p=out/'predictions'/f'{name}_{stem}.npy'
            if not p.is_file():raise FileNotFoundError(f'Missing baseline output: {p}')
            preds[name]=np.load(p)
        if any(value.shape!=raw['truth'].shape for value in preds.values()):
            raise RuntimeError(f'Prediction and ground truth do not align for {stem}')
        missed=candidate_crops(raw['truth'],preds,'missed')
        severity=candidate_crops(raw['truth'],preds,'severity',exclude=(missed['rect'] if missed else None))
        if severity is None:
            severity=candidate_crops(raw['truth'],preds,'severity')
        areas=[missed,severity]
        write_figure(out,stem,raw,preds,areas)
        row={'test_stem':stem,'crop_selection':'highest error pixel count on fixed grid; prioritize RS-BDNet, then baseline',
             'missed':missed,'severity':severity,
             'error_pixel_counts':{name:{kind:int(error_mask(raw['truth'],prediction,kind).sum())
                                   for kind in ('missed','severity')}
                                   for name,prediction in preds.items()}}
        report.append(row)
        say(f'Error crop selection: {stem}: missed={missed}; severity={severity}')
    target=out/'figure_audit.json'
    if target.exists():raise FileExistsError(f'Refusing to overwrite {target}')
    target.write_text(json.dumps({
        'disclaimer':('GPU-stage baseline outputs from previously benchmarked complete ensemble adapters. '
                      'Second-place CPU watershed and native CPU export/postprocessing omitted. '
                      'Error counts are pixel-level, not whole-building counts.'),
        'scene_alignment':'All outputs resized to the original xBD image grid with nearest-neighbor interpolation',
        'input_resolutions':{'rsbdnet':896,'first':1024,'second':1024,'third':1024},
        'cases':report},indent=2)+'\n',encoding='utf-8')
    say(f'Saved {target}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['rsbdnet','first','second','third','assemble'],required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--xbd-root',type=Path,default=Path('/homes/j244s673/documents/wsu/phd/xview2'))
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_dir():raise FileNotFoundError(f'Create unique output directory first: {out}')
    if args.stage=='rsbdnet':run_rsbdnet(out,args.xbd_root)
    elif args.stage in ('first','second','third'):run_baseline(out,args.xbd_root,args.stage)
    else:run_assemble(out)


if __name__=='__main__':main()
