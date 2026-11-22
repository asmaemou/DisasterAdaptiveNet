#!/usr/bin/env python3
"""Reviewer 2 comment 7: xView2 winner inference smoke tests and GPU benchmarks.

Run on the WSU GPU compute node, from the original DisasterAdaptiveNet checkout.
Do not edit or overwrite the three original winner repositories or checkpoints.

Backend 'first': first-place 24 checkpoint models, 4 flip TTA (batched), GPU weighted
averages and the published localization/damage thresholds + dilation.
Backend 'second': second-place 21 checkpoint models, 4 flip TTA (sequential),
weighted GPU averaging and simple damage/localization reductions. The original
CPU watershed postprocessing is NOT included; this is explicitly a GPU-scope
benchmark matching RS-BDNet's GPU timing scope (not end-to-end deployment).
Backend 'third': third-place 13 checkpoint models, weighted averaging,
HFlipTTA + multiscale (-128/+128) as in docker_submission_37.py, GPU
classification. The original CPU segmentation formatting is excluded.

IMPORTANT: The model-native resolution for all xView2 systems is 1024x1024;
RS-BDNet measured 896x896. Report this as a practical pipeline comparison,
not isolated architecture speed at identical resolution.

'--mode smoke' loads ONE native checkpoint and executes ONE forward.
'--mode full' loads ALL checkpoints; rejects partial checkpoint sets.
Models stay resident on GPU, so allocated GPU memory includes model weights.
CPU image loading/normalization is excluded from all timed iterations, and
model load time is excluded. Includes H2D tensor transfers, GPU TTA, full
ensemble forward, GPU accumulation / mask postprocessing, and synchronization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT = Path('/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet')
WINNERS = PROJECT / 'baselines' / 'xview2_winners'
DATA_IMAGES = Path('/homes/j244s673/documents/wsu/phd/xview2/test/images')
MIB = 1024.0**2
MEAN = torch.tensor([.485,.456,.406,.485,.456,.406]).view(6,1,1)
STD = torch.tensor([.229,.224,.225,.229,.224,.225]).view(6,1,1)


def stamp(t0, msg):
    print(f'[+{time.perf_counter()-t0:8.2f}s] {msg}', flush=True)


def checkpoints_first():
    loc = [
        ('Res34_Unet_Loc', f'res34_loc_{i}_1_best') for i in range(3)
    ] + [
        ('SeResNext50_Unet_Loc', f'res50_loc_{i}_tuned_best') for i in range(3)
    ] + [
        ('Dpn92_Unet_Loc', f'dpn92_loc_{i}_tuned_best') for i in range(3)
    ] + [
        ('SeNet154_Unet_Loc', f'se154_loc_{i}_1_best') for i in range(3)
    ]
    cls = [
        ('Res34_Unet_Double', f'res34_cls2_{i}_tuned_best') for i in range(3)
    ] + [
        ('SeResNext50_Unet_Double', f'res50_cls_cce_{i}_tuned_best') for i in range(3)
    ] + [
        ('Dpn92_Unet_Double', f'dpn92_cls_cce_{i}_tuned_best') for i in range(3)
    ] + [
        ('SeNet154_Unet_Double', f'se154_cls_cce_{i}_tuned_best') for i in range(3)
    ]
    return loc, cls


def extract_dict(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    raise RuntimeError('Checkpoint must contain a state dictionary')


def load_first(device, full, t0):
    repo = WINNERS / 'xview2_first_place'
    sys.path.insert(0, str(repo))
    from zoo import models as zoo
    loc, cls = checkpoints_first()
    chosen = loc + cls if full else loc[:1]
    weights = repo / 'weights' / 'weights'
    missing = [str(weights/name) for _,name in chosen if not (weights/name).is_file()]
    if missing: raise FileNotFoundError('Missing first-place checkpoints: ' + '\n'.join(missing))
    first_loc=[]; first_cls=[]; paths=[]
    for index, (model_cls, fname) in enumerate(chosen,1):
        stamp(t0, f'FIRST checkpoint {index}/{len(chosen)}: {fname}')
        m = getattr(zoo, model_cls)(pretrained=(False if model_cls.startswith('Res34_') else None))
        ckpt = torch.load(weights/fname, map_location='cpu', weights_only=False)
        state = extract_dict(ckpt)
        state = {k[7:] if k.startswith('module.') else k: v for k,v in state.items()}
        own = m.state_dict()
        matching = {k:v for k,v in state.items() if k in own and own[k].shape==v.shape}
        ratio = len(matching) / max(len(own),1)
        if ratio < .90:
            raise RuntimeError(f'First place {fname}: only {len(matching)}/{len(own)} state tensors match')
        own.update(matching)
        m.load_state_dict(own, strict=True)
        m = m.to(device).eval().requires_grad_(False)
        if 'Loc' in model_cls: first_loc.append(m)
        else: first_cls.append(m)
        paths.append(str(weights/fname))
        del ckpt, state, own, matching
    return FirstBackend(first_loc, first_cls, device, paths, full)


def four_tta(x):
    return torch.cat([x, x.flip(-2), x.flip(-1), x.flip((-2,-1))],dim=0)


def undo_four(x):
    return (x[0:1]+x[1:2].flip(-2)+x[2:3].flip(-1)+x[3:4].flip((-2,-1))) / 4.0


class FirstBackend:
    def __init__(self, loc, cls, device, paths, full):
        self.loc,self.cls,self.device,self.paths,self.full=loc,cls,device,paths,full
        self.n_models = len(paths)
        self.input_resolution = 1024
        self.protocol = '24 models / 4-flip batched TTA / GPU ensemble and threshold / exclude CPU I-O'
    @torch.inference_mode()
    def predict(self, pair):
        pre,post=pair
        img = torch.cat([pre,post],dim=0).unsqueeze(0).to(self.device,non_blocking=False)
        xp=four_tta(img[:,:3]); both=four_tta(img)
        loc=None; dmg=None
        for m in self.loc:
            p=undo_four(m(xp).sigmoid())
            loc = p if loc is None else loc + p
        if not self.full:
            return loc
        for m in self.cls:
            p=undo_four(m(both).sigmoid())
            dmg = p if dmg is None else dmg + p
        loc = loc / len(self.loc)
        dmg = dmg / len(self.cls)
        d = dmg[:,1:].argmax(dim=1) + 1
        b = (loc[:,0] > .38) | ((loc[:,0] > .13)&(d>1)&(d<4)) | ((loc[:,0] > .14)&(d>1))
        mask=d*b.long()
        # First-place submission code dilates class-2 (minor damage) into class-1 (no damage).
        from torch.nn import functional as F
        mask_minor=F.max_pool2d((mask==2).float().unsqueeze(1),5,stride=1,padding=2)[:,0]>0
        mask = torch.where(mask_minor&(mask==1),torch.tensor(2,device=self.device),mask)
        return b,mask


def second_configs():
    # Parse the official 21-entry list without importing predict_single.py,
    # which includes legacy scikit-image imports irrelevant to model inference.
    import ast
    from types import SimpleNamespace
    repo=WINNERS/'xview2_second_place'
    f=repo/'predict_single.py'
    tree=ast.parse(f.read_text(encoding='utf-8'))
    entries=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='configs' for t in node.targets):
            entries=node.value.elts
            break
    if entries is None: raise RuntimeError(f'Unable to find second-place configs in {f}')
    configs=[]
    for call in entries:
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='ModelConfig':
            raise RuntimeError('Second-place ModelConfig list format changed')
        fields=[ast.literal_eval(arg) for arg in call.args]
        if len(fields)!=4: raise RuntimeError('Unexpected second-place ModelConfig fields')
        configs.append(SimpleNamespace(config_path=fields[0],weight_path=fields[1],type=fields[2],weight=fields[3]))
    return configs


def load_second(device, full, t0):
    repo=WINNERS/'xview2_second_place'
    confs=second_configs()
    if full and len(confs)!=21: raise RuntimeError(f'Second official list has {len(confs)} configs, expected 21')
    chosen=confs if full else confs[:1]
    import models
    from tools.config import load_config
    loc=[];dmg=[];paths=[]
    for i,c in enumerate(chosen,1):
        path=repo/'weights'/c.weight_path
        if not path.is_file(): raise FileNotFoundError(f'Checkpoint not found: {path}')
        stamp(t0,f'SECOND checkpoint {i}/{len(chosen)}: {c.weight_path}')
        conf=load_config(str(repo/c.config_path))
        m=models.__dict__[conf['network']](seg_classes=(1 if c.type=='localization' else 5),
                                          backbone_arch=conf['encoder'])
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        state=extract_dict(ckpt)
        state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
        m.load_state_dict(state,strict=True)
        m=m.to(device).eval().requires_grad_(False)
        if c.type=='localization': loc.append((m,c))
        else: dmg.append((m,c))
        paths.append(str(path))
        del ckpt,state
    return SecondBackend(loc,dmg,device,paths,full)


class SecondBackend:
    def __init__(self,loc,dmg,device,paths,full):
        self.loc,self.dmg,self.device,self.paths,self.full=loc,dmg,device,paths,full
        self.n_models=len(paths)
        self.input_resolution=1024
        self.protocol='21 models / 4-flip sequential TTA / GPU averaged masks; CPU watershed omitted'
    def tta(self,model,x,dpn_pad=False,damage=False):
        from torch.nn import functional as F
        if dpn_pad: x=F.pad(x,(16,16,16,16),'reflect')
        pred=None
        for idx in range(4):
            xx = x if idx==0 else (x.flip(-2) if idx==1 else (x.flip(-1) if idx==2 else x.flip((-2,-1))))
            raw=model(xx)
            out = raw.softmax(1) if damage else raw.sigmoid()
            if idx==1:out=out.flip(-2)
            elif idx==2:out=out.flip(-1)
            elif idx==3:out=out.flip((-2,-1))
            pred=out if pred is None else pred + out
        pred = pred / 4.0
        if dpn_pad: pred=pred[:,:,16:-16,16:-16]
        return pred
    @torch.inference_mode()
    def predict(self,pair):
        pre,post=pair
        xx=torch.cat([pre,post],dim=0).unsqueeze(0).to(self.device)
        loc=None; dmg=None; wsum=0
        for m,c in self.loc:
            pred=self.tta(m,xx[:,:3],dpn_pad=('dpn' in c.weight_path),damage=False)
            loc=pred if loc is None else loc + pred
        if not self.full: return loc
        for m,c in self.dmg:
            pred=self.tta(m,xx,damage=True)
            w=float(c.weight)
            dmg=pred*w if dmg is None else dmg+pred*w
            wsum += w
        loc/=len(self.loc)
        dmg/=wsum
        # Original second-place CPU watershed postprocess is not included.
        # This GPU-stage summary is not an exact exported mask reproduction.
        positive=dmg[:,1:].max(1).values
        building=(loc[:,0]>.25) | (positive>.25)
        damage=dmg[:,1:].argmax(1)+1
        return building, damage*building.long()


def third_weights():
    # Extract the published 13 checkpoint/weight pairs directly from the
    # original implementation's AST. Do NOT hardcode a potentially wrong list.
    import ast
    f=WINNERS/'xview2_third_place'/'docker_submission_37.py'
    tree=ast.parse(f.read_text())
    env={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id.startswith('fold_') and target.id.endswith('_models_dict'):
                    env[target.id]=ast.literal_eval(node.value)
    labels=[f'fold_{i}_models_dict' for i in range(5)]
    if any(k not in env for k in labels):raise RuntimeError('Could not locate third-place ensemble model lists in docker_submission_37.py')
    pairs=sum((env[k] for k in labels),[])
    if len(pairs)!=13:raise RuntimeError(f'Third model list count={len(pairs)}, expected 13')
    return pairs


def load_third(device, full, t0):
    repo=WINNERS/'xview2_third_place'
    sys.path.insert(0,str(repo))
    from xview.inference import model_from_checkpoint,ApplyWeights,Ensembler, HFlipTTA, MultiscaleTTA
    from xview.dataset import OUTPUT_MASK_KEY
    import xview.inference as inf
    class QuietProgress:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def update(self,*a):pass
    inf.tqdm=QuietProgress
    pairs=third_weights()
    chosen=pairs if full else pairs[:1]
    models=[]; paths=[]
    weights_dir=repo/'weights'/'extracted'/'ensemble'
    for i,(fname,weights) in enumerate(chosen,1):
        path=weights_dir/fname
        if not path.is_file():raise FileNotFoundError(f'Missing third-place checkpoint: {path}')
        stamp(t0,f'THIRD checkpoint {i}/{len(chosen)}: {fname}')
        m,info=model_from_checkpoint(str(path),activation_after='model',report=False,classifiers=False)
        m=ApplyWeights(m,weights).to(device).eval().requires_grad_(False)
        models.append(m); paths.append(str(path))
    ensemble=Ensembler(models,outputs=[OUTPUT_MASK_KEY])
    # Preserve the official docker inference chain: HFlipTTA then MultiscaleTTA.
    if full:
        ensemble=HFlipTTA(ensemble,outputs=[OUTPUT_MASK_KEY],average=True)
        ensemble=MultiscaleTTA(ensemble,outputs=[OUTPUT_MASK_KEY],size_offsets=[-128,+128],average=True)
    ensemble=ensemble.to(device).eval().requires_grad_(False)
    return ThirdBackend(ensemble,device,paths,full)


class ThirdBackend:
    def __init__(self,ensemble,device,paths,full):
        self.ensemble,self.device,self.paths,self.full=ensemble,device,paths,full
        self.n_models=len(paths)
        self.input_resolution=1024
        self.protocol='13 models / weighted average / horizontal-flip + +/-128 multiscale TTA / GPU argmax'
    @torch.inference_mode()
    def predict(self,pair):
        from xview.dataset import OUTPUT_MASK_KEY
        pre,post=pair
        x=torch.cat([pre,post],0).unsqueeze(0).to(self.device)
        p=self.ensemble(x)[OUTPUT_MASK_KEY]
        pred=p.argmax(1)
        return pred>0,pred


def find_images(root):
    folders=[root/'test'/'images',root/'images',root]
    d=next((p for p in folders if p.is_dir()),None)
    if d is None:raise FileNotFoundError(f'No xBD test images directory under {root}')
    candidates=sorted(p for p in d.iterdir() if p.is_file() and '_pre_' in p.name and p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'))
    pairs=[]
    for p in candidates:
        post=p.with_name(p.name.replace('_pre_','_post_',1))
        if post.is_file():pairs.append((p,post))
    if not pairs:raise FileNotFoundError(f'No paired *_pre_* and *_post_* files under {d}')
    return pairs


def preprocess_pair(pre,post,backend):
    import cv2
    a=cv2.imread(str(pre),cv2.IMREAD_COLOR)
    b=cv2.imread(str(post),cv2.IMREAD_COLOR)
    if a is None or b is None:raise RuntimeError(f'Cannot decode {pre} or {post}')
    if a.shape[:2]!=(1024,1024):a=cv2.resize(a,(1024,1024),interpolation=cv2.INTER_LINEAR)
    if b.shape[:2]!=(1024,1024):b=cv2.resize(b,(1024,1024),interpolation=cv2.INTER_LINEAR)
    if backend=='first':
        # First-place preprocessing explicitly uses BGR pixels / 127 - 1.
        aa=(a.astype(np.float32)/127.0)-1
        bb=(b.astype(np.float32)/127.0)-1
        t1=torch.from_numpy(aa.transpose(2,0,1).copy()).contiguous()
        t2=torch.from_numpy(bb.transpose(2,0,1).copy()).contiguous()
    else:
        # Second and third official inference uses RGB + ImageNet statistics.
        ar=a[:,:,::-1].copy(); br=b[:,:,::-1].copy()
        aa=torch.from_numpy(ar.transpose(2,0,1).copy()).float()/255.
        bb=torch.from_numpy(br.transpose(2,0,1).copy()).float()/255.
        t1=(aa-MEAN[:3])/STD[:3];t2=(bb-MEAN[:3])/STD[:3]
    return t1.contiguous(),t2.contiguous()


def build_backend(name, device, full, t0):
    return {'first':load_first,'second':load_second,'third':load_third}[name](device,full,t0)


def run(args):
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required')
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    full=args.mode=='full'
    expected={'first':24,'second':21,'third':13}[args.backend]
    if full and args.repeats<2:raise ValueError('Full benchmark must use at least 2 timed iterations')
    output_root=PROJECT/'output'/'Reviewer2_Comment7_InferenceBench'
    output_dir=(output_root/args.output_name).resolve()
    if output_root.resolve() not in output_dir.parents:raise ValueError('Output must be inside reviewer-specific directory')
    output_dir.mkdir(parents=True,exist_ok=False)
    t0=time.perf_counter()
    stamp(t0,f'Backend={args.backend}; mode={args.mode}; cudnn.enabled={torch.backends.cudnn.enabled}')
    stamp(t0,f'GPU={torch.cuda.get_device_name(0)}; torch={torch.__version__}')
    stamp(t0,'Loading native models and checkpoints (excluded from timed inference)')
    torch.cuda.reset_peak_memory_stats()
    backend=build_backend(args.backend,device,full,t0)
    if full and backend.n_models!=expected:raise RuntimeError(f'Only {backend.n_models}/{expected} checkpoints loaded; refusing partial ensemble benchmark')
    torch.cuda.synchronize()
    stamp(t0,f'Loaded {backend.n_models} of {expected} checkpoints; GPU allocation={torch.cuda.memory_allocated()/MIB:.2f} MiB')
    allpairs=find_images(args.xbd_root)
    # Reuse same sorted xBD test samples for every ensemble.
    n_samples=1 if args.mode=='smoke' else args.samples
    if len(allpairs)<n_samples:raise ValueError(f'Only {len(allpairs)} image pairs, need {n_samples}')
    selected=allpairs[:n_samples]
    cpu_pairs=[]
    for i,(pre,post) in enumerate(selected):
        cpu_pairs.append(preprocess_pair(pre,post,args.backend))
        if (i+1)%10==0:stamp(t0,f'Prepared {i+1}/{len(selected)} pairs on CPU')
    stamp(t0,f'Loaded {len(cpu_pairs)} preprocessed xBD pairs to CPU cache')
    warmup=1 if args.mode=='smoke' else args.warmup
    repeats=1 if args.mode=='smoke' else args.repeats
    with torch.inference_mode():
        for i in range(warmup):
            t=time.perf_counter()
            output=backend.predict(cpu_pairs[i%len(cpu_pairs)])
            torch.cuda.synchronize()
            stamp(t0,f'Warmup {i+1}/{warmup} finished in {1000*(time.perf_counter()-t):.2f} ms')
        durations=[]
        for i in range(repeats):
            t=time.perf_counter()
            output=backend.predict(cpu_pairs[i%len(cpu_pairs)])
            torch.cuda.synchronize()
            dt=(time.perf_counter()-t)*1000
            durations.append(dt)
            if args.mode=='smoke' or i<3 or (i+1)%max(1,repeats//10)==0:
                stamp(t0,f'Timed {i+1}/{repeats}: {dt:.2f} ms')
        del output
    torch.cuda.synchronize()
    mean=statistics.mean(durations)
    memory_alloc=torch.cuda.max_memory_allocated()/MIB
    memory_res=torch.cuda.max_memory_reserved()/MIB
    summary={
        'backend':f'xView2 {args.backend}-place ensemble',
        'mode':args.mode,
        'complete_ensemble':full,
        'checkpoints_loaded':backend.n_models,
        'expected_checkpoints':expected,
        'input_size':1024,
        'batch_size':1,
        'cudnn_enabled':False,
        'precision':'float32',
        'warmup':warmup,
        'iterations':repeats,
        'num_unique_test_pairs':len(selected),
        'mean_latency_ms_pair':round(mean,4),
        'median_latency_ms_pair':round(statistics.median(durations),4),
        'p95_latency_ms_pair':round(float(np.percentile(durations,95)),4),
        'pairs_per_second':round(1000./mean,4),
        'torch_peak_allocated_mib':round(memory_alloc,3),
        'torch_peak_reserved_mib':round(memory_res,3),
        'torch_post_load_allocated_mib':round(torch.cuda.memory_allocated()/MIB,3),
        'torch_version':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(0),
        'scope':backend.protocol,
        'no_disk_io_in_timing':True,
        'notes':'Smoke: ONLY ONE checkpoint; NOT a complete ensemble performance result' if not full else 'Full checkpoint set; GPU-only segmentation reductions, not native CPU watershed/image export; 1024 native input vs RS-BDNet 896',
        'checkpoints':backend.paths,
        'xBD_pair_names':[p.name for p,_ in selected],
    }
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output_dir/'summary.csv').open('w',newline='') as f:
        cols=[k for k,v in summary.items() if isinstance(v,(int,float,str,bool))]
        w=csv.DictWriter(f,cols); w.writeheader();w.writerow({k:summary[k] for k in cols})
    with (output_dir/'iterations.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['iteration','latency_ms']);w.writerows((i+1,x) for i,x in enumerate(durations))
    print('\nSUMMARY: '+json.dumps({k:v for k,v in summary.items() if k in ('backend','mode','complete_ensemble','mean_latency_ms_pair','pairs_per_second','torch_peak_allocated_mib','torch_peak_reserved_mib')},indent=2),flush=True)
    stamp(t0,f'Saved to {output_dir}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['first','second','third'],required=True)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--xbd-root',type=Path,default=Path('/homes/j244s673/documents/wsu/phd/xview2'))
    p.add_argument('--samples',type=int,default=100)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--output-name',required=True)
    args=p.parse_args()
    if min(args.samples,args.warmup,args.repeats)<1: p.error('samples, warmup, repeats must be >=1')
    try: run(args)
    except Exception as e:
        print(f'BENCHMARK FAILED: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        traceback.print_exc()
        raise

if __name__=='__main__':main()
