#!/usr/bin/env python3
"""Reviewer 3 Comment 4: reproducible BUILDING-POLYGON count audit for xBD.

No GPU, model checkpoints, dataset modification, or network connection required.
Counts individual *post-disaster JSON building-polygon features* by their recorded
'damage subtype', NOT image pairs, pixels, masks or inferred connected components.
Produces independent per-scene class-support counts as well.

Expected original xBD layout:
 <xbd_root>/{train,tier3,hold,test}/labels/*_post_disaster.json
Set --xbd-root to the original xview2 root. Missing JSON is explicitly shown;
no mask-component proxy is substituted for polygon counts.

Optional --ebd-base scans five EBD _preprocessed directories and reports
image-pair presence and 8-connected mask components as a separately labeled
PROXY (not actual building polygons). Not used in the xBD polygon table.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, sys
from pathlib import Path

CLASSES = ('no_damage', 'minor_damage', 'major_damage', 'destroyed')
SPLITS = ('train', 'tier3', 'hold', 'test')
EXPECTED_PAIRS = {'train_tier3':9168, 'hold':933, 'test':933}
EBD = collections.OrderedDict([
  ('Earthquake Turkey', ('earthquake_turkey_preprocessed', (755,94,95))),
  ('Mount Semeru Eruption', ('mount_semeru_eruption_preprocessed', (473,59,60))),
  ('Texas Tornadoes', ('texas_tornadoes_preprocessed', (363,45,46))),
  ('Hurricane Delta', ('hurricane_delta_preprocessed', (1091,136,137))),
  ('Pakistan Flooding', ('pakistan_flooding_preprocessed', (2832,354,354))),
])


def canonical_split(p: str) -> str | None:
    p=p.lower().replace('-', '').replace('_','')
    if p in ('train','training'):return 'train'
    if p in ('tier3',):return 'tier3'
    if p in ('hold','holdout','val','validation'):return 'hold'
    if p in ('test','testing'):return 'test'
    return None


def norm_label(value):
    s = str(value).strip().lower().replace('_','-').replace(' ','-')
    mapping = {'no-damage':'no_damage','no-damage-label':'unknown',
        'undamaged':'no_damage','minor-damage':'minor_damage',
        'major-damage':'major_damage','destroyed':'destroyed',
        'un-classified':'unclassified','unclassified':'unclassified',
        'none':'unknown','null':'unknown','':'unknown'}
    return mapping.get(s, 'other:'+s)


def features_from_json(doc, src):
    # xBD has both features.xy and features.lng_lat: use xy ONLY to avoid
    # counting the same polygon in two coordinate systems.
    obj=doc.get('features',{})
    if isinstance(obj,dict):
        features=obj.get('xy')
        if features is None:
            raise ValueError(f'{src}: JSON has no features.xy; cannot safely count polygons')
    elif isinstance(obj,list):
        features=obj
    else:
        raise ValueError(f'{src}: unexpected features structure: {type(obj).__name__}')
    if not isinstance(features,list):
        raise ValueError(f'{src}: polygon features must be a list')
    return features


def tally_feature(feature):
    if not isinstance(feature,dict):return 'other:invalid-feature'
    props=feature.get('properties') or {}
    ftype=str(props.get('feature_type','building')).strip().lower()
    if ftype and ftype!='building':return 'other:nonbuilding:'+ftype
    return norm_label(props.get('subtype',props.get('damage',props.get('damage_type'))))


def write_csv(dst, rows):
    if not rows:return
    fields=sorted(set().union(*(r.keys() for r in rows)))
    fields=[k for k in ('dataset','split','unit','image_pairs','json_label_pairs','classified_polygon_instances',
                        *CLASSES,'unclassified','unknown','other_labels','notes') if k in fields] + [k for k in fields if k not in ('dataset','split','unit','image_pairs','json_label_pairs','classified_polygon_instances',*CLASSES,'unclassified','unknown','other_labels','notes')]
    with dst.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def base_row(dataset,split,unit):
    return {'dataset':dataset,'split':split,'unit':unit,'image_pairs':0,'json_label_pairs':0,
            'classified_polygon_instances':0,**{k:0 for k in CLASSES},
            'unclassified':0,'unknown':0,'other_labels':0,'notes':''}


def scan_xbd(root):
    allrows=[]; pairrows=[]; manifest=[]; available=[]
    for split in SPLITS:
        dirs=[p for p in root.iterdir() if p.is_dir() and canonical_split(p.name)==split]
        row=base_row('xBD',split,'post-disaster JSON building polygons')
        if len(dirs)!=1:
            row['notes']=f'Expected exactly one {split} directory, found: '+', '.join(str(x) for x in dirs)
            allrows.append(row);continue
        d=dirs[0]; labels=d/'labels'; images=d/'images'
        # Locate only actual xBD JSONs, never file contents in another directory.
        candidates=sorted(labels.rglob('*_post_disaster.json')) if labels.exists() else []
        # Count image pairs independently.
        image_files=sorted(images.rglob('*_post_disaster.png')) if images.exists() else []
        row['image_pairs']=len(image_files)
        row['json_label_pairs']=len(candidates)
        if not candidates:
            row['notes']='NO POST-DISASTER JSON LABELS: cannot calculate building-polygon counts (do not report zeros)'
            allrows.append(row);continue
        seen=set()
        for p in candidates:
            stem=p.name.removesuffix('_post_disaster.json')
            if stem in seen:raise ValueError(f'Duplicate scene {stem} in xBD {split}')
            seen.add(stem)
            with p.open(encoding='utf-8') as f: doc=json.load(f)
            fs=features_from_json(doc,p)
            s=collections.Counter(tally_feature(z) for z in fs)
            for cls in CLASSES: row[cls] += s[cls]
            row['unclassified'] += s['unclassified']
            row['unknown'] += s['unknown']
            row['other_labels'] += sum(v for k,v in s.items() if k.startswith('other:'))
            prow={'split':split,'stem':stem,'polygon_total_all_subtypes':sum(s.values()),
                  **{c:int(s[c]) for c in CLASSES},'unclassified':int(s['unclassified']),
                  'unknown':int(s['unknown']),
                  'other_labels':sum(v for k,v in s.items() if k.startswith('other:')),
                  **{'pairs_with_'+c:int(s[c]>0) for c in CLASSES},
                  'annotation_path':str(p)}
            pairrows.append(prow)
            st=p.stat();manifest.append((str(p.relative_to(root)),st.st_size))
        row['classified_polygon_instances']=sum(row[c] for c in CLASSES)
        if row['json_label_pairs']!=row['image_pairs']:
            row['notes']='WARNING: Number of post-label files differs from image pairs; inspect data inventory'
        available.append(split)
        allrows.append(row)
    if 'train' in available and 'tier3' in available:
        combo=base_row('xBD','train_tier3','post-disaster JSON building polygons')
        for row in allrows[:2]:
            for k in ('image_pairs','json_label_pairs','classified_polygon_instances',*CLASSES,
                      'unclassified','unknown','other_labels'):combo[k]+=row[k]
        combo['notes']='Sum of original Train and Tier3; manuscript training split'
        allrows.append(combo)
    # The one combined table to cite is train_tier3, hold, test, ONLY if complete.
    source_lookup={r['split']:r for r in allrows}
    partition_rows=[]
    for manuscript_split,source in (('Training','train_tier3'),('Validation','hold'),('Test','test')):
        if source not in source_lookup or source not in available and source!='train_tier3':
            continue
        r=dict(source_lookup[source]);r['split']=manuscript_split
        partition_rows.append(r)
    support_rows=[]
    for s in SPLITS+('train_tier3',):
        if s=='train_tier3':
            subset=[r for r in pairrows if r['split'] in ('train','tier3')]
        else:subset=[r for r in pairrows if r['split']==s]
        if not subset:continue
        support_rows.append({'dataset':'xBD','split':s,'labeled_image_pairs':len(subset),
            **{'pairs_containing_'+c:sum(int(r[c]>0) for r in subset) for c in CLASSES}})
    return allrows,partition_rows,pairrows,support_rows,manifest


def scan_ebd_proxy(base):
    # The EBD subsets were preprocessed into image-pair / class-mask folders;
    # touching buildings are merged; components are NOT verified building counts.
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return [],[f'EBD proxy unavailable (optional cv2/numpy import failed): {exc}']
    rows=[];issues=[]
    for name,(dirname,expected) in EBD.items():
        d=base/dirname
        for split,exp in zip(('train','val','test'),expected):
            dd=d/split; target=dd/'targets'
            row=base_row(name,split,'8-connected per-class mask regions (PROXY; not building polygons)')
            paths=sorted(target.glob('*_post_disaster_target.png')) if target.exists() else []
            row['image_pairs']=len(paths)
            row['json_label_pairs']='N/A'
            if not paths:
                row['notes']='Missing preprocessed post masks; counts unavailable'
                issues.append(f'{name}/{split}: no target masks at {target}')
                rows.append(row);continue
            if len(paths)!=exp:row['notes']=f'WARNING expected {exp} samples, found {len(paths)}'
            for p in paths:
                a=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
                if a is None:raise FileNotFoundError(p)
                if len(a.shape)!=2:raise ValueError(f'Expected 2D class-code mask: {p}, shape={a.shape}')
                codes=set(int(x) for x in np.unique(a))
                unexpected=codes.difference({0,1,2,3,4,255})
                if unexpected:raise ValueError(f'Unexpected EBD mask classes {unexpected} in {p}')
                # Count components within each class separately. This may
                # merge adjacent roofs or fragment a building: proxy ONLY.
                for i,c in enumerate(CLASSES,1):
                    n,_=cv2.connectedComponents((a==i).astype(np.uint8),connectivity=8)
                    row[c]+=int(n-1)
            row['classified_polygon_instances']='N/A'
            row['notes']+='; '.join(filter(None,[row['notes'],
                'Eight-connected regions from post-disaster target masks; NOT verified building-polygon counts']))
            rows.append(row)
    return rows,issues


def markdown_table(rows):
    s=['| Dataset | Split | Pairs | No damage | Minor | Major | Destroyed | Total classified |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        s.append('| '+ ' | '.join(str(r.get(k,'—')) for k in ('dataset','split','image_pairs',*CLASSES,'classified_polygon_instances'))+' |')
    return '\n'.join(s)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xbd-root',type=Path,default=Path('/homes/j244s673/documents/wsu/phd/xview2'))
    p.add_argument('--ebd-base',type=Path,default=None,help='Optional five-EBD mask-component proxy; NOT polygons')
    p.add_argument('--out',type=Path,default=Path('output/Reviewer3_Comment4_ClassCountAudit'))
    a=p.parse_args()
    if not a.xbd_root.is_dir():p.error(f'xBD root does not exist: {a.xbd_root}')
    a.out.mkdir(parents=True,exist_ok=True)
    rows,part,pairs,support,manifest=scan_xbd(a.xbd_root)
    write_csv(a.out/'xbd_original_split_polygon_counts.csv',rows)
    write_csv(a.out/'xbd_manuscript_train_val_test_polygon_counts.csv',part)
    write_csv(a.out/'xbd_per_image_polygon_counts.csv',pairs)
    write_csv(a.out/'xbd_pairs_containing_class.csv',support)
    ebd=[];issues=[]
    if a.ebd_base is not None:
        ebd,issues=scan_ebd_proxy(a.ebd_base)
        if ebd:write_csv(a.out/'ebd_mask_component_proxy_not_buildings.csv',ebd)
    digest=hashlib.sha256(''.join(f'{x}\t{y}\n' for x,y in manifest).encode()).hexdigest()
    report=['# Reviewer 3 Comment 4 class-count audit',
            f'xBD root: `{a.xbd_root.resolve()}`',
            'Statistical unit: **one post-disaster building polygon feature** '
            'from `features.xy` in each xBD JSON; pre-disaster polygons are never counted.',
            'All four classified labels count as one polygon each. `un-classified`, '
            '`unknown`, and other subtype values are reported separately.',
            f'Size-only manifest SHA-256 (not a data version): `{digest}`',
            '', '## xBD original partitions', markdown_table(rows), '',
            '## xBD experimental partitions (Train+Tier3 / Hold / Test)',
            markdown_table(part),'',
            '## Remaining limitations',
            'A zero in a row with NO LABELS is *missing data*, not evidence of no buildings.',
            'xBD public 850,736 total annotations and 411,357 four-class damage polygons '
            'are differently defined totals; do not infer split counts from either published total.',
            'Image-pair counts and polygon counts are different statistical units.',
            'The `xbd_pairs_containing_class.csv` file reports the number of labeled '
            'image pairs with at least one polygon from each class.',
            ''];
    if ebd:
        report+=['## EBD component-based proxy (NOT building polygons)',markdown_table(ebd),'',
                 'Do NOT merge these component counts into a table labeled "number of buildings".',
                 'Original EBD instance annotations are needed for validated polygon counts.','']
    if issues:report+=['Warnings:']+issues+['']
    (a.out/'README_AUDIT_RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report),flush=True)
    print(f'OUTPUT: {a.out.resolve()}')
    print('Upload xbd_manuscript_train_val_test_polygon_counts.csv and README_AUDIT_RESULTS.md')

if __name__=='__main__':main()
