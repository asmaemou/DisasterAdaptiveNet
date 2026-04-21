from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xview2_stca_lib import (
    BCEPlusCELoss,
    RunningAverage,
    XView2Dataset,
    build_model,
    create_optimizer,
    evaluate_model,
    format_metrics,
    load_checkpoint,
    sample_source_damage_features,
    sample_target_building_features,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Source training + STCA adaptation on xView2.')
    p.add_argument('--root', type=str, required=True)
    p.add_argument('--work-dir', type=str, required=True)
    p.add_argument('--cache-dir', type=str, default=None)
    p.add_argument('--source-split', type=str, default='tier3')
    p.add_argument('--target-split', type=str, default='train')
    p.add_argument('--val-split', type=str, default='test')
    p.add_argument('--eval-split', type=str, default='hold')
    p.add_argument('--crop-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--source-epochs', type=int, default=12)
    p.add_argument('--adapt-epochs', type=int, default=6)
    p.add_argument('--source-lr', type=float, default=1e-4)
    p.add_argument('--adapt-lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--stca-weight', type=float, default=0.2)
    p.add_argument('--target-threshold', type=float, default=0.5)
    p.add_argument('--target-feature-samples', type=int, default=64)
    p.add_argument('--source-feature-samples', type=int, default=64)
    p.add_argument('--resume-source', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-train-items', type=int, default=None)
    p.add_argument('--max-target-items', type=int, default=None)
    p.add_argument('--max-val-items', type=int, default=None)
    return p.parse_args()


def make_loader(ds, batch_size: int, workers: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, drop_last=shuffle)


@torch.no_grad()
def dump_metrics(model, loader, device, out_path: Path, split_name: str):
    metrics = evaluate_model(model, loader, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'split': split_name, 'metrics': metrics}, f, indent=2)
    return metrics


def train_source(model, train_loader, val_loader, device, work_dir: Path, args):
    criterion = BCEPlusCELoss()
    optimizer = create_optimizer(model, lr=args.source_lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_score = -1.0

    if args.resume_source:
        ckpt = load_checkpoint(args.resume_source, model, optimizer)
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_score = float(ckpt.get('best_score', -1.0))
        print(f"[source] resumed from {args.resume_source}")

    best_path = work_dir / 'checkpoints' / 'source_best.pt'
    last_path = work_dir / 'checkpoints' / 'source_last.pt'

    for epoch in range(start_epoch, args.source_epochs):
        model.train()
        losses = RunningAverage()
        for batch in train_loader:
            pre = batch['pre'].to(device, non_blocking=True)
            post = batch['post'].to(device, non_blocking=True)
            loc = batch['loc'].to(device, non_blocking=True)
            dam = batch['dam'].to(device, non_blocking=True)

            out = model(pre, post)
            loss = criterion(out['loc_logits'], loc, out['dam_logits'], dam)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.update(loss.item(), n=pre.size(0))

        val_metrics = evaluate_model(model, val_loader, device)
        print(f"[source] epoch={epoch+1}/{args.source_epochs} loss={losses.avg:.4f} | {format_metrics(val_metrics)}")

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_score': best_score,
            'stage': 'source',
            'args': vars(args),
        }
        save_checkpoint(last_path, state)
        if val_metrics['f1_avg'] > best_score:
            best_score = val_metrics['f1_avg']
            state['best_score'] = best_score
            save_checkpoint(best_path, state)
            print(f"[source] new best checkpoint -> {best_path}")

    return best_path if best_path.exists() else last_path


def adapt_stca(model, source_loader, target_loader, val_loader, device, work_dir: Path, args):
    optimizer = create_optimizer(model, lr=args.adapt_lr, weight_decay=args.weight_decay)
    ce_loss = torch.nn.CrossEntropyLoss()
    best_score = -1.0
    best_path = work_dir / 'checkpoints' / 'stca_best.pt'
    last_path = work_dir / 'checkpoints' / 'stca_last.pt'

    for epoch in range(args.adapt_epochs):
        model.train()
        losses = RunningAverage()
        tgt_iter = iter(target_loader)
        for src_batch in source_loader:
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                tgt_batch = next(tgt_iter)

            src_post = src_batch['post'].to(device, non_blocking=True)
            src_dam = src_batch['dam'].to(device, non_blocking=True)
            tgt_pre = tgt_batch['pre'].to(device, non_blocking=True)

            pre_feat, loc_low = model.encode_pre(tgt_pre)
            post_feat = model.encode_post(src_post)

            tgt_feats = sample_target_building_features(pre_feat, loc_low, args.target_threshold, args.target_feature_samples)
            src_feats, src_labels = sample_source_damage_features(post_feat, src_dam, args.source_feature_samples, include_classes=(2, 3, 4))
            if tgt_feats is None or src_feats is None or src_labels is None:
                continue

            logits = model.classify_pair_features(src_feats, tgt_feats)
            labels = src_labels[:, None].expand(src_feats.size(0), tgt_feats.size(0)).reshape(-1)
            loss = args.stca_weight * ce_loss(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.update(loss.item(), n=1)

        val_metrics = evaluate_model(model, val_loader, device)
        print(f"[stca] epoch={epoch+1}/{args.adapt_epochs} loss={losses.avg:.4f} | {format_metrics(val_metrics)}")

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_score': best_score,
            'stage': 'stca',
            'args': vars(args),
        }
        save_checkpoint(last_path, state)
        if val_metrics['f1_avg'] > best_score:
            best_score = val_metrics['f1_avg']
            state['best_score'] = best_score
            save_checkpoint(best_path, state)
            print(f"[stca] new best checkpoint -> {best_path}")

    return best_path if best_path.exists() else last_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = str(work_dir / 'mask_cache')

    source_train = XView2Dataset(args.root, args.source_split, 'supervised', args.crop_size, True, args.cache_dir, args.max_train_items)
    source_adapt = XView2Dataset(args.root, args.source_split, 'source_post', args.crop_size, True, args.cache_dir, args.max_train_items)
    target_train = XView2Dataset(args.root, args.target_split, 'target_pre', args.crop_size, True, args.cache_dir, args.max_target_items)
    val_set = XView2Dataset(args.root, args.val_split, 'supervised', args.crop_size, False, args.cache_dir, args.max_val_items)
    eval_set = XView2Dataset(args.root, args.eval_split, 'supervised', args.crop_size, False, args.cache_dir, None)

    source_train_loader = make_loader(source_train, args.batch_size, args.workers, True)
    source_adapt_loader = make_loader(source_adapt, args.batch_size, args.workers, True)
    target_loader = make_loader(target_train, args.batch_size, args.workers, True)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False)
    eval_loader = make_loader(eval_set, args.batch_size, args.workers, False)

    model = build_model(device)

    source_ckpt = train_source(model, source_train_loader, val_loader, device, work_dir, args)
    load_checkpoint(source_ckpt, model)

    val_metrics = dump_metrics(model, val_loader, device, work_dir / 'metrics' / 'source_test.json', args.val_split)
    print(f"[eval] source-only {args.val_split}: {format_metrics(val_metrics)}")
    hold_metrics = dump_metrics(model, eval_loader, device, work_dir / 'metrics' / 'source_hold.json', args.eval_split)
    print(f"[eval] source-only {args.eval_split}: {format_metrics(hold_metrics)}")

    stca_ckpt = adapt_stca(model, source_adapt_loader, target_loader, val_loader, device, work_dir, args)
    load_checkpoint(stca_ckpt, model)

    val_metrics = dump_metrics(model, val_loader, device, work_dir / 'metrics' / 'stca_test.json', args.val_split)
    print(f"[eval] STCA {args.val_split}: {format_metrics(val_metrics)}")
    hold_metrics = dump_metrics(model, eval_loader, device, work_dir / 'metrics' / 'stca_hold.json', args.eval_split)
    print(f"[eval] STCA {args.eval_split}: {format_metrics(hold_metrics)}")

    with open(work_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
