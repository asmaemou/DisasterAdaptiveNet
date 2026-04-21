from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xview2_stca_lib import XView2Dataset, build_model, evaluate_model, format_metrics, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate a trained checkpoint on an xView2 split.')
    p.add_argument('--root', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--split', type=str, default='hold')
    p.add_argument('--crop-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cache-dir', type=str, default=None)
    p.add_argument('--out', type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)
    load_checkpoint(args.checkpoint, model)
    ds = XView2Dataset(args.root, args.split, 'supervised', args.crop_size, False, args.cache_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    metrics = evaluate_model(model, loader, device)
    print(format_metrics(metrics))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'split': args.split, 'metrics': metrics}, f, indent=2)


if __name__ == '__main__':
    main()
