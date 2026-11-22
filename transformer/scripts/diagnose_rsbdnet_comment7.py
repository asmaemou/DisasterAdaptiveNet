#!/usr/bin/env python3
"""Stage-by-stage diagnostic for Reviewer 2, Comment 7 timeout.

One xBD test image pair; NO training, NO checkpoint writes, NO full benchmark.
Use when the original benchmark stalls in warm-up. Logs each stage with flush.
"""
from __future__ import annotations
import argparse
import faulthandler
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0,p)

import torch
from benchmark_rsbdnet_comment7 import (
    DEFAULT_DATA, DEFAULT_CHECKPOINT, RsbdNetBackend, cpu_sample
)

T0 = time.perf_counter()
def stage(message: str):
    print(f'[+{time.perf_counter()-T0:9.2f}s] {message}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xbd-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--sample-index', type=int, default=0)
    args = p.parse_args()
    args.expected_sha256 = None

    faulthandler.enable()
    # Repeated stack traces if Python or a CUDA synchronization stalls for >2 minutes.
    faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)

    stage('BEGIN; this is a one-pair diagnostic, not a publishable benchmark')
    stage(f'PyTorch={torch.__version__}; CUDA available={torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check SLURM GPU allocation.')
    dev = torch.device('cuda:0')
    torch.cuda.set_device(0)
    stage(f'GPU={torch.cuda.get_device_name(dev)}')
    stage(f'CUDA_VISIBLE_DEVICES={__import__("os").environ.get("CUDA_VISIBLE_DEVICES")}')

    stage('Stage 1: instantiate xBD test dataset')
    from transformer.scripts.train_xbd_supervised_disasteradaptivenet import XBDOriginalDataset
    ds = XBDOriginalDataset(root=args.xbd_root, split='test', image_size=896,
                            training=False, conditioning_id=0)
    stage(f'Stage 1 done; dataset length={len(ds)}')
    if not (0 <= args.sample_index < len(ds)):
        raise IndexError(f'--sample-index {args.sample_index} out of range.')

    stage(f'Stage 2: decode and prepare ONE CPU pair (index={args.sample_index})')
    sample = cpu_sample(ds[args.sample_index])
    stage(f'Stage 2 done; image={tuple(sample["img"].shape)}, dtype={sample["img"].dtype}, '
          f'conditioning={tuple(sample["cond_id"].shape)}')

    stage('Stage 3: load model and original checkpoint')
    b = RsbdNetBackend(dev, args)
    torch.cuda.synchronize(dev)
    stage(f'Stage 3 done; parameters={b.parameters_million:.3f}M; '
          f'current allocated GPU MiB={torch.cuda.memory_allocated(dev)/1024**2:.1f}')

    with torch.inference_mode():
        stage('Stage 4: transfer image and conditioning tensor to CUDA')
        x = sample['img'].unsqueeze(0).to(dev)
        cond = sample['cond_id'].unsqueeze(0).to(dev)
        torch.cuda.synchronize(dev)
        stage(f'Stage 4 done; input shape={tuple(x.shape)}')

        stage('Stage 5: FIRST network forward pass (possible expensive step)')
        t = time.perf_counter()
        logits = b.model(x, cond)
        torch.cuda.synchronize(dev)
        stage(f'Stage 5 done; forward={time.perf_counter()-t:.2f}s; '
              f'output shape={tuple(logits.shape)}')

        stage('Stage 6: building threshold and damage argmax')
        t = time.perf_counter()
        building = torch.sigmoid(logits[:, 0]) > b.threshold
        damage = (logits[:, 1:5].argmax(dim=1) + 1) * building.long()
        torch.cuda.synchronize(dev)
        stage(f'Stage 6 done; postprocess={time.perf_counter()-t:.2f}s; '
              f'masks={tuple(building.shape)}, {tuple(damage.shape)}')
        stage(f'Peak allocated GPU MiB={torch.cuda.max_memory_allocated(dev)/1024**2:.1f}; '
              f'peak reserved MiB={torch.cuda.max_memory_reserved(dev)/1024**2:.1f}')

        stage('Stage 7: SECOND forward pass (check whether warm-up is costly)')
        t = time.perf_counter()
        logits = b.model(x, cond)
        torch.cuda.synchronize(dev)
        stage(f'Stage 7 done; second forward={time.perf_counter()-t:.2f}s')

    faulthandler.cancel_dump_traceback_later()
    stage('COMPLETE. Send this stdout and stderr to diagnose the full benchmark.')

if __name__ == '__main__':
    main()
