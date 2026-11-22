#!/usr/bin/env python3
"""Short, isolated diagnostic for RS-BDNet's stalled first CUDA inference.

No weights are modified, no training, no benchmark numbers are reported.
Run each mode in a SEPARATE process with a shell timeout.
"""
from __future__ import annotations
import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'transformer' / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

T0 = time.perf_counter()

def say(msg):
    print(f'[{time.perf_counter() - T0:8.2f}s] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', choices=('conv_default','conv_no_cudnn','model_default','model_no_cudnn'), required=True)
    args = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
    assert torch.cuda.is_available(), 'No CUDA GPU visible inside SLURM'
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = not args.test.endswith('no_cudnn')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('cuda:0')
    say(f'Test: {args.test}')
    say(f'PyTorch={torch.__version__}, CUDA={torch.version.cuda}, cuDNN={torch.backends.cudnn.version()}')
    say(f'GPU={torch.cuda.get_device_name(0)}; capability={torch.cuda.get_device_capability(0)}; cudnn.enabled={torch.backends.cudnn.enabled}; cudnn.benchmark={torch.backends.cudnn.benchmark}')

    if args.test.startswith('conv_'):
        # Basic Conv2d shape similar to the first ResNet CNN encoder layers.
        with torch.inference_mode():
            x = torch.randn((1, 3, 896, 896), device=device)
            conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).to(device).eval()
            say('ENTER standalone Conv2d(3->64, kernel=7, stride=2), input 1x3x896x896')
            y = conv1(x)
            torch.cuda.synchronize()
            say(f'EXIT first Conv2d, output={tuple(y.shape)}')
            conv2 = torch.nn.Conv2d(64, 64, 3, padding=1).to(device).eval()
            say('ENTER standalone Conv2d(64->64, kernel=3), input 1x64x448x448')
            z = conv2(y)
            torch.cuda.synchronize()
            say(f'EXIT second Conv2d, output={tuple(z.shape)}')
    else:
        from benchmark_rsbdnet_comment7 import DEFAULT_CHECKPOINT, RsbdNetBackend
        from transformer.scripts.train_xbd_supervised_disasteradaptivenet import XBDOriginalDataset
        from types import SimpleNamespace
        data_root = Path('/homes/j244s673/documents/wsu/phd/xview2')
        say('Load one xBD test image pair')
        ds = XBDOriginalDataset(root=data_root, split='test', image_size=896, training=False, conditioning_id=0)
        sample = ds[0]
        x = sample['img'].unsqueeze(0).to(device)
        cond = sample['cond_id'].unsqueeze(0).to(device)
        say(f'Image tensor {tuple(x.shape)}')
        say('Load original RS-BDNet checkpoint')
        backend = RsbdNetBackend(device, SimpleNamespace(checkpoint=DEFAULT_CHECKPOINT, expected_sha256=None))
        say('Checkpoint loaded; run ONE complete model forward')
        with torch.inference_mode():
            logits = backend.model(x, cond)
            torch.cuda.synchronize()
            say(f'EXIT complete model forward, output={tuple(logits.shape)}')

    say(f'SUCCESS: {args.test}; peak PyTorch GPU memory={torch.cuda.max_memory_allocated(0)/1024**3:.3f} GiB (diagnostic only)')
    faulthandler.cancel_dump_traceback_later()

if __name__ == '__main__':
    main()
