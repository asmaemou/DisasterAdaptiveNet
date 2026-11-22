#!/usr/bin/env python3
"""Reviewer 2 Comment 7: real GPU inference-time and memory benchmark (cuDNN disabled).

Run from <DisasterAdaptiveNet>/transformer/scripts/ so PROJECT_ROOT resolves.

Built-in backend 'rsbdnet' loads the existing original xBD checkpoint and runs
pre/post inference with damage-map postprocessing. Additional complete xView2
ensembles can be tested using an explicitly implemented --backend module:factory.
The runner NEVER substitutes one constituent backbone for an ensemble.

Scope: batch-one, preprocessed-image inference (CPU dataset decoding excluded),
including host-to-GPU copy, network forward and GPU postprocessing. This is not
end-to-end deployment time including image decoding / disk I/O. Compare only
runs with the same declared scope and hardware, and report any native-resolution
or TTA differences in the ensemble adapter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

DEFAULT_DATA = Path('/homes/j244s673/documents/wsu/phd/xview2')
DEFAULT_CHECKPOINT = (PROJECT_ROOT / 'output' /
                      'BitemporalBuildingCrossAttentionOrdinal_xBD_TrainTier3' /
                      'checkpoints' / 'best.pt')
EXPECTED_ORIGINAL_XBD_SHA256 = ('c781c1d9539618e2d5c67bc6365daa24557e9'
                                 'c01e64f87fc7d92282fe90aea77')
MIB = 1024.0**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def cmd_output(command):
    try:
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=15, check=False)
        return p.stdout.strip()[:16000]
    except Exception as e:
        return f'unavailable: {e}'


class NvmlProcessPeak:
    """Best-effort NVML memory for THIS PID (includes CUDA context).

    Optional: pip install nvidia-ml-py. Samples asynchronously; very brief peaks
    can be missed, so also report PyTorch's exact allocator high-water mark.
    """
    def __init__(self, device_index: int, sample_ms: float):
        self.device_index = device_index
        self.sample_s = sample_ms / 1000.0
        self.pid = os.getpid()
        self.maximum_bytes = None
        self.samples = 0
        self.error = None
        self._thread = None
        self._stop = threading.Event()
        self._nvml = None
        self._handle = None

    def start(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            try:
                uuid = torch.cuda.get_device_properties(self.device_index).uuid
                self._handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid).encode('ascii'))
            except Exception:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as e:
            self.error = f'NVML unavailable ({e}); install nvidia-ml-py for process GPU memory.'
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
                    for proc in procs:
                        if int(proc.pid) == self.pid:
                            used = int(proc.usedGpuMemory)
                            if 0 <= used < (1 << 62):
                                if self.maximum_bytes is None or used > self.maximum_bytes:
                                    self.maximum_bytes = used
                                self.samples += 1
                except Exception as e:
                    self.error = f'NVML process sampling error: {e}'
                    return
                self._stop.wait(self.sample_s)
        except Exception as e:
            self.error = str(e)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def summary(self):
        return {
            'nvml_peak_process_mib_sampled': (
                round(self.maximum_bytes / MIB, 3) if self.maximum_bytes is not None else None),
            'nvml_sample_count': self.samples,
            'nvml_sample_period_ms': round(self.sample_s * 1000, 3),
            'nvml_warning': self.error,
        }


class RsbdNetBackend:
    name = 'RS-BDNet (original xBD checkpoint)'
    architecture_scope = 'one complete RS-BDNet checkpoint; pre+post inputs'
    input_size = 896
    model_count = 1
    has_tta = False

    def __init__(self, device, args):
        from transformer.scripts.train_xbd_bitemporal_building_crossattention_ordinal import (
            BuildingGuidedCrossAttentionOrdinalNet,
        )
        self.device = device
        self.ckpt_path = args.checkpoint.resolve()
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f'RS-BDNet checkpoint not found: {self.ckpt_path}')
        self.checkpoint_sha256 = sha256(self.ckpt_path)
        # Prevent accidental benchmark of a different original checkpoint.
        if self.ckpt_path == DEFAULT_CHECKPOINT.resolve():
            if self.checkpoint_sha256 != EXPECTED_ORIGINAL_XBD_SHA256:
                raise RuntimeError('Original xBD checkpoint SHA-256 has changed.')
        if args.expected_sha256 and self.checkpoint_sha256 != args.expected_sha256:
            raise RuntimeError('Checkpoint SHA-256 does not match --expected-sha256.')
        try:
            ckpt = torch.load(self.ckpt_path, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(self.ckpt_path, map_location='cpu')
        if not isinstance(ckpt, dict) or 'model' not in ckpt:
            raise ValueError('Expected a training checkpoint dict containing model state.')
        recorded_img_size = (ckpt.get('args') or {}).get('img_size')
        if recorded_img_size is not None and int(recorded_img_size) != 896:
            raise ValueError(f'Unexpected checkpoint input size: {recorded_img_size}')
        self.threshold = float(ckpt['best_threshold'])
        self.model = BuildingGuidedCrossAttentionOrdinalNet(image_size=896, width=96)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.parameters_million = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.checkpoint_paths = [str(self.ckpt_path)]
        self.checkpoint_bytes = self.ckpt_path.stat().st_size

    def predict(self, item):
        image = item['img'].unsqueeze(0).to(self.device)
        cond = item['cond_id'].unsqueeze(0).to(self.device)
        logits = self.model(image, cond)
        if tuple(logits.shape) != (1, 8, 896, 896):
            raise RuntimeError(f'Unexpected RS-BDNet logits shape: {tuple(logits.shape)}')
        building = torch.sigmoid(logits[:, 0]) > self.threshold
        damage = (logits[:, 1:5].argmax(dim=1) + 1) * building.long()
        return building, damage



def create_backend(device, args):
    if args.backend == 'rsbdnet':
        return RsbdNetBackend(device, args)
    if ':' not in args.backend:
        raise ValueError('Other backends must use --backend python.module:factory')
    module_name, factory_name = args.backend.split(':', 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    backend = factory(device=device, args=args)
    if not callable(getattr(backend, 'predict', None)):
        raise TypeError('Adapter must return an object with predict(item) method.')
    if not getattr(backend, 'name', None):
        raise ValueError('Adapter must set a descriptive name; no anonymous baseline.')
    if int(getattr(backend, 'model_count', 0)) < 1:
        raise ValueError('Adapter must record actual model_count (all ensemble checkpoints).')
    return backend


def cpu_sample(item):
    # Keep ALL image decoding/preprocessing out of the timed section.
    result = {}
    for key in ('img', 'cond_id'):
        val = item[key]
        if not isinstance(val, torch.Tensor):
            val = torch.as_tensor(val)
        if val.device.type != 'cpu':
            val = val.cpu()
        result[key] = val.contiguous()
    # Pass along optional metadata; adapter may need it.
    for key, value in item.items():
        if key not in result:
            result[key] = value
    return result


def run(args):
    # Confirmed on WSU RTX PRO 6000 Blackwell, PyTorch 2.11.0+cu128:
    # cuDNN-enabled conv2d stalled >65 s; disabled conv and full RS-BDNet
    # forward both succeeded. Force this known-working execution path.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    print('Runtime workaround: cuDNN DISABLED for all model operations; '
          f'cuDNN library version={torch.backends.cudnn.version()}', flush=True)
    if torch.backends.cudnn.enabled:
        raise RuntimeError('Expected cuDNN to be disabled for this benchmark.')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the paper benchmark.')
    if min(args.warmup, args.repeats, args.samples) < 1:
        raise ValueError('--warmup, --repeats, and --samples must all be positive.')
    if args.input_size != 896:
        raise ValueError('This data loader configuration uses 896. Set up a separate '
                         'adapter if an ensemble requires native 1024 inputs.')
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite or reuse output folder: {output_dir}')
    if output_dir == PROJECT_ROOT or output_dir in PROJECT_ROOT.parents:
        raise ValueError('Output must not point to a project or ancestor directory.')
    # Disallow writing anywhere in existing training outputs except an
    # explicit new, unique subdirectory of this reviewer-specific directory.
    safe_parent = (PROJECT_ROOT / 'output' / 'Reviewer2_Comment7_InferenceBench').resolve()
    if safe_parent not in output_dir.parents:
        raise ValueError(f'Output must be a new folder inside {safe_parent}')
    if not args.xbd_root.is_dir():
        raise FileNotFoundError(f'Missing xBD data directory: {args.xbd_root}')
    output_dir.mkdir(parents=True, exist_ok=False)
    print('Output:', output_dir, flush=True)

    torch.cuda.set_device(args.device_index)
    device = torch.device(f'cuda:{args.device_index}')
    torch.cuda.init()
    torch.cuda.empty_cache()
    sampler = NvmlProcessPeak(args.device_index, args.nvml_sample_ms)

    from transformer.scripts.train_xbd_supervised_disasteradaptivenet import XBDOriginalDataset
    dataset = XBDOriginalDataset(root=args.xbd_root, split='test', image_size=args.input_size,
                                 training=False, conditioning_id=0)
    if len(dataset) == 0:
        raise RuntimeError('The xBD test dataset has no samples.')
    indices = np.linspace(0, len(dataset)-1, min(args.samples, len(dataset)), dtype=int).tolist()
    samples = [cpu_sample(dataset[i]) for i in indices]
    print(f'Loaded {len(samples)} xBD test pairs on CPU (timing excludes image decoding).', flush=True)
    print('Initializing backend:', args.backend, flush=True)
    load_start = time.perf_counter()
    backend = create_backend(device, args)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    init_gpu_alloc_mib = torch.cuda.memory_allocated(device) / MIB
    print(f'Loaded: {backend.name}; constituent checkpoints={backend.model_count}; '
          f'initial torch allocation={init_gpu_alloc_mib:.1f} MiB', flush=True)
    # Measure inference peak, not temporary buffers used during weight loading.
    torch.cuda.reset_peak_memory_stats(device)
    sampler.start()
    print('Warm-up inference (first pass is also a model execution check)...', flush=True)
    with torch.inference_mode():
        for i in range(args.warmup):
            output = backend.predict(samples[i % len(samples)])
            if output is None:
                raise RuntimeError('Backend.predict returned None: cannot verify inference.')
            torch.cuda.synchronize(device)
            if i == 0 or (i + 1) % max(1, args.warmup // 4) == 0:
                print(f'  warmup {i+1}/{args.warmup} completed', flush=True)
    print('Timed inference...', flush=True)
    latency_ms = []
    with torch.inference_mode():
        for iteration in range(args.repeats):
            idx = iteration % len(samples)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = backend.predict(samples[idx])
            if output is None:
                raise RuntimeError('Backend.predict returned None during timed inference.')
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start)*1000.0
            latency_ms.append(elapsed)
            if (iteration + 1) % max(1, args.repeats // 5) == 0:
                print(f'  {iteration+1}/{args.repeats}: {elapsed:.2f} ms', flush=True)
    memory_summary = {
        'torch_peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / MIB, 3),
        'torch_peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / MIB, 3),
        'torch_post_init_allocated_mib': round(init_gpu_alloc_mib, 3),
        'torch_end_allocated_mib': round(torch.cuda.memory_allocated(device) / MIB, 3),
    }
    sampler.stop()
    memory_summary.update(sampler.summary())
    with (output_dir/'iterations.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['iteration', 'sample_index_in_test', 'latency_ms'])
        writer.writerows((i + 1, indices[i % len(samples)], round(t, 5))
                         for i, t in enumerate(latency_ms))
    latency_p95 = float(np.percentile(latency_ms, 95))
    details = {
        'status': 'completed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'backend': args.backend, 'name': backend.name,
        'model_count': int(backend.model_count),
        'model_scope': str(getattr(backend, 'architecture_scope', 'not documented')),
        'has_tta': getattr(backend, 'has_tta', None),
        'model_input_size': getattr(backend, 'input_size', 'adapter-dependent'),
        'checkpoint_paths': getattr(backend, 'checkpoint_paths', []),
        'checkpoint_sha256': getattr(backend, 'checkpoint_sha256', None),
        'checkpoint_bytes': getattr(backend, 'checkpoint_bytes', None),
        'parameters_million': getattr(backend, 'parameters_million', None),
        'model_loading_seconds': round(load_seconds, 5),
        'protocol': {
            'dataset': 'xBD original Test', 'root': str(args.xbd_root.resolve()),
            'dataset_size': len(dataset), 'distinct_pairs_cached': len(samples),
            'sample_indices_in_test': indices,
            'preprocessed_input_size': args.input_size,
            'timing_scope': ('H2D transfer + complete GPU model predictions + GPU postprocessing; '
                             'not input file decode and CPU dataset preprocessing'),
            'batch_size': 1, 'warmup_iterations': args.warmup,
            'timed_iterations': args.repeats, 'precision': 'fp32 unless adapter says otherwise',
            'cuda_synchronized_each_iteration': True,
            'conditions': 'torch.inference_mode, eval(), GPU, no optimizer, no training',
            'cudnn_enabled': False,
            'cudnn_workaround': ('Disabled due to standalone Conv2d and full-model stalls '
                                 'with cuDNN enabled on this WSU Blackwell environment'),
        },
        'latency': {
            'mean_ms_per_pair': round(statistics.mean(latency_ms), 5),
            'median_ms_per_pair': round(statistics.median(latency_ms), 5),
            'p95_ms_per_pair': round(latency_p95, 5),
            'std_ms_per_pair': round(statistics.pstdev(latency_ms), 5),
            'throughput_pairs_per_second': round(1000.0/statistics.mean(latency_ms), 4),
            'min_ms_per_pair': round(min(latency_ms), 5),
            'max_ms_per_pair': round(max(latency_ms), 5),
        },
        'memory': memory_summary,
        'environment': {
            'hostname': platform.node(), 'python': platform.python_version(),
            'pytorch': torch.__version__, 'cuda_build': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'gpu_name': torch.cuda.get_device_name(device),
            'gpu_device_index': args.device_index,
            'slurm_job_id': os.getenv('SLURM_JOB_ID'),
            'nvidia_smi': cmd_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                      '--format=csv,noheader']),
        },
    }
    json_dump(output_dir/'summary.json', details)
    with (output_dir/'summary.csv').open('w', newline='', encoding='utf-8') as fh:
        row = {
            'backend': backend.name, 'checkpoints': backend.model_count,
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'parameters_million': details['parameters_million'],
            'mean_latency_ms_pair': details['latency']['mean_ms_per_pair'],
            'median_latency_ms_pair': details['latency']['median_ms_per_pair'],
            'p95_latency_ms_pair': details['latency']['p95_ms_per_pair'],
            'pairs_per_second': details['latency']['throughput_pairs_per_second'],
            'torch_peak_reserved_mib': memory_summary['torch_peak_reserved_mib'],
            'torch_peak_allocated_mib': memory_summary['torch_peak_allocated_mib'],
            'nvml_peak_process_mib_sampled': memory_summary['nvml_peak_process_mib_sampled'],
        }
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps({'name': details['name'], 'latency': details['latency'],
                      'memory': memory_summary}, indent=2), flush=True)
    print('Saved results:', output_dir, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', default='rsbdnet',
                   help='rsbdnet OR installed python.module:factory for a complete ensemble')
    p.add_argument('--xbd-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--expected-sha256', default=None)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device-index', type=int, default=0)
    p.add_argument('--input-size', type=int, default=896)
    p.add_argument('--samples', type=int, default=100,
                   help='Distinct test pairs cached on CPU and repeated during timing')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--nvml-sample-ms', type=float, default=10)
    p.add_argument('--ensemble-checkpoint-dir', type=Path, default=None,
                   help='Made available as args.ensemble_checkpoint_dir to adapter factory')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
