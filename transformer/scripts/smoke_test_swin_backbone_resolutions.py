#!/usr/bin/env python3
"""
Fast, GPU-free smoke test for SwinPretrainedBackbone.

Imports the real backbone class from train_xbd_hrtbda_v5_swin_pretrained_cascade.py
(the actual training script, unmodified by this test) and exercises exactly the
two resolutions HRTBDAPhase2 needs to support with a SINGLE backbone instance:
  - --img-size      (896 by default): used for validation/test, and for Phase I
                     end to end.
  - --phase2-crop-size (672 by default): used for Phase II training crops.

This does not need a GPU allocation -- it runs on CPU with batch size 1, so it
can be run directly on the login node right after `conda activate
disasteradaptivenet_cu128`, in under a couple of minutes, instead of waiting
in the SLURM queue to find out whether the resolution/format fixes actually
work.

Usage:
    conda activate disasteradaptivenet_cu128
    python transformer/scripts/smoke_test_swin_backbone_resolutions.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / "train_xbd_hrtbda_v5_swin_pretrained_cascade.py"


def load_backbone_class():
    import importlib.util

    spec = importlib.util.spec_from_file_location("hrtbda_swin_module", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Must be registered in sys.modules BEFORE exec_module runs: the training
    # script defines @dataclass classes (e.g. XBDSample), and dataclasses
    # resolves annotation types via sys.modules[cls.__module__] while the
    # class body executes. A normal `python train_xbd_...py` run registers
    # __main__ automatically; this manual importlib load does not, unless we
    # do it here ourselves.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SwinPretrainedBackbone


def check_one_resolution(backbone, size: int, label: str) -> bool:
    import torch

    print(f"\n--- Forward pass at {size}x{size} ({label}) ---", flush=True)
    try:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            feats = backbone(x)
    except Exception:
        print(f"FAILED at {size}x{size} ({label}):", flush=True)
        traceback.print_exc()
        return False

    ok = True
    for i, (feat, expected_c) in enumerate(zip(feats, backbone.channels)):
        actual_c = feat.shape[1]
        status = "OK" if actual_c == expected_c else "WRONG CHANNEL POSITION/COUNT"
        if actual_c != expected_c:
            ok = False
        print(
            f"  stage {i}: shape={tuple(feat.shape)} expected_channels={expected_c} "
            f"actual_channels(dim=1)={actual_c} [{status}]",
            flush=True,
        )

    if ok:
        print(f"PASS: all {len(feats)} stages have correct NCHW shape at {size}x{size}.", flush=True)
    return ok


def main() -> None:
    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: could not find {TRAIN_SCRIPT}", flush=True)
        sys.exit(1)

    print(f"Loading SwinPretrainedBackbone from: {TRAIN_SCRIPT}", flush=True)
    SwinPretrainedBackbone = load_backbone_class()

    img_size = 896
    phase2_crop_size = 672

    print(f"\nConstructing backbone with img_size={img_size} (this downloads ImageNet "
          f"weights on first run if not already cached -- may take a minute) ...", flush=True)
    backbone = SwinPretrainedBackbone(
        in_channels=3,
        variant="swin_tiny_patch4_window7_224",
        pretrained=True,
        img_size=img_size,
        patch_size=4,
        window_size=7,
    )
    backbone.eval()
    print(f"Backbone channels per stage: {backbone.channels}", flush=True)

    results = {
        "img_size (Phase I + Phase II val/test)": check_one_resolution(backbone, img_size, "img_size"),
        "phase2_crop_size (Phase II train crops)": check_one_resolution(backbone, phase2_crop_size, "phase2_crop_size"),
    }

    print("\n===== SUMMARY =====", flush=True)
    all_ok = True
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {label}", flush=True)
        all_ok = all_ok and ok

    if not all_ok:
        print("\nAt least one resolution failed. Do not submit the full sbatch job yet.", flush=True)
        sys.exit(1)

    print("\nBoth resolutions passed on CPU. Safe to submit the full sbatch job.", flush=True)


if __name__ == "__main__":
    main()
