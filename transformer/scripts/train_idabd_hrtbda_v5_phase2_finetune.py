#!/usr/bin/env python3
"""
HRTBDA v5 Phase-II fine-tuning on IDA-BD.

This wrapper reads the original xBD v5 training script:
    train_xbd_hrtbda_v5_multilabel_crop_cascade.py

Then it creates a temporary patched runtime script that adds:
    --init-phase2-from

This allows Phase II to be initialized from the xBD-trained Phase-II checkpoint
before fine-tuning on IDA-BD.

Main experiment:
    Fixed xBD-trained Phase I
    xBD-trained Phase II initialization
    Fine-tune Phase II on IDA-BD train
    Validate on IDA-BD val
    Test on IDA-BD test
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def add_init_argument(text: str) -> str:
    """Add --init-phase2-from to the original v5 argument parser."""
    if "--init-phase2-from" in text:
        return text

    marker = "    return parser.parse_args()"
    if marker not in text:
        raise RuntimeError("Could not find parser return marker: return parser.parse_args()")

    insert_arg = '''
    parser.add_argument(
        "--init-phase2-from",
        type=str,
        default=None,
        help="Optional xBD-trained Phase-II checkpoint used to initialize Phase II before IDA-BD fine-tuning.",
    )

'''
    return text.replace(marker, insert_arg + marker)


def add_phase2_loader_helper(text: str) -> str:
    """Add a flexible Phase-II checkpoint loader to the original v5 script."""
    if "def load_phase2_init_weights_flexible" in text:
        return text

    helper = r'''
# ---------------------------------------------------------------------
# Added for IDA-BD supervised fine-tuning from xBD-trained Phase-II
# ---------------------------------------------------------------------
def load_phase2_init_weights_flexible(model, checkpoint_path, device):
    """
    Load an existing xBD-trained Phase-II checkpoint before fine-tuning on IDA-BD.
    This is used for:
        HRTBDA-v5-xBDInit-PhaseII-Finetune_IDA
    """
    import torch
    from pathlib import Path
    import torch.nn as nn

    checkpoint_path = Path(checkpoint_path)

    print(f"Loading external Phase-II initialization checkpoint: {checkpoint_path}", flush=True)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint is not a dictionary: {checkpoint_path}")

    state = None

    candidate_keys = [
        "model",
        "model_state_dict",
        "state_dict",
        "phase2_model_state_dict",
        "phase2_state_dict",
    ]

    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            print(f"Using Phase-II init checkpoint key: {key}", flush=True)
            break

    if state is None:
        # Some checkpoints are saved directly as state_dict.
        if all(hasattr(v, "shape") for v in ckpt.values()):
            state = ckpt

    if state is None:
        print("Checkpoint keys:", sorted(list(ckpt.keys())), flush=True)
        raise RuntimeError("Could not find Phase-II model weights in checkpoint.")

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            clean_state[k[len("module."):]] = v
        else:
            clean_state[k] = v

    target_model = model.module if isinstance(model, nn.DataParallel) else model
    missing, unexpected = target_model.load_state_dict(clean_state, strict=False)

    print(f"Loaded Phase-II initialization checkpoint: {checkpoint_path}", flush=True)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}", flush=True)

    if missing:
        print("First missing keys:", missing[:10], flush=True)
    if unexpected:
        print("First unexpected keys:", unexpected[:10], flush=True)

'''

    idx = text.find("\ndef train_phase2")
    if idx == -1:
        raise RuntimeError("Could not find def train_phase2 in the original v5 script.")

    return text[:idx] + helper + text[idx:]


def add_phase2_init_call(text: str) -> str:
    """
    Insert the external Phase-II checkpoint loading call inside train_phase2.

    We insert it immediately before the class_weights line because at that point
    the Phase-II model has already been constructed and the original Phase-I
    backbone loading has already happened.
    """
    sentinel = "Applying external Phase-II initialization before fine-tuning"
    if sentinel in text:
        return text

    lines = text.splitlines(keepends=True)

    target_index = None
    target_indent = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("class_weights = make_damage4_class_weights"):
            target_index = i
            target_indent = line[: len(line) - len(stripped)]
            break

    if target_index is None:
        raise RuntimeError(
            "Could not find class_weights = make_damage4_class_weights(...) line. "
            "The v5 script structure may have changed."
        )

    inner_indent = target_indent + "    "

    block = [
        f'{target_indent}if getattr(args, "init_phase2_from", None):\n',
        f'{inner_indent}print("Applying external Phase-II initialization before fine-tuning.", flush=True)\n',
        f'{inner_indent}load_phase2_init_weights_flexible(\n',
        f'{inner_indent}    model=model,\n',
        f'{inner_indent}    checkpoint_path=Path(args.init_phase2_from),\n',
        f'{inner_indent}    device=device,\n',
        f'{inner_indent})\n',
        "\n",
    ]

    lines[target_index:target_index] = block
    return "".join(lines)


def build_runtime_script() -> Path:
    here = Path(__file__).resolve().parent

    source_script = here / "train_xbd_hrtbda_v5_multilabel_crop_cascade.py"
    runtime_script = here / "_runtime_train_idabd_hrtbda_v5_phase2_finetune.py"

    if not source_script.exists():
        raise FileNotFoundError(f"Missing original v5 script: {source_script}")

    text = source_script.read_text()

    text = add_init_argument(text)
    text = add_phase2_loader_helper(text)
    text = add_phase2_init_call(text)

    header = '''#!/usr/bin/env python3
# Auto-generated runtime script.
# Do not edit manually.
# Generated by train_idabd_hrtbda_v5_phase2_finetune.py.

'''

    runtime_script.write_text(header + text)

    print(f"Wrote runtime fine-tuning script: {runtime_script}", flush=True)
    return runtime_script


def main() -> None:
    runtime_script = build_runtime_script()

    # Execute the patched runtime script exactly as if Slurm called it directly.
    sys.argv[0] = str(runtime_script)
    runpy.run_path(str(runtime_script), run_name="__main__")


if __name__ == "__main__":
    main()