#!/usr/bin/env python3
"""
HRTBDA v5 Phase-II fine-tuning on Earthquake Turkey.

This wrapper reads the original xBD v5 training script:
    train_xbd_hrtbda_v5_multilabel_crop_cascade.py

Then it creates a temporary patched runtime script that adds:
    --init-phase2-from

This allows Phase II to be initialized from the xBD-trained Phase-II checkpoint
before fine-tuning on the Earthquake Turkey dataset.

Main experiment:
    xBD-trained Phase I initialization -> fine-tune Phase I on Earthquake Turkey
    xBD-trained Phase II initialization -> fine-tune Phase II on Earthquake Turkey
    Validate on Earthquake Turkey val
    Test on Earthquake Turkey test
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


DATASET_TAG = "earthquake_turkey"


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
        help="Optional xBD-trained Phase-II checkpoint used to initialize Phase II before target-dataset fine-tuning.",
    )

'''
    return text.replace(marker, insert_arg + marker)


def add_phase2_loader_helper(text: str) -> str:
    """Add a flexible Phase-II checkpoint loader to the original v5 script."""
    if "def load_phase2_init_weights_flexible" in text:
        return text

    helper = r'''
# ---------------------------------------------------------------------
# Added for Earthquake Turkey supervised fine-tuning from xBD Phase-II
# ---------------------------------------------------------------------
def load_phase2_init_weights_flexible(model, checkpoint_path, device):
    """
    Load an existing xBD-trained Phase-II checkpoint before fine-tuning
    on a target disaster dataset.
    """
    import torch
    import torch.nn as nn
    from pathlib import Path

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
    """
    sentinel = "Applying external Phase-II initialization before fine-tuning"
    if sentinel in text:
        return text

    lines = text.splitlines(keepends=True)

    in_phase2 = False
    target_index = None
    target_indent = None

    for i, line in enumerate(lines):
        if line.startswith("def train_phase2"):
            in_phase2 = True
        elif in_phase2 and line.startswith("def ") and not line.startswith("def train_phase2"):
            break

        if in_phase2:
            stripped = line.lstrip()
            if stripped.startswith("class_weights = make_damage4_class_weights"):
                target_index = i
                target_indent = line[: len(line) - len(stripped)]
                break

    if target_index is None:
        raise RuntimeError(
            "Could not find class_weights = make_damage4_class_weights(...) inside train_phase2. "
            "The v5 script structure may have changed."
        )

    inner_indent = target_indent + "    "

    block = [
        f'{target_indent}if getattr(args, "init_phase2_from", None):\n',
        f'{inner_indent}print("Applying external Phase-II initialization before fine-tuning.", flush=True)\n',
        f'{inner_indent}load_phase2_init_weights_flexible(\n',
        f'{inner_indent}    model=model,\n',
        f'{inner_indent}    checkpoint_path=args.init_phase2_from,\n',
        f'{inner_indent}    device=device,\n',
        f'{inner_indent})\n',
        "\n",
    ]

    lines[target_index:target_index] = block
    return "".join(lines)


def build_runtime_script() -> Path:
    here = Path(__file__).resolve().parent

    source_script = here / "train_xbd_hrtbda_v5_multilabel_crop_cascade.py"
    runtime_script = here / f"_runtime_train_{DATASET_TAG}_hrtbda_v5_phase2_finetune.py"

    if not source_script.exists():
        raise FileNotFoundError(f"Missing original v5 script: {source_script}")

    text = source_script.read_text()

    text = add_init_argument(text)
    text = add_phase2_loader_helper(text)
    text = add_phase2_init_call(text)

    header = f'''#!/usr/bin/env python3
# Auto-generated runtime script.
# Do not edit manually.
# Generated by train_{DATASET_TAG}_hrtbda_v5_phase2_finetune.py.

'''

    runtime_script.write_text(header + text)

    print(f"Wrote runtime fine-tuning script: {runtime_script}", flush=True)
    return runtime_script


def main() -> None:
    runtime_script = build_runtime_script()

    sys.argv[0] = str(runtime_script)
    runpy.run_path(str(runtime_script), run_name="__main__")


if __name__ == "__main__":
    main()