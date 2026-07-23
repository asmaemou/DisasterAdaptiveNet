#!/usr/bin/env python3
"""
HRTBDA v5 Phase-II ROBUST fine-tuning on Earthquake Turkey.

This is a sibling of train_earthquake_turkey_hrtbda_v5_phase2_finetune.py.
It does NOT modify that file or the original v5 training script; instead it
generates its own separate runtime-patched copy, so nothing about the
existing (already-run) experiment is touched.

Why this variant exists
------------------------
The first full xBD-init fine-tune run on Earthquake Turkey
(train_earthquake_turkey_hrtbda_v5_full_finetune_xbdinit.sbatch) trained
Phase II well (localization F1 0.93 on test) but Phase II damage
classification lagged badly on the minor/major classes (test minor F1
0.27, major F1 0.18), while the xView2-winner baselines fine-tuned on the
same official split did noticeably better on those classes. Reading the
training log for that run surfaced two protocol-level problems specific to
Phase II, on top of the target dataset simply being small and imbalanced:

  1. "Best epoch" selection used a single-epoch argmax over
     hold_score_cascade computed on a 94-image val split with only a few
     thousand minor/major/destroyed pixels. That score swings by +/-0.13
     between adjacent epochs with no trend (see history_phase2.json from the
     original run), so picking the single best-looking epoch is close to a
     lottery -- the epoch that won on val (epoch 19, hold major F1 0.315)
     scored much worse on test (major F1 0.179).

  2. Phase II started from an xBD-pretrained checkpoint but immediately
     applied LR=5e-5 with only a 3-epoch warmup. The damage head collapsed
     to near-zero minor/major/destroyed F1 for epochs 1-5 before recovering,
     wasting roughly a sixth of the 30-epoch budget re-learning what it
     already knew. A further unconditional 3-epoch "warm restart" tail at
     the end (LR reset back to 5e-5) then added more instability right when
     the model should have been settling, and did not beat the epoch-19
     checkpoint on this run.

This wrapper keeps the exact same HRTBDA v5 architecture, data pipeline,
and loss functions from train_xbd_hrtbda_v5_multilabel_crop_cascade.py --
only the Phase II *training protocol* changes:

  - Adds --init-phase2-from (same mechanism as the original wrapper) so
    Phase II can still start from the xBD-trained Phase-II checkpoint.
  - Adds --smoothing-window N: "best epoch" is now selected using a
    trailing moving average of hold_score_cascade over the last N
    evaluated epochs instead of the raw single-epoch value. This directly
    targets problem (1) without touching model weights, BatchNorm
    statistics, or any other part of the training loop -- it only changes
    what value is compared against best_score. N=1 reproduces the original
    (unsmoothed) behavior exactly.
  - Everything else (warmup length, initial LR, whether the warm-restart
    tail runs at all, class-imbalance weights) is already exposed as plain
    CLI flags on the original script (--warmup-epochs, --lr,
    --finetune-epochs, --finetune-lr, --minor-damage-boost,
    --major-damage-boost, --max-damage-class-weight) -- problem (2) is
    fixed by calling this wrapper with different values for those flags
    from the new sbatch, not by patching code.

Main experiment:
    Already-fine-tuned Turkey Phase I checkpoint (reused, not retrained)
    xBD-trained Phase II initialization -> robust fine-tune Phase II on
    Earthquake Turkey with smoothed checkpoint selection
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


def add_smoothing_argument(text: str) -> str:
    """Add --smoothing-window to the original v5 argument parser."""
    if "--smoothing-window" in text:
        return text

    marker = "    return parser.parse_args()"
    if marker not in text:
        raise RuntimeError("Could not find parser return marker: return parser.parse_args()")

    insert_arg = '''
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help=(
            "Number of trailing evaluated epochs (including the current one) averaged "
            "together to decide whether a Phase II checkpoint is the new best. "
            "1 reproduces the original single-epoch argmax behavior. Use a larger value "
            "(e.g. 3) when the hold/val split is small and the per-epoch cascade score "
            "is noisy, so checkpoint selection is not a lottery over one lucky epoch."
        ),
    )

'''
    return text.replace(marker, insert_arg + marker)


def add_phase2_loader_helper(text: str) -> str:
    """Add a flexible Phase-II checkpoint loader to the original v5 script."""
    if "def load_phase2_init_weights_flexible" in text:
        return text

    helper = r'''
# ---------------------------------------------------------------------
# Added for Earthquake Turkey robust supervised fine-tuning from xBD Phase-II
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


def add_smoothed_selection(text: str) -> str:
    """
    Replace the single-epoch-argmax "best checkpoint" criterion inside
    train_phase2's validate_and_save closure with a trailing moving average
    of hold_score_cascade over --smoothing-window epochs. Model weights,
    BatchNorm statistics, checkpoint file formats, and everything else about
    the training loop are left exactly as in the original script.
    """
    sentinel = "smoothed_score = sum(_recent) / len(_recent)"
    if sentinel in text:
        return text

    old_score_line = '        val_score = float(val_results["score"])\n'
    if text.count(old_score_line) != 1:
        raise RuntimeError(
            "Expected exactly one occurrence of the hold-score assignment line "
            "inside validate_and_save. The v5 script structure may have changed."
        )

    new_score_block = (
        '        val_score = float(val_results["score"])\n'
        '        smoothing_window = max(1, int(getattr(args, "smoothing_window", 1)))\n'
        '        _recent = [float(h["hold_score_cascade"]) for h in history[-(smoothing_window - 1):]] '
        'if smoothing_window > 1 else []\n'
        '        _recent.append(val_score)\n'
        '        smoothed_score = sum(_recent) / len(_recent)\n'
    )
    text = text.replace(old_score_line, new_score_block)

    old_promotion_block = (
        '        if val_score > best_score:\n'
        '            best_score = val_score\n'
        '            best_epoch = epoch_label\n'
        '            no_improve = 0\n'
        '            save_checkpoint(checkpoints_dir / "phase2_best.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)\n'
        '            print(f"Saved Phase II best checkpoint | epoch={epoch_label} | cascade_score={best_score:.6f}", flush=True)\n'
        '        else:\n'
        '            no_improve += 1\n'
        '            print(f"Phase II no improvement for {no_improve} epoch(s). Best epoch={best_epoch}", flush=True)\n'
    )
    if text.count(old_promotion_block) != 1:
        raise RuntimeError(
            "Expected exactly one occurrence of the best-checkpoint promotion block "
            "inside validate_and_save. The v5 script structure may have changed."
        )

    new_promotion_block = (
        '        if smoothed_score > best_score:\n'
        '            best_score = smoothed_score\n'
        '            best_epoch = epoch_label\n'
        '            no_improve = 0\n'
        '            save_checkpoint(checkpoints_dir / "phase2_best.pt", model, optimizer, scheduler, scaler, epoch_label, best_score, args, extra=extra)\n'
        '            print(f"Saved Phase II best checkpoint | epoch={epoch_label} | raw_cascade_score={val_score:.6f} | '
        'smoothed_cascade_score={best_score:.6f} (window={smoothing_window})", flush=True)\n'
        '        else:\n'
        '            no_improve += 1\n'
        '            print(f"Phase II no improvement for {no_improve} epoch(s). Best epoch={best_epoch} | '
        'raw_cascade_score={val_score:.6f} | smoothed_cascade_score={smoothed_score:.6f}", flush=True)\n'
    )
    text = text.replace(old_promotion_block, new_promotion_block)

    return text


def build_runtime_script() -> Path:
    here = Path(__file__).resolve().parent

    source_script = here / "train_xbd_hrtbda_v5_multilabel_crop_cascade.py"
    runtime_script = here / f"_runtime_train_{DATASET_TAG}_hrtbda_v5_phase2_robust_finetune.py"

    if not source_script.exists():
        raise FileNotFoundError(f"Missing original v5 script: {source_script}")

    text = source_script.read_text()

    text = add_init_argument(text)
    text = add_smoothing_argument(text)
    text = add_phase2_loader_helper(text)
    text = add_phase2_init_call(text)
    text = add_smoothed_selection(text)

    header = f'''#!/usr/bin/env python3
# Auto-generated runtime script.
# Do not edit manually.
# Generated by train_{DATASET_TAG}_hrtbda_v5_phase2_robust_finetune.py.

'''

    runtime_script.write_text(header + text)

    print(f"Wrote runtime robust fine-tuning script: {runtime_script}", flush=True)
    return runtime_script


def main() -> None:
    runtime_script = build_runtime_script()

    sys.argv[0] = str(runtime_script)
    runpy.run_path(str(runtime_script), run_name="__main__")


if __name__ == "__main__":
    main()
