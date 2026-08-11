#!/usr/bin/env python3
"""Count parameters stored in released xView2 winner checkpoints.

This utility operates directly on checkpoint state dictionaries, so it does
not require constructing the legacy model classes or downloading pretrained
weights. It reports every loadable checkpoint and the sum across all ensemble
members in each supplied directory.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch


STATE_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "network",
    "weights",
)
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and any(
        torch.is_tensor(item) for item in value.values()
    )


def find_state_dict(checkpoint: Any) -> Optional[Mapping[str, Any]]:
    """Extract the most likely model state dictionary from a checkpoint."""
    if not isinstance(checkpoint, Mapping):
        return None

    for key in STATE_KEYS:
        candidate = checkpoint.get(key)
        if is_tensor_mapping(candidate):
            return candidate

    if is_tensor_mapping(checkpoint):
        return checkpoint
    return None


def parameter_statistics(state: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Return learned parameters, state elements, and tensor count.

    Batch-normalization running statistics are buffers rather than learned
    parameters, so they are included in state elements but excluded from the
    learned-parameter estimate.
    """
    learned = 0
    state_elements = 0
    tensors = 0
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        count = value.numel()
        tensors += 1
        state_elements += count
        clean_name = name.removeprefix("module.")
        if not clean_name.endswith(BUFFER_SUFFIXES):
            learned += count
    return learned, state_elements, tensors


def candidate_files(root: Path) -> Iterable[Path]:
    """Yield regular files while skipping obvious metadata and result files."""
    ignored_suffixes = {
        ".csv", ".json", ".log", ".md", ".png", ".jpg", ".jpeg",
        ".txt", ".yaml", ".yml", ".zip", ".tar", ".gz",
    }
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() not in ignored_suffixes:
            yield path


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        return torch.load(path, map_location="cpu")


def millions(value: int) -> str:
    return f"{value / 1_000_000:.3f} M"


def gibibytes(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def inspect_directory(label: str, root: Path) -> Dict[str, int]:
    print(f"\n{'=' * 88}\n{label}\nDirectory: {root}\n{'=' * 88}")
    if not root.is_dir():
        print("ERROR: directory does not exist")
        return {"models": 0, "parameters": 0, "state_elements": 0, "bytes": 0}

    totals = {"models": 0, "parameters": 0, "state_elements": 0, "bytes": 0}
    seen_inodes = set()
    for path in candidate_files(root):
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            print(f"SKIP alias/hardlink: {path.relative_to(root)}")
            continue

        try:
            checkpoint = load_checkpoint(path)
        except Exception:
            continue
        state = find_state_dict(checkpoint)
        if state is None:
            del checkpoint
            gc.collect()
            continue

        learned, state_elements, tensors = parameter_statistics(state)
        seen_inodes.add(inode)
        totals["models"] += 1
        totals["parameters"] += learned
        totals["state_elements"] += state_elements
        totals["bytes"] += stat.st_size
        print(
            f"{path.relative_to(root)}\n"
            f"  learned parameters: {learned:,} ({millions(learned)})\n"
            f"  state elements:     {state_elements:,} ({millions(state_elements)})\n"
            f"  tensors: {tensors:,}; checkpoint size: {gibibytes(stat.st_size)}"
        )
        del state, checkpoint
        gc.collect()

    print("-" * 88)
    print(f"Loadable model checkpoints: {totals['models']}")
    print(
        "SUM OF DEPLOYED ENSEMBLE PARAMETERS: "
        f"{totals['parameters']:,} ({millions(totals['parameters'])})"
    )
    print(
        "Sum of all model-state elements:     "
        f"{totals['state_elements']:,} ({millions(totals['state_elements'])})"
    )
    print(f"Total checkpoint storage: {gibibytes(totals['bytes'])}")
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--third", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"PyTorch: {torch.__version__}")
    print("Counting learned tensor elements; BatchNorm running statistics are excluded.")
    summaries = {
        "First-place xView2 winner": inspect_directory("First-place xView2 winner", args.first),
        "Second-place xView2 winner": inspect_directory("Second-place xView2 winner", args.second),
        "Third-place xView2 winner": inspect_directory("Third-place xView2 winner", args.third),
    }

    print(f"\n{'=' * 88}\nFINAL COMPARISON\n{'=' * 88}")
    for label, values in summaries.items():
        print(
            f"{label}: {values['models']} checkpoints, "
            f"{values['parameters']:,} parameters ({millions(values['parameters'])})"
        )


if __name__ == "__main__":
    main()
