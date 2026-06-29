#!/usr/bin/env python3
"""
prepare_hrtbda_LOO_E0061_E0068.py

Prepare leave-one-disaster-out datasets for HRTBDA-v5 experiments E0061-E0068.

Experiments:
  E0061/E0062: all non-xBD except Hurricane Irma -> Hurricane Irma
  E0063/E0064: all non-xBD except Mount Semeru -> Mount Semeru
  E0065/E0066: all non-xBD except Earthquake Turkey -> Earthquake Turkey
  E0067/E0068: all non-xBD except Texas Tornadoes -> Texas Tornadoes

For each experiment:
  - train/val = all non-xBD source disasters except the held-out target
  - test      = held-out target disaster only


This script prepares the combined train, validation, and test folders for the
HRTBDA-v5 leave-one-disaster-out generalization experiments E0061 to E0068.
The goal of these experiments is to evaluate whether HRTBDA-v5 can generalize
to a completely unseen disaster when that disaster is excluded from training.

For each experiment, one disaster dataset is held out as the target test dataset.
The training and validation folders are created by combining all other available
non-xBD disaster datasets. The held-out disaster is used only for testing and is
not included in the training or validation splits. This keeps the evaluation clean
and measures cross-disaster generalization to an unseen disaster event.

The script supports both training strategies used in the sbatch file: training
HRTBDA-v5 from scratch and initializing HRTBDA-v5 from xBD-pretrained weights.
However, this script only prepares the dataset folders. The sbatch script decides
whether the model is trained from scratch or initialized from xBD-pretrained
weights.

For each experiment, this script:
  - prepares each individual disaster dataset using the existing single-dataset
    preparation script;
  - combines the train splits from all source datasets into one train folder;
  - combines the validation splits from all source datasets into one validation
    folder;
  - uses only the held-out target disaster test split as the final test folder;
  - prefixes filenames with the dataset name to avoid overwriting files when
    multiple datasets contain the same image or mask names.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ALL_NON_XBD_DATASETS = [
    "HURRICANE-IDA",
    "HURRICANE-IAN",
    "EARTHQUAKE-TURKEY",
    "HURRICANE-LAURA",
    "MOUNT-SEMERU-ERUPTION",
    "TEXAS-TORNADOES",
    "STVINCENT-VOLCANO",
    "TONGA-VOLCANO",
    "HURRICANE-DELTA",
    "HURRICANE-IRMA",
    "HURRICANE-DORIAN",
    "PAKISTAN-FLOODING",
]


EXPERIMENTS = {
    "E0061": {
        "name": "Scratch_LOO_AllExceptHIRMA_to_HIRMA",
        "test_dataset": "HURRICANE-IRMA",
    },
    "E0062": {
        "name": "xBDTL_LOO_AllExceptHIRMA_to_HIRMA",
        "test_dataset": "HURRICANE-IRMA",
    },
    "E0063": {
        "name": "Scratch_LOO_AllExceptMSEMERU_to_MSEMERU",
        "test_dataset": "MOUNT-SEMERU-ERUPTION",
    },
    "E0064": {
        "name": "xBDTL_LOO_AllExceptMSEMERU_to_MSEMERU",
        "test_dataset": "MOUNT-SEMERU-ERUPTION",
    },
    "E0065": {
        "name": "Scratch_LOO_AllExceptETURKEY_to_ETURKEY",
        "test_dataset": "EARTHQUAKE-TURKEY",
    },
    "E0066": {
        "name": "xBDTL_LOO_AllExceptETURKEY_to_ETURKEY",
        "test_dataset": "EARTHQUAKE-TURKEY",
    },
    "E0067": {
        "name": "Scratch_LOO_AllExceptTEXAS_to_TEXAS",
        "test_dataset": "TEXAS-TORNADOES",
    },
    "E0068": {
        "name": "xBDTL_LOO_AllExceptTEXAS_to_TEXAS",
        "test_dataset": "TEXAS-TORNADOES",
    },
}


def run_cmd(cmd):
    print("Running:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.check_call(cmd)


def prepare_single_dataset(dataset, staging_root, args):
    out_root = staging_root / dataset

    if out_root.exists() and not args.refresh_staging:
        print(f"[SKIP] staging already exists for {dataset}: {out_root}", flush=True)
        return out_root

    if out_root.exists():
        shutil.rmtree(out_root)

    cmd = [
        sys.executable,
        str(args.single_prep_script),
        "--dataset-base",
        str(args.dataset_base),
        "--dataset",
        dataset,
        "--output-root",
        str(out_root),
        "--mode",
        args.mode,
    ]

    run_cmd(cmd)
    return out_root


def safe_link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def merge_split(source_root, source_split, dest_root, dest_split, dataset_name, mode):
    src_dir = source_root / source_split
    dst_dir = dest_root / dest_split

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {src_dir}")

    count = 0

    for src in src_dir.rglob("*"):
        if src.is_dir():
            continue

        rel = src.relative_to(src_dir)

        # Prefix files by dataset name to avoid overwriting duplicate names.
        new_name = f"{dataset_name}__{rel.name}"
        dst = dst_dir / rel.parent / new_name

        safe_link_or_copy(src.resolve(), dst, mode)
        count += 1

    print(f"Merged {count} files: {dataset_name}/{source_split} -> {dest_split}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True, choices=sorted(EXPERIMENTS.keys()))
    parser.add_argument("--project-root", default="/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    parser.add_argument("--dataset-base", default="/homes/j244s673/documents/wsu/phd")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--single-prep-script", default="cross_DS/scripts/prepare_hrtbda_single_dataset_finetune.py")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--refresh-staging", action="store_true")
    args = parser.parse_args()

    args.project_root = Path(args.project_root)
    args.dataset_base = Path(args.dataset_base)
    args.output_root = Path(args.output_root)
    args.staging_root = Path(args.staging_root)
    args.single_prep_script = args.project_root / args.single_prep_script

    exp = EXPERIMENTS[args.experiment_id]
    test_dataset = exp["test_dataset"]
    train_datasets = [d for d in ALL_NON_XBD_DATASETS if d != test_dataset]

    print("================================================", flush=True)
    print(f"Experiment ID: {args.experiment_id}", flush=True)
    print(f"Experiment name: {exp['name']}", flush=True)
    print(f"Train datasets: {train_datasets}", flush=True)
    print(f"Held-out test dataset: {test_dataset}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print("================================================", flush=True)

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.staging_root.mkdir(parents=True, exist_ok=True)

    needed_datasets = sorted(set(train_datasets + [test_dataset]))

    staged = {}
    for ds in needed_datasets:
        staged[ds] = prepare_single_dataset(ds, args.staging_root, args)

    # Train and validation come from all datasets except the held-out target.
    for ds in train_datasets:
        merge_split(staged[ds], "train", args.output_root, "train", ds, args.mode)
        merge_split(staged[ds], "val", args.output_root, "val", ds, args.mode)

    # Test comes only from the held-out unseen target.
    merge_split(staged[test_dataset], "test", args.output_root, "test", test_dataset, args.mode)

    print("================================================", flush=True)
    print("Leave-one-disaster-out dataset prepared successfully.", flush=True)
    print("Final root:", args.output_root, flush=True)
    print("================================================", flush=True)


if __name__ == "__main__":
    main()