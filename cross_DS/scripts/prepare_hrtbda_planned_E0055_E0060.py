#!/usr/bin/env python3
"""
prepare_hrtbda_planned_E0055_E0060.py

This script prepares the combined train/validation/test folders for the planned
HRTBDA-v5 cross-disaster generalization experiments E0055 to E0060.

The purpose of these experiments is to evaluate whether HRTBDA-v5 can generalize
to an unseen disaster dataset after being trained on selected source disasters.
The script supports three experiment types:

1. Selected Dataset A + Dataset B -> unseen Dataset C experiments:
   - Same-hazard transfer:
     Train on Hurricane Laura + Hurricane Delta, then test on Hurricane Irma.
   - Different-hazard transfer:
     Train on Earthquake Turkey + Mount Semeru, then test on Pakistan flooding.

2. Leave-one-disaster-out generalization:
   - Train on all available non-xBD disaster datasets except Pakistan flooding,
     then test on Pakistan flooding as the unseen target dataset.

3. Scratch and xBD-pretrained settings:
   - The script only prepares the dataset folders.
   - The sbatch script decides whether the model is trained from scratch or
     initialized from xBD-pretrained weights.

For each experiment, this script:
   - Uses the existing single-dataset preparation script to stage each dataset
     into the expected HRTBDA/xBD-style format.
   - Combines the training splits from the selected source datasets into one
     train folder.
   - Combines the validation splits from the selected source datasets into one
     val folder.
   - Uses only the held-out target dataset test split as the final test folder.
   - Prefixes filenames with the dataset name to avoid overwriting files when
     multiple datasets contain the same image or mask names.

Important:
The target test dataset is not included in the training or validation splits.
This keeps the evaluation clean and measures cross-disaster generalization to
an unseen disaster.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_EXACT_PATHS = {
    "HURRICANE-IDA": "/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet",
    "IDA-BD": "/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet",

    "HURRICANE-IAN": "/homes/j244s673/documents/wsu/phd/hurrican-ian",
    "IAN-BD": "/homes/j244s673/documents/wsu/phd/hurrican-ian",

    "EARTHQUAKE-TURKEY": "/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed",
    "HURRICANE-LAURA": "/homes/j244s673/documents/wsu/phd/hurricane_laura_preprocessed",
    "MOUNT-SEMERU-ERUPTION": "/homes/j244s673/documents/wsu/phd/mount_semeru_eruption_preprocessed",
    "TEXAS-TORNADOES": "/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed",
    "STVINCENT-VOLCANO": "/homes/j244s673/documents/wsu/phd/stvincent_volcano_preprocessed",
    "TONGA-VOLCANO": "/homes/j244s673/documents/wsu/phd/tonga_volcano_preprocessed",
    "HURRICANE-DELTA": "/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed",
    "HURRICANE-IRMA": "/homes/j244s673/documents/wsu/phd/hurricane_irma_preprocessed",
    "HURRICANE-DORIAN": "/homes/j244s673/documents/wsu/phd/hurricane_dorian_preprocessed",
    "PAKISTAN-FLOODING": "/homes/j244s673/documents/wsu/phd/pakistan_flooding_preprocessed",
}


EXPERIMENTS = {
    "E0055": {
        "name": "Scratch_HLAURA_HDELTA_to_HIRMA",
        "train_datasets": ["HURRICANE-LAURA", "HURRICANE-DELTA"],
        "test_dataset": "HURRICANE-IRMA",
    },
    "E0056": {
        "name": "xBDTL_HLAURA_HDELTA_to_HIRMA",
        "train_datasets": ["HURRICANE-LAURA", "HURRICANE-DELTA"],
        "test_dataset": "HURRICANE-IRMA",
    },
    "E0057": {
        "name": "Scratch_ETURKEY_MSEMERU_to_PAKFLOOD",
        "train_datasets": ["EARTHQUAKE-TURKEY", "MOUNT-SEMERU-ERUPTION"],
        "test_dataset": "PAKISTAN-FLOODING",
    },
    "E0058": {
        "name": "xBDTL_ETURKEY_MSEMERU_to_PAKFLOOD",
        "train_datasets": ["EARTHQUAKE-TURKEY", "MOUNT-SEMERU-ERUPTION"],
        "test_dataset": "PAKISTAN-FLOODING",
    },
    "E0059": {
        "name": "Scratch_LOO_AllExceptPAKFLOOD_to_PAKFLOOD",
        "train_datasets": [
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
        ],
        "test_dataset": "PAKISTAN-FLOODING",
    },
    "E0060": {
        "name": "xBDTL_LOO_AllExceptPAKFLOOD_to_PAKFLOOD",
        "train_datasets": [
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
        ],
        "test_dataset": "PAKISTAN-FLOODING",
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

        # Prefix filenames so different datasets do not overwrite each other.
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
    args.single_prep_script = Path(args.project_root) / args.single_prep_script

    exp = EXPERIMENTS[args.experiment_id]
    train_datasets = exp["train_datasets"]
    test_dataset = exp["test_dataset"]

    print("================================================", flush=True)
    print(f"Experiment ID: {args.experiment_id}", flush=True)
    print(f"Experiment name: {exp['name']}", flush=True)
    print(f"Train datasets: {train_datasets}", flush=True)
    print(f"Test dataset: {test_dataset}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print("================================================", flush=True)

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.staging_root.mkdir(parents=True, exist_ok=True)

    needed_datasets = sorted(set(train_datasets + [test_dataset]))

    staged = {}
    for ds in needed_datasets:
        if ds not in DATASET_EXACT_PATHS:
            raise KeyError(f"Unknown dataset: {ds}")
        staged[ds] = prepare_single_dataset(ds, args.staging_root, args)

    # Train and validation come only from source datasets.
    for ds in train_datasets:
        merge_split(staged[ds], "train", args.output_root, "train", ds, args.mode)
        merge_split(staged[ds], "val", args.output_root, "val", ds, args.mode)

    # Test comes only from the unseen target dataset.
    merge_split(staged[test_dataset], "test", args.output_root, "test", test_dataset, args.mode)

    print("================================================", flush=True)
    print("Combined planned dataset prepared successfully.", flush=True)
    print("Final root:", args.output_root, flush=True)
    print("================================================", flush=True)


if __name__ == "__main__":
    main()