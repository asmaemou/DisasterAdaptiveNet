#!/usr/bin/env python3
"""
Prepare one disaster dataset as an xBD-style root for HRTBDA single-dataset experiments.

It creates:

  output_root/train/images, output_root/train/targets
  output_root/val/images,   output_root/val/targets
  output_root/test/images,  output_root/test/targets

Files are symlinked by default. Use --mode copy if needed.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# Exact preprocessed dataset paths you provided.
# These are checked first before aliases.
DATASET_EXACT_PATHS = {
    "IDA-BD": "/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet",
    "HURRICANE-IDA": "/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet",

    "IAN-BD": "/homes/j244s673/documents/wsu/phd/hurrican-ian",
    "HURRICANE-IAN": "/homes/j244s673/documents/wsu/phd/hurrican-ian",

    "EARTHQUAKE-TURKEY": "/homes/j244s673/documents/wsu/phd/earthquake_turkey_preprocessed",
    "HURRICANE-LAURA": "/homes/j244s673/documents/wsu/phd/hurricane_laura_preprocessed",
    "MOUNT-SEMERU-ERUPTION": "/homes/j244s673/documents/wsu/phd/mount_semeru_eruption_preprocessed",
    "TEXAS-TORNADOES": "/homes/j244s673/documents/wsu/phd/texas_tornadoes_preprocessed",
    "STVINCENT-VOLCANO": "/homes/j244s673/documents/wsu/phd/stvincent_volcano_preprocessed",
    "ST-VINCENT-VOLCANO": "/homes/j244s673/documents/wsu/phd/stvincent_volcano_preprocessed",
    "TONGA-VOLCANO": "/homes/j244s673/documents/wsu/phd/tonga_volcano_preprocessed",
    "HURRICANE-DELTA": "/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed",
    "HURRICANE-IRMA": "/homes/j244s673/documents/wsu/phd/hurricane_irma_preprocessed",
    "HURRICANE-DORIAN": "/homes/j244s673/documents/wsu/phd/hurricane_dorian_preprocessed",
    "PAKISTAN-FLOODING": "/homes/j244s673/documents/wsu/phd/pakistan_flooding_preprocessed",
}


DATASET_ALIASES = {
    "IDA-BD": [
        "idabd_real_disasteradaptivenet",
        "ida-bd",
        "IDA-BD",
        "hurricane_ida_preprocessed",
        "hurricane-ida_preprocessed",
        "hurricane_ida",
        "hurricane-ida",
        "HURRICANE-IDA",
        "HURRICANE_IDA",
    ],
    "HURRICANE-IDA": [
        "idabd_real_disasteradaptivenet",
        "ida-bd",
        "IDA-BD",
        "hurricane_ida_preprocessed",
        "hurricane-ida_preprocessed",
        "hurricane_ida",
        "hurricane-ida",
        "HURRICANE-IDA",
        "HURRICANE_IDA",
    ],
    "IAN-BD": [
        "ian-bd",
        "IAN-BD",
        "hurrican-ian",
        "hurricane_ian_preprocessed",
        "hurricane-ian_preprocessed",
        "hurricane_ian",
        "hurricane-ian",
        "HURRICANE-IAN",
    ],
    "HURRICANE-IAN": [
        "ian-bd",
        "IAN-BD",
        "hurrican-ian",
        "hurricane_ian_preprocessed",
        "hurricane-ian_preprocessed",
        "hurricane_ian",
        "hurricane-ian",
        "HURRICANE-IAN",
    ],
    "EARTHQUAKE-TURKEY": [
        "earthquake_turkey_preprocessed",
        "earthquake-turkey_preprocessed",
        "earthquake_turkey",
        "earthquake-turkey",
        "EARTHQUAKE-TURKEY",
    ],
    "HURRICANE-LAURA": [
        "hurricane_laura_preprocessed",
        "hurricane-laura_preprocessed",
        "hurricane_laura",
        "hurricane-laura",
        "HURRICANE-LAURA",
    ],
    "MOUNT-SEMERU-ERUPTION": [
        "mount_semeru_eruption_preprocessed",
        "mount-semeru-eruption_preprocessed",
        "mount_semeru_eruption",
        "mount-semeru-eruption",
        "MOUNT-SEMERU-ERUPTION",
    ],
    "TEXAS-TORNADOES": [
        "texas_tornadoes_preprocessed",
        "texas-tornadoes_preprocessed",
        "texas_tornadoes",
        "texas-tornadoes",
        "TEXAS-TORNADOES",
    ],
    "STVINCENT-VOLCANO": [
        "stvincent_volcano_preprocessed",
        "stvincent-volcano_preprocessed",
        "st_vincent_volcano_preprocessed",
        "st-vincent-volcano_preprocessed",
        "stvincent_volcano",
        "stvincent-volcano",
        "st_vincent_volcano",
        "st-vincent-volcano",
        "STVINCENT-VOLCANO",
        "ST-VINCENT-VOLCANO",
    ],
    "ST-VINCENT-VOLCANO": [
        "stvincent_volcano_preprocessed",
        "stvincent-volcano_preprocessed",
        "st_vincent_volcano_preprocessed",
        "st-vincent-volcano_preprocessed",
        "stvincent_volcano",
        "stvincent-volcano",
        "st_vincent_volcano",
        "st-vincent-volcano",
        "STVINCENT-VOLCANO",
        "ST-VINCENT-VOLCANO",
    ],
    "TONGA-VOLCANO": [
        "tonga_volcano_preprocessed",
        "tonga-volcano_preprocessed",
        "tonga_volcano",
        "tonga-volcano",
        "TONGA-VOLCANO",
    ],
    "HURRICANE-DELTA": [
        "hurricane_delta_preprocessed",
        "hurricane-delta_preprocessed",
        "hurricane_delta",
        "hurricane-delta",
        "HURRICANE-DELTA",
    ],
    "HURRICANE-IRMA": [
        "hurricane_irma_preprocessed",
        "hurricane-irma_preprocessed",
        "hurricane_irma",
        "hurricane-irma",
        "HURRICANE-IRMA",
    ],
    "HURRICANE-DORIAN": [
        "hurricane_dorian_preprocessed",
        "hurricane-dorian_preprocessed",
        "hurricane_dorian",
        "hurricane-dorian",
        "HURRICANE-DORIAN",
    ],
    "PAKISTAN-FLOODING": [
        "pakistan_flooding_preprocessed",
        "pakistan-flooding_preprocessed",
        "pakistan_flooding",
        "pakistan-flooding",
        "pakflood",
        "pak_flood",
        "PAKISTAN-FLOODING",
    ],
}


SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation", "hold"],
    "test": ["test", "testing"],
}


def sanitize_prefix(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_dataset_root(dataset_base: Path, dataset_name: str) -> Path:
    dataset_name = dataset_name.strip()
    dataset_upper = dataset_name.upper()

    # 1. Exact path mapping first
    if dataset_name in DATASET_EXACT_PATHS:
        p = Path(DATASET_EXACT_PATHS[dataset_name])
        if p.exists() and p.is_dir():
            print(f"Resolved {dataset_name} -> {p}")
            return p
        raise FileNotFoundError(f"Exact path for {dataset_name} does not exist: {p}")

    if dataset_upper in DATASET_EXACT_PATHS:
        p = Path(DATASET_EXACT_PATHS[dataset_upper])
        if p.exists() and p.is_dir():
            print(f"Resolved {dataset_name} -> {p}")
            return p
        raise FileNotFoundError(f"Exact path for {dataset_upper} does not exist: {p}")

    # 2. Absolute path passed directly
    direct = Path(dataset_name)
    if direct.is_absolute() and direct.exists() and direct.is_dir():
        print(f"Resolved {dataset_name} -> {direct}")
        return direct

    # 3. Alias/candidate search under dataset_base
    candidates = []
    candidates.extend(DATASET_ALIASES.get(dataset_name, []))
    candidates.extend(DATASET_ALIASES.get(dataset_upper, []))
    candidates.extend(
        [
            dataset_name,
            dataset_name.lower(),
            dataset_name.upper(),
            dataset_name.lower().replace("-", "_"),
            dataset_name.lower().replace("-", "_") + "_preprocessed",
            dataset_name.lower().replace("_", "-"),
            dataset_name.lower().replace("_", "-") + "_preprocessed",
        ]
    )

    candidates = unique_keep_order(candidates)

    for c in candidates:
        p = dataset_base / c
        if p.exists() and p.is_dir():
            print(f"Resolved {dataset_name} -> {p}")
            return p

    raise FileNotFoundError(
        f"Could not resolve dataset '{dataset_name}' under {dataset_base}\n"
        f"Tried candidates: {candidates}"
    )


def resolve_split_dir(dataset_root: Path, split: str) -> Path:
    candidates = SPLIT_ALIASES.get(split, [split])

    for c in candidates:
        p = dataset_root / c
        if p.exists() and p.is_dir():
            return p

    raise FileNotFoundError(
        f"Could not find split '{split}' under {dataset_root}. Tried {candidates}"
    )


def find_dir(split_root: Path, candidates):
    for c in candidates:
        p = split_root / c
        if p.exists() and p.is_dir():
            return p

    raise FileNotFoundError(f"Could not find any of {candidates} under {split_root}")


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_files(src_dir: Path):
    return sorted(
        [
            p
            for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def transfer_folder(
    src_dir: Path,
    dst_dir: Path,
    prefix: str,
    mode: str,
    manifest_rows,
    split: str,
    dataset: str,
    kind: str,
):
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(src_dir)

    for src in files:
        dst_name = f"{prefix}__{src.name}"
        dst = dst_dir / dst_name

        link_or_copy(src, dst, mode)

        manifest_rows.append(
            {
                "split": split,
                "dataset": dataset,
                "kind": kind,
                "source": str(src),
                "destination": str(dst),
            }
        )

    return len(files)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare single-dataset xBD-style root for HRTBDA fine-tuning/scratch experiments."
    )

    parser.add_argument("--dataset-base", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. HURRICANE-IAN or PAKISTAN-FLOODING",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")

    args = parser.parse_args()

    dataset_base = Path(args.dataset_base)
    output_root = Path(args.output_root)
    dataset = args.dataset.strip()

    print("====================================================")
    print("Preparing single-dataset HRTBDA root")
    print("Dataset base:", dataset_base)
    print("Dataset:", dataset)
    print("Output root:", output_root)
    print("Mode:", args.mode)
    print("====================================================")

    if not dataset_base.exists():
        raise FileNotFoundError(f"Dataset base does not exist: {dataset_base}")

    dataset_root = resolve_dataset_root(dataset_base, dataset)

    for split in ["train", "val", "test"]:
        reset_dir(output_root / split / "images")
        reset_dir(output_root / split / "targets")

    manifest_rows = []
    prefix = sanitize_prefix(dataset)
    counts = {}

    for split in ["train", "val", "test"]:
        split_root = resolve_split_dir(dataset_root, split)

        image_dir = find_dir(split_root, ["images", "image"])
        target_dir = find_dir(
            split_root,
            ["targets", "masks", "labels", "target", "label"],
        )

        n_images = transfer_folder(
            image_dir,
            output_root / split / "images",
            prefix,
            args.mode,
            manifest_rows,
            split,
            dataset,
            "image",
        )

        n_targets = transfer_folder(
            target_dir,
            output_root / split / "targets",
            prefix,
            args.mode,
            manifest_rows,
            split,
            dataset,
            "target",
        )

        counts[split] = (n_images, n_targets)

        print(
            f"{split:5s} | {dataset:25s} | "
            f"source={split_root} | images={n_images:6d} | targets={n_targets:6d}"
        )

        if n_images == 0:
            raise RuntimeError(f"No image files found in {image_dir}")

        if n_targets == 0:
            raise RuntimeError(f"No target files found in {target_dir}")

        if n_images != n_targets:
            print(
                f"WARNING: image/target count mismatch for {dataset} {split}: "
                f"images={n_images}, targets={n_targets}"
            )

    manifest_path = output_root / "manifest.csv"

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "dataset", "kind", "source", "destination"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_root / "dataset_counts.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "split", "images", "targets"])

        for split, (n_images, n_targets) in counts.items():
            writer.writerow([dataset, split, n_images, n_targets])

    print("====================================================")
    print("Done preparing single-dataset root.")
    print("Dataset root:", dataset_root)
    print("Manifest:", manifest_path)
    print("Counts:", summary_path)
    print("====================================================")


if __name__ == "__main__":
    main()