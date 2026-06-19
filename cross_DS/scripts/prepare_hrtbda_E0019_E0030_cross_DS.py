#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


DATASET_ALIASES = {
    "EARTHQUAKE-TURKEY": [
        "earthquake_turkey_preprocessed",
        "earthquake-turkey_preprocessed",
        "earthquake_turkey",
        "earthquake-turkey",
        "EARTHQUAKE-TURKEY",
    ],
    "HURRICANE-DELTA": [
        "hurricane_delta_preprocessed",
        "hurricane-delta_preprocessed",
        "hurricane_delta",
        "hurricane-delta",
        "HURRICANE-DELTA",
    ],
    "HURRICANE-IAN": [
        "hurrican-ian",
        "hurricane_ian_preprocessed",
        "hurricane-ian_preprocessed",
        "hurricane_ian",
        "hurricane-ian",
        "HURRICANE-IAN",
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
    ],
    "TEXAS-TORNADOES": [
        "texas_tornadoes_preprocessed",
        "texas-tornadoes_preprocessed",
        "texas_tornadoes",
        "texas-tornadoes",
        "TEXAS-TORNADOES",
    ],
    "TONGA-VOLCANO": [
        "tonga_volcano_preprocessed",
        "tonga-volcano_preprocessed",
        "tonga_volcano",
        "tonga-volcano",
        "TONGA-VOLCANO",
    ],
    "xBD": [
        "xview2",
        "xbd",
        "xBD",
        "XBD",
    ],
    "XBD": [
        "xview2",
        "xbd",
        "xBD",
        "XBD",
    ],
}


def sanitize_prefix(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def find_dir(root: Path, split: str, candidates):
    for c in candidates:
        p = root / split / c
        if p.exists() and p.is_dir():
            return p

    raise FileNotFoundError(
        f"Could not find any of {candidates} under {root / split}"
    )


def resolve_dataset_root(dataset_base: Path, dataset_name: str) -> Path:
    dataset_name = dataset_name.strip()

    direct = Path(dataset_name)
    if direct.is_absolute() and direct.exists():
        return direct

    candidates = []

    if dataset_name in DATASET_ALIASES:
        candidates.extend(DATASET_ALIASES[dataset_name])

    upper_name = dataset_name.upper()
    if upper_name in DATASET_ALIASES:
        candidates.extend(DATASET_ALIASES[upper_name])

    lower_dash = dataset_name.lower()
    lower_under = lower_dash.replace("-", "_")

    candidates.extend(
        [
            dataset_name,
            lower_dash,
            lower_under,
            lower_dash + "_preprocessed",
            lower_under + "_preprocessed",
        ]
    )

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    for c in unique_candidates:
        p = dataset_base / c
        if p.exists() and p.is_dir():
            print(f"Resolved {dataset_name} -> {p}")
            return p

    raise FileNotFoundError(
        f"Could not resolve dataset '{dataset_name}' under {dataset_base}\n"
        f"Tried candidates: {unique_candidates}"
    )


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_files(src_dir: Path):
    return sorted(
        [
            p for p in src_dir.iterdir()
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
        description="Prepare combined cross-dataset xBD-style root for HRTBDA E0019-E0030."
    )

    parser.add_argument("--dataset-base", required=True)
    parser.add_argument("--train-datasets", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")

    args = parser.parse_args()

    dataset_base = Path(args.dataset_base)
    train_datasets = [
        x.strip() for x in args.train_datasets.split(",") if x.strip()
    ]
    test_dataset = args.test_dataset.strip()
    output_root = Path(args.output_root)

    print("====================================================")
    print("Preparing HRTBDA cross-dataset root")
    print("Dataset base:", dataset_base)
    print("Train datasets:", train_datasets)
    print("Test dataset:", test_dataset)
    print("Output root:", output_root)
    print("Mode:", args.mode)
    print("====================================================")

    if not dataset_base.exists():
        raise FileNotFoundError(f"Dataset base does not exist: {dataset_base}")

    for split in ["train", "val", "test"]:
        reset_dir(output_root / split / "images")
        reset_dir(output_root / split / "targets")

    manifest_rows = []

    for dataset in train_datasets:
        dataset_root = resolve_dataset_root(dataset_base, dataset)
        prefix = sanitize_prefix(dataset)

        for split in ["train", "val"]:
            image_dir = find_dir(dataset_root, split, ["images"])
            target_dir = find_dir(dataset_root, split, ["targets", "masks", "labels"])

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

            print(
                f"{split:5s} | {dataset:25s} | "
                f"source={dataset_root} | images={n_images:6d} | targets={n_targets:6d}"
            )

    dataset_root = resolve_dataset_root(dataset_base, test_dataset)
    prefix = sanitize_prefix(test_dataset)

    image_dir = find_dir(dataset_root, "test", ["images"])
    target_dir = find_dir(dataset_root, "test", ["targets", "masks", "labels"])

    n_images = transfer_folder(
        image_dir,
        output_root / "test" / "images",
        prefix,
        args.mode,
        manifest_rows,
        "test",
        test_dataset,
        "image",
    )

    n_targets = transfer_folder(
        target_dir,
        output_root / "test" / "targets",
        prefix,
        args.mode,
        manifest_rows,
        "test",
        test_dataset,
        "target",
    )

    print(
        f"{'test':5s} | {test_dataset:25s} | "
        f"source={dataset_root} | images={n_images:6d} | targets={n_targets:6d}"
    )

    manifest_path = output_root / "manifest.csv"

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "dataset", "kind", "source", "destination"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("====================================================")
    print("Done preparing combined dataset.")
    print("Manifest:", manifest_path)
    print("====================================================")


if __name__ == "__main__":
    main()
