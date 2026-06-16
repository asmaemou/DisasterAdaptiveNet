#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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
        description="Prepare combined cross-dataset xBD-style root for HRTBDA."
    )

    parser.add_argument(
        "--dataset-base",
        required=True,
        help="Base folder containing disaster dataset folders.",
    )

    parser.add_argument(
        "--train-datasets",
        required=True,
        help="Comma-separated training datasets, e.g. EARTHQUAKE-TURKEY,HURRICANE-DELTA",
    )

    parser.add_argument(
        "--test-dataset",
        required=True,
        help="Target test dataset, e.g. HURRICANE-IAN",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="Output combined xBD-style dataset root.",
    )

    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlink to save space or copy files physically.",
    )

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

    for split in ["train", "val", "test"]:
        reset_dir(output_root / split / "images")
        reset_dir(output_root / split / "targets")

    manifest_rows = []

    # Train and validation come from Dataset A + Dataset B
    for dataset in train_datasets:
        dataset_root = dataset_base / dataset
        prefix = sanitize_prefix(dataset)

        for split in ["train", "val"]:
            image_dir = find_dir(dataset_root, split, ["images"])
            target_dir = find_dir(dataset_root, split, ["targets", "masks", "labels"])

            n_images = transfer_folder(
                src_dir=image_dir,
                dst_dir=output_root / split / "images",
                prefix=prefix,
                mode=args.mode,
                manifest_rows=manifest_rows,
                split=split,
                dataset=dataset,
                kind="image",
            )

            n_targets = transfer_folder(
                src_dir=target_dir,
                dst_dir=output_root / split / "targets",
                prefix=prefix,
                mode=args.mode,
                manifest_rows=manifest_rows,
                split=split,
                dataset=dataset,
                kind="target",
            )

            print(
                f"{split:5s} | {dataset:25s} | "
                f"images={n_images:6d} | targets={n_targets:6d}"
            )

    # Test comes only from Dataset C
    dataset_root = dataset_base / test_dataset
    prefix = sanitize_prefix(test_dataset)

    image_dir = find_dir(dataset_root, "test", ["images"])
    target_dir = find_dir(dataset_root, "test", ["targets", "masks", "labels"])

    n_images = transfer_folder(
        src_dir=image_dir,
        dst_dir=output_root / "test" / "images",
        prefix=prefix,
        mode=args.mode,
        manifest_rows=manifest_rows,
        split="test",
        dataset=test_dataset,
        kind="image",
    )

    n_targets = transfer_folder(
        src_dir=target_dir,
        dst_dir=output_root / "test" / "targets",
        prefix=prefix,
        mode=args.mode,
        manifest_rows=manifest_rows,
        split="test",
        dataset=test_dataset,
        kind="target",
    )

    print(
        f"{'test':5s} | {test_dataset:25s} | "
        f"images={n_images:6d} | targets={n_targets:6d}"
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