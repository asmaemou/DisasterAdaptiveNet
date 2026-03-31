import json
import shutil
from pathlib import Path
import cv2

SRC_ROOT = Path("/homes/j244s673/documents/wsu/phd/xFBD/data/xfbd/data 7/xfbd/random_building")
OUT_ROOT = Path("/homes/j244s673/documents/wsu/phd/xfbd_random_disasteradaptivenet")

TRAIN_SOURCE_SPLITS = ["train", "tier1", "tier3"]
VAL_SOURCE_SPLITS = ["hold"]
TEST_SOURCE_SPLITS = ["test"]


def reset_output_root():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ["train", "val", "test"]:
        for sub in ["images", "masks", "targets"]:
            (OUT_ROOT / split / sub).mkdir(parents=True, exist_ok=True)


def source_png_dir(split_name: str) -> Path:
    return SRC_ROOT / split_name / "pngs"


def collect_pairs_from_split(split_name: str):
    png_dir = source_png_dir(split_name)
    if not png_dir.exists():
        print(f"Skipping missing split folder: {png_dir}")
        return []

    pairs = []
    for pre_path in sorted(png_dir.glob("*_pre_disaster.png")):
        stem = pre_path.stem.replace("_pre_disaster", "")
        event, patch_id = stem.rsplit("_", 1)

        post_img = png_dir / f"{event}_{patch_id}_post_disaster.png"
        pre_msk = png_dir / f"{event}_{patch_id}_pre_disaster.png.msk"
        post_msk = png_dir / f"{event}_{patch_id}_post_disaster.png.msk"

        if post_img.exists() and pre_msk.exists() and post_msk.exists():
            pairs.append((split_name, event, patch_id))

    return pairs


def collect_pairs(split_names):
    all_pairs = []
    for split_name in split_names:
        pairs = collect_pairs_from_split(split_name)
        print(f"{split_name}: found {len(pairs)} valid pairs")
        all_pairs.extend(pairs)
    return all_pairs


def copy_sample(src_split: str, dst_split: str, event: str, patch_id: str):
    png_dir = source_png_dir(src_split)

    for suffix in ["pre", "post"]:
        img_name = f"{event}_{patch_id}_{suffix}_disaster.png"
        shutil.copy2(
            png_dir / img_name,
            OUT_ROOT / dst_split / "images" / img_name
        )

    for suffix in ["pre", "post"]:
        msk_src = png_dir / f"{event}_{patch_id}_{suffix}_disaster.png.msk"
        msk_dst = OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_{suffix}_disaster.png"
        shutil.copy2(msk_src, msk_dst)

    shutil.copy2(
        png_dir / f"{event}_{patch_id}_pre_disaster.png.msk",
        OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_pre_disaster_target.png"
    )
    shutil.copy2(
        png_dir / f"{event}_{patch_id}_post_disaster.png.msk",
        OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_post_disaster_target.png"
    )


def summarize_sample(dst_split: str, event: str, patch_id: str):
    post_mask_path = OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_post_disaster.png"
    m = cv2.imread(str(post_mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read {post_mask_path}")

    return {
        "event": event,
        "patch_id": patch_id,
        "subset": dst_split,
        "loc": int((m > 0).sum() > 0),
        "cls_1": int((m == 1).sum() > 0),
        "cls_2": int((m == 2).sum() > 0),
        "cls_3": int((m == 3).sum() > 0),
        "cls_4": int((m == 4).sum() > 0),
    }


def build_split(dst_split: str, source_splits):
    pairs = collect_pairs(source_splits)
    entries = []

    for src_split, event, patch_id in pairs:
        copy_sample(src_split, dst_split, event, patch_id)
        entries.append(summarize_sample(dst_split, event, patch_id))

    print(f"{dst_split}: wrote {len(entries)} samples")
    return {"patches": entries}


def main():
    reset_output_root()

    metadata = {
        "train": build_split("train", TRAIN_SOURCE_SPLITS),
        "val": build_split("val", VAL_SOURCE_SPLITS),
        "test": build_split("test", TEST_SOURCE_SPLITS),
    }

    with open(OUT_ROOT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prepared xFBD dataset at: {OUT_ROOT}")


if __name__ == "__main__":
    main()