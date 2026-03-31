import json
import shutil
from pathlib import Path
import cv2
import numpy as np

SRC_ROOT = Path("/homes/j244s673/documents/wsu/phd/xFBD/data/xfbd/data 7/xfbd/random_building")
OUT_ROOT = Path("/homes/j244s673/documents/wsu/phd/xfbd_random_disasteradaptivenet")

TRAIN_SOURCE_SPLITS = ["tier1", "tier3"]
VAL_SOURCE_SPLITS = ["hold"]
TEST_SOURCE_SPLITS = ["test"]


def reset_output_root():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ["train", "val", "test"]:
        for sub in ["images", "masks", "targets"]:
            (OUT_ROOT / split / sub).mkdir(parents=True, exist_ok=True)


def read_mask_any(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def write_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def collect_pairs_from_pngs_split(split_name: str):
    png_dir = SRC_ROOT / split_name / "pngs"
    if not png_dir.exists():
        return []

    pairs = []
    for pre_path in sorted(png_dir.glob("*_pre_disaster.png")):
        stem = pre_path.stem.replace("_pre_disaster", "")
        event, patch_id = stem.rsplit("_", 1)

        post_img = png_dir / f"{event}_{patch_id}_post_disaster.png"
        pre_msk = png_dir / f"{event}_{patch_id}_pre_disaster.png.msk"
        post_msk = png_dir / f"{event}_{patch_id}_post_disaster.png.msk"

        if post_img.exists() and pre_msk.exists() and post_msk.exists():
            pairs.append(("pngs", split_name, event, patch_id))

    return pairs


def collect_pairs_from_tif_split(split_name: str):
    img_dir = SRC_ROOT / split_name / "images"
    msk_dir = SRC_ROOT / split_name / "masks"

    if not img_dir.exists() or not msk_dir.exists():
        return []

    pairs = []
    for pre_path in sorted(img_dir.glob("*_pre_disaster.tif")):
        stem = pre_path.stem.replace("_pre_disaster", "")
        event, patch_id = stem.rsplit("_", 1)

        post_img = img_dir / f"{event}_{patch_id}_post_disaster.tif"
        post_msk = msk_dir / f"{event}_{patch_id}_post_disaster.tif"

        if post_img.exists() and post_msk.exists():
            pairs.append(("tif", split_name, event, patch_id))

    return pairs


def collect_pairs(split_names):
    all_pairs = []
    for split_name in split_names:
        pairs_png = collect_pairs_from_pngs_split(split_name)
        pairs_tif = collect_pairs_from_tif_split(split_name)

        pairs = pairs_png + pairs_tif
        print(f"{split_name}: found {len(pairs)} valid pairs")
        all_pairs.extend(pairs)
    return all_pairs


def copy_sample_from_pngs(src_split: str, dst_split: str, event: str, patch_id: str):
    png_dir = SRC_ROOT / src_split / "pngs"

    # copy images directly
    for suffix in ["pre", "post"]:
        img_name = f"{event}_{patch_id}_{suffix}_disaster.png"
        shutil.copy2(
            png_dir / img_name,
            OUT_ROOT / dst_split / "images" / img_name
        )

    # copy masks directly, renaming .msk -> .png
    pre_mask_src = png_dir / f"{event}_{patch_id}_pre_disaster.png.msk"
    post_mask_src = png_dir / f"{event}_{patch_id}_post_disaster.png.msk"

    pre_mask_dst = OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_pre_disaster.png"
    post_mask_dst = OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_post_disaster.png"

    shutil.copy2(pre_mask_src, pre_mask_dst)
    shutil.copy2(post_mask_src, post_mask_dst)

    shutil.copy2(
        pre_mask_src,
        OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_pre_disaster_target.png"
    )
    shutil.copy2(
        post_mask_src,
        OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_post_disaster_target.png"
    )


def copy_sample_from_tif(src_split: str, dst_split: str, event: str, patch_id: str):
    img_dir = SRC_ROOT / src_split / "images"
    msk_dir = SRC_ROOT / src_split / "masks"

    # read tif images, save as png
    pre_img = cv2.imread(str(img_dir / f"{event}_{patch_id}_pre_disaster.tif"), cv2.IMREAD_COLOR)
    post_img = cv2.imread(str(img_dir / f"{event}_{patch_id}_post_disaster.tif"), cv2.IMREAD_COLOR)

    if pre_img is None:
        raise FileNotFoundError(f"Could not read pre-image tif for {event}_{patch_id}")
    if post_img is None:
        raise FileNotFoundError(f"Could not read post-image tif for {event}_{patch_id}")

    write_png(OUT_ROOT / dst_split / "images" / f"{event}_{patch_id}_pre_disaster.png", pre_img)
    write_png(OUT_ROOT / dst_split / "images" / f"{event}_{patch_id}_post_disaster.png", post_img)

    # read post mask tif
    post_mask = read_mask_any(msk_dir / f"{event}_{patch_id}_post_disaster.tif")

    # create pre mask as binary localization from post mask
    pre_mask = np.where(post_mask > 0, 255, 0).astype(np.uint8)

    write_png(OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_pre_disaster.png", pre_mask)
    write_png(OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_post_disaster.png", post_mask.astype(np.uint8))

    write_png(OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_pre_disaster_target.png", pre_mask)
    write_png(OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_post_disaster_target.png", post_mask.astype(np.uint8))


def summarize_sample(dst_split: str, event: str, patch_id: str):
    post_mask_path = OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_post_disaster.png"
    m = read_mask_any(post_mask_path)

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

    for fmt, src_split, event, patch_id in pairs:
        if fmt == "pngs":
            copy_sample_from_pngs(src_split, dst_split, event, patch_id)
        elif fmt == "tif":
            copy_sample_from_tif(src_split, dst_split, event, patch_id)
        else:
            raise ValueError(f"Unknown format: {fmt}")

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