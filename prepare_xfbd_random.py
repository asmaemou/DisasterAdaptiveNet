import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

random.seed(42)

SRC_ROOT = Path("/homes/j244s673/documents/wsu/phd/xFBD/data/xfbd/data 7/xfbd/random_building")
OUT_ROOT = Path("/homes/j244s673/documents/wsu/phd/xfbd_random_disasteradaptivenet")


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


def collect_tier1_pairs():
    img_dir = SRC_ROOT / "tier1" / "images"
    msk_dir = SRC_ROOT / "tier1" / "masks"

    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"tier1 images/masks not found under {SRC_ROOT}")

    pairs = []
    for pre_path in sorted(img_dir.glob("*_pre_disaster.tif")):
        stem = pre_path.stem.replace("_pre_disaster", "")
        event, patch_id = stem.rsplit("_", 1)

        post_img = img_dir / f"{event}_{patch_id}_post_disaster.tif"
        post_msk = msk_dir / f"{event}_{patch_id}_post_disaster.tif"

        if post_img.exists() and post_msk.exists():
            pairs.append((event, patch_id))

    return pairs


def collect_test_pairs():
    png_dir = SRC_ROOT / "test" / "pngs"
    if not png_dir.exists():
        raise FileNotFoundError(f"test/pngs not found under {SRC_ROOT}")

    pairs = []
    for pre_path in sorted(png_dir.glob("*_pre_disaster.png")):
        stem = pre_path.stem.replace("_pre_disaster", "")
        event, patch_id = stem.rsplit("_", 1)

        post_img = png_dir / f"{event}_{patch_id}_post_disaster.png"
        pre_msk = png_dir / f"{event}_{patch_id}_pre_disaster.png.msk"
        post_msk = png_dir / f"{event}_{patch_id}_post_disaster.png.msk"

        if post_img.exists() and pre_msk.exists() and post_msk.exists():
            pairs.append((event, patch_id))

    return pairs


def copy_sample_from_tif(dst_split: str, event: str, patch_id: str):
    img_dir = SRC_ROOT / "tier1" / "images"
    msk_dir = SRC_ROOT / "tier1" / "masks"

    pre_img = cv2.imread(str(img_dir / f"{event}_{patch_id}_pre_disaster.tif"), cv2.IMREAD_COLOR)
    post_img = cv2.imread(str(img_dir / f"{event}_{patch_id}_post_disaster.tif"), cv2.IMREAD_COLOR)

    if pre_img is None:
        raise FileNotFoundError(f"Could not read pre-image tif for {event}_{patch_id}")
    if post_img is None:
        raise FileNotFoundError(f"Could not read post-image tif for {event}_{patch_id}")

    write_png(OUT_ROOT / dst_split / "images" / f"{event}_{patch_id}_pre_disaster.png", pre_img)
    write_png(OUT_ROOT / dst_split / "images" / f"{event}_{patch_id}_post_disaster.png", post_img)

    post_mask = read_mask_any(msk_dir / f"{event}_{patch_id}_post_disaster.tif")
    pre_mask = np.where(post_mask > 0, 255, 0).astype(np.uint8)

    write_png(OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_pre_disaster.png", pre_mask)
    write_png(OUT_ROOT / dst_split / "masks" / f"{event}_{patch_id}_post_disaster.png", post_mask.astype(np.uint8))

    write_png(OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_pre_disaster_target.png", pre_mask)
    write_png(OUT_ROOT / dst_split / "targets" / f"{event}_{patch_id}_post_disaster_target.png", post_mask.astype(np.uint8))


def copy_sample_from_pngs(dst_split: str, event: str, patch_id: str):
    png_dir = SRC_ROOT / "test" / "pngs"

    for suffix in ["pre", "post"]:
        img_name = f"{event}_{patch_id}_{suffix}_disaster.png"
        shutil.copy2(
            png_dir / img_name,
            OUT_ROOT / dst_split / "images" / img_name
        )

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


def build_train_val():
    pairs = collect_tier1_pairs()
    print(f"tier1: found {len(pairs)} valid pairs")

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(0.9 * n)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    train_entries = []
    val_entries = []

    for event, patch_id in train_pairs:
        copy_sample_from_tif("train", event, patch_id)
        train_entries.append(summarize_sample("train", event, patch_id))

    for event, patch_id in val_pairs:
        copy_sample_from_tif("val", event, patch_id)
        val_entries.append(summarize_sample("val", event, patch_id))

    print(f"train: wrote {len(train_entries)} samples")
    print(f"val: wrote {len(val_entries)} samples")

    return {"patches": train_entries}, {"patches": val_entries}


def build_test():
    pairs = collect_test_pairs()
    print(f"test: found {len(pairs)} valid pairs")

    test_entries = []
    for event, patch_id in pairs:
        copy_sample_from_pngs("test", event, patch_id)
        test_entries.append(summarize_sample("test", event, patch_id))

    print(f"test: wrote {len(test_entries)} samples")
    return {"patches": test_entries}


def main():
    reset_output_root()

    train_meta, val_meta = build_train_val()
    test_meta = build_test()

    metadata = {
        "train": train_meta,
        "val": val_meta,
        "test": test_meta,
    }

    with open(OUT_ROOT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prepared xFBD dataset at: {OUT_ROOT}")


if __name__ == "__main__":
    main()