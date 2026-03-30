import json
import random
import shutil
from pathlib import Path
import cv2

random.seed(42)

RAW_ROOT = Path(
    "/homes/j244s673/documents/wsu/phd/idabd_real/PRJ-3563/"
    "Project--ida-bd-pre-and-post-disaster-high-resolution-satellite-imagery-for-building-damage-assessment-from-hurricane-ida"
)

IMG_DIR = RAW_ROOT / "images"
MSK_DIR = RAW_ROOT / "masks"

OUT_ROOT = Path("/homes/j244s673/documents/wsu/phd/idabd_real_disasteradaptivenet")

for split in ["train", "val", "test"]:
    for sub in ["images", "masks", "targets"]:
        (OUT_ROOT / split / sub).mkdir(parents=True, exist_ok=True)


def collect_pairs():
    pairs = []
    for pre_path in sorted(IMG_DIR.glob("*_pre_disaster.png")):
        stem = pre_path.stem.replace("_pre_disaster", "")
        event, patch_id = stem.rsplit("_", 1)

        post_img = IMG_DIR / f"{event}_{patch_id}_post_disaster.png"
        pre_msk = MSK_DIR / f"{event}_{patch_id}_pre_disaster.png"
        post_msk = MSK_DIR / f"{event}_{patch_id}_post_disaster.png"

        if post_img.exists() and pre_msk.exists() and post_msk.exists():
            pairs.append((event, patch_id))
    return pairs


def summarize_sample(event: str, patch_id: str, subset: str):
    post_mask_path = OUT_ROOT / subset / "masks" / f"{event}_{patch_id}_post_disaster.png"
    m = cv2.imread(str(post_mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read {post_mask_path}")

    return {
        "event": event,
        "patch_id": patch_id,
        "subset": subset,
        "loc": int((m > 0).sum() > 0),
        "cls_1": int((m == 1).sum() > 0),
        "cls_2": int((m == 2).sum() > 0),
        "cls_3": int((m == 3).sum() > 0),
        "cls_4": int((m == 4).sum() > 0),
    }


pairs = collect_pairs()
print(f"Found {len(pairs)} valid pre/post pairs.")
random.shuffle(pairs)

n = len(pairs)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

split_map = {
    "train": pairs[:n_train],
    "val": pairs[n_train:n_train + n_val],
    "test": pairs[n_train + n_val:],
}

metadata = {}

for split, split_pairs in split_map.items():
    patch_entries = []

    for event, patch_id in split_pairs:
        # copy paired images
        for suffix in ["pre", "post"]:
            img_name = f"{event}_{patch_id}_{suffix}_disaster.png"
            shutil.copy2(IMG_DIR / img_name, OUT_ROOT / split / "images" / img_name)

        # copy paired masks
        for suffix in ["pre", "post"]:
            msk_name = f"{event}_{patch_id}_{suffix}_disaster.png"
            shutil.copy2(MSK_DIR / msk_name, OUT_ROOT / split / "masks" / msk_name)

        # create targets from masks
        shutil.copy2(
            MSK_DIR / f"{event}_{patch_id}_pre_disaster.png",
            OUT_ROOT / split / "targets" / f"{event}_{patch_id}_pre_disaster_target.png",
        )
        shutil.copy2(
            MSK_DIR / f"{event}_{patch_id}_post_disaster.png",
            OUT_ROOT / split / "targets" / f"{event}_{patch_id}_post_disaster_target.png",
        )

        patch_entries.append(summarize_sample(event, patch_id, split))

    metadata[split] = {"patches": patch_entries}
    print(f"{split}: {len(patch_entries)} samples")

with open(OUT_ROOT / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Prepared Ida-BD dataset at: {OUT_ROOT}")