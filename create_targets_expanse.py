import json
import re
import timeit
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Expanse xView2/xBD root
dataset_dir = Path("/home/amouradi/expanse/xview2")
split = "tier3"

DAMAGE_DICT = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1,
    "unclassified": 1,
}


def parse_wkt_rings(wkt: str):
    """
    Lightweight parser for xBD-style POLYGON/MULTIPOLYGON WKT.
    Returns a list of polygon rings, each as [(x, y), ...].
    Avoids shapely to prevent GEOS segfaults on the cluster.
    """
    if not wkt:
        return []

    # Extract coordinate rings inside parentheses.
    # Example: POLYGON ((x y, x y, ...))
    rings_text = re.findall(r"\(\s*([+-]?\d+(?:\.\d+)?\s+[+-]?\d+(?:\.\d+)?(?:\s*,\s*[+-]?\d+(?:\.\d+)?\s+[+-]?\d+(?:\.\d+)?)+)\s*\)", wkt)

    rings = []
    for ring_text in rings_text:
        pts = []
        for pair in ring_text.split(","):
            nums = pair.strip().split()
            if len(nums) < 2:
                continue
            try:
                x = int(round(float(nums[0])))
                y = int(round(float(nums[1])))
            except ValueError:
                continue

            x = max(0, min(1023, x))
            y = max(0, min(1023, y))
            pts.append((x, y))

        if len(pts) >= 3:
            rings.append(pts)

    return rings


def draw_polygons(mask: Image.Image, features, value_getter):
    draw = ImageDraw.Draw(mask)

    for feat in features:
        wkt = feat.get("wkt", "")
        rings = parse_wkt_rings(wkt)

        if not rings:
            continue

        value = value_getter(feat)

        for ring in rings:
            draw.polygon(ring, fill=int(value))


def process_image(pre_json_file: Path):
    subset_dir = pre_json_file.parent.parent
    labels_dir = subset_dir / "labels"
    targets_dir = subset_dir / "targets"

    prefix = pre_json_file.stem.replace("_pre_disaster", "")
    post_json_file = labels_dir / f"{prefix}_post_disaster.json"

    if not post_json_file.exists():
        print(f"SKIP missing post label: {post_json_file}")
        return False

    with open(pre_json_file, "r") as f:
        pre_data = json.load(f)

    with open(post_json_file, "r") as f:
        post_data = json.load(f)

    pre_features = pre_data.get("features", {}).get("xy", [])
    post_features = post_data.get("features", {}).get("xy", [])

    # Phase I localization target: 0 background, 1 building
    loc_mask = Image.new("L", (1024, 1024), 0)
    draw_polygons(loc_mask, pre_features, lambda feat: 1)

    # Phase II damage target:
    # 0 background, 1 no damage, 2 minor, 3 major, 4 destroyed
    dmg_mask = Image.new("L", (1024, 1024), 0)

    def damage_value(feat):
        subtype = feat.get("properties", {}).get("subtype", "no-damage")
        return DAMAGE_DICT.get(subtype, 1)

    draw_polygons(dmg_mask, post_features, damage_value)

    loc_file = targets_dir / f"{prefix}_pre_disaster_target.png"
    dmg_file = targets_dir / f"{prefix}_post_disaster_target.png"

    loc_mask.save(loc_file, compress_level=9)
    dmg_mask.save(dmg_file, compress_level=9)

    return True


def main():
    t0 = timeit.default_timer()

    split_dir = dataset_dir / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    targets_dir = split_dir / "targets"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    targets_dir.mkdir(parents=True, exist_ok=True)

    pre_images = sorted(images_dir.glob("*_pre_disaster.png"))
    if not pre_images:
        raise RuntimeError(f"No *_pre_disaster.png files found in {images_dir}")

    pre_json_files = []
    for img in pre_images:
        jf = labels_dir / f"{img.stem}.json"
        if jf.exists():
            pre_json_files.append(jf)
        else:
            print(f"SKIP missing pre label: {jf}")

    print(f"Found pre images: {len(pre_images)}")
    print(f"Found pre labels: {len(pre_json_files)}")
    print(f"Writing targets to: {targets_dir}")

    made = 0
    for i, jf in enumerate(pre_json_files, 1):
        ok = process_image(jf)
        if ok:
            made += 1

        if i % 100 == 0:
            print(f"Processed {i}/{len(pre_json_files)} | created pairs: {made}")

    elapsed = timeit.default_timer() - t0
    print(f"Done. Created target pairs: {made}")
    print("Time: {:.3f} min".format(elapsed / 60))


if __name__ == "__main__":
    main()