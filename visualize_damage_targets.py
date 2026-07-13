
import cv2

import numpy as np

from pathlib import Path

import random



split_root = Path("/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed/train")

image_dir = split_root / "images"

target_dir = split_root / "targets"

out_dir = Path("target_visual_audit_hurricane_delta_train")

out_dir.mkdir(exist_ok=True)



label_names = {

    1: "no_damage",

    2: "minor_damage",

    3: "major_damage",

    4: "destroyed",

}



# BGR colors for OpenCV overlay

colors = {

    1: (0, 255, 0),      # green

    2: (0, 255, 255),    # yellow

    3: (0, 165, 255),    # orange

    4: (0, 0, 255),      # red

}



def find_image(stem):

    candidates = [

        image_dir / f"{stem}.png",

        image_dir / f"{stem}.jpg",

        image_dir / f"{stem}.jpeg",

        image_dir / f"{stem}.tif",

        image_dir / f"{stem}.tiff",

    ]

    for p in candidates:

        if p.exists():

            return p

    return None



def make_overlay(img, mask):

    overlay = img.copy()

    color_mask = np.zeros_like(img)



    for cls, color in colors.items():

        color_mask[mask == cls] = color



    nonzero = mask > 0

    overlay[nonzero] = cv2.addWeighted(img, 0.55, color_mask, 0.45, 0)[nonzero]

    return overlay



def add_title(img, title):

    out = img.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)

    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return out



post_targets = sorted(target_dir.glob("*_post_disaster_target.png"))



for cls, cls_name in label_names.items():

    cls_files = []



    for target_path in post_targets:

        mask = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)

        if mask is None:

            continue

        if mask.ndim == 3:

            mask = mask[..., 0]



        count = int((mask == cls).sum())

        if count > 0:

            cls_files.append((target_path, count))



    cls_files = sorted(cls_files, key=lambda x: x[1], reverse=True)

    selected = cls_files[:20]



    cls_out = out_dir / f"class_{cls}_{cls_name}"

    cls_out.mkdir(parents=True, exist_ok=True)



    print(f"\nClass {cls} = {cls_name}")

    print(f"Files containing this class: {len(cls_files)}")

    print(f"Saving top {len(selected)} examples to: {cls_out}")



    for i, (target_path, count) in enumerate(selected):

        post_stem = target_path.name.replace("_target.png", "")

        pre_stem = post_stem.replace("_post_disaster", "_pre_disaster")



        post_img_path = find_image(post_stem)

        pre_img_path = find_image(pre_stem)



        if post_img_path is None or pre_img_path is None:

            print("Missing image for:", target_path.name)

            continue



        post_img = cv2.imread(str(post_img_path), cv2.IMREAD_COLOR)

        pre_img = cv2.imread(str(pre_img_path), cv2.IMREAD_COLOR)

        mask = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)



        if mask.ndim == 3:

            mask = mask[..., 0]



        if post_img is None or pre_img is None:

            continue



        if post_img.shape[:2] != mask.shape[:2]:

            post_img = cv2.resize(post_img, (mask.shape[1], mask.shape[0]))

        if pre_img.shape[:2] != mask.shape[:2]:

            pre_img = cv2.resize(pre_img, (mask.shape[1], mask.shape[0]))



        overlay = make_overlay(post_img, mask)



        pre_vis = add_title(pre_img, "pre-disaster image")

        post_vis = add_title(post_img, "post-disaster image")

        overlay_vis = add_title(overlay, f"target overlay: {cls_name}, pixels={count}")



        combined = np.concatenate([pre_vis, post_vis, overlay_vis], axis=1)



        out_path = cls_out / f"{i:02d}_{target_path.stem}_pixels_{count}.png"

        cv2.imwrite(str(out_path), combined)



print("\nDone.")

print("Open this folder to inspect examples:")

print(out_dir.resolve())

