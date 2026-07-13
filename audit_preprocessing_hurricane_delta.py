
import cv2

import numpy as np

from pathlib import Path

from collections import Counter

import csv



split_root = Path("/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed/train")



image_dir = split_root / "images"

target_dir = split_root / "targets"



out_dir = Path("preprocessing_audit_hurricane_delta_train")

out_dir.mkdir(exist_ok=True)



report_csv = out_dir / "preprocessing_audit_report.csv"



expected_labels = {0, 1, 2, 3, 4, 255}



label_names = {

    0: "background",

    1: "no_damage",

    2: "minor_damage",

    3: "major_damage",

    4: "destroyed",

    255: "ignore",

}



image_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]



def find_image(stem):

    for ext in image_exts:

        p = image_dir / f"{stem}{ext}"

        if p.exists():

            return p

    return None



def read_mask(path):

    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if m is None:

        return None

    if m.ndim == 3:

        m = m[..., 0]

    return m



def blur_score(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())



def brightness_score(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return float(gray.mean())



def estimate_shift(pre, post):

    if pre.shape[:2] != post.shape[:2]:

        post = cv2.resize(post, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_LINEAR)



    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY).astype(np.float32)

    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY).astype(np.float32)



    pre_gray = cv2.GaussianBlur(pre_gray, (5, 5), 0)

    post_gray = cv2.GaussianBlur(post_gray, (5, 5), 0)



    shift, response = cv2.phaseCorrelate(pre_gray, post_gray)

    return float(shift[0]), float(shift[1]), float(response)



post_targets = sorted(target_dir.glob("*_post_disaster_target.png"))



print("Split root:", split_root)

print("Number of post target masks:", len(post_targets))

print("Writing report to:", report_csv)



class_counter = Counter()

file_counter = Counter()



rows = []



missing_pairs = 0

unreadable = 0

shape_mismatch = 0

unexpected_label_files = 0

blurry_files = 0

brightness_problem_files = 0

large_shift_files = 0



for target_post_path in post_targets:

    sample_post_stem = target_post_path.name.replace("_target.png", "")

    sample_pre_stem = sample_post_stem.replace("_post_disaster", "_pre_disaster")



    pre_img_path = find_image(sample_pre_stem)

    post_img_path = find_image(sample_post_stem)

    target_pre_path = target_dir / f"{sample_pre_stem}_target.png"



    status = []



    if pre_img_path is None or post_img_path is None or not target_pre_path.exists():

        missing_pairs += 1

        status.append("missing_pair")

        rows.append({

            "sample": sample_post_stem,

            "status": ";".join(status),

            "pre_blur": "",

            "post_blur": "",

            "pre_brightness": "",

            "post_brightness": "",

            "shift_x": "",

            "shift_y": "",

            "shift_response": "",

            "labels": "",

            "class_pixels": "",

        })

        continue



    pre_img = cv2.imread(str(pre_img_path), cv2.IMREAD_COLOR)

    post_img = cv2.imread(str(post_img_path), cv2.IMREAD_COLOR)

    pre_mask = read_mask(target_pre_path)

    post_mask = read_mask(target_post_path)



    if pre_img is None or post_img is None or pre_mask is None or post_mask is None:

        unreadable += 1

        status.append("unreadable")

        continue



    shapes = {

        "pre_img": pre_img.shape[:2],

        "post_img": post_img.shape[:2],

        "pre_mask": pre_mask.shape[:2],

        "post_mask": post_mask.shape[:2],

    }



    if len(set(shapes.values())) > 1:

        shape_mismatch += 1

        status.append(f"shape_mismatch_{shapes}")



    labels, counts = np.unique(post_mask, return_counts=True)

    labels_set = set(int(v) for v in labels)



    if not labels_set.issubset(expected_labels):

        unexpected_label_files += 1

        status.append(f"unexpected_labels_{sorted(labels_set - expected_labels)}")



    class_pixel_dict = {}

    for v, c in zip(labels, counts):

        v = int(v)

        c = int(c)

        class_counter[v] += c

        file_counter[v] += 1

        class_pixel_dict[v] = c



    pre_blur = blur_score(pre_img)

    post_blur = blur_score(post_img)

    pre_brightness = brightness_score(pre_img)

    post_brightness = brightness_score(post_img)



    if pre_blur < 20 or post_blur < 20:

        blurry_files += 1

        status.append("very_blurry")



    if pre_brightness < 25 or post_brightness < 25 or pre_brightness > 230 or post_brightness > 230:

        brightness_problem_files += 1

        status.append("brightness_problem")



    shift_x, shift_y, shift_response = estimate_shift(pre_img, post_img)



    if abs(shift_x) > 8 or abs(shift_y) > 8:

        large_shift_files += 1

        status.append("possible_misalignment")



    if not status:

        status.append("ok")



    rows.append({

        "sample": sample_post_stem,

        "status": ";".join(status),

        "pre_blur": round(pre_blur, 3),

        "post_blur": round(post_blur, 3),

        "pre_brightness": round(pre_brightness, 3),

        "post_brightness": round(post_brightness, 3),

        "shift_x": round(shift_x, 3),

        "shift_y": round(shift_y, 3),

        "shift_response": round(shift_response, 5),

        "labels": sorted(labels_set),

        "class_pixels": class_pixel_dict,

    })



with open(report_csv, "w", newline="") as f:

    writer = csv.DictWriter(f, fieldnames=[

        "sample",

        "status",

        "pre_blur",

        "post_blur",

        "pre_brightness",

        "post_brightness",

        "shift_x",

        "shift_y",

        "shift_response",

        "labels",

        "class_pixels",

    ])

    writer.writeheader()

    writer.writerows(rows)



total_pixels = sum(class_counter.values())



print("\n================ DATASET AUDIT SUMMARY ================")

print("Total samples checked:", len(post_targets))

print("Missing pairs:", missing_pairs)

print("Unreadable files:", unreadable)

print("Shape mismatches:", shape_mismatch)

print("Files with unexpected labels:", unexpected_label_files)

print("Very blurry files:", blurry_files)

print("Brightness problem files:", brightness_problem_files)

print("Possible misalignment files:", large_shift_files)



print("\nClass distribution:")

print("-" * 80)

print(f"{'Label':>8} {'Name':<20} {'Pixels':>14} {'Percent':>10} {'Files':>8}")

print("-" * 80)



for label in sorted(class_counter):

    pixels = class_counter[label]

    pct = 100.0 * pixels / total_pixels if total_pixels > 0 else 0

    name = label_names.get(label, "unknown")

    files = file_counter[label]

    print(f"{label:>8} {name:<20} {pixels:>14} {pct:>9.4f}% {files:>8}")



print("-" * 80)

print("Total pixels:", total_pixels)



print("\nCSV report saved to:")

print(report_csv.resolve())



print("\nTo see suspicious samples:")

print(f"grep 'possible_misalignment\\|very_blurry\\|brightness_problem\\|shape_mismatch\\|unexpected_labels' {report_csv}")

