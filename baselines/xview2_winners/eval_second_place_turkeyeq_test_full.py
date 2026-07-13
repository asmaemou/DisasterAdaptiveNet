import os
import sys
import csv
import json
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm
from albumentations.pytorch.transforms import img_to_tensor

BASE = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners")
REPO = BASE / "xview2_second_place"
MANIFEST = BASE / "second_place_full_solution_manifest.tsv"

DATASET = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_earthquake_turkey_TEST_ONLY")
TEST_IMAGES = DATASET / "images"
TEST_MASKS = DATASET / "masks"

OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_earthquake_turkey_TEST_ONLY_ZERO_SHOT_full_solution")
PRED_DIR = OUT / "predictions"
LOC_DIR = PRED_DIR / "localization"
DMG_DIR = PRED_DIR / "damage"
PROB_DIR = OUT / "probabilities"

for d in [OUT, PRED_DIR, LOC_DIR, DMG_DIR, PROB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
os.chdir(REPO)

import models
from tools.config import load_config


def read_manifest(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def load_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1]


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def strip_module_state(state):
    return {k.replace("module.", ""): v for k, v in state.items()}

def get_normalize_for_channels(conf, channels):
    """
    Some second-place configs store 6-channel normalization for pre+post images.
    Localization uses only the 3-channel pre image, so we slice mean/std to 3.
    Damage uses 6 channels, so it keeps all 6 values.
    """
    norm = conf["input"].get("normalize", None)
    if norm is None:
        return None

    norm = dict(norm)

    for key in ["mean", "std"]:
        if key in norm and norm[key] is not None:
            vals = list(norm[key])
            if len(vals) > channels:
                vals = vals[:channels]
            elif len(vals) < channels:
                vals = vals * (channels // len(vals))
            norm[key] = vals

    return norm


def build_model(config_path, weight_path, task):
    conf = load_config(config_path)

    if task == "loc":
        model = models.__dict__[conf["network"]](seg_classes=1, backbone_arch=conf["encoder"])
    else:
        model = models.__dict__[conf["network"]](seg_classes=5, backbone_arch=conf["encoder"])

    print(f"=> loading checkpoint: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = strip_module_state(state)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("WARNING: strict load failed, retrying strict=False")
        print(str(e)[:1000])
        model.load_state_dict(state, strict=False)

    model.eval()
    model.cuda()
    return model, conf


def tta_predict_loc(model, conf, image_rgb, weight_name):
    x = img_to_tensor(image_rgb, get_normalize_for_channels(conf, 3)).cpu().numpy()

    # Original 2nd-place code pads DPN localization inputs by 16 pixels.
    use_dpn_pad = "dpn" in weight_name.lower()
    if use_dpn_pad:
        x = np.pad(x, [(0, 0), (16, 16), (16, 16)], mode="reflect")

    variants = [
        ("none", x),
        ("vflip", x[:, ::-1, :]),
        ("hflip", x[:, :, ::-1]),
        ("vhflip", x[:, ::-1, ::-1]),
    ]

    preds = []

    with torch.no_grad():
        for mode, arr in variants:
            inp = torch.from_numpy(arr.copy()[None]).cuda().float()
            logits = model(inp)
            pred = torch.sigmoid(logits)[0].detach().cpu().numpy()

            if mode == "vflip":
                pred = pred[:, ::-1, :]
            elif mode == "hflip":
                pred = pred[:, :, ::-1]
            elif mode == "vhflip":
                pred = pred[:, ::-1, ::-1]

            preds.append(pred)

    pred = np.mean(preds, axis=0)

    if use_dpn_pad:
        pred = pred[:, 16:-16, 16:-16]

    return pred[0].astype(np.float32)


def tta_predict_damage(model, conf, pre_rgb, post_rgb):
    image = np.concatenate([pre_rgb, post_rgb], axis=-1)
    x = img_to_tensor(image, get_normalize_for_channels(conf, 6)).cpu().numpy()

    variants = [
        ("none", x),
        ("vflip", x[:, ::-1, :]),
        ("hflip", x[:, :, ::-1]),
        ("vhflip", x[:, ::-1, ::-1]),
    ]

    preds = []

    with torch.no_grad():
        for mode, arr in variants:
            inp = torch.from_numpy(arr.copy()[None]).cuda().float()
            logits = model(inp)
            pred = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            if mode == "vflip":
                pred = pred[:, ::-1, :]
            elif mode == "hflip":
                pred = pred[:, :, ::-1]
            elif mode == "vhflip":
                pred = pred[:, ::-1, ::-1]

            preds.append(pred)

    return np.mean(preds, axis=0).astype(np.float32)


def postprocess_prediction(loc_prob, damage_prob):
    """
    Dynamic 512-safe version of the second-place post-processing idea.

    loc_prob: H x W
    damage_prob: 5 x H x W, channels [background, no-damage, minor, major, destroyed]
    """
    h, w = loc_prob.shape

    damage_1to4 = np.moveaxis(damage_prob[1:5], 0, -1)  # H,W,4

    background = 1.0 - np.sum(damage_1to4, axis=2, keepdims=True)
    background = np.clip(background, 0, 1) * 0.8

    damage_pred_full = np.concatenate([background, damage_1to4], axis=2)

    # Boost damage classes like original 2nd-place post-processing.
    # Channel index: 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed.
    damage_pred_full[:, :, 2] *= 2.0
    damage_pred_full[:, :, 3] *= 2.0
    damage_pred_full[:, :, 4] *= 2.0

    argmax_full = np.argmax(damage_pred_full, axis=2)

    loc_pred = ((loc_prob > 0.25) | (argmax_full > 0)).astype(np.uint8)

    damage_cls = np.argmax(damage_1to4, axis=2).astype(np.uint8) + 1
    damage_cls[loc_pred == 0] = 0

    return loc_pred, damage_cls


def binary_f1(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    denom = (2 * tp + fp + fn)
    if denom == 0:
        return np.nan
    return (2 * tp) / denom


def main():
    torch.backends.cudnn.benchmark = True

    rows = read_manifest(MANIFEST)
    loc_rows = [r for r in rows if r["task"] == "loc"]
    dmg_rows = [r for r in rows if r["task"] == "damage"]

    print("================================================")
    print("ZERO-SHOT xView2 2nd-place full available ensemble on Earthquake Turkey TEST")
    print("No Earthquake Turkey TEST fine-tuning is performed.")
    print("================================================")
    print("Localization models:", len(loc_rows))
    print("Damage models:", len(dmg_rows))
    print("Output:", OUT)

    pre_images = sorted(TEST_IMAGES.glob("*_pre_disaster.png"))
    samples = []

    for pre in pre_images:
        sid = pre.name.replace("_pre_disaster.png", "")
        post = TEST_IMAGES / f"{sid}_post_disaster.png"
        pre_mask = TEST_MASKS / f"{sid}_pre_disaster.png"
        post_mask = TEST_MASKS / f"{sid}_post_disaster.png"

        if post.exists() and pre_mask.exists() and post_mask.exists():
            samples.append((sid, pre, post, pre_mask, post_mask))
        else:
            print("WARNING missing pair/mask:", sid)

    print("Test samples:", len(samples))
    if len(samples) == 0:
        raise RuntimeError("No test samples found.")

    shapes = {}
    loc_acc = {}
    dmg_acc = {}

    for sid, pre, post, pre_mask, post_mask in samples:
        img = load_image_rgb(pre)
        h, w = img.shape[:2]
        shapes[sid] = (h, w)
        loc_acc[sid] = np.zeros((h, w), dtype=np.float32)
        dmg_acc[sid] = np.zeros((5, h, w), dtype=np.float32)

    print("================================================")
    print("Run localization ensemble")
    print("================================================")

    for r in loc_rows:
        tag = r["tag"]
        config = r["config"]
        weight = r["weight"]

        print("Localization model:", tag)
        model, conf = build_model(config, weight, "loc")

        for sid, pre, post, pre_mask, post_mask in tqdm(samples, desc=f"loc {tag}"):
            pre_rgb = load_image_rgb(pre)
            pred = tta_predict_loc(model, conf, pre_rgb, Path(weight).name)
            loc_acc[sid] += pred

        del model
        torch.cuda.empty_cache()

    for sid in loc_acc:
        loc_acc[sid] /= max(len(loc_rows), 1)

    print("================================================")
    print("Run damage ensemble")
    print("================================================")

    total_damage_weight = 0

    for r in dmg_rows:
        tag = r["tag"]
        config = r["config"]
        weight = r["weight"]

        ensemble_weight = 2 if "resnext101" in Path(weight).name.lower() else 1
        total_damage_weight += ensemble_weight

        print("Damage model:", tag, "ensemble_weight:", ensemble_weight)
        model, conf = build_model(config, weight, "damage")

        for sid, pre, post, pre_mask, post_mask in tqdm(samples, desc=f"damage {tag}"):
            pre_rgb = load_image_rgb(pre)
            post_rgb = load_image_rgb(post)
            pred = tta_predict_damage(model, conf, pre_rgb, post_rgb)
            dmg_acc[sid] += ensemble_weight * pred

        del model
        torch.cuda.empty_cache()

    for sid in dmg_acc:
        dmg_acc[sid] /= max(total_damage_weight, 1)

    print("================================================")
    print("Save predictions and compute F1 scores")
    print("================================================")

    all_loc_gt = []
    all_loc_pr = []

    all_dmg_gt = []
    all_dmg_pr = []

    per_image_rows = []

    for sid, pre, post, pre_mask_path, post_mask_path in tqdm(samples, desc="metrics"):
        loc_prob = loc_acc[sid]
        dmg_prob = dmg_acc[sid]

        loc_pred, dmg_pred = postprocess_prediction(loc_prob, dmg_prob)

        loc_gt = (load_mask(pre_mask_path) > 0).astype(np.uint8)
        dmg_gt = load_mask(post_mask_path).astype(np.uint8)

        if loc_gt.shape != loc_pred.shape:
            raise RuntimeError(f"Shape mismatch for {sid}: gt {loc_gt.shape}, pred {loc_pred.shape}")

        cv2.imwrite(str(LOC_DIR / f"{sid}_localization_prediction.png"), loc_pred.astype(np.uint8) * 255)
        cv2.imwrite(str(DMG_DIR / f"{sid}_damage_prediction.png"), dmg_pred.astype(np.uint8))

        np.save(PROB_DIR / f"{sid}_loc_prob.npy", loc_prob.astype(np.float16))
        np.save(PROB_DIR / f"{sid}_damage_prob.npy", dmg_prob.astype(np.float16))

        all_loc_gt.append(loc_gt.reshape(-1))
        all_loc_pr.append(loc_pred.reshape(-1))

        all_dmg_gt.append(dmg_gt.reshape(-1))
        all_dmg_pr.append(dmg_pred.reshape(-1))

        row = {"id": sid}
        row["loc_f1"] = binary_f1(loc_gt > 0, loc_pred > 0)

        for cls_id, cls_name in [
            (1, "no_damage"),
            (2, "minor_damage"),
            (3, "major_damage"),
            (4, "destroyed"),
        ]:
            row[f"{cls_name}_f1"] = binary_f1(dmg_gt == cls_id, dmg_pred == cls_id)

        per_image_rows.append(row)

    loc_gt = np.concatenate(all_loc_gt)
    loc_pr = np.concatenate(all_loc_pr)
    dmg_gt = np.concatenate(all_dmg_gt)
    dmg_pr = np.concatenate(all_dmg_pr)

    summary = {}
    summary["Localization_F1"] = binary_f1(loc_gt > 0, loc_pr > 0)

    class_scores = []
    for cls_id, cls_name in [
        (1, "No_damage_F1"),
        (2, "Minor_damage_F1"),
        (3, "Major_damage_F1"),
        (4, "Destroyed_F1"),
    ]:
        f1 = binary_f1(dmg_gt == cls_id, dmg_pr == cls_id)
        summary[cls_name] = f1
        class_scores.append(f1)

    summary["Macro_F1_damage_classes"] = float(np.nanmean(class_scores))

    with open(OUT / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "metrics_summary.txt", "w") as f:
        f.write("2nd-place xView2 full available ensemble ZERO-SHOT on Earthquake Turkey TEST\n")
        f.write("No Earthquake Turkey TEST fine-tuning used.\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v:.6f}\n")

    with open(OUT / "per_image_metrics.csv", "w", newline="") as f:
        fieldnames = [
            "id",
            "loc_f1",
            "no_damage_f1",
            "minor_damage_f1",
            "major_damage_f1",
            "destroyed_f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow(row)

    print("================================================")
    print("FINAL ZERO-SHOT RESULTS")
    print("================================================")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    print("Saved:")
    print(OUT / "metrics_summary.txt")
    print(OUT / "metrics_summary.json")
    print(OUT / "per_image_metrics.csv")
    print(LOC_DIR)
    print(DMG_DIR)


if __name__ == "__main__":
    main()
