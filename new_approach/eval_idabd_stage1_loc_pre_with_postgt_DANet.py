#!/usr/bin/env python3
"""
================================================================================
for IDABD stage-1 localization, you can do the same trick as xView2:

Input: use PRE images (because they’re cleaner).

Ground truth: use the POST mask as the label source, but convert it to a binary building mask:

building = (post_mask > 0) and (post_mask != 255)

ignore pixels: post_mask == 255

That lets you report pixel-level Precision/Recall/F1 for “can the model localize buildings”, even though PRE masks are not labeled.

Below is a full script that:

runs inference on pre (or post)

evaluates against post masks (or pre masks if you want)

pairs pre↔post correctly by tile id

includes debug printing and optional threshold sweep
IDABD STAGE-1 LOCALIZATION EVAL
Run inference on PRE images but evaluate using POST masks as GT (building = mask>0).

Why?
- IDABD PRE masks contain only {0,255} => not labeled for buildings.
- IDABD POST masks contain {0,1,2,3,...} => real labels.
- For stage-1 localization, we only need building footprint:
    GT_building = (mask > 0) & (mask != 255)

Usage examples:
  # Pre images, Post masks (recommended for IDABD stage-1)
  python eval_idabd_stage1_loc_pre_with_postgt.py --input_split pre --gt_split post --thresh 0.1 --thresh_sweep

  # Post images, Post masks (standard)
  python eval_idabd_stage1_loc_pre_with_postgt.py --input_split post --gt_split post --thresh 0.5

  # Debug first 5 evaluated pairs, then every 20
  python eval_idabd_stage1_loc_pre_with_postgt.py --input_split pre --gt_split post --debug --debug_first 5 --debug_every 20

================================================================================
"""

import os
from os import path
import sys
from pathlib import Path
import glob
import argparse
import numpy as np
import cv2
import torch

# ---------------------------------------------------------------------
# Make sure "zoo" is importable even when this script is run from
# /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/new_approach
# while the xView2 1st-place weights/code may live elsewhere.
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
XVIEW2_NEW_APPROACH_DIR = Path("/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach")
DISASTER_NEW_APPROACH_DIR = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/new_approach")

ZOO_SEARCH_DIRS = [
    THIS_DIR,
    THIS_DIR.parent,
    THIS_DIR.parent.parent,
    DISASTER_NEW_APPROACH_DIR,
    DISASTER_NEW_APPROACH_DIR.parent,
    XVIEW2_NEW_APPROACH_DIR,
    XVIEW2_NEW_APPROACH_DIR.parent,
]

for cand in ZOO_SEARCH_DIRS:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        print(f"[PATH] Using zoo from: {cand / 'zoo'}")
        break
else:
    print("[WARN] Could not find a zoo/ directory in the expected locations.")
    print("[WARN] Current sys.path head:", sys.path[:5])

# Defaults
IMG_DIR = "/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/images"
MASK_DIR = "/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/masks"
WEIGHTS_DIR = "/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/weights"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -------------------------
# Filename helpers
# -------------------------
def split_from_name(fname: str):
    f = fname.lower()
    if "_pre_disaster" in f:
        return "pre"
    if "_post_disaster" in f:
        return "post"
    return "unknown"

def tile_id_from_name(fname: str):
    base = path.splitext(path.basename(fname))[0]
    base = base.replace("_pre_disaster", "").replace("_post_disaster", "")
    return base

def list_images(img_dir: str, which_split: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(path.join(img_dir, e)))
    files = sorted(files)

    if which_split == "pre":
        files = [f for f in files if "_pre_disaster" in path.basename(f)]
    elif which_split == "post":
        files = [f for f in files if "_post_disaster" in path.basename(f)]
    return files

def find_mask(mask_dir: str, tile_id: str, gt_split: str):
    # we expect: <tile_id>_<gt_split>_disaster.png
    # ex: AOI1-tile_1-3_post_disaster.png
    patterns = [
        f"{tile_id}_{gt_split}_disaster.png",
        f"{tile_id}_{gt_split}_disaster.jpg",
        f"{tile_id}_{gt_split}_disaster.jpeg",
        f"{tile_id}_{gt_split}_disaster.tif",
        f"{tile_id}_{gt_split}_disaster.tiff",
        f"{tile_id}_{gt_split}_disaster.bmp",
    ]
    for ptn in patterns:
        cand = path.join(mask_dir, ptn)
        if path.exists(cand):
            return cand
    return None


# -------------------------
# Image preprocessing
# -------------------------
def pad_to_factor(img, factor=32):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / factor) * factor)
    new_w = int(np.ceil(w / factor) * factor)
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    return padded, (pad_h, pad_w)

def unpad(arr2d, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return arr2d
    h, w = arr2d.shape[:2]
    return arr2d[: h - pad_h, : w - pad_w]

def preprocess_rgb(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).unsqueeze(0)  # 1x3xHxW


# -------------------------
# Mask loading
# -------------------------
def load_mask_binary(mask_path: str):
    """
    Building GT = (mask > 0) & (mask != 255)
    Ignore = (mask == 255)
    """
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = m.astype(np.int32)
    valid = (m != 255)
    gt = (m > 0) & valid
    return gt, valid, m


# -------------------------
# Checkpoint loading
# -------------------------
def strip_prefixes(sd: dict):
    prefixes = ["module.", "model.", "net."]
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def load_state_dict_any(weight_path: str):
    # PyTorch 2.6+: must use weights_only=False for older checkpoints (only if you trust them)
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint did not contain a state_dict mapping: {weight_path}")

    return strip_prefixes(sd)

def build_model_for_checkpoint(weight_path: str, device: torch.device):
    import zoo.models as zm
    fname = path.basename(weight_path).lower()

    # Use no-download constructors only. The checkpoint contains the trained weights,
    # so we should not ask the HPC to download ImageNet/pretrained weights.
    if fname.startswith("dpn92"):
        model = zm.Dpn92_Unet_Loc(pretrained=None)
    elif fname.startswith("res34"):
        model = zm.Res34_Unet_Loc(pretrained=False)
    elif fname.startswith("res50"):
        model = zm.SeResNext50_Unet_Loc(pretrained=None)  # this repo’s “res50” loc is SeResNeXt50
    elif fname.startswith("se154"):
        model = zm.SeNet154_Unet_Loc(pretrained=None)
    else:
        raise ValueError(f"Unrecognized weight prefix: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def predict_prob(model, x):
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if y.ndim == 4 and y.shape[1] > 1:
        y = y[:, 0:1, :, :]
    y = torch.sigmoid(y)
    return y.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

def accumulate_stats(pred_bin, gt_bin, valid):
    pred = pred_bin & valid
    gt = gt_bin & valid
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return int(tp), int(fp), int(fn)

def prf(tp, fp, fn):
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return prec, rec, f1

def describe_prob(prob):
    p = prob.astype(np.float32)
    return {
        "min": float(np.min(p)),
        "mean": float(np.mean(p)),
        "max": float(np.max(p)),
        "p50": float(np.percentile(p, 50)),
        "p90": float(np.percentile(p, 90)),
        "p95": float(np.percentile(p, 95)),
        "p99": float(np.percentile(p, 99)),
    }

def sweep_thresholds(prob_list, gt_list, valid_list, thresholds):
    best = (-1.0, None, None)  # f1, t, (p,r)
    for t in thresholds:
        tp = fp = fn = 0
        for prob, gt, valid in zip(prob_list, gt_list, valid_list):
            pred = (prob >= t)
            a, b, c = accumulate_stats(pred, gt, valid)
            tp += a; fp += b; fn += c
        p, r, f1 = prf(tp, fp, fn)
        if f1 > best[0]:
            best = (f1, t, (p, r))
    return best


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--mask_dir", default=MASK_DIR)
    ap.add_argument("--weights_dir", default=WEIGHTS_DIR)
    ap.add_argument("--thresh", type=float, default=0.5)

    # Key options for your request:
    ap.add_argument("--input_split", choices=["pre", "post"], default="pre",
                    help="Which images to run inference on.")
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post",
                    help="Which masks to use as ground truth labels (recommended: post).")

    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--out_dir", default="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/new_approach/pred_idabd_loc_stage1")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_first", type=int, default=5)
    ap.add_argument("--debug_every", type=int, default=20)

    ap.add_argument("--thresh_sweep", action="store_true")

    args = ap.parse_args()

    print(f"[PATHS] img_dir={args.img_dir}")
    print(f"[PATHS] mask_dir={args.mask_dir}")
    print(f"[PATHS] weights_dir={args.weights_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[MODE] inference on: {args.input_split} images | GT from: {args.gt_split} masks")
    print(f"[PATH] IMG_DIR={args.img_dir}")
    print(f"[PATH] MASK_DIR={args.mask_dir}")
    print(f"[PATH] WEIGHTS_DIR={args.weights_dir}")

    # weights
    patterns = ["dpn92_loc*.pth", "res34_loc*.pth", "res50_loc*.pth", "se154_loc*.pth"]
    weight_files = []
    for ptn in patterns:
        weight_files += glob.glob(path.join(args.weights_dir, ptn))
    weight_files = sorted(weight_files)
    if not weight_files:
        raise FileNotFoundError(f"No localization weights found in {args.weights_dir}")

    print("[WEIGHTS] Localization checkpoints:")
    for w in weight_files:
        print("  -", w)

    models = []
    for w in weight_files:
        print(f"[LOAD] {w}")
        models.append(build_model_for_checkpoint(w, device))
    print(f"[OK] Loaded {len(models)} localization models.")

    if args.save_preds:
        os.makedirs(args.out_dir, exist_ok=True)

    # image list
    img_files = list_images(args.img_dir, args.input_split)
    if not img_files:
        raise FileNotFoundError(f"No {args.input_split} images found in {args.img_dir}")

    # metrics accumulators
    total_tp = total_fp = total_fn = 0
    per_image_f1 = []

    n_eval = 0
    missing_gt = 0
    empty_gt = 0
    empty_pred = 0
    empty_both = 0

    sweep_probs, sweep_gts, sweep_valids = [], [], []

    for idx, img_path in enumerate(img_files, 1):
        fname = path.basename(img_path)
        tile_id = tile_id_from_name(fname)

        gt_mask_path = find_mask(args.mask_dir, tile_id, args.gt_split)
        if gt_mask_path is None:
            missing_gt += 1
            if args.debug and idx <= args.debug_first:
                print(f"[DBG] missing GT mask for tile={tile_id} (expected {args.gt_split})")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            if args.debug:
                print(f"[DBG] failed to read image {img_path}")
            continue

        img_pad, pad_hw = pad_to_factor(img, factor=32)
        x = preprocess_rgb(img_pad).to(device)

        probs = [predict_prob(m, x) for m in models]
        prob = np.mean(probs, axis=0)
        prob = unpad(prob, pad_hw)

        gt_bin, valid, raw_mask = load_mask_binary(gt_mask_path)

        # resize prob to mask if needed
        if prob.shape != gt_bin.shape:
            prob = cv2.resize(prob, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_LINEAR)

        pred_bin = (prob >= args.thresh)

        tp, fp, fn = accumulate_stats(pred_bin, gt_bin, valid)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        p_i, r_i, f1_i = prf(tp, fp, fn)
        per_image_f1.append(f1_i)
        n_eval += 1

        gt_ratio = float((gt_bin & valid).mean())
        pred_ratio = float((pred_bin & valid).mean())
        if gt_ratio == 0.0:
            empty_gt += 1
        if pred_ratio == 0.0:
            empty_pred += 1
        if gt_ratio == 0.0 and pred_ratio == 0.0:
            empty_both += 1

        if args.thresh_sweep:
            sweep_probs.append(prob.copy())
            sweep_gts.append(gt_bin.copy())
            sweep_valids.append(valid.copy())

        # debug prints
        do_dbg = args.debug and (n_eval <= args.debug_first or (args.debug_every > 0 and n_eval % args.debug_every == 0))
        if do_dbg:
            st = describe_prob(prob)
            uniq = np.unique(raw_mask)[:20]
            print("\n---------------- DEBUG ----------------")
            print(f"[DBG] #{n_eval} input: {fname}")
            print(f"[DBG] GT mask: {path.basename(gt_mask_path)}  (gt_split={args.gt_split})")
            print(f"[DBG] shapes img={img.shape} prob={prob.shape} mask={raw_mask.shape}")
            print(f"[DBG] raw mask unique(first20): {uniq}")
            print(f"[DBG] gt_ratio(building): {gt_ratio:.6f} | pred_ratio@t: {pred_ratio:.6f} (t={args.thresh})")
            print(f"[DBG] prob: min={st['min']:.4f} mean={st['mean']:.4f} max={st['max']:.4f} "
                  f"p90={st['p90']:.4f} p95={st['p95']:.4f} p99={st['p99']:.4f}")
            print(f"[DBG] TP={tp} FP={fp} FN={fn} | P={p_i:.4f} R={r_i:.4f} F1={f1_i:.4f}")
            print("--------------------------------------\n")

        if n_eval % 10 == 0 or idx == len(img_files):
            print(f"[{idx}/{len(img_files)}] last F1={f1_i:.4f}")

        if args.save_preds:
            stem = path.splitext(path.basename(img_path))[0]
            prob_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
            bin_u8 = (pred_bin.astype(np.uint8) * 255)
            cv2.imwrite(path.join(args.out_dir, f"{stem}_prob.png"), prob_u8)
            cv2.imwrite(path.join(args.out_dir, f"{stem}_bin.png"), bin_u8)

    # final report
    prec, rec, f1 = prf(total_tp, total_fp, total_fn)
    mean_f1 = float(np.mean(per_image_f1)) if per_image_f1 else 0.0

    print("\n==================== RESULTS (IDABD STAGE-1 LOC) ====================")
    print(f"Inference images: {args.input_split}")
    print(f"GT masks:         {args.gt_split}")
    print(f"Threshold:        {args.thresh}")
    print(f"Images evaluated: {n_eval} (missing GT masks skipped: {missing_gt})")
    print(f"Global Precision: {prec:.6f}")
    print(f"Global Recall:    {rec:.6f}")
    print(f"Global F1:        {f1:.6f}")
    print(f"Mean per-image F1:{mean_f1:.6f}")
    print("---------------------------------------------------------------------")
    print(f"Empty GT masks (no building pixels): {empty_gt}/{n_eval}")
    print(f"Empty predictions at threshold:      {empty_pred}/{n_eval}")
    print(f"Empty both:                          {empty_both}/{n_eval}")
    print("=====================================================================\n")

    if args.thresh_sweep and n_eval > 0:
        thresholds = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        best_f1, best_t, (best_p, best_r) = sweep_thresholds(sweep_probs, sweep_gts, sweep_valids, thresholds)
        print("========== THRESHOLD SWEEP ==========")
        print(f"Best F1:      {best_f1:.6f} at threshold {best_t}")
        print(f"Precision:    {best_p:.6f}")
        print(f"Recall:       {best_r:.6f}")
        print(f"Thresholds:   {thresholds}")
        print("=====================================\n")


if __name__ == "__main__":
    main()