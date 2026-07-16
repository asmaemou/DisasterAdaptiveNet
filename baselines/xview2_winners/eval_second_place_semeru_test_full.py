import argparse
import csv
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm
from albumentations.pytorch.transforms import img_to_tensor
from skimage import measure

try:
    from skimage.segmentation import watershed
except Exception:
    from skimage.morphology import watershed


BASE = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/baselines/xview2_winners")
REPO = BASE / "xview2_second_place"
MANIFEST = BASE / "second_place_full_solution_manifest.tsv"

DATASET = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baseline_datasets/second_place_mount_semeru_TEST_ONLY")
TEST_IMAGES = DATASET / "images"
TEST_MASKS = DATASET / "masks"
PRED_SIZE = 512

ZERO_SHOT_OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_mount_semeru_TEST_ONLY_ZERO_SHOT_full_solution")
FT_EXP = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_mount_semeru_FULL_SOLUTION_finetune_official_split")
FINETUNED_OUT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/xview2_baselines/second_place_mount_semeru_FINE_TUNED_OFFICIAL_SPLIT_full_solution")

OUT = ZERO_SHOT_OUT
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


ModelConfig = namedtuple("ModelConfig", "tag task config_path weight_path ensemble_weight")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FALLBACK_MODELS = [
    ModelConfig("loc_d161_0", "localization", "configs/d161_loc.json", "weights/localization_densenet_unet_densenet161_3_0_best_dice", 1),
    ModelConfig("loc_d161_1", "localization", "configs/d161_loc.json", "weights/localization_densenet_unet_densenet161_3_1_best_dice", 1),
    ModelConfig("loc_d92_0", "localization", "configs/d92_loc.json", "weights/localization_dpn_unet_dpn92_0_best_dice", 1),
    ModelConfig("loc_d92_1", "localization", "configs/d92_loc.json", "weights/localization_dpn_unet_dpn92_1_best_dice", 1),
    ModelConfig("loc_d92_2", "localization", "configs/d92_loc.json", "weights/localization_dpn_unet_dpn92_2_best_dice", 1),
    ModelConfig("loc_d92_3", "localization", "configs/d92_loc.json", "weights/localization_dpn_unet_dpn92_3_best_dice", 1),

    ModelConfig("damage_d161_0", "damage", "configs/d161_softmax.json", "weights/pseudo_densenet_seamese_unet_shared_densenet161_0_best_xview", 1),
    ModelConfig("damage_d161_1", "damage", "configs/d161_softmax.json", "weights/pseudo_densenet_seamese_unet_shared_densenet161_2_best_xview", 1),
    ModelConfig("damage_d92_0", "damage", "configs/d92_softmax.json", "weights/pseudo_dpn_seamese_unet_shared_dpn92_0_best_xview", 1),
    ModelConfig("damage_d92_1", "damage", "configs/d92_softmax.json", "weights/pseudo_dpn_seamese_unet_shared_dpn92_2_best_xview", 1),
    ModelConfig("damage_se50_0", "damage", "configs/se50_softmax.json", "weights/pseudo_scseresnext_seamese_unet_shared_seresnext50_0_best_xview", 1),
    ModelConfig("damage_se50_1", "damage", "configs/se50_softmax.json", "weights/pseudo_scseresnext_seamese_unet_shared_seresnext50_1_best_xview", 1),
    ModelConfig("damage_se50_2", "damage", "configs/se50_softmax.json", "weights/pseudo_scseresnext_seamese_unet_shared_seresnext50_2_best_xview", 1),
    ModelConfig("damage_se50_3", "damage", "configs/se50_softmax.json", "weights/pseudo_scseresnext_seamese_unet_shared_seresnext50_3_best_xview", 1),
    ModelConfig("damage_r101_0", "damage", "configs/r101_softmax_sgd.json", "weights/sgd_resnext_seamese_unet_shared_resnext101_0_best_xview", 2),
    ModelConfig("damage_d161_2", "damage", "configs/d161_softmax.json", "weights/softmax_densenet_seamese_unet_shared_densenet161_0_best_xview", 1),
    ModelConfig("damage_d161_3", "damage", "configs/d161_softmax.json", "weights/softmax_densenet_seamese_unet_shared_densenet161_2_best_xview", 1),
    ModelConfig("damage_d92_2", "damage", "configs/d92_softmax.json", "weights/softmax_dpn_seamese_unet_shared_dpn92_0_best_xview", 1),
    ModelConfig("damage_d92_3", "damage", "configs/d92_softmax.json", "weights/softmax_dpn_seamese_unet_shared_dpn92_2_best_xview", 1),
    ModelConfig("damage_b2_0", "damage", "configs/b2_softmax.json", "weights/softmax_sampling_efficient_seamese_unet_shared_efficientnet-b2_0_best_xview", 1),
    ModelConfig("damage_b2_1", "damage", "configs/b2_softmax.json", "weights/softmax_sampling_efficient_seamese_unet_shared_efficientnet-b2_1_best_xview", 1),
]


def resolve_path(p):
    p = Path(str(p))
    if p.is_absolute():
        return p
    if (REPO / p).exists():
        return REPO / p
    if (BASE / p).exists():
        return BASE / p
    return REPO / p


def read_manifest():
    if not MANIFEST.exists():
        print(f"WARNING: manifest not found: {MANIFEST}")
        print("Using fallback full available model list.")
        return FALLBACK_MODELS

    try:
        rows = []
        with open(MANIFEST, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                lower = {k.lower().strip(): v for k, v in r.items() if k is not None}

                tag = (
                    lower.get("tag")
                    or lower.get("name")
                    or lower.get("model")
                    or lower.get("model_name")
                    or ""
                ).strip()

                task = (
                    lower.get("task")
                    or lower.get("type")
                    or lower.get("kind")
                    or ""
                ).strip().lower()

                config_path = (
                    lower.get("config_path")
                    or lower.get("config")
                    or lower.get("config_file")
                    or ""
                ).strip()

                weight_path = (
                    lower.get("weight_path")
                    or lower.get("checkpoint")
                    or lower.get("checkpoint_path")
                    or lower.get("weights")
                    or lower.get("model_path")
                    or lower.get("weight")
                    or ""
                ).strip()

                ensemble_weight = (
                    lower.get("ensemble_weight")
                    or "1"
                )

                if task == "loc":
                    task = "localization"
                elif not task:
                    if tag.startswith("loc"):
                        task = "localization"
                    elif tag.startswith("damage"):
                        task = "damage"

                if not tag or not task or not config_path or not weight_path:
                    continue

                if not weight_path.startswith("weights/") and not Path(weight_path).is_absolute():
                    weight_path = "weights/" + weight_path

                rows.append(
                    ModelConfig(
                        tag=tag,
                        task=task,
                        config_path=config_path,
                        weight_path=weight_path,
                        ensemble_weight=float(ensemble_weight),
                    )
                )

        loc = [m for m in rows if m.task == "localization"]
        dmg = [m for m in rows if m.task == "damage"]

        if len(loc) == 6 and len(dmg) == 15:
            return rows

        print("WARNING: manifest parse did not find 6 localization and 15 damage models.")
        print(f"Parsed localization={len(loc)}, damage={len(dmg)}")
        print("Using fallback full available model list.")
        return FALLBACK_MODELS

    except Exception as e:
        print("WARNING: could not parse manifest:", e)
        print("Using fallback full available model list.")
        return FALLBACK_MODELS


def torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_finetuned_checkpoint(model_config):
    if model_config.task == "localization":
        folder = FT_EXP / "weights_localization" / model_config.tag
        patterns = ["*best_dice*", "*last*"]
    else:
        folder = FT_EXP / "weights_damage" / model_config.tag
        patterns = ["*best_xview*", "*best_dice*", "*last*"]

    if not folder.exists():
        raise FileNotFoundError(f"Missing fine-tuned checkpoint folder: {folder}")

    for pattern in patterns:
        candidates = sorted(p for p in folder.glob(pattern) if p.is_file())
        if candidates:
            return str(candidates[0])

    raise FileNotFoundError(f"No fine-tuned checkpoint found inside: {folder}")


def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_model(model_config, seg_classes):
    config_path = resolve_path(model_config.config_path)
    weight_path = resolve_path(model_config.weight_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    if not weight_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weight_path}")

    conf = load_config(str(config_path))

    model = models.__dict__[conf["network"]](
        seg_classes=seg_classes,
        backbone_arch=conf["encoder"],
    )

    print(f"=> loading checkpoint: {weight_path}")
    checkpoint = torch_load_checkpoint(weight_path)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    try:
        model.load_state_dict(clean_state_dict(state_dict), strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(DEVICE)
    return model, conf


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img[:, :, ::-1]


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def to_tensor_numpy(image, conf):
    normalize = conf.get("input", {}).get("normalize", None)

    # Some xView2 configs have 6-channel normalization values.
    # Localization uses 3-channel RGB, damage uses 6-channel pre+post.
    if isinstance(normalize, dict) and image.ndim == 3:
        c = image.shape[-1]
        normalize = dict(normalize)

        if "mean" in normalize and hasattr(normalize["mean"], "__len__"):
            if len(normalize["mean"]) != c:
                normalize["mean"] = normalize["mean"][:c]

        if "std" in normalize and hasattr(normalize["std"], "__len__"):
            if len(normalize["std"]) != c:
                normalize["std"] = normalize["std"][:c]

    tensor = img_to_tensor(image, normalize)
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu().numpy()
    return tensor

def predict_localization_one(model, conf, image_pre_rgb, image_post_rgb, model_config):
    # Localization model expects 3-channel pre-disaster RGB.
    image_pre_rgb = cv2.resize(
        image_pre_rgb,
        (PRED_SIZE, PRED_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    image = to_tensor_numpy(image_pre_rgb, conf)

    pad = 0
    if "dpn" in str(model_config.weight_path).lower():
        pad = 16
        image = np.pad(image, [(0, 0), (pad, pad), (pad, pad)], mode="reflect")

    images = np.array([
        image,
        image[:, ::-1, :],
        image[:, :, ::-1],
        image[:, ::-1, ::-1],
    ])

    images = np.ascontiguousarray(images)
    images = torch.from_numpy(images).to(DEVICE).float()

    with torch.no_grad():
        logits = model(images)
        preds = torch.sigmoid(logits).detach().cpu().numpy()

    prediction_masks = []
    for i in range(4):
        pred = preds[i]
        if i == 1:
            pred = pred.copy()[:, ::-1, :]
        elif i == 2:
            pred = pred.copy()[:, :, ::-1]
        elif i == 3:
            pred = pred.copy()[:, ::-1, ::-1]
        prediction_masks.append(pred)

    pred = np.average(prediction_masks, axis=0)

    if pad > 0:
        pred = pred[:, pad:-pad, pad:-pad]

    return pred[0].astype(np.float32)

def predict_damage_one(model, conf, image_pre_rgb, image_post_rgb):
    image_pre_rgb = cv2.resize(
        image_pre_rgb,
        (PRED_SIZE, PRED_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    image_post_rgb = cv2.resize(
        image_post_rgb,
        (PRED_SIZE, PRED_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    image = np.concatenate([image_pre_rgb, image_post_rgb], axis=-1)
    image = to_tensor_numpy(image, conf)

    images = np.array([
        image,
        image[:, ::-1, :],
        image[:, :, ::-1],
        image[:, ::-1, ::-1],
    ])

    images = np.ascontiguousarray(images)
    images = torch.from_numpy(images).to(DEVICE).float()

    with torch.no_grad():
        logits = model(images)
        preds = torch.softmax(logits, dim=1).detach().cpu().numpy()

    prediction_masks = []
    for i in range(4):
        pred = preds[i]
        if i == 1:
            pred = pred.copy()[:, ::-1, :]
        elif i == 2:
            pred = pred.copy()[:, :, ::-1]
        elif i == 3:
            pred = pred.copy()[:, ::-1, ::-1]
        prediction_masks.append(pred)

    pred = np.average(prediction_masks, axis=0)
    return pred.astype(np.float32)

def safe_label(binary):
    try:
        return measure.label(binary, connectivity=2, background=0)
    except TypeError:
        return measure.label(binary, neighbors=8, background=0)


def label_mask(loc, labels, intensity, mask, seed_threshold=0.8):
    av_pred = (loc > seed_threshold).astype(np.uint8)
    y_pred = safe_label(av_pred)

    nucl_msk = (1 - loc).astype(np.uint8)

    try:
        y_pred = watershed(nucl_msk, y_pred, mask=mask, watershed_line=False)
    except TypeError:
        y_pred = watershed(nucl_msk, y_pred, mask=mask)

    props = measure.regionprops(y_pred)
    max_label = int(np.max(y_pred))

    for i in range(1, max_label):
        reg_labels = labels[y_pred == i]
        if reg_labels.size == 0:
            continue

        unique, counts = np.unique(reg_labels, return_counts=True)
        max_idx = int(np.argmax(counts))
        out_label = unique[max_idx]

        if out_label <= 0:
            continue

        if i - 1 >= len(props):
            continue

        prop = props[i - 1]
        if (
            counts[max_idx] > 0.6 * sum(counts)
            and prop.eccentricity < 1.5
            and prop.euler_number == 1
        ):
            labels[(y_pred == i) & (intensity < 0.6)] = out_label

    return y_pred


def post_process(loc_prob, damage_prob):
    loc_prob = np.clip(loc_prob.astype(np.float32), 0, 1)
    damage_prob = np.clip(damage_prob.astype(np.float32), 0, 1)

    # damage_prob channels are expected as 0 background + 1 no_damage + 2 minor + 3 major + 4 destroyed.
    damage_classes = np.moveaxis(damage_prob[1:5], 0, -1)

    first = np.zeros((damage_classes.shape[0], damage_classes.shape[1], 1), dtype=np.float32)
    first[:, :, 0] = 1.0 - np.sum(damage_classes, axis=2)
    first = np.clip(first, 0, 1)
    first *= 0.8

    damage_pred = np.concatenate([first, damage_classes], axis=-1)

    # Match the original 2nd-place post-processing: boost damaged classes.
    damage_pred[:, :, 2] *= 2.0
    damage_pred[:, :, 3] *= 2.0
    damage_pred[:, :, 4] *= 2.0

    argmax_with_bg = np.argmax(damage_pred, axis=-1)

    loc_mask = ((loc_prob > 0.25) | (argmax_with_bg > 0)).astype(np.uint8)

    damage_labels = np.argmax(damage_classes, axis=-1).astype(np.uint8) + 1
    intensity = np.max(damage_classes, axis=-1)

    label_mask(loc_prob, damage_labels, intensity, loc_mask)

    return loc_mask.astype(np.uint8), damage_labels.astype(np.uint8)


def f1_from_counts(tp, fp, fn):
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0 or recall == 0:
        return 0.0

    return (2.0 * precision * recall) / (precision + recall)


def compute_tp_fn_fp(pred, targ, cls):
    tp = int(np.logical_and(pred == cls, targ == cls).sum())
    fn = int(np.logical_and(pred != cls, targ == cls).sum())
    fp = int(np.logical_and(pred == cls, targ != cls).sum())
    return tp, fn, fp


def compute_metrics(samples):
    loc_tp = 0
    loc_fn = 0
    loc_fp = 0

    dmg_counts = {
        1: {"tp": 0, "fn": 0, "fp": 0},
        2: {"tp": 0, "fn": 0, "fp": 0},
        3: {"tp": 0, "fn": 0, "fp": 0},
        4: {"tp": 0, "fn": 0, "fp": 0},
    }

    for sample in tqdm(samples, desc="metrics"):
        sid = sample["id"]

        pred_loc = read_mask(LOC_DIR / f"{sid}_localization.png")
        pred_dmg = read_mask(DMG_DIR / f"{sid}_damage.png")

        target_loc = read_mask(sample["pre_mask"])
        target_dmg = read_mask(sample["post_mask"])

        if pred_loc.shape != target_loc.shape:
            pred_loc = cv2.resize(
                pred_loc,
                (target_loc.shape[1], target_loc.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if pred_dmg.shape != target_dmg.shape:
            pred_dmg = cv2.resize(
                pred_dmg,
                (target_dmg.shape[1], target_dmg.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        pred_loc_b = (pred_loc > 0).astype(np.uint8)
        target_loc_b = (target_loc > 0).astype(np.uint8)
        target_dmg_b = (target_dmg > 0).astype(np.uint8)

        tp, fn, fp = compute_tp_fn_fp(pred_loc_b, target_loc_b, 1)
        loc_tp += tp
        loc_fn += fn
        loc_fp += fp

        # xView2 damage scoring only evaluates damage where target buildings exist.
        pred_dmg_scored = pred_dmg * pred_loc_b
        pred_dmg_scored = pred_dmg_scored[target_dmg_b == 1]
        target_dmg_scored = target_dmg[target_dmg_b == 1]

        for cls in [1, 2, 3, 4]:
            tp, fn, fp = compute_tp_fn_fp(pred_dmg_scored, target_dmg_scored, cls)
            dmg_counts[cls]["tp"] += tp
            dmg_counts[cls]["fn"] += fn
            dmg_counts[cls]["fp"] += fp

    loc_f1 = f1_from_counts(loc_tp, loc_fp, loc_fn)

    no_damage_f1 = f1_from_counts(dmg_counts[1]["tp"], dmg_counts[1]["fp"], dmg_counts[1]["fn"])
    minor_f1 = f1_from_counts(dmg_counts[2]["tp"], dmg_counts[2]["fp"], dmg_counts[2]["fn"])
    major_f1 = f1_from_counts(dmg_counts[3]["tp"], dmg_counts[3]["fp"], dmg_counts[3]["fn"])
    destroyed_f1 = f1_from_counts(dmg_counts[4]["tp"], dmg_counts[4]["fp"], dmg_counts[4]["fn"])

    macro_damage = float(np.mean([no_damage_f1, minor_f1, major_f1, destroyed_f1]))

    harmonic_damage = 4.0 / sum((x + 1e-6) ** -1 for x in [no_damage_f1, minor_f1, major_f1, destroyed_f1])
    xview2_score = 0.3 * loc_f1 + 0.7 * harmonic_damage

    return {
        "Localization_F1": loc_f1,
        "No_damage_F1": no_damage_f1,
        "Minor_damage_F1": minor_f1,
        "Major_damage_F1": major_f1,
        "Destroyed_F1": destroyed_f1,
        "Macro_F1_damage_classes": macro_damage,
        "Harmonic_F1_damage_classes_xView2": harmonic_damage,
        "xView2_score": xview2_score,
        "counts": {
            "localization": {"tp": loc_tp, "fp": loc_fp, "fn": loc_fn},
            "damage": dmg_counts,
        },
    }


def collect_samples():
    if not TEST_IMAGES.exists():
        raise FileNotFoundError(f"Missing TEST_IMAGES folder: {TEST_IMAGES}")

    if not TEST_MASKS.exists():
        raise FileNotFoundError(f"Missing TEST_MASKS folder: {TEST_MASKS}")

    pre_images = sorted(TEST_IMAGES.glob("*_pre_disaster.png"))

    samples = []
    missing = []

    for pre in pre_images:
        sid = pre.name.replace("_pre_disaster.png", "")

        post = TEST_IMAGES / f"{sid}_post_disaster.png"
        pre_mask = TEST_MASKS / f"{sid}_pre_disaster.png"
        post_mask = TEST_MASKS / f"{sid}_post_disaster.png"

        for p in [post, pre_mask, post_mask]:
            if not p.exists():
                missing.append(str(p))

        samples.append({
            "id": sid,
            "pre": pre,
            "post": post,
            "pre_mask": pre_mask,
            "post_mask": post_mask,
        })

    if missing:
        print("ERROR: missing required files.")
        for m in missing[:50]:
            print(m)
        print("Total missing:", len(missing))
        raise SystemExit(2)

    if len(samples) == 0:
        raise SystemExit("ERROR: no test samples found.")

    return samples


def main():
    global OUT, PRED_DIR, LOC_DIR, DMG_DIR, PROB_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="zero_shot",
        choices=["zero_shot", "zeroshot", "finetuned"],
    )
    args = parser.parse_args()

    is_finetuned = args.mode == "finetuned"
    if is_finetuned:
        OUT = FINETUNED_OUT
        label = "2nd-place xView2 full available ensemble FINE-TUNED on Mount Semeru official split"
        note = "Fine-tuned on Semeru train, selected on Semeru validation, evaluated on held-out Semeru test."
    else:
        OUT = ZERO_SHOT_OUT
        label = "2nd-place xView2 full available ensemble ZERO-SHOT on Mount Semeru TEST"
        note = "No Mount Semeru fine-tuning used."

    PRED_DIR = OUT / "predictions"
    LOC_DIR = PRED_DIR / "localization"
    DMG_DIR = PRED_DIR / "damage"
    PROB_DIR = OUT / "probabilities"
    for directory in [OUT, PRED_DIR, LOC_DIR, DMG_DIR, PROB_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("================================================")
    print(label)
    print(note)
    print("================================================")

    print("Device:", DEVICE)
    print("Dataset:", DATASET)
    print("Images:", TEST_IMAGES)
    print("Masks:", TEST_MASKS)
    print("Output:", OUT)

    model_configs = read_manifest()
    if is_finetuned:
        model_configs = [
            model_config._replace(weight_path=find_finetuned_checkpoint(model_config))
            for model_config in model_configs
        ]
    loc_models = [m for m in model_configs if m.task == "localization"]
    damage_models = [m for m in model_configs if m.task == "damage"]

    print("Localization models:", len(loc_models))
    print("Damage models:", len(damage_models))

    if len(loc_models) != 6:
        raise SystemExit(f"ERROR: expected 6 localization models, found {len(loc_models)}")

    if len(damage_models) != 15:
        raise SystemExit(f"ERROR: expected 15 damage models, found {len(damage_models)}")

    samples = collect_samples()
    print("Test samples:", len(samples))

    loc_sum = {}
    damage_sum = {}

    for sample in samples:
        loc_sum[sample["id"]] = np.zeros((PRED_SIZE, PRED_SIZE), dtype=np.float32)
        damage_sum[sample["id"]] = np.zeros((5, PRED_SIZE, PRED_SIZE), dtype=np.float32)

    print()
    print("================================================")
    print("Run localization ensemble")
    print("================================================")

    for model_config in loc_models:
        print(f"Localization model: {model_config.tag}")
        model, conf = load_model(model_config, seg_classes=1)

        for sample in tqdm(samples, desc=f"loc {model_config.tag}"):
            pre = read_rgb(sample["pre"])
            post = read_rgb(sample["post"])
            pred = predict_localization_one(model, conf, pre, post, model_config)

            sid = sample["id"]
            if pred.shape != loc_sum[sid].shape:
                pred = cv2.resize(
                    pred,
                    (loc_sum[sid].shape[1], loc_sum[sid].shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            loc_sum[sid] += pred * float(model_config.ensemble_weight)

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    loc_total_weight = sum(float(m.ensemble_weight) for m in loc_models)

    print()
    print("================================================")
    print("Run damage ensemble")
    print("================================================")

    for model_config in damage_models:
        print(f"Damage model: {model_config.tag} ensemble_weight: {model_config.ensemble_weight}")
        model, conf = load_model(model_config, seg_classes=5)

        for sample in tqdm(samples, desc=f"damage {model_config.tag}"):
            pre = read_rgb(sample["pre"])
            post = read_rgb(sample["post"])
            pred = predict_damage_one(model, conf, pre, post)

            sid = sample["id"]
            expected_h, expected_w = damage_sum[sid].shape[1:]

            if pred.shape[1:] != (expected_h, expected_w):
                resized = np.zeros((5, expected_h, expected_w), dtype=np.float32)
                for c in range(5):
                    resized[c] = cv2.resize(
                        pred[c],
                        (expected_w, expected_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                pred = resized

            damage_sum[sid] += pred * float(model_config.ensemble_weight)

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    damage_total_weight = sum(float(m.ensemble_weight) for m in damage_models)

    print()
    print("================================================")
    print("Post-process predictions")
    print("================================================")

    for sample in tqdm(samples, desc="postprocess"):
        sid = sample["id"]

        loc_prob = loc_sum[sid] / loc_total_weight
        damage_prob = damage_sum[sid] / damage_total_weight

        loc_mask, damage_mask = post_process(loc_prob, damage_prob)

        cv2.imwrite(str(LOC_DIR / f"{sid}_localization.png"), loc_mask.astype(np.uint8))
        cv2.imwrite(str(DMG_DIR / f"{sid}_damage.png"), damage_mask.astype(np.uint8))

    print()
    print("================================================")
    print("Compute metrics")
    print("================================================")

    metrics = compute_metrics(samples)

    with open(OUT / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUT / "metrics_summary.txt", "w") as f:
        f.write(label + "\n")
        f.write(note + "\n\n")
        f.write(f"Localization_F1: {metrics['Localization_F1']:.6f}\n")
        f.write(f"No_damage_F1: {metrics['No_damage_F1']:.6f}\n")
        f.write(f"Minor_damage_F1: {metrics['Minor_damage_F1']:.6f}\n")
        f.write(f"Major_damage_F1: {metrics['Major_damage_F1']:.6f}\n")
        f.write(f"Destroyed_F1: {metrics['Destroyed_F1']:.6f}\n")
        f.write(f"Macro_F1_damage_classes: {metrics['Macro_F1_damage_classes']:.6f}\n")
        f.write(f"Harmonic_F1_damage_classes_xView2: {metrics['Harmonic_F1_damage_classes_xView2']:.6f}\n")
        f.write(f"xView2_score: {metrics['xView2_score']:.6f}\n")

    print()
    print("DONE")
    print("Metrics TXT:", OUT / "metrics_summary.txt")
    print("Metrics JSON:", OUT / "metrics_summary.json")
    print()

    with open(OUT / "metrics_summary.txt") as f:
        print(f.read())


if __name__ == "__main__":
    main()
