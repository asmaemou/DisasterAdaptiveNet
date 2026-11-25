#!/usr/bin/env python3
"""Select Strong Baseline localization threshold on xBD Hold; evaluate Test ONCE.

Requires the existing, validated transformer/scripts/evaluate_gerard_official_xbd.py
and the frozen model checkpoint trained by SLURM job 32359.

- Hold: select threshold using 0.3*localization F1 + 0.7*harmonic damage F1.
- Test: apply that fixed threshold, report arithmetic damage Macro-F1.
- No training, no modification of model checkpoint or previous result directories.
- The threshold grid follows the RS-BDNet manuscript's documented candidate grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from transformer.scripts import evaluate_gerard_official_xbd as ref

THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def get_hold_samples(root: Path):
    """Read Hold targets only, preserving exactly the test scorer's conventions."""
    with (root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert len(metadata["train"]["patches"]) + len(metadata["tier3"]["patches"]) == 9168
    assert len(metadata["hold"]["patches"]) == 933
    assert len(metadata["test"]["patches"]) == 933
    keys = {
        part: {(str(s["event"]), str(s["patch_id"])) for s in metadata[part]["patches"]}
        for part in ("train", "tier3", "hold", "test")
    }
    assert len(keys["hold"]) == 933 and len(keys["test"]) == 933
    assert not (keys["train"] & keys["hold"] or keys["tier3"] & keys["hold"])
    assert not (keys["train"] & keys["test"] or keys["tier3"] & keys["test"])
    assert not (keys["hold"] & keys["test"])
    samples = []
    for s in sorted(metadata["hold"]["patches"], key=lambda x: (str(x["event"]), str(x["patch_id"]))):
        name = f'{s["event"]}_{s["patch_id"]}'
        paths = {
            "pre": root / "hold/images" / f"{name}_pre_disaster.png",
            "post": root / "hold/images" / f"{name}_post_disaster.png",
            "loc": root / "hold/targets" / f"{name}_pre_disaster_target.png",
            "dmg": root / "hold/targets" / f"{name}_post_disaster_target.png",
        }
        for kind, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing Hold {kind}: {path}")
        samples.append((name, paths))
    return samples


def new_counts():
    return {"localization": {"tp": 0, "fp": 0, "fn": 0},
            **{label: {"tp": 0, "fp": 0, "fn": 0} for label in ref.CLASSES}}


def accumulate(counts, probs: np.ndarray, damage_class: np.ndarray,
               true_loc: np.ndarray, true_dmg: np.ndarray, threshold: float):
    """Same pixel-level definitions as original Gerard test evaluation."""
    pred_loc = probs > threshold
    loc = counts["localization"]
    loc["tp"] += int(np.count_nonzero(pred_loc & true_loc))
    loc["fp"] += int(np.count_nonzero(pred_loc & ~true_loc))
    loc["fn"] += int(np.count_nonzero(~pred_loc & true_loc))

    valid = true_loc & (true_dmg >= 1) & (true_dmg <= 4)
    predicted = np.where(pred_loc[valid], damage_class[valid], 0)
    target = true_dmg[valid]
    for i, label in enumerate(ref.CLASSES, start=1):
        score = counts[label]
        score["tp"] += int(np.count_nonzero((predicted == i) & (target == i)))
        score["fp"] += int(np.count_nonzero((predicted == i) & (target != i)))
        score["fn"] += int(np.count_nonzero((predicted != i) & (target == i)))


def scores_for_counts(counts):
    records = {key: ref.f1_record(**ct) for key, ct in counts.items()}
    loc_f1 = records["localization"]["f1"]
    classwise = {label: records[label]["f1"] for label in ref.CLASSES}
    macro = sum(classwise.values()) / 4
    harmonic = 4 / sum(1 / (f1 + 1e-6) for f1 in classwise.values())
    # This is the validation selection score documented in the manuscript.
    selection_score = 0.3 * loc_f1 + 0.7 * harmonic
    return {
        "localization_f1": loc_f1,
        "damage_macro_f1": macro,
        "damage_harmonic_f1": harmonic,
        "validation_selection_score": selection_score,
        "classwise_damage_f1": classwise,
        "counts": records,
    }


def build_frozen_model(checkpoint_path: Path):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required; submit using SLURM")
    print(f"CALIB: device={device}, cudnn_enabled={torch.backends.cudnn.enabled}", flush=True)

    from utils import models
    original = models.torchvision.models.resnet34
    def no_download(*args, **kwargs):
        kwargs["weights"] = None
        kwargs.pop("pretrained", None)
        return original(*args, **kwargs)
    models.torchvision.models.resnet34 = no_download
    try:
        cfg = SimpleNamespace(MODEL=SimpleNamespace(TYPE="strongbaseline", OUT_CHANNELS=5))
        net = models.create_network(cfg)
    finally:
        models.torchvision.models.resnet34 = original

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "network" not in checkpoint or "epoch" not in checkpoint:
        raise RuntimeError("Checkpoint must contain 'network' and 'epoch'")
    net.load_state_dict(checkpoint["network"], strict=True)
    epoch = float(checkpoint["epoch"])
    del checkpoint
    net.to(device).eval().requires_grad_(False)
    parameters = sum(p.numel() for p in net.module.parameters())
    print(f"CALIB: frozen checkpoint epoch={epoch:g}, params={parameters}", flush=True)
    return net, device, epoch, parameters


def score_split(samples, net, device, image_size, thresholds, split_name):
    cumulative = {t: new_counts() for t in thresholds}
    with torch.inference_mode():
        for i, (name, paths) in enumerate(samples, 1):
            img, true_loc, true_dmg = ref.read_pair(paths, image_size)
            logits = net(img.unsqueeze(0).to(device))
            if logits.shape != (1, 5, image_size, image_size):
                raise RuntimeError(f"Unexpected logits {tuple(logits.shape)}: {name}")
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"Nonfinite logits: {name}")
            probs = torch.sigmoid(logits[0, 0]).cpu().numpy()
            damage_class = (logits[0, 1:5].argmax(dim=0) + 1).cpu().numpy().astype(np.uint8)
            for t in thresholds:
                accumulate(cumulative[t], probs, damage_class, true_loc, true_dmg, t)
            if i == 1 or i % 25 == 0 or i == len(samples):
                print(f"{split_name}: {i}/{len(samples)} pairs; latest={name}", flush=True)
    return {t: scores_for_counts(cumulative[t]) for t in thresholds}


def write_json_exclusive(path: Path, obj):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ref.DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=ref.DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.image_size <= 0 or args.image_size % 32:
        parser.error("--image-size must be a positive multiple of 32")
    root = args.data_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    print(f"CALIB: checkpoint found: {checkpoint}", flush=True)
    hold_samples = get_hold_samples(root)
    # Test metadata / image path checks do not inspect test ground truth values.
    test_samples = ref.discover_samples(root)
    print(f"CALIB PREFLIGHT PASSED: Hold={len(hold_samples)} Test={len(test_samples)}; 896-grid default", flush=True)
    if args.preflight:
        print("CALIB: preflight only; no GPU inference or threshold selection", flush=True)
        return
    if args.output_dir is None:
        parser.error("--output-dir required except for --preflight")
    output = args.output_dir.absolute()
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Missing output parent: {output.parent}")
    if output == checkpoint or output in checkpoint.parents or checkpoint in output.parents:
        raise RuntimeError("Output directory overlaps checkpoint")
    output.mkdir(exist_ok=False)
    sha = ref.sha256_file(checkpoint)
    net, device, epoch, parameters = build_frozen_model(checkpoint)

    print("CALIB: BEGIN HOLD INFERENCE — test labels not used for selection", flush=True)
    hold_results = score_split(hold_samples, net, device, args.image_size, THRESHOLDS, "HOLD")
    # Maximize validation composite score. Deterministic ties: closest to 0.50, then lower threshold.
    chosen = max(THRESHOLDS, key=lambda t: (hold_results[t]["validation_selection_score"], -abs(t - 0.5), -t))
    calib = {
        "model": "Gerard-style Strong Baseline (Hafner implementation)",
        "calibration_split": "official_xbd_hold",
        "num_hold_pairs": len(hold_samples),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "selected_val_epoch": epoch,
        "parameter_count": parameters,
        "image_size": args.image_size,
        "selection_rule": "maximize 0.3 * localization_f1 + 0.7 * harmonic_damage_f1 on Hold",
        "candidate_thresholds": list(THRESHOLDS),
        "selected_threshold": chosen,
        "hold_results_by_threshold": {f"{t:.2f}": hold_results[t] for t in THRESHOLDS},
    }
    write_json_exclusive(output / "hold_calibration.json", calib)
    with (output / "hold_threshold_sweep.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "hold_selection_score", "hold_localization_f1", "hold_damage_harmonic_f1", "hold_damage_macro_f1", *ref.CLASSES])
        for t in THRESHOLDS:
            r = hold_results[t]
            writer.writerow([f"{t:.2f}", r["validation_selection_score"], r["localization_f1"], r["damage_harmonic_f1"], r["damage_macro_f1"], *[r["classwise_damage_f1"][k] for k in ref.CLASSES]])
    print(f"CALIB: HOLD SELECTED threshold={chosen:.2f}, score={hold_results[chosen]['validation_selection_score']:.8f}", flush=True)
    print(f"CALIB: HOLD frozen selection saved: {output / 'hold_calibration.json'}", flush=True)

    # The threshold is frozen on Hold before any test image or GT is loaded.
    print("CALIB: BEGIN TEST INFERENCE with frozen Hold-selected threshold", flush=True)
    test_thresholds = (chosen,) if chosen == 0.5 else (chosen, 0.5)
    test_scores = score_split(test_samples, net, device, args.image_size, test_thresholds, "TEST")
    selected = test_scores[chosen]
    # Regression check: the unchanged 0.50 threshold must reproduce job 39123.
    # This comparison never affects threshold selection, which is already frozen.
    old_scores = PROJECT / "output/reviewer3_gerard_eval_39123/scores.json"
    if old_scores.is_file():
        with old_scores.open("r", encoding="utf-8") as handle:
            original_scores = json.load(handle)
        if original_scores.get("checkpoint_sha256") != sha:
            raise RuntimeError("Original evaluation checkpoint SHA256 differs")
        check_fields = ("localization_f1", "damage_macro_f1")
        for field in check_fields:
            delta = abs(test_scores[0.5][field] - original_scores[field])
            if delta > 1e-8:
                raise RuntimeError(f"Fixed 0.50 regression mismatch for {field}: {delta}")
        for name in ref.CLASSES:
            delta = abs(test_scores[0.5]["classwise_damage_f1"][name] - original_scores["classwise_damage_f1"][name])
            if delta > 1e-8:
                raise RuntimeError(f"Fixed 0.50 regression mismatch for {name}: {delta}")
        print("CALIB: fixed-0.50 scores exactly reproduce original job 39123", flush=True)
    else:
        print("CALIB: original fixed-0.50 scores.json absent; regression check skipped", flush=True)
    result = {
        "model": "Gerard-style Strong Baseline (Hafner implementation)",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "selected_val_epoch": epoch,
        "parameter_count": parameters,
        "evaluated_split": "official_xbd_test",
        "num_test_pairs": len(test_samples),
        "image_size": args.image_size,
        "input_preprocessing": "BGR, concatenate pre/post, float32 / 127 - 1",
        "loc_threshold": chosen,
        "loc_threshold_source": "official_xbd_hold",
        "threshold_selection_rule": calib["selection_rule"],
        "threshold_sweep": list(THRESHOLDS),
        "damage_gated_by_localization": True,
        "damage_valid_gt": "pre-localization-positive and damage target in 1..4",
        "localization_f1": selected["localization_f1"],
        "damage_macro_f1": selected["damage_macro_f1"],
        "damage_harmonic_f1_for_reference_only": selected["damage_harmonic_f1"],
        "classwise_damage_f1": selected["classwise_damage_f1"],
        "counts": selected["counts"],
        "fixed_threshold_0.50_reference": {
            "localization_f1": test_scores[0.5]["localization_f1"],
            "damage_macro_f1": test_scores[0.5]["damage_macro_f1"],
            "classwise_damage_f1": test_scores[0.5]["classwise_damage_f1"],
        },
    }
    write_json_exclusive(output / "scores.json", result)
    with (output / "scores.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "checkpoint_epoch", "num_test_pairs", "parameters_M", "loc_threshold", "localization_f1", "damage_macro_f1", *ref.CLASSES])
        writer.writerow(["gerard_strong_baseline", epoch, len(test_samples), parameters / 1e6, chosen, selected["localization_f1"], selected["damage_macro_f1"], *[selected["classwise_damage_f1"][c] for c in ref.CLASSES]])
    print("CALIB: TEST RESULTS", json.dumps({k: result[k] for k in ("num_test_pairs", "loc_threshold", "localization_f1", "damage_macro_f1", "classwise_damage_f1")}, indent=2), flush=True)
    print("CALIB: saved new results to", output, flush=True)


if __name__ == "__main__":
    main()
