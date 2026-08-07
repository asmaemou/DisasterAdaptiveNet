#!/usr/bin/env python3
"""Export Texas-fine-tuned xView2 second-place ensemble probabilities."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("second_place_texas_export_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import second-place source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def disable_constructor_pretraining() -> None:
    """Prevent legacy encoders from downloading weights before checkpoint load."""
    def offline_constructor(original):
        def construct():
            try:
                return original(pretrained=None)
            except TypeError as error:
                if "pretrained" not in str(error):
                    raise
                return original()
        return construct

    registry_names = ("models.unet", "models.siamese_unet")
    patched = 0
    for registry_name in registry_names:
        registry_module = sys.modules.get(registry_name)
        if registry_module is None or not hasattr(registry_module, "encoder_params"):
            raise RuntimeError(f"Could not locate second-place encoder registry: {registry_name}")
        for encoder_name, parameters in registry_module.encoder_params.items():
            if encoder_name.startswith("efficientnet-"):
                # EfficientNet.from_pretrained downloads internally and has no
                # `pretrained` keyword. from_name constructs the identical
                # architecture without network access.
                parameters["init_op"] = (
                    lambda name=encoder_name, module=registry_module:
                    module.EfficientNet.from_name(name)
                )
            else:
                parameters["init_op"] = offline_constructor(parameters["init_op"])
            parameters["url"] = None
            patched += 1
        remaining_urls = [
            name for name, parameters in registry_module.encoder_params.items()
            if parameters.get("url") is not None
        ]
        if remaining_urls:
            raise RuntimeError(
                f"Constructor downloads remain enabled in {registry_name}: {remaining_urls}"
            )
    print(
        f"Disabled redundant constructor downloads for "
        f"{patched} second-place encoders across localization and Siamese registries; "
        "complete fine-tuned checkpoints will be loaded next.",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")
    parser.add_argument(
        "--source-script", type=Path,
        default=project / "baselines/xview2_winners/eval_second_place_semeru_test_full.py",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=project / "output/xview2_baseline_datasets/second_place_texas_tornadoes",
    )
    parser.add_argument(
        "--finetune-root", type=Path,
        default=project / "output/xview2_baselines/second_place_texas_tornadoes_FULL_SOLUTION_finetune_official_split",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--expected-val-samples", type=int, default=45)
    parser.add_argument("--expected-test-samples", type=int, default=46)
    parser.add_argument("--use-released-checkpoints", action="store_true", help="Use original checkpoint paths from the second-place manifest")
    return parser.parse_args()


def main():
    args = parse_args()
    required_paths = [args.source_script, args.data_root]
    if not args.use_released_checkpoints:
        required_paths.append(args.finetune_root)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the second-place ensemble exporter")

    source = load_source(args.source_script)
    disable_constructor_pretraining()
    source.FT_EXP = args.finetune_root
    source.DEVICE = torch.device("cuda")

    folds = pd.read_csv(args.data_root / "folds.csv", dtype={"id": str})
    samples_by_split = {}
    for split in args.splits:
        rows = folds[folds["split"].astype(str).str.lower() == split]
        samples = []
        for tile_id in rows["id"].astype(str):
            sample = {
                "id": tile_id,
                "pre": args.data_root / "images" / f"{tile_id}_pre_disaster.png",
                "post": args.data_root / "images" / f"{tile_id}_post_disaster.png",
                "pre_mask": args.data_root / "masks" / f"{tile_id}_pre_disaster.png",
                "post_mask": args.data_root / "masks" / f"{tile_id}_post_disaster.png",
                "split": split,
            }
            for key in ("pre", "post", "pre_mask", "post_mask"):
                if not sample[key].is_file():
                    raise FileNotFoundError(sample[key])
            samples.append(sample)
        samples_by_split[split] = samples

    expected = {"val": args.expected_val_samples, "test": args.expected_test_samples}
    for split, samples in samples_by_split.items():
        if split in expected and len(samples) != expected[split]:
            raise RuntimeError(f"Expected {expected[split]} {split} samples, found {len(samples)}")
    all_samples = [sample for split in args.splits for sample in samples_by_split[split]]
    print("Second-place export samples:", {k: len(v) for k, v in samples_by_split.items()}, flush=True)

    model_configs = source.read_manifest()
    if not args.use_released_checkpoints:
        model_configs = [
            config._replace(weight_path=source.find_finetuned_checkpoint(config))
            for config in model_configs
        ]
    localization_models = [model for model in model_configs if model.task == "localization"]
    damage_models = [model for model in model_configs if model.task == "damage"]
    if len(localization_models) != 6 or len(damage_models) != 15:
        raise RuntimeError(
            f"Expected 6 localization and 15 damage models; got "
            f"{len(localization_models)} and {len(damage_models)}"
        )
    checkpoint_label = "Released xBD" if args.use_released_checkpoints else "Fine-tuned"
    print(f"{checkpoint_label} models: localization=6, damage=15", flush=True)

    loc_sum = {sample["id"]: np.zeros((source.PRED_SIZE, source.PRED_SIZE), np.float32) for sample in all_samples}
    damage_sum = {sample["id"]: np.zeros((5, source.PRED_SIZE, source.PRED_SIZE), np.float32) for sample in all_samples}

    for config in localization_models:
        print(f"Localization model: {config.tag} checkpoint={config.weight_path}", flush=True)
        model, conf = source.load_model(config, seg_classes=1)
        for sample in tqdm(all_samples, desc=f"loc {config.tag}"):
            prediction = source.predict_localization_one(
                model, conf, source.read_rgb(sample["pre"]), source.read_rgb(sample["post"]), config
            )
            if prediction.shape != loc_sum[sample["id"]].shape:
                prediction = cv2.resize(prediction, (source.PRED_SIZE, source.PRED_SIZE), interpolation=cv2.INTER_LINEAR)
            loc_sum[sample["id"]] += prediction * float(config.ensemble_weight)
        del model
        torch.cuda.empty_cache()

    for config in damage_models:
        print(f"Damage model: {config.tag} checkpoint={config.weight_path}", flush=True)
        model, conf = source.load_model(config, seg_classes=5)
        for sample in tqdm(all_samples, desc=f"damage {config.tag}"):
            prediction = source.predict_damage_one(
                model, conf, source.read_rgb(sample["pre"]), source.read_rgb(sample["post"])
            )
            if prediction.shape[1:] != (source.PRED_SIZE, source.PRED_SIZE):
                prediction = np.stack([
                    cv2.resize(channel, (source.PRED_SIZE, source.PRED_SIZE), interpolation=cv2.INTER_LINEAR)
                    for channel in prediction
                ])
            damage_sum[sample["id"]] += prediction * float(config.ensemble_weight)
        del model
        torch.cuda.empty_cache()

    loc_weight = sum(float(model.ensemble_weight) for model in localization_models)
    damage_weight = sum(float(model.ensemble_weight) for model in damage_models)
    for sample in all_samples:
        tile_id = sample["id"]
        split = sample["split"]
        loc_probability = np.clip(loc_sum[tile_id] / loc_weight, 0, 1)
        damage_probability5 = np.clip(damage_sum[tile_id] / damage_weight, 0, 1)
        foreground = damage_probability5[1:5]
        foreground /= np.maximum(foreground.sum(axis=0, keepdims=True), 1e-7)
        loc_prediction, damage_prediction = source.post_process(loc_probability, damage_probability5)

        loc_true = (source.read_mask(sample["pre_mask"]) > 0).astype(np.uint8)
        damage_true = source.read_mask(sample["post_mask"]).astype(np.uint8)
        if loc_true.shape != (source.PRED_SIZE, source.PRED_SIZE):
            loc_true = cv2.resize(loc_true, (source.PRED_SIZE, source.PRED_SIZE), interpolation=cv2.INTER_NEAREST)
            damage_true = cv2.resize(damage_true, (source.PRED_SIZE, source.PRED_SIZE), interpolation=cv2.INTER_NEAREST)

        destination = args.output_root / split / f"{tile_id}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            loc_probability=loc_probability.astype(np.float16),
            damage_probability=foreground.astype(np.float16),
            damage_probability5=damage_probability5.astype(np.float16),
            loc_prediction=loc_prediction.astype(np.uint8),
            damage_prediction=damage_prediction.astype(np.uint8),
            loc_true=loc_true,
            damage_true=damage_true,
        )
    for split in args.splits:
        (args.output_root / split / "_SUCCESS_V1").touch()
    print(f"Wrote second-place probability maps under: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
