#!/usr/bin/env python3
"""Select an official Keras xView2 classifier checkpoint on Texas validation."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-model-dir", required=True)
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.official_model_dir)
    from model import generate_xBD_baseline_model  # type: ignore
    import keras

    frame = pd.read_csv(args.val_csv)
    frame["labels"] = frame["labels"].astype(str)
    generator_factory = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
    checkpoints = sorted(glob.glob(args.checkpoint_glob))
    if not checkpoints:
        raise RuntimeError(f"No classifier checkpoints match {args.checkpoint_glob}")
    model = generate_xBD_baseline_model()
    results = []
    for checkpoint in checkpoints:
        model.load_weights(checkpoint)
        generator = generator_factory.flow_from_dataframe(
            dataframe=frame, directory=args.val_data, x_col="uuid", y_col="labels",
            batch_size=64, shuffle=False, class_mode="categorical", target_size=(128, 128),
        )
        probability = model.predict_generator(generator, steps=int(np.ceil(len(frame) / 64.0)))
        prediction = probability[:len(frame)].argmax(1)
        score = float(f1_score(generator.classes[:len(frame)], prediction, average="weighted"))
        results.append({"checkpoint": checkpoint, "validation_weighted_f1": score})
        print(f"{checkpoint}: validation weighted F1={score:.6f}")
    best = max(results, key=lambda item: item["validation_weighted_f1"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["checkpoint"], output)
    output.with_suffix(".selection.json").write_text(json.dumps({"best": best, "all": results}, indent=2) + "\n")
    print(f"Selected: {best}")


if __name__ == "__main__":
    main()
