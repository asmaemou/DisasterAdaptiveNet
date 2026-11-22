#!/usr/bin/env python3
"""Compare a new HBG-CAON test-score JSON against archived Aug 10 Slurm results.

The reference values were transcribed from original completed job logs.
The four-class macro F1 is an arithmetic mean, *not* JSON's `damage_f1`.
"""
import argparse
import json
from pathlib import Path

REFERENCE = {
    "turkey": {
        "original_job": "1173291",
        "localization_f1": 0.9242687264811915,
        "damage_f1_no_damage": 0.9610982329585656,
        "damage_f1_minor_damage": 0.49762640902966704,
        "damage_f1_major_damage": 0.350770983636526,
        "damage_f1_destroyed": 0.6363747404421644,
    },
    "semeru": {
        "original_job": "1173317",
        "localization_f1": 0.9078565504784151,
        "damage_f1_no_damage": 0.9569443653766599,
        "damage_f1_minor_damage": 0.9140819219938962,
        "damage_f1_major_damage": 0.4511568269241679,
        "damage_f1_destroyed": 0.06414134469957376,
    },
}

def macro(results):
    return sum(results[f"damage_f1_{cls}"] for cls in
               ("no_damage", "minor_damage", "major_damage", "destroyed")) / 4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=REFERENCE)
    parser.add_argument("--output-dir", required=True,
                        help="New HBG_CAON_*_REPRO_JOBID output directory")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    path = Path(args.output_dir) / "scores" / "scores_xbd_test.json"
    scores = json.loads(path.read_text())
    reference = REFERENCE[args.experiment]
    print(f"Original August Slurm job: {reference['original_job']}")
    print(f"New scores file: {path}")
    print(f"{'Metric':30} {'August':>12} {'Rerun':>12} {'Absolute delta':>15}")
    print("-" * 73)
    for key in ("localization_f1", "damage_f1_no_damage",
                "damage_f1_minor_damage", "damage_f1_major_damage",
                "damage_f1_destroyed"):
        old = reference[key]
        new = scores[key]
        print(f"{key:30} {old:12.6f} {new:12.6f} {abs(new-old):15.6f}")
    old_macro = macro(reference)
    new_macro = macro(scores)
    print(f"{'4-class macro F1':30} {old_macro:12.6f} {new_macro:12.6f} "
          f"{abs(new_macro-old_macro):15.6f}")
    print(f"All individual metrics within {args.tolerance}? ",
          all(abs(scores[key] - reference[key]) <= args.tolerance for key in
              ("localization_f1", "damage_f1_no_damage",
               "damage_f1_minor_damage", "damage_f1_major_damage",
               "damage_f1_destroyed")))
    print("NOTE: Similar F1 scores alone do not prove identical model weights.")

if __name__ == "__main__":
    main()
