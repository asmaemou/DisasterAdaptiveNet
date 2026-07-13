
import cv2

import numpy as np

from pathlib import Path

from collections import Counter



target_dir = Path("/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed/train/targets")



label_names = {

    0: "background / non-building",

    1: "no damage",

    2: "minor damage",

    3: "major damage",

    4: "destroyed",

    255: "ignore / unknown / unlabeled",

}



counter = Counter()

file_counter = Counter()

bad_files = []



for p in sorted(target_dir.glob("*_post_disaster_target.png")):

    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)



    if m is None:

        bad_files.append(str(p))

        continue



    if m.ndim == 3:

        m = m[..., 0]



    values, counts = np.unique(m, return_counts=True)



    for v, c in zip(values, counts):

        v = int(v)

        counter[v] += int(c)

        file_counter[v] += 1



total = sum(counter.values())



print("Target directory:", target_dir)

print("Number of unreadable files:", len(bad_files))

print()

print("Overall class distribution:")

print("-" * 85)

print(f"{'Value':>8}  {'Meaning':<35} {'Pixels':>14} {'Percent':>10} {'Files':>8}")

print("-" * 85)



for v in sorted(counter):

    meaning = label_names.get(v, "unknown label value")

    pixels = counter[v]

    pct = 100.0 * pixels / total

    files = file_counter[v]

    print(f"{v:>8}  {meaning:<35} {pixels:>14} {pct:>9.4f}% {files:>8}")



print("-" * 85)

print("Total pixels:", total)



if bad_files:

    print("\nUnreadable files:")

    for f in bad_files[:20]:

        print(f)

