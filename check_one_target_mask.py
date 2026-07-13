
import cv2

import numpy as np

from pathlib import Path



mask_path = Path("/homes/j244s673/documents/wsu/phd/hurricane_delta_preprocessed/train/targets/HURRICANE-DELTA_000000_post_disaster_target.png")



label_names = {

    0: "background / non-building",

    1: "no damage",

    2: "minor damage",

    3: "major damage",

    4: "destroyed",

    255: "ignore / unknown / unlabeled",

}



m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)



if m is None:

    raise FileNotFoundError(f"Could not read mask: {mask_path}")



if m.ndim == 3:

    print(f"Mask has {m.shape[2]} channels. Using first channel only.")

    m = m[..., 0]



print("Mask path:", mask_path)

print("Shape:", m.shape)

print("Dtype:", m.dtype)

print()



values, counts = np.unique(m, return_counts=True)

total = counts.sum()



print("Unique classes found:")

print("-" * 80)

print(f"{'Value':>8}  {'Meaning':<35} {'Pixels':>12} {'Percent':>10}")

print("-" * 80)



for v, c in zip(values, counts):

    v_int = int(v)

    meaning = label_names.get(v_int, "unknown label value")

    pct = 100.0 * c / total

    print(f"{v_int:>8}  {meaning:<35} {int(c):>12} {pct:>9.4f}%")



print("-" * 80)

print("Total pixels:", int(total))

