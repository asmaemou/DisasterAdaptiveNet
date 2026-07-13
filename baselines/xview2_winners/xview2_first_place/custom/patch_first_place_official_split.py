#!/usr/bin/env python3

from pathlib import Path
import re
import subprocess

TRAIN_TUNE_FILES = [
    "train154_cls_cce.py", "train154_loc.py",
    "train34_cls.py", "train34_loc.py",
    "train50_cls_cce.py", "train50_loc.py",
    "train92_cls_cce.py", "train92_loc.py",
    "tune154_cls_cce.py", "tune34_cls.py",
    "tune50_cls_cce.py", "tune50_loc.py",
    "tune92_cls_cce.py", "tune92_loc.py",
]

PREDICT_FILES = [
    "predict154cls.py", "predict154_loc.py",
    "predict34cls.py", "predict34_loc.py",
    "predict50cls.py", "predict50_loc.py",
    "predict92cls.py", "predict92_loc.py",
]

FILES = TRAIN_TUNE_FILES + PREDICT_FILES + ["predict_loc_val.py"]

APEX_BLOCK = '''try:
    from apex import amp
except ImportError:
    from contextlib import contextmanager

    class _DummyAmp:
        def initialize(self, model, optimizer, opt_level="O1"):
            print("WARNING: apex is not installed. Running without mixed precision.")
            return model, optimizer

        def master_params(self, optimizer):
            for group in optimizer.param_groups:
                for p in group["params"]:
                    yield p

        @contextmanager
        def scale_loss(self, loss, optimizer):
            yield loss

    amp = _DummyAmp()
'''

OFFICIAL_COLLECTION = '''all_files = []
train_len = None

for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(path.join(d, 'images', f))

    if os.environ.get('USE_OFFICIAL_VAL', '0') == '1' and d == 'train':
        train_len = len(all_files)

if train_len is None:
    train_len = len(all_files)

def get_official_split_idxs():
    train_idxs = np.array([i for i, fn in enumerate(all_files) if fn.startswith('train/')])
    val_idxs = np.array([i for i, fn in enumerate(all_files) if fn.startswith('val/')])

    if len(train_idxs) == 0:
        raise RuntimeError("Official split mode is enabled, but no train images were found.")
    if len(val_idxs) == 0:
        raise RuntimeError("Official split mode is enabled, but no val images were found.")

    print("USE_OFFICIAL_VAL=1")
    print("Official train images:", len(train_idxs))
    print("Official val images:", len(val_idxs))

    return train_idxs, val_idxs

'''


def patch_file(path):
    s = path.read_text()
    old = s

    if "import os\n" not in s:
        s = s.replace("from os import", "import os\nfrom os import", 1)

    s = re.sub(
        r"^from apex import amp\s*$",
        APEX_BLOCK,
        s,
        flags=re.MULTILINE,
    )

    s = s.replace(
        "models_folder = 'weights'",
        "models_folder = __import__('os').environ.get('MODELS_FOLDER', 'weights')",
    )

    s = re.sub(
        r"torch\.load\((path\.join\(models_folder,\s*snap_to_load\)),\s*map_location='cpu'\)",
        r"torch.load(\1, map_location='cpu', weights_only=False)",
        s,
    )

    s = s.replace(
        "train_dirs = ['train', 'tier3']",
        "train_dirs = ['train', 'val'] if os.environ.get('USE_OFFICIAL_VAL', '0') == '1' else ['train', 'tier3']",
    )

    s = re.sub(
        r"all_files = \[\]\n"
        r"for d in train_dirs:\n"
        r"    for f in sorted\(listdir\(path.join\(d, 'images'\)\)\):\n"
        r"        if '_pre_disaster.png' in f:\n"
        r"            all_files.append\(path.join\(d, 'images', f\)\)\n"
        r"(?:train_len = len\(all_files\)\n)?",
        OFFICIAL_COLLECTION,
        s,
    )

    s = s.replace(
        "train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)",
        "if os.environ.get('USE_OFFICIAL_VAL', '0') == '1':\n"
        "        train_idxs, val_idxs = get_official_split_idxs()\n"
        "    else:\n"
        "        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)",
    )

    s = s.replace(
        "_, val_idxs = train_test_split(np.arange(train_len), test_size=0.1, random_state=seed)",
        "if os.environ.get('USE_OFFICIAL_VAL', '0') == '1':\n"
        "        train_idxs, val_idxs = get_official_split_idxs()\n"
        "    else:\n"
        "        _, val_idxs = train_test_split(np.arange(train_len), test_size=0.1, random_state=seed)",
    )

    s = s.replace(
        "train_idxs = np.arange(len(all_files)) # Use all train",
        "if os.environ.get('USE_OFFICIAL_VAL', '0') != '1':\n"
        "        train_idxs = np.arange(len(all_files)) # Use all train",
    )

    s = s.replace(
        "train_idxs = np.arange(len(all_files))  # Use all train",
        "if os.environ.get('USE_OFFICIAL_VAL', '0') != '1':\n"
        "        train_idxs = np.arange(len(all_files))  # Use all train",
    )

    s = s.replace(
        "    train_idxs = []\n"
        "    for i in np.arange(len(all_files)):\n"
        "        train_idxs.append(i)\n"
        "        if file_classes[i, 1:].max():\n"
        "            train_idxs.append(i)\n"
        "    train_idxs = np.asarray(train_idxs)\n",
        "    train_idxs_aug = []\n"
        "    base_train_idxs = train_idxs if os.environ.get('USE_OFFICIAL_VAL', '0') == '1' else np.arange(len(all_files))\n"
        "    for i in base_train_idxs:\n"
        "        train_idxs_aug.append(i)\n"
        "        if file_classes[i, 1:].max():\n"
        "            train_idxs_aug.append(i)\n"
        "    train_idxs = np.asarray(train_idxs_aug)\n",
    )

    if path.name.startswith("tune"):
        s = re.sub(
            r"for\s+epoch\s+in\s+range\(\s*0\s*,\s*\d+\s*\):",
            "for epoch in range(0, int(__import__('os').environ.get('FT_EPOCHS', '10'))):",
            s,
        )
        s = re.sub(
            r"for\s+epoch\s+in\s+range\(\s*\d+\s*\):",
            "for epoch in range(int(__import__('os').environ.get('FT_EPOCHS', '10'))):",
            s,
        )

    if s != old:
        path.write_text(s)
        print("patched", path)


def main():
    print("Restoring original scripts from git before patching...")
    subprocess.run(["git", "checkout", "--"] + FILES, check=True)

    for name in FILES:
        p = Path(name)
        if p.exists():
            patch_file(p)

    print("Checking syntax...")
    subprocess.run(["python", "-m", "py_compile"] + FILES, check=True)
    print("Official split patch complete.")


if __name__ == "__main__":
    main()
