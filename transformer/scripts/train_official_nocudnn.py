"""Official xBD Strong Baseline training with cuDNN disabled.

Thin wrapper around transformer/scripts/train_official.py. Keeps the in-memory
Train+Tier3 / Hold / Test split and original training code intact.
"""
from pathlib import Path
import runpy
import torch

# Apply BEFORE loading the baseline or running the training entry point.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
assert not torch.backends.cudnn.enabled
print(f"NO_CUDNN_LAUNCHER: cuDNN enabled: {torch.backends.cudnn.enabled}", flush=True)

original = Path(__file__).resolve().with_name("train_official.py")
if not original.is_file():
    raise FileNotFoundError(f"Missing original launcher: {original}")

# Preserve sys.argv; original launcher implements --preflight and trains via
# runpy.run_path(train_network.py, run_name='__main__').
runpy.run_path(str(original), run_name="__main__")
