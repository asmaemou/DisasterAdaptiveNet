from pathlib import Path
import re

# -----------------------------
# Patch xview/inference.py
# -----------------------------
p = Path("xview/inference.py")
s = p.read_text()

backup = p.with_suffix(".py.bak_full_patch")
if not backup.exists():
    backup.write_text(s)

tile_block = '''try:
    from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
except Exception:
    from pytorch_toolbelt.inference.tiles import ImageSlicer
    CudaTileMerger = None
'''

# Fix broken or old CudaTileMerger import block.
s = re.sub(
    r'''try:\s*\n\s*from pytorch_toolbelt\.inference\.tiles import CudaTileMerger, ImageSlicer\s*\nexcept Exception:\s*\n(?:[ \t].*\n)*?[ \t]*CudaTileMerger = None\s*\n''',
    tile_block,
    s,
    count=1,
    flags=re.S,
)

s = s.replace(
'''try:
from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
''',
tile_block
)

s = s.replace(
    "from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer\n",
    tile_block
)

# PyTorch 2.6+ compatibility
s = s.replace(
    'torch.load(model_checkpoint, map_location="cpu")',
    'torch.load(model_checkpoint, map_location="cpu", weights_only=False)'
)

p.write_text(s)
print("Patched xview/inference.py")


# -----------------------------
# Patch xview/dataset.py
# -----------------------------
p = Path("xview/dataset.py")
s = p.read_text()

backup = p.with_suffix(".py.bak_full_patch")
if not backup.exists():
    backup.write_text(s)

old = "from pytorch_toolbelt.utils.catalyst import PseudolabelDatasetMixin"
new = '''try:
    from pytorch_toolbelt.utils.catalyst import PseudolabelDatasetMixin
except Exception:
    class PseudolabelDatasetMixin:
        pass'''

if old in s:
    s = s.replace(old, new)

p.write_text(s)
print("Patched xview/dataset.py")


# -----------------------------
# Patch xview/metric.py
# -----------------------------
p = Path("xview/metric.py")
s = p.read_text()

backup = p.with_suffix(".py.bak_full_patch")
if not backup.exists():
    backup.write_text(s)

old = "from catalyst.dl import Callback, RunnerState, CallbackOrder"
new = '''try:
    from catalyst.dl import Callback, RunnerState, CallbackOrder
except Exception:
    class Callback:
        def __init__(self, *args, **kwargs):
            pass
    class RunnerState:
        pass
    class CallbackOrder:
        Metric = 0'''

if old in s:
    s = s.replace(old, new)

p.write_text(s)
print("Patched xview/metric.py")


# -----------------------------
# Patch xview/train_utils.py
# -----------------------------
p = Path("xview/train_utils.py")
s = p.read_text()

backup = p.with_suffix(".py.bak_full_patch")
if not backup.exists():
    backup.write_text(s)

old = "from catalyst.dl import CriterionCallback"
new = '''try:
    from catalyst.dl import CriterionCallback
except Exception:
    class CriterionCallback:
        pass'''

if old in s:
    s = s.replace(old, new)

p.write_text(s)
print("Patched xview/train_utils.py")


# -----------------------------
# Patch xview/model_wrapper.py
# -----------------------------
p = Path("xview/model_wrapper.py")
s = p.read_text()

backup = p.with_suffix(".py.bak_full_patch")
if not backup.exists():
    backup.write_text(s)

old1 = "from catalyst.dl import CallbackOrder, logger, RunnerState, Callback"
new1 = '''try:
    from catalyst.dl import CallbackOrder, logger, RunnerState, Callback
except Exception:
    class Callback:
        def __init__(self, *args, **kwargs):
            pass
    class RunnerState:
        pass
    class CallbackOrder:
        Metric = 0
    class _Logger:
        def info(self, *args, **kwargs):
            pass
    logger = _Logger()'''

if old1 in s:
    s = s.replace(old1, new1)

old2 = "from catalyst.dl.callbacks.criterion import _add_loss_to_state, CriterionCallback"
new2 = '''try:
    from catalyst.dl.callbacks.criterion import _add_loss_to_state, CriterionCallback
except Exception:
    def _add_loss_to_state(*args, **kwargs):
        return None
    class CriterionCallback:
        pass'''

if old2 in s:
    s = s.replace(old2, new2)

p.write_text(s)
print("Patched xview/model_wrapper.py")

print("Finished all compatibility patches.")