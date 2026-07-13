from pathlib import Path

p = Path("xview/models/__init__.py")
s = p.read_text()

old = "from . import fpn, unet, hrnet, unetv2, cls, fpn_v2, hrnet_v2, fpn_v3, unetv3"

new = '''# Some source files from the original repo are not present in this extracted 3rd-place package.
# Import available model modules and skip missing optional ones.
import importlib

for _name in ["fpn", "unet", "hrnet", "unetv2", "cls", "fpn_v2", "hrnet_v2", "fpn_v3", "unetv3"]:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception as e:
        print(f"[WARN] Skipping optional model module {_name}: {e}")'''

if old not in s:
    print("Original import line not found. Showing first 40 lines:")
    for i, line in enumerate(s.splitlines()[:40], 1):
        print(f"{i}: {line}")
    raise SystemExit(1)

s = s.replace(old, new)
p.write_text(s)
print("Patched xview/models/__init__.py optional imports.")
