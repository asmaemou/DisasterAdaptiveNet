from pathlib import Path

p = Path("xview/inference.py")
s = p.read_text()

backup = Path("xview/inference.py.bak_fix_import")
if not backup.exists():
    backup.write_text(s)

lines = s.splitlines()

# Find the pytorch_toolbelt tile import block, even if it is badly indented.
idx = None
for i, line in enumerate(lines):
    if "pytorch_toolbelt.inference.tiles" in line:
        idx = i
        break

if idx is None:
    raise SystemExit("ERROR: could not find pytorch_toolbelt.inference.tiles import line")

start = idx
if idx > 0 and lines[idx - 1].strip() == "try:":
    start = idx - 1

end = idx + 1
for j in range(idx + 1, min(len(lines), idx + 15)):
    if "CudaTileMerger = None" in lines[j]:
        end = j + 1
        break

clean_block = [
    "try:",
    "    from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer",
    "except Exception:",
    "    from pytorch_toolbelt.inference.tiles import ImageSlicer",
    "    CudaTileMerger = None",
]

new_lines = lines[:start] + clean_block + lines[end:]
p.write_text("\n".join(new_lines) + "\n")

print("Fixed xview/inference.py import block.")
