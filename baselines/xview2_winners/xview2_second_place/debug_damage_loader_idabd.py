import sys
import time
import faulthandler
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)

print("Starting damage DataLoader debug...", flush=True)

# Reuse train.py imports/classes by running only the loading part is hard,
# so first we simply import train.py dependencies to see where it freezes.
import train

print("Imported train.py successfully.", flush=True)
print("Now run the real train.py command with faulthandler if needed.", flush=True)
