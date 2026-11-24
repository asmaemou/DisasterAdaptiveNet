#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --job-name=gerard_cudnn_diag
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/cudnn_diag_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/cudnn_diag_%j.err

set -euo pipefail

cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet

source /homes/j244s673/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128

export PYTHONUNBUFFERED=1

for MODE in off on; do
    echo "===== CUDNN $MODE ====="

    set +e
    timeout -k 5s 240s python -u - "$MODE" <<'PY'
import sys
import time
import torch
from types import SimpleNamespace

mode = sys.argv[1]

torch.backends.cudnn.enabled = (mode == "on")
torch.backends.cudnn.benchmark = False
torch.set_num_threads(4)

print("PyTorch:", torch.__version__, flush=True)
print("CUDA:", torch.version.cuda, flush=True)
print("cuDNN:", torch.backends.cudnn.version(), flush=True)
print("cuDNN enabled:", torch.backends.cudnn.enabled, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

def check(label, fn):
    print("BEFORE:", label, flush=True)
    t = time.monotonic()
    result = fn()
    torch.cuda.synchronize()
    print(
        "PASSED:", label,
        f"{time.monotonic()-t:.2f}s",
        flush=True
    )
    return result

# Basic CUDA convolution test.
conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).cuda()
x_small = torch.randn(1, 3, 128, 128, device="cuda")

check("simple GPU convolution", lambda: conv(x_small))

# Original Strong Baseline architecture.
from utils.models import StrongBaselineNet

cfg = SimpleNamespace(
    MODEL=SimpleNamespace(OUT_CHANNELS=5),
    DATASET=SimpleNamespace(CONDITIONING_KEY={})
)

model = StrongBaselineNet(cfg).cuda()
model.train()

x = torch.randn(1, 6, 128, 128, device="cuda")

check("StrongBaseline batch=1 size=128", lambda: model(x))

del x

# Test original training dimensions.
x = torch.randn(16, 6, 512, 512, device="cuda")

check("StrongBaseline batch=16 size=512", lambda: model(x))

print("ALL TESTS PASSED: CUDNN", mode, flush=True)
PY

    RC=$?
    set -e

    echo "CUDNN $MODE exit code: $RC"
done
