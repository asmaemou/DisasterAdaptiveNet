#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=gerard_bw_nocudnn
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/backward_nocudnn_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/output/reviewer3_gerard_logs/backward_nocudnn_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
source /homes/j244s673/miniforge3/etc/profile.d/conda.sh
conda activate disasteradaptivenet_cu128
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# Only one training batch; no results or checkpoint files are written.
timeout -k 5s 780s python -u - <<'PY'
import time
import faulthandler
from pathlib import Path
from types import SimpleNamespace

faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)

import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from transformer.scripts import train_official  # patches official xBD split in memory
from utils import datasets, experiment_manager, losses, models

root = Path.cwd()
args = SimpleNamespace(
    config_file='strong_baseline_xview2',
    output_dir=str(root / 'output/reviewer3_gerard_preflight'),
    dataset_dir='/homes/j244s673/documents/wsu/phd/xview2',
    opts=['DATALOADER.NUM_WORKER', '0', 'LOG_FREQ', '10'],
)
cfg = experiment_manager.setup_cfg(args)

def say(message):
    print('BACKWARD_DIAG:', message, flush=True)

say(f'cuDNN enabled: {torch.backends.cudnn.enabled}')
say('Building model')
net = models.create_network(cfg).to('cuda')
net.train()
say('Model ready')

ds = datasets.xBDDataset(cfg, run_type='train')
loader = DataLoader(ds, batch_size=cfg.TRAINER.BATCH_SIZE,
                    shuffle=True, num_workers=0, drop_last=True,
                    pin_memory=False)
say('Loading first complete batch')
batch = next(iter(loader))
x, msk = batch['img'].to('cuda'), batch['msk'].to('cuda')
say(f'Batch on GPU: {tuple(x.shape)}')

optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LEARNING_RATE,
                              weight_decay=0.01)
criterion = losses.ComboLoss(weights=cfg.TRAINER.LOSS.WEIGHTS)
class_weights = losses.loss_class_weights(cfg.TRAINER.LOSS.CLASS_WEIGHTS,
                                         ds.get_class_counts())

say('Before forward')
t0 = time.monotonic()
logits = net(x)
torch.cuda.synchronize()
say(f'Forward passed in {time.monotonic() - t0:.2f}s')

say('Before original training loss')
loss = criterion(logits[:, 0], msk[:, 0].long()) * class_weights[0]
for c in range(1, logits.size(1)):
    loss = loss + criterion(logits[:, c], msk[:, c].long()) * class_weights[c]
torch.cuda.synchronize()
say(f'Loss passed: {loss.item():.6f}')

say('Before backward')
loss.backward()
torch.cuda.synchronize()
say('Backward passed')

say('Before optimizer step')
optimizer.step()
torch.cuda.synchronize()
say('Optimizer step passed')
say('ONE-BATCH TRAINING DIAGNOSTIC PASSED')
PY
