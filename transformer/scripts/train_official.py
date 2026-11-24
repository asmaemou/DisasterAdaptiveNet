"""Train Gerard-style Strong Baseline on xBD's official Train+Tier3/Hold/Test partitions.

The split is patched in memory only. Original project files are never modified.
"""
from pathlib import Path
from types import SimpleNamespace
import runpy
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from utils import datasets, experiment_manager, parsers  # noqa: E402


def official_xbd_split(self):
    """Same official split used for RS-BDNet; keep native baseline oversampling."""
    train = list(self.metadata['train']['patches']) + list(self.metadata['tier3']['patches'])
    val = list(self.metadata['hold']['patches'])
    test = list(self.metadata['test']['patches'])

    if (len(train), len(val), len(test)) != (9168, 933, 933):
        raise RuntimeError(f"Unexpected source counts: {len(train)}, {len(val)}, {len(test)}")

    def ids(records):
        return {(str(s['event']), str(s['patch_id'])) for s in records}

    train_ids, val_ids, test_ids = ids(train), ids(val), ids(test)
    if len(train_ids) != 9168 or len(val_ids) != 933 or len(test_ids) != 933:
        raise RuntimeError("Duplicate event/patch_id values within a source partition")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Train/Hold/Test overlap detected; refusing to train")
    if self.cfg.DATASET.EXCLUDE_UNDAMAGED:
        raise RuntimeError("EXCLUDE_UNDAMAGED must be False for the complete official split")

    if self.cfg.DATASET.OVERSAMPLE_BUILDINGS:
        train += [s for s in train if any(s[f'cls_{c}'] > 0 for c in (2, 3, 4))]

    print(f"OFFICIAL_XBD_SPLIT: train_unique=9168 train_records={len(train)} "
          f"hold_val={len(val)} test={len(test)}", flush=True)
    return {'train': train, 'val': val, 'test': test}


datasets.AbstractxBDDataset.get_samples_legacy = official_xbd_split

if __name__ == '__main__':
    preflight = '--preflight' in sys.argv
    if preflight:
        sys.argv.remove('--preflight')

    args = parsers.argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    if cfg.MODEL.TYPE != 'strongbaseline' or cfg.DATASET.SPLIT != 'xview2':
        raise RuntimeError('Wrong configuration: expected strongbaseline + xview2')

    if preflight:
        train_ds = datasets.xBDDataset(cfg, run_type='train', disable_augmentations=True)
        val_ds = datasets.xBDDataset(cfg, run_type='val', disable_augmentations=True)
        test_ds = datasets.xBDDataset(cfg, run_type='test', disable_augmentations=True)
        unique = {(str(s['event']), str(s['patch_id'])) for s in train_ds.samples}
        assert (len(unique), len(val_ds), len(test_ds)) == (9168, 933, 933)
        print(f"PREFLIGHT PASSED: model={cfg.MODEL.TYPE}, "
              f"train_unique={len(unique)}, val={len(val_ds)}, test={len(test_ds)}")
    else:
        # Execute original training entry point with dataset split patched in memory.
        runpy.run_path(str(PROJECT / 'train_network.py'), run_name='__main__')
