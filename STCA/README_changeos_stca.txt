This package gives you a much closer-to-paper setup than the earlier simple STCA script:

- it uses the authors' trainable ChangeOS implementation from the torchange repo
- it uses the authors' official ChangeOS backbones/configs (r18/r34/r50/r101/swint)
- it keeps the paper's benchmark split protocol:
  source = xBD tier3
  unlabeled target adaptation = xBD train
  validation = xBD test
  final evaluation = xBD hold

Files:
- scripts/prepare_xview2_for_changeos.py
  Converts raw xView2 JSON labels into torchange-compatible PNG masks and symlinks images.
- scripts/train_changeos_stca.py
  Stage 1 supervised ChangeOS training, then Stage 2 STCA adaptation.
- sbatch/run_changeos_stca_xview2.sbatch
  Example Slurm job for your WSU cluster.

Important honesty note:
This is NOT the authors' released STCA training script, because I could not find a public STCA code release.
It DOES use the authors' trainable ChangeOS model implementation and official backbone configs from torchange.
The STCA loop is a faithful custom wrapper built from the STCA paper's algorithm description.

Expected raw xView2 layout:
  /path/to/xview2/
    tier3/
      images/*.png
      labels/*.json
    train/
      images/*.png
      labels/*.json
    test/
      images/*.png
      labels/*.json
    hold/
      images/*.png
      labels/*.json

Prepared layout created by the prep script:
  <prepared_root>/<split>/images   (symlink)
  <prepared_root>/<split>/targets  (*.png masks)

Results go to:
  <work_dir>/checkpoints/
  <work_dir>/metrics/

Default run in sbatch:
  backbone = r18
which matches the paper's common benchmark reference model.

To switch to Swin-T, change:
  --backbone r18
to:
  --backbone swint
and reduce batch size if needed.
