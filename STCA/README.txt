Practical STCA-style xView2 experiment package.

Included:
- scripts/xview2_stca_lib.py
- scripts/train_stca_xview2.py
- scripts/eval_stca_xview2.py
- sbatch/run_stca_xview2.sh

Important:
This is a practical STCA-style experiment runner for your HPC.
It follows the paper's split protocol:
- source = tier3
- target adaptation = train
- validation = test
- final evaluation = hold

It is not an exact reproduction of the paper's original ChangeOS code.
It uses a simpler shared-encoder segmentation model plus an STCA pair-classification adaptation stage.
