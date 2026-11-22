"""Template, NOT a working xView2 ensemble implementation.

Copy this file to transformer/scripts/ and implement load_full_ensemble and
predict_full_ensemble using the EXACT xView2 pipeline used in the paper.

Each winning solution has its own architecture(s), checkpoint structure, native
image dimensions, TTA, aggregation and postprocessing. Do not represent a
single backbone as a full ensemble. Once implemented, call:

python -m transformer.scripts.benchmark_rsbdnet_comment7 \
  --backend transformer.scripts.benchmark_xview2_adapter_template:build_backend \
  --ensemble-checkpoint-dir /path/to/THIS/ensemble/checkpoints \
  --output-dir /.../output/Reviewer2_Comment7_InferenceBench/ensemble1_<JOB_ID>

Input to predict(item): item['img'] is a CPU tensor (6,896,896) with ImageNet
normalization, and item['cond_id'] is a scalar CPU tensor. If your official
ensemble expects full 1024x1024 RGB imagery and native preprocessing, adapt
this module AND the input dataset source accordingly. Do not silently resize
preprocessed 896 inputs to 1024 while claiming a native-input comparison.
"""


def build_backend(device, args):
    raise NotImplementedError(
        'You must integrate the exact already-used xView2 ensemble loader, all '
        'constituent checkpoints, prediction aggregation, any TTA, and GPU '
        'postprocessing here. Please supply the ensemble inference scripts and '
        'weights directory for a defensible real comparison. This template '
        'deliberately generates no fake numbers.')
