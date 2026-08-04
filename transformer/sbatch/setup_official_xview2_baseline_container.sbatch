#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --job-name=xview2_setup
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
ENV_DIR=$PROJECT_ROOT/.conda/xview2_baseline_py37_v2
OFFICIAL=$PROJECT_ROOT/third_party/xview2_baseline
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs .conda third_party

echo "Creating rootless official xView2 environment: $ENV_DIR"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  # Use a fixed Python 3.7 Miniconda distribution. This bypasses Conda's
  # repodata solver, which repeatedly stalled for hours on the HPC filesystem.
  LOCAL_JOB_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
  INSTALLER=$LOCAL_JOB_TMP/miniconda_py37.sh
  mkdir -p "$LOCAL_JOB_TMP"
  curl -fL --retry 3 \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh \
    -o "$INSTALLER"
  bash "$INSTALLER" -b -p "$ENV_DIR"
  "$ENV_DIR/bin/python" -m pip install --no-cache-dir --upgrade pip==21.3.1
fi

"$ENV_DIR/bin/pip" install --no-cache-dir \
  numpy==1.18.5 scipy==1.4.1 h5py==2.10.0 pandas==1.0.5 \
  scikit-learn==0.23.2 pillow==7.2.0 \
  tensorflow==1.15.5 keras==2.2.4 chainer==7.8.1 \
  matplotlib==3.2.2 tqdm scikit-image==0.16.2 \
  opencv-python-headless==4.2.0.34 imgaug==0.4.0 \
  shapely==1.7.1 geopandas==0.8.1 imantics \
  simplification==0.5.7 tensorboardX

if [ ! -d "$OFFICIAL/.git" ]; then
  git clone --branch v1.0 --depth 1 \
    https://github.com/DIUx-xView/xView2_baseline.git "$OFFICIAL"
fi
mkdir -p "$OFFICIAL/weights" "$HOME/.keras/models"
if [ ! -s "$OFFICIAL/weights/localization.h5" ]; then
  curl -fL https://github.com/DIUx-xView/xView2_baseline/releases/download/v1.0/localization.h5 \
    -o "$OFFICIAL/weights/localization.h5"
fi
if [ ! -s "$OFFICIAL/weights/classification.hdf5" ]; then
  curl -fL https://github.com/DIUx-xView/xView2_baseline/releases/download/v1.0/classification.hdf5 \
    -o "$OFFICIAL/weights/classification.hdf5"
fi
if [ ! -s "$HOME/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5" ]; then
  curl -fL https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -o "$HOME/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
fi

"$ENV_DIR/bin/python" -c 'import tensorflow, keras, chainer; print("TensorFlow", tensorflow.__version__, "Keras", keras.__version__, "Chainer", chainer.__version__)'
test -s "$OFFICIAL/weights/localization.h5"
test -s "$OFFICIAL/weights/classification.hdf5"
echo "READY: environment=$ENV_DIR repository=$OFFICIAL"
