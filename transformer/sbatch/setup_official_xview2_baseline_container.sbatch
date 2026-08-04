#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=xview2_setup
#SBATCH --output=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet/transformer/sbatch/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet
ENV_DIR=$PROJECT_ROOT/.conda/xview2_baseline_py37
OFFICIAL=$PROJECT_ROOT/third_party/xview2_baseline
cd "$PROJECT_ROOT"
mkdir -p transformer/sbatch/logs .conda third_party

if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniforge/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
else
  echo "ERROR: conda not found"
  exit 1
fi

echo "Creating rootless official xView2 environment: $ENV_DIR"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  # Resolve only Python and pip with Conda. Asking Conda to solve the complete
  # 2019 scientific stack against today's conda-forge metadata exceeded the
  # setup job's time limit. The pinned Linux wheels are installed below.
  "$CONDA_EXE" create -y -p "$ENV_DIR" \
    --override-channels -c defaults \
    python=3.7 pip=21.3.1
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
