#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda env create -n $CONDA_ENVNAME -f ${TEAM_CODE_ROOT}/environment.yml
conda activate $CONDA_ENVNAME
easy_install ${CARLA910_EGG_PATH}

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install mmcv-full==1.5.3