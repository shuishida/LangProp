#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda env create -n $CONDA_ENVNAME -f environment.yaml
conda activate $CONDA_ENVNAME
easy_install ${CARLA911_EGG_PATH}

if [ ! -f "${SETTING_DIR}/weights/mile.ckpt" ]; then
  wget https://github.com/wayveai/mile/releases/download/v1.0/mile.ckpt -P ${SETTING_DIR}/weights
fi
