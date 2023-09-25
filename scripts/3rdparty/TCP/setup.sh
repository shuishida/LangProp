#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda env create -n $CONDA_ENVNAME -f ${TEAM_CODE_ROOT}/environment.yml
conda activate $CONDA_ENVNAME
easy_install ${CARLA910_EGG_PATH}
