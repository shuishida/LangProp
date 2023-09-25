#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh

cd $TEAM_CODE_ROOT

source $(conda info --base)/etc/profile.d/conda.sh
conda create -n $CONDA_ENVNAME python=3.7 -y
conda activate $CONDA_ENVNAME
pip install -r requirements.txt
cd interfuser

pip install setuptools==49.6.0
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
pip install -U pip setuptools
python setup.py develop

cd ..
cd dataset
python init_dir.py
cd ..
cd data_collection
python generate_yamls.py # You can modify fps, waypoints distribution strength ...

# If you don't need all weather, you can modify the following script
python generate_bashs.py
python generate_batch_collect.py
cd ..