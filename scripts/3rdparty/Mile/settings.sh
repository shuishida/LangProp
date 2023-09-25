#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

export CONDA_ENVNAME=mile
export TEAM_CODE_ROOT=${ROOT}/3rdparty/Mile
export CARLA_ROOT=$CARLA911_ROOT
#export SCENARIO_RUNNER_ROOT=${TEAM_CODE_ROOT}/scenario_runner
#export LEADERBOARD_ROOT=${TEAM_CODE_ROOT}/leaderboard
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${CARLA911_EGG_PATH}:${PYTHONPATH}
export DATA_ROOT=${DATA_ROOT_BASE}/mile
mkdir -p ${DATA_ROOT}

export CHECKPOINT_PATH=$SETTING_DIR/weights/mile.ckpt