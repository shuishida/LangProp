#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

export CONDA_ENVNAME=tcp
export TEAM_CODE_ROOT=${ROOT}/3rdparty/TCP
export CARLA_ROOT=$CARLA910_ROOT
export SCENARIO_RUNNER_ROOT=${TEAM_CODE_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${TEAM_CODE_ROOT}/leaderboard
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${CARLA910_EGG_PATH}:${PYTHONPATH}
export DATA_ROOT=${DATA_ROOT_BASE}/tcp
mkdir -p ${DATA_ROOT}