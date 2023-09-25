#!/usr/bin/env bash
export ROOT=$(dirname "$(realpath -s "$BASH_SOURCE")")
export SCENARIO_RUNNER_ROOT=${ROOT}/scenario_runner
export LEADERBOARD_ROOT=${ROOT}/leaderboard

export CARLA910_ROOT=${ROOT}/carla/910
export CARLA910_EGG_PATH=${CARLA910_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

export CARLA911_ROOT=${ROOT}/carla/911
export CARLA911_EGG_PATH=${CARLA911_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg

export CARLA912_ROOT=${ROOT}/carla/912
export CARLA912_EGG_PATH=${CARLA912_ROOT}/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg

export CARLA913a_ROOT=${ROOT}/carla/913a
export CARLA913a_EGG_PATH=${CARLA913a_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

#export CARLA910_PYTHONPATH=${CARLA910_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${CARLA910_EGG_PATH}:${PYTHONPATH}
. ${ROOT}/.env
