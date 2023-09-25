#!/bin/bash

if [ -z "$CARLA_ROOT" ]
then
    echo "Error $CARLA_ROOT is empty. Set \$CARLA_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$SCENARIO_RUNNER_ROOT" ]
then echo "Error $SCENARIO_RUNNER_ROOT is empty. Set \$SCENARIO_RUNNER_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$LEADERBOARD_ROOT" ]
then echo "Error $LEADERBOARD_ROOT is empty. Set \$LEADERBOARD_ROOT as an environment variable first."
    exit 1
fi

mkdir -p .tmp

[[ -z "$TEAM_CODE_ROOT" ]] && export TEAM_CODE_ROOT=${ROOT}/src

cp -fr ${CARLA_ROOT}/PythonAPI  .tmp
mv .tmp/PythonAPI/carla/dist/carla*-py2*.egg .tmp/PythonAPI/carla/dist/carla-leaderboard-py2.7.egg
mv .tmp/PythonAPI/carla/dist/carla*-py3*.egg .tmp/PythonAPI/carla/dist/carla-leaderboard-py3x.egg

cp -fr ${SCENARIO_RUNNER_ROOT}/ .tmp
cp -fr ${LEADERBOARD_ROOT}/ .tmp
cp -fr ${TEAM_CODE_ROOT}/ .tmp/team_code
cp -fr ${ROOT}/weights/ .tmp/team_code/weights

[[ -z "$DOCKER_NAME" ]] && export DOCKER_NAME=leaderboard-user

# build docker image
docker build --force-rm -t ${DOCKER_NAME} -f ${ROOT}/scripts/submit/Dockerfile.master .

rm -fr .tmp
