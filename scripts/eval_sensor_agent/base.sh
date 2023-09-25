#!/bin/bash
export PYTHONPATH=${CARLA910_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${CARLA910_EGG_PATH}:${ROOT}/src:${PYTHONPATH}

[[ -z "$CHALLENGE_TRACK_CODENAME" ]] && export CHALLENGE_TRACK_CODENAME=SENSORS
[[ -z "$DEBUG_CHALLENGE" ]] && export DEBUG_CHALLENGE=0
[[ -z "$REPETITIONS" ]] && export REPETITIONS=1

[[ -z "$SCENARIOS" ]] && export SCENARIOS=${ROOT}/dataset/benchmark/eval_scenarios.json

mkdir -p ${DATA_ROOT}/

[[ -z "$PORT" ]] && export PORT=2000
[[ -z "$TM_PORT" ]] && export TM_PORT=8000

[[ -z "$ROUTES" ]] && export ROUTES=${ROOT}/dataset/benchmark/longest6.xml

[[ -z "$CHECKPOINT_ENDPOINT" ]] && export CHECKPOINT_ENDPOINT=${DATA_ROOT}/longest6.json
[[ -z "$SAVE_PATH" ]] && export SAVE_PATH=${DATA_ROOT}/longest6/

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}
