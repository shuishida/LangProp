#!/bin/bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh

export CHALLENGE_TRACK_CODENAME=MAP
export DEBUG_CHALLENGE=1
export REPETITIONS=1 # multiple evaluation runs
export DATAGEN=1
# export RESUME=True

export TEAM_AGENT=${TEAM_CODE_ROOT}/team_code_autopilot/data_agent.py
export TEAM_CONFIG=${TEAM_CODE_ROOT}/roach/config/config_agent.yaml
# data collection
export ROUTES=${ROOT}/leaderboard/data/routes_training.xml
export SCENARIOS=${ROOT}/leaderboard/data/all_towns_traffic_scenarios_public.json
export CHECKPOINT_ENDPOINT=${DATA_ROOT}/routes_training.json
export SAVE_PATH=${DATA_ROOT}/routes_training/

[[ -z "$PORT" ]] && export PORT=2000
[[ -z "$TM_PORT" ]] && export TM_PORT=8000


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}
