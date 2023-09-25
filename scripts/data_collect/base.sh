#!/bin/bash
export CUDA_VISIBLE_DEVICES=$GPU

[[ -z "$CHALLENGE_TRACK_CODENAME" ]] && export CHALLENGE_TRACK_CODENAME=SENSORS
[[ -z "$DEBUG_CHALLENGE" ]] && export DEBUG_CHALLENGE=0
[[ -z "$REPETITIONS" ]] && export REPETITIONS=1

[[ -z "$ROUTE_NAME" ]] && export ROUTE_NAME="training"
[[ -z "$ROUTES" ]] && export ROUTES=${LEADERBOARD_ROOT}/data/routes_${ROUTE_NAME}.xml
[[ -z "$SCENARIOS" ]] && export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json

[[ -z "$TIMEOUT" ]] && export TIMEOUT=60
[[ -z "$PORT" ]] && export PORT=2000

[[ -z "$WEATHER" ]] && export WEATHER=$2

if [ -z "$RUN_NAME" ]; then
  export RUN_NAME=data_$(date +%m%d_%H%M%S)_$1
  if [ -n "$WEATHER" ]; then
    echo "Using weather ${WEATHER}"
    export RUN_NAME=${RUN_NAME}_weather-$WEATHER
  else
    echo "Weather hasn't been set. Using default weather."
  fi
fi

echo "RUN_NAME: ${RUN_NAME}"

[[ -z "$CHECKPOINT_ENDPOINT" ]] && export CHECKPOINT_ENDPOINT=${DATA_ROOT}/${ROUTE_NAME}/${RUN_NAME}/summary.json
[[ -z "$SAVE_PATH" ]] && export SAVE_PATH=${DATA_ROOT}/${ROUTE_NAME}/${RUN_NAME}

mkdir -p ${SAVE_PATH}

#export JUST_ONE=true

python3 ${ROOT}/src/data_collect/data_collector.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=True \
--port=${PORT} \
--weather=${WEATHER} \
--just-one=${JUST_ONE} \
--timeout=${TIMEOUT}
