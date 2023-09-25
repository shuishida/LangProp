#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

SCENARIO_NAME=$1

if [[ -z $SCENARIO_NAME ]] ; then
    echo 'Please specify scenarios: ll lr rl rr Scenario1 Scenario3 Scenario4 Scenario7 Scenario8 Scenario9 Scenario10.'
    exit 1
fi

[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/tfuse_full

[[ -z "$WEATHER" ]] && export WEATHER=$3

if [ -z "RUN_NAME" ]; then
  export RUN_NAME=data_$(date +%m%d_%H%M%S)_$2_scenario_${SCENARIO_NAME}
  if [ -n "$WEATHER" ]; then
    echo "Using weather ${WEATHER}"
    export RUN_NAME=${RUN_NAME}_weather-$WEATHER
  else
    echo "Weather hasn't been set. Using default weather."
  fi
fi

LANE_CHANGE_SCENARIO=(ll lr rl rr)

echo $SCENARIO_NAME
for ROUTES in ${ROOT}/dataset/routes/${SCENARIO_NAME}/*.xml; do
    echo $ROUTES
    export ROUTES
    export ROUTE_NAME=$(basename -- ${ROUTES%.*})
    if [[ ${LANE_CHANGE_SCENARIO[*]} =~ $SCENARIO_NAME ]]; then
        SCENARIOS=${ROOT}/dataset/scenarios/no_scenarios.json
    else
        SCENARIOS=${ROOT}/dataset/scenarios/${SCENARIO_NAME}/${ROUTE_NAME}.json
    fi
    bash ${ROOT}/scripts/eval_expert/tfuse_expert.sh
done
