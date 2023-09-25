#!/bin/bash
export ROOT="$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")"
. ${ROOT}/set_path.sh

export ROUTE_NAME="training"
export ROUTES=${LEADERBOARD_ROOT}/data/routes_${ROUTE_NAME}.xml
export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json

export PENALTY_STOP=0.8
export DENSE_TRAFFIC=""