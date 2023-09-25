#!/bin/bash
export ROOT="$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")"
. ${ROOT}/set_path.sh

export ROUTE_NAME="longest6"
export ROUTES=${ROOT}/dataset/benchmark/longest6.xml
export SCENARIOS=${ROOT}/dataset/benchmark/eval_scenarios.json

export PENALTY_STOP=1.0
export DENSE_TRAFFIC="true"
