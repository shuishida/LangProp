#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/baselines/interfuser_expert

export TEAM_AGENT=${ROOT}/src/baselines/interfuser/expert/auto_pilot.py
export TEAM_CONFIG=${ROOT}/src/baselines/interfuser/expert/config/eval_expert.yaml

bash ${ROOT}/scripts/data_collect/base.sh $@
