#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/langprop/expert

export TEAM_AGENT=${ROOT}/src/expert/expert_agent.py
export TEAM_CONFIG=${ROOT}/src/expert/config/expert.yaml

bash ${ROOT}/scripts/data_collect/base.sh $@
