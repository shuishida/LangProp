#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/baselines/tcp_expert

export TEAM_AGENT=${ROOT}/src/baselines/tcp/expert/roach_ap_agent.py
export TEAM_CONFIG=${ROOT}/src/baselines/tcp/expert/config/config_agent.yaml

bash ${ROOT}/scripts/data_collect/base.sh $@
