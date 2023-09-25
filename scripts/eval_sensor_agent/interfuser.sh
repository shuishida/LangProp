#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

export TEAM_AGENT=${ROOT}/src/baselines/interfuser/learner/interfuser_agent.py
export TEAM_CONFIG=${ROOT}/src/baselines/interfuser/learner/interfuser_config.py
[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/interfuser/eval_$(date +%m%d_%H%M%S)

bash ${ROOT}/scripts/eval/base.sh
