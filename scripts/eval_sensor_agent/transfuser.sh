#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

export TEAM_AGENT=${ROOT}/src/baselines/transfuser/learner/submission_agent.py
export TEAM_CONFIG=${ROOT}/weights/tfuser/models_2022/transfuser
[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/tfuse/eval_$(date +%m%d_%H%M%S)

bash ${ROOT}/scripts/eval/base.sh
