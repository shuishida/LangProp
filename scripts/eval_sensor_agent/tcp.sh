#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

export TEAM_AGENT=${ROOT}/src/baselines/tcp/learner/tcp_agent.py
export TEAM_CONFIG=${ROOT}/weights/tcp/TCP.ckpt
[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/tcp/eval_$(date +%m%d_%H%M%S)

bash ${ROOT}/scripts/eval/base.sh
