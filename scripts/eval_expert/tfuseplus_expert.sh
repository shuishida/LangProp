#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/baselines/tfuseplus_expert

export CHALLENGE_TRACK_CODENAME=MAP
export TEAM_AGENT=${ROOT}/src/baselines/tfuseplus/expert/data_agent.py

export DATAGEN=1

bash ${ROOT}/scripts/data_collect/base.sh $@
