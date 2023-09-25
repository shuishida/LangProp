#!/bin/bash
export SETTING_DIR=$(dirname "$(realpath -s "$BASH_SOURCE")")
export ROOT="$(dirname "$(dirname "$SETTING_DIR")")"
. ${ROOT}/settings.sh

[[ -z "$DATA_ROOT" ]] && export DATA_ROOT=${DATA_ROOT_BASE}/interfuser_full

SHORT_TOWNS=(07 10)

for TOWN in 01 02 03 04 05 06 07 10
do
    if [[ ${SHORT_TOWNS[*]} =~ $TOWN ]]
    then
        LENGTHS=(short tiny)
    else
        LENGTHS=(long short tiny)
    fi

    for LENGTH in ${LENGTHS[@]}
    do
        export ROUTE_NAME="town${TOWN}_${LENGTH}"
        export ROUTES=${ROOT}/dataset/routes/core/routes_${ROUTE_NAME}.xml
        bash ${ROOT}/scripts/eval_expert/interfuser_expert.sh $@
    done
done
