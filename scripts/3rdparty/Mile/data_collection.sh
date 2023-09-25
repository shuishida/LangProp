#!/bin/bash
# Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license.

export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh


[[ -z "$SAVE_PATH" ]] && export SAVE_PATH=${DATA_ROOT}/random_routes_training/
[[ -z "$PORT" ]] && export PORT=2000
[[ -z "$TEST_SUITE" ]] && export TEST_SUITE=lb_data

data_collect () {
  python -u ${TEAM_CODE_ROOT}/data_collect.py --config-name data_collect carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh dataset_root=${SAVE_PATH} port=${PORT} test_suites=${TEST_SUITE}
}

# Remove checkpoint files
rm -f outputs/port_${PORT}_checkpoint.txt
rm -f outputs/port_${PORT}_wb_run_id.txt
rm -f outputs/port_${PORT}_ep_stat_buffer_*.json


# Resume benchmark in case carla crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  data_collect
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

echo "Bash script done."
