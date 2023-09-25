#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh

CKPT_DIR=${ROOT}/weights/tfuser

wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip -P ${CKPT_DIR}
unzip ${CKPT_DIR}/models_2022.zip -d ${CKPT_DIR}
