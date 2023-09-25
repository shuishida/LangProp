#!/usr/bin/env bash
export SETTING_DIR=$(dirname "$(dirname "$(realpath -s "$BASH_SOURCE")")")
. ${SETTING_DIR}/settings.sh

mkdir -p ${DATA_ROOT}/data
cd ${DATA_ROOT}/data

# Download license file
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/LICENSE.txt

# Download 2022 dataset
for scenario in left ll lr right rl rr s1 s3 s4 s7 s8 s9 s10
do
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/2022_data/${scenario}.zip
	unzip -q ${scenario}.zip
	rm ${scenario}.zip
done