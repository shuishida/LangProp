#!/bin/bash
[[ -z "$DOCKER_NAMET" ]] && export DOCKER_NAME=leaderboard-user

alpha benchmark:submit  --split 3 ${DOCKER_NAME}:latest