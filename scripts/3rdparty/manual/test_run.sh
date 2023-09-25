#!/usr/bin/env bash
# Parameterization settings. These will be explained in 2.2. Now simply copy them to run the test.
export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export ROUTES=${LEADERBOARD_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=${LEADERBOARD_ROOT}/leaderboard/autoagents/human_agent.py
export CHECKPOINT_ENDPOINT=${ROOT}/results/test_run.json
export CHALLENGE_TRACK_CODENAME=SENSORS

mkdir -p ${ROOT}/results

${ROOT}/leaderboard/scripts/run_evaluation.sh