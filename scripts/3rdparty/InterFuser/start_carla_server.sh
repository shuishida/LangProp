#!/usr/bin/env bash
. settings.sh

CUDA_VISIBLE_DEVICES=0 ${CARLA_PATH} --world-port=20000 -opengl &
CUDA_VISIBLE_DEVICES=1 ${CARLA_PATH} --world-port=20002 -opengl &
CUDA_VISIBLE_DEVICES=2 ${CARLA_PATH} --world-port=20004 -opengl &
CUDA_VISIBLE_DEVICES=2 ${CARLA_PATH} --world-port=20006 -opengl &
CUDA_VISIBLE_DEVICES=3 ${CARLA_PATH} --world-port=20008 -opengl &
CUDA_VISIBLE_DEVICES=3 ${CARLA_PATH} --world-port=20010 -opengl &
CUDA_VISIBLE_DEVICES=4 ${CARLA_PATH} --world-port=20012 -opengl &
CUDA_VISIBLE_DEVICES=4 ${CARLA_PATH} --world-port=20014 -opengl &
CUDA_VISIBLE_DEVICES=5 ${CARLA_PATH} --world-port=20016 -opengl &
CUDA_VISIBLE_DEVICES=5 ${CARLA_PATH} --world-port=20018 -opengl &
CUDA_VISIBLE_DEVICES=6 ${CARLA_PATH} --world-port=20020 -opengl &
CUDA_VISIBLE_DEVICES=6 ${CARLA_PATH} --world-port=20022 -opengl &
CUDA_VISIBLE_DEVICES=7 ${CARLA_PATH} --world-port=20024 -opengl &
CUDA_VISIBLE_DEVICES=7 ${CARLA_PATH} --world-port=20026 -opengl &