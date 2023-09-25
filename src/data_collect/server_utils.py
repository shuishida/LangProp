"""Adapted from https://github.com/zhejz/carla-roach CC-BY-NC 4.0 license."""

import subprocess
import os
import time
import logging
log = logging.getLogger(__name__)


class CarlaServerManager:
    def __init__(self, carla_path="", port=2000, carla_fps=20, gpu=None, t_sleep=2):
        self.carla_path = carla_path or os.environ.get('CARLA_PATH')
        assert self.carla_path, "CARLA PATH not set."
        self.gpu = gpu if gpu is not None else os.environ.get('CUDA_VISIBLE_DEVICES')
        self.port = port
        self.carla_fps = carla_fps
        self.t_sleep = t_sleep

    def start(self):
        cmd = f'CUDA_VISIBLE_DEVICES={self.gpu} bash {self.carla_path} -fps={self.carla_fps} --world-port={self.port}'
        log.info(cmd)
        server_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        time.sleep(self.t_sleep)

    def stop(self):
        # This one only kills processes linked to a certain port
        kill_process = subprocess.Popen(f'fuser -k {self.port}/tcp', shell=True)
        log.info(f"Killed Carla Servers on port {self.port}!")
        kill_process.wait()
        time.sleep(self.t_sleep)

    def __del__(self):
        self.stop()
