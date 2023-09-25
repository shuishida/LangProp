import argparse
from collections import deque
import matplotlib.pyplot as plt

import numpy as np


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self.n = n
        self.reset()

    def reset(self):
        self._window = deque([0 for _ in range(self.n)], maxlen=self.n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


if __name__ == "__main__":

    class Vehicle:
        def __init__(self, fps):
            self.speed = 0
            self.fps = fps

        def step(self, accel):
            self.speed += accel / self.fps
            return self.speed


    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--i", type=float, default=0.0)
    parser.add_argument("--d", type=float, default=0.0)
    parser.add_argument("--target", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=5.0)
    args = parser.parse_args()

    ts = np.arange(args.tmin, args.tmax, 1 / args.fps)

    target = (ts > 0).astype(float) * args.target

    vehicle = Vehicle(args.fps)
    controller = PIDController(args.p, args.i, args.d)

    speed = ts.copy()
    for i, t in enumerate(ts):
        if t < 0:
            speed[i] = 0
        else:
            error = args.target - speed[i-1]
            accel = controller.step(error)
            speed[i] = vehicle.step(accel)

    plt.plot(ts, target)
    plt.plot(ts, speed)
    plt.show()
