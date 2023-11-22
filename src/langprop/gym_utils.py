import numpy as np
import cv2

from langprop.module import LPModule, RunConfig
from langprop.trainer import LPTrainer
from langprop.utils import set_timeout


def train_gym_model(env, model_root, model_name, run_config: RunConfig, batch_size: int = 2, epochs=10, timeout=10):
    env_data_loader = [[env] * batch_size]
    model = LPModule.from_template(name=model_name, root=model_root, return_function=True)
    trainer = GymTrainer(model, run_config, forward_timeout=timeout)
    trainer.fit(env_data_loader, epochs=epochs)
    return model


def eval_gym_policy(policy, env, render=False, timeout=10):
    with set_timeout(timeout):
        images = []
        obs, info = env.reset()
        max_steps = env.spec.max_episode_steps
        total_reward = 0
        for _ in range(max_steps):
            action = policy(*tuple(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if render:
                images.append(env.render())
            if terminated or truncated:
                break
        return images, total_reward


def make_video(filepath, images):
    if isinstance(images, list):
        images = np.stack(images)
    images = images[..., ::-1]
    B, H, W, C = images.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video = cv2.VideoWriter(str(filepath), fourcc, 20, (W, H))

    for img in images:
        video.write(img)

    video.release()


class GymTrainer(LPTrainer):
    def __init__(self, *args, forward_timeout=10, **kwargs):
        self.forward_timeout = forward_timeout
        super().__init__(*args, **kwargs)

    def preprocess(self, env):
        return (), None, env

    def score(self, policy, env) -> float:
        _, total_reward = eval_gym_policy(policy, env, render=False, timeout=self.forward_timeout)
        return float(total_reward)
