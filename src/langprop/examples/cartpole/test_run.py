from datetime import datetime
from pathlib import Path
import gymnasium as gym

from langprop.gym_utils import eval_gym_policy, make_video, train_gym_model
from langprop.module import RunConfig


if __name__ == "__main__":
    model_name = "solve_cartpole"
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + model_name
    run_config = RunConfig(run_name=run_name, n_responses=3, n_top_choices=3, max_keep=6)
    model = train_gym_model(env, Path(__file__).parent, model_name, run_config)
    images, total_reward = eval_gym_policy(model(), env, render=True)
    make_video(Path(__file__).parent / 'output.mp4', images)
