import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper

if __name__ == '__main__':
    # Important first step to prevent program from failing
    # where it tries to save files into the dir that doesn't exist
    # Creates temporary folder to save agent's training outputs 
    os.makedirs("tmp/td3", exist_ok=True)

    # Load joint velocity controller
    controller_config = suite.load_controller_config(
        default_controller="JOINT_VELOCITY"
    )

    print("Controller configuration loaded!")

    # Create Panda Door environment

    env = suite.make(
        env_name="Door",
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20
    )

    print("Robosuite environment created!")

    # Apply Gym wrapper
    env = GymWrapper(env)

    print("GymWrapper applied successfully!")

    # Reset environment
    obs = env.reset()

    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)

    env.close()
