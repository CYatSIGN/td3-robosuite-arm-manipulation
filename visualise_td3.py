# Run this to visualise trained Franka robot

import time
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent


if __name__ == "__main__":

    # Load the same joint-velocity controller used during training
    controller_config = suite.load_controller_config(
        default_controller="JOINT_VELOCITY"
    )

    # Create the Panda Door environment with an interactive viewer
    env = suite.make(
        env_name="Door",
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=300,
        render_camera="frontview",
        reward_shaping=True,
        control_freq=20
    )

    # Convert Robosuite observation into the flattened format
    # expected by the trained TD3 neural networks
    env = GymWrapper(env)

    # Recreate the TD3 agent with the same network architecture
    # and observation/action dimensions used during training
    agent = Agent(
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        n_actions=env.action_space.shape[0],
        layer_1_size=256, 
        layer_2_size=128,
        batch_size=128
    )

    # Load the six network checkpoints saved during training
    agent.load_models()

    print("Trained TD3 model loaded!")

    try:
        # Reset the environment to its initial configuration
        observation = env.reset()

        # Render the initial state of the robot
        env.render()

        done = False
        episode_reward = 0

        while not done:

            # Use the trained actor to select an action
            # validation=True disables exploration noise
            action = agent.choose_actions(
                observation,
                validation=True
            )

            # Execute the action in the simulation
            next_observation, reward, done, info = env.step(action)

            # Accumulate reward for this evaluation episode
            episode_reward += reward

            # Display the updated robot and door position
            env.render()

            # Update the observation for the next action
            observation = next_observation

            # Slow down the display so the movement is easier to watch
            time.sleep(0.05)

        print(f"Episode reward: {episode_reward:.3f}")

        # Keep the final frame visible briefly
        time.sleep(10)

    finally:
        # Close the simulation and release graphics resources
        env.close()
    