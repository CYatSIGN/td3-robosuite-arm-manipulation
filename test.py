import robosuite as suite
from robosuite.wrappers import GymWrapper

# Check that Robosuite environment works before using it for RL

"""
test.py checks:
1. The Panda Door simulation resets successfully
2. The action space accepts valid actions
3. The simulation advances
4. Robosuite returns observations and rewards
5. The environment can be closed

"""
if __name__ == '__main__':

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
        render_camera="frontview",
        reward_shaping=True,
        control_freq=20
    )

    print("Robosuite environment created!")

    # Apply Gym wrapper
    env = GymWrapper(env)

    print("GymWrapper applied successfully!")

    try: 
        # Reset the robot and obtain its initial observation
        observation = env.reset()

        # Check the dimensions expected by the TD3 agent
        print("Observation shape:", observation.shape)
        print("Action space:", env.action_space)

        # Execute a few random actions to verify the simulation
        for step in range(10):

            # Sample an action within the environment's valid limits
            action = env.action_space.sample()

            # Apply the action and obtain the resulting transition
            observation, reward, done, info = env.step(action)

            print(
                f"Step {step + 1}: "
                f"Reward = {reward:.4f}"
            )

            # Stop if the episode has ended
            if done:
                 break

        print("Environment test completed!")

    finally:
        # Release the simulation and rendering resources
        env.close()
