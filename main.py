import os
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

# Train a Franka Panda robot on Robosuite's Door manipulation task
# The JOINT_VELOCITY controller accepts continuous control commands
# Each episode has a maximum horizon of 300 environment steps
# Reward shaping provides intermediate feedback during the task

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

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer_1_size = 256
    layer_2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                  tau=0.005, input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer_1_size=layer_1_size,
                  layer_2_size=layer_2_size, batch_size=batch_size)

    writer = SummaryWriter('logs/td3_continue_200')

    # Initial smoke test: verify the complete training loop
    n_games = 10000
    #best_score = 0
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate}\ncritic_learning_rate={critic_learning_rate}\nlayer_1_size={layer_1_size}\nlayer_2_size={layer_2_size}"

    # ***IMPORTANT***
    # Set True only when continuing from existing saved model weights
    load_checkpoint = False
    if load_checkpoint:
        #Restore previously saved network parameters
        agent.load_models()
    try:
        for i in range(n_games):
            # Start with reset when starting with Gym environment
            observation = env.reset()
            done = False
            score = 0

            while not done: 

                # Select a continuous action using the current observation
                action = agent.choose_actions(observation)

                # Execute the action and observe its consequences
                next_observation, reward, done, info = env.step(action)

                # Accumulate the episode's total reward
                score += reward

                # Store the transition for off-policy learning
                agent.remember(observation, action, reward, next_observation, done)

                # Update the TD3 networks when enough data is available
                agent.learn()

                # Use the latest observation for the next action
                observation = next_observation

            # Record each episode's total reward for training analysis
            writer.add_scalar(
                "Training/Episode_Reward",
                score,
                global_step=i+1
            )

            # Save network checkpoints every 10 completed episodes
            if (i + 1) % 10 == 0:
                agent.save_models()

            print(
                f"Episode {i + 1}/{n_games} | "
                f"Reward: {score:.3f} | "
                f"Buffer size: {min(agent.memory.mem_ctr, agent.memory.mem_size)} | "
                f"Learning steps: {agent.learn_step_ctr}"
            )

    finally:
        # Release TensorBoard and simulation process
        writer.close()
        env.close()
