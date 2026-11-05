import numpy as np

# Store the robot's previous experiences so TD3 can learn from them
# Every time the Panda robot takes an action, the environment returns information
# about what happened, TD3 stores this information as a transition
# (state, action, reward, next_state, done)


class ReplayBuffer():

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0

        # Store robot observations before taking actions
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32
                                     )

        #Store the observation received after executing each action
        self.next_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32
                                         )
        # PyTorch networks uses float32 by default, NumPy uses float64

        #Store the continuous actions executed by the Panda robot
        self.action_memory = np.zeros((self.mem_size, n_actions),
                                      dtype=np.float32
                                      )

        # Store the reward received after each action
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        # Record whether each transition ended an episode
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)


    def store_transition(self, state, action, reward, next_state, done):
        # Use a circular buffer so new transitions overwrite the
        # oldest experiences when the maximum capacity is reached
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # Count the total number of transitions collected
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        # Only sample from memory locations containing valid transitions
        max_mem = min(self.mem_ctr, self.mem_size)

        # Randomly select unique experiences from the populated buffer
        # This works as long as max_mem >= batch_size
        batch = np.random.choice(
            max_mem, 
            batch_size,
            replace=False
            )

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones

