import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import CriticNetwork, ActorNetwork

# Implement the TD3 agent, including action selection, critic training,
# actor training and target-network updates.

class Agent:
    """
    6 networks
    1. Actor - determines which action to take
    2. Critic 1 - estimates the Q-value of an action
    3. Critic 2 - independently estimates the Q-value
    4. Target actor - produces more stable actions for calculating training targets
    5. Target critic 1 - estimates future Q-values for training
    6. Target critic 2 - provides a second estimate of future Q-values
    """

    def __init__(self, actor_learning_rate, critic_learning_rate,
                 input_dims, tau, env, 
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=1000000, layer_1_size=256,
                 layer_2_size=128, batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_ctr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        # Create the networks
        # Create the actor, which maps observations to continuous actions
        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer_1_size,
                                  fc2_dims=layer_2_size, n_actions=n_actions,
                                  name='actor', learning_rate=actor_learning_rate)

        # Create two independent critics to reduce Q-value estimation
        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size,
                                       fc2_dims=layer_2_size, n_actions=n_actions,
                                       name='critic_1', 
                                       learning_rate=critic_learning_rate)

        self.critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size,
                                       fc2_dims=layer_2_size, n_actions=n_actions,
                                       name='critic_2', 
                                       learning_rate=critic_learning_rate)

        # Target networks are slowly updated copies of the main networks
        # They provide more stable learning targets during TD3 training
        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer_1_size,
                                  fc2_dims=layer_2_size, n_actions=n_actions,
                                  name='target_actor', learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size,
                                       fc2_dims=layer_2_size, n_actions=n_actions,
                                       name='target_critic_1', 
                                       learning_rate=critic_learning_rate)

        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size,
                                       fc2_dims=layer_2_size, n_actions=n_actions,
                                       name='target_critic_2', 
                                       learning_rate=critic_learning_rate)

        self.noise = noise

        # Initialise this function to copy the online networks' initial weights
        # into their correspoinding target networks
        self.update_network_parameters(tau=1)


    def choose_actions(self, observation, validation=False):
        """
        TD3 is an off-policy algorithm - it needs experiences stored in the
        replay buffer before it can learn effectively. During warmup, 
        the agent collects experiences using random actions rather than 
        relying on an untrained actor.

        e.g., if warmup = 1000, the first 1000 action selections during training
              use the warmup behaviour.
        """
        # Warmup: collect experienecs using random actions
        if self.time_step < self.warmup and not validation:
            action = np.random.uniform(
                low=self.min_action,
                high=self.max_action
            )

        else: # Convert the observation into a PyTorch tensor
            state = T.as_tensor(
                observation,
                dtype=T.float32,
                device=self.actor.device
            )

            # Disable gradient tracking because action selection
            # does not require updating the actor's parameters
            with T.no_grad():
                # Obtain the actor's state from the cpu and assign 
                # variable as numpy format
                action = self.actor(state).cpu().numpy()

            if not validation:
                # Add exploration noise during traning only
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.noise, 
                    size=self.n_actions
                )

                action = action + noise

        # Ensure all actions stay within the environment's limits
        action = np.clip(
            action,
            self.min_action,
            self.max_action
        )

        # Only count actions collected during training
        if not validation:
            self.time_step += 1

        return action.astype(np.float32)


    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    # Contains all the learning logic
    def learn(self):
        # Delay learnining until the replay buffer contains at least
        # 10 batches of experieneces, providing an initial data pool
        if self.memory.mem_ctr < self.batch_size * 10:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # Target values are used to train the critics
        # Do not calculate gradients through the target networks
        with T.no_grad():
            target_actions = self.target_actor(next_state)

            # Add clipped noise to smooth the target policy
            target_noise = T.randn_like(target_actions) * 0.2
            target_noise = T.clamp(
                target_noise,
                min=-0.5,
                max=0.5
            )

            target_actions = target_actions + target_noise

            # Keep target actions inside the valid action range
            target_actions = T.clamp(
                target_actions,
                min=-1.0,
                max=1.0
            )

            next_q1 = self.target_critic_1(
                next_state,
                target_actions
            )

            next_q2 = self.target_critic_2(
                next_state,
                target_actions
            )

            # TD3 uses the smaller of the two target-critic predictions
            # to reduce overestimation of future rewards
            next_critic_value = T.min(next_q1, next_q2)

            # Convert rewards and termination flags to column vectors
            reward_column = reward.view(-1, 1)
            done_column = done.float().view(-1, 1)

            # Bellman target: immediate reward plus discounted future value
            # No future value is added after a true terminal transition
            target = reward_column + (
                self.gamma * (1.0 - done_column)
                * next_critic_value
            )

        # Reset previously accumulated critic gradients before learning
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # Calculate the Q-values predicted by the two critics
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # Measure how far each critic's prediciton is from
        # the TD3 Bellman target
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        # Train both critics using their combined prediction error
        critic_loss = q1_loss + q2_loss

        # Calculate gradients for the critics parameters        
        critic_loss.backward()

        # Update critic parameters using their respective optimisers
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Count the number of completed critic-learning steps
        self.learn_step_ctr += 1

        # TD3 updates the actor less frequently than the criticsF
        # With an interval of 2, the actor updates every second step
        if self.learn_step_ctr % self.update_actor_iter != 0:
            return 

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        # By default, PyTorch optimisers minimise losses, however here neg is used
        # to maximise the estimated Q-value
        # Negating the mean Q-value turns maximisation into a
        # Minimisation problem suitable for PyTorch optimisers
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()

        self.actor.optimizer.step()
        self.update_network_parameters()


            
    def update_network_parameters(self, tau=None):

        # Use the agent's configured tau unless another value is supplied
        if tau is None:
            tau = self.tau

        # Pair each online network with its corresponding target network
        network_pairs = [
            (self.actor, self.target_actor),
            (self.critic_1, self.target_critic_1),
            (self.critic_2, self.target_critic_2)
        ]

        # Target updates do not require gradient tracking
        with T.no_grad():

            for online_networks, target_network in network_pairs:

                #Update each target parameters using Polyak averaging
                for online_param, target_param in zip(
                    online_networks.parameters(),
                    target_network.parameters()
                ):
                    # In-place PyTorch operations
                    # multplication
                    target_param.mul_(1.0 - tau)
                    # addition
                    target_param.add_(tau * online_param)


        
    def save_models(self):
        # Save all six neural networks so training can be resumed later
        # This saves model weights, not the replay buffer or optimiser states
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

        


    def load_models(self):
        # Restore all six neural networks from saved checkpoints
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()

        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

        print("Successfully loaded models!")

