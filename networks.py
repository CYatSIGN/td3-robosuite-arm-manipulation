import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module):

    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128,
                 name='critic', checkpoint_dir='tmp/td3', 
                 learning_rate=10e-3):
        # Initialise the critic network
        super(CriticNetwork,self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        #The critic receives both the robot observation and its action
        # Panda Door observation: 46 values
        # Panda actions: 8 values
        # Total critic input: 46 + 8 = 54 values
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)

        # Second hidden layer extracts more complex state-action relationships
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Output one Q-value representing the estimated future return
        self.q1 = nn.Linear(self.fc2_dims, out_features=1)


        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, 
                                     weight_decay=0.005)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        print(f"Created Critic Network on device: {self.device}")

        self.to(self.device)

    def forward(self, state, action):
        # Combine observations and actions into a single input
        action_value = self.fc1(T.cat(tensors=[state, action], dim=1))

        # Apply nonlinear activation to the first hidden layer
        action_value = F.relu(action_value)

        # Process the combined information through the second hidden layer
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        # Predict the expected discounted future reward
        q1 = self.q1(action_value)

        return q1

    def save_checkpoint(self):
        #Save the critic's learned parameters
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # Load saved network parameters onto the selected device
        checkpoint = T.load(
            self.checkpoint_file,
            map_location=self.device,
            weights_only=True
        )

        self.load_state_dict(checkpoint)

class ActorNetwork(nn.Module):

    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, 
                 learning_rate=10e-3, n_actions=2, name='actor',
                 checkpoint_dir='tmp/td3'):

        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        # Convert the 46-dimensional observation into 256 learned features
        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)

        # Further process the observation features
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Produce one control command for each action dimension
        # The Panda Door environment currently uses 8 action values
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        print(f"Created Actor Network on device: {self.device}")

        self.to(self.device)


    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # Restricts the actor's actions to [-1, 1], matching the
        # normalised action range of the Panda Door environment. 
        # May need to scale this if using another robot/controller with different limits
        x = T.tanh(self.output(x))

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # Load saved network parameters onto the selected device
        checkpoint = T.load(
            self.checkpoint_file,
            map_location=self.device,
            weights_only=True
        )

        self.load_state_dict(checkpoint)