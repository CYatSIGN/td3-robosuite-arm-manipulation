# Robotic Arm Manipulation Project

Training **Franka robotic arm** to open a door in simulation using **Twin Delayed Deep Deterministic Policy Gradient (TD3)**.

## Project Overview
This project explores reinforcement learning (RL) for robotics manipulation in **Robosuite**, using **MuJoCo** for physics simulation

The objective is to train a Franka Panda robot to interact with a door handle and open the door using continous joint-velocity commands. 

TD3 is an off-policy reinforcement learning algorithm designed for continous action spaces. It uses an actor network to select actions and two critics networks to estimate their value. 

## TD3 Implementation and Features
The TD3 implementation consists of six neural network:

1. **Actor Network** - Predicts continuous actions from the current observation
2. **Critic Network 1** - Estimates the expected future reward of a state-action pair. 
3. **Critic Network 2** - Provides an independent Q-value estimate to reduce overestimation.
4. **Target Actor** - Generates target actions during training. 
5. **Target Critic 1** - Estimates the future Q-values for the Bellman target.
6. **Target Critic 2** - Provides a second estimate for clipped double Q-learning. 

The implementation also includes an experience replay buffer that stores previous transitions for off-policy learning. 

### Key Features
> Twin Critics: Uses the minimum of two target Q-value estimates to reduce overestimation. 
> Delayed Policy Updates: Updates the actor less frequently than the critics.
> Target Policy Smoothing: Adds clipped Gaussian noise to target actions. 
> Experience Replay: Samples previously collected transitions for training. 
> Soft Target Updates: Uses Polyak averaging to update target-network parameters gradually. 

## Main Technologies
- Python
- MuJoCo
- Robosuite
- TD3 Algorithm
- Numpy
- PyTorch
- TensorBoard

## Results and Discussion



https://github.com/user-attachments/assets/0870b5ad-6d4b-4624-9490-30bbcc60cd9c


![Training Chart](training_chart.png)





## Installaton and Simulation Setup
To run the trained agent model on MuJoCo to watch the Frank robot arm opening the door, follow the steps below:

### 1. Clone the Repository

*git clone https://github.com/CYatSIGN/td3-robosuite-arm-manipulation.git*

*cd td3-robosuite-arm-manipulation*

### 2. Create a Python 3.10 virtual environment

*uv venv --python 3.10 .venv*

*source .venv/bin/activate*

### 3. Install dependancies

*uv pip install -r requirements.txt*

### 4. Run the trained robot

*MUJOCO_GL=glfw python visualise_td3.py*

## Credits/Appendix
Tutorial provided by Robert Cowher
https://youtu.be/z1Lnlw2m8dg?si=5LAFK7Euc1j79fNw