# Robotic Arm Manipulation Project

Training Franka Robot to open a door in Robosuite Environment

# Project Overview
This project investigates agent training for Franka arm manipulation to open a door using MuJoCo simulation.

The objective was to train an agent using Twin Delayed Deep Deterministic Policy Gradient (TD3) - an off policy reinforced learning algorithm for continous action spaces, to open a door in simulation. 

# Main Technologies
- Python
- MuJoCo
- Robosuite
- TD3 Algorithm
- Numpy
- Torch

# Results and Discussion



https://github.com/user-attachments/assets/0870b5ad-6d4b-4624-9490-30bbcc60cd9c








# Simulation Setup
To run the trained agent model on MuJoCo to watch the Frank robot arm opening the door, follow the steps below:

*git clone https://github.com/CYatSIGN/td3-robosuite-arm-manipulation.git*

*cd td3-robosuite-arm-manipulation*

Create a Python 3.10 virtual environment
*uv venv --python 3.10 .venv*

*source .venv/bin/activate*

Install dependancies
*uv pip install -r requirements.txt*

Run the trained robot
*MUJOCO_GL=glfw python visualise_td3.py*

