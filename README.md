# Robotic Arm Continuous Control

## 1. Project Details:
The project is a part of **Udacity Deep Reinforcement Learning Nanodegree Project**. This is the 2nd Project Main Project, where we trained DQN with discrete state space and discrete action.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

<p align=center><img src="images/reacher.gif" alt="scores" width="500"/></p>

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Solving the Environment 

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

  - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
  - This yields an average score for each episode (where the average is over all 20 agents).
As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.


## 2. Getting Started:

### Prerequisite

A good understanding of artificial neural network and Q-learning will help you navigate throughout the documentation. In addition, python skillset must be at near intermediate for understanding the structure of the program. 

### Installation 

Download the environment from one of the links below.  You need only select the environment that matches your operating system:
  - **Twenty (20) Agents**
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

## 3. Instructions:

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

## 4. Running the Tests:

<p align=center><img src="images/graph.png" width="750"/></p>
