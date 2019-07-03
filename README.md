# Robotic-Arm

## 1. Project Details:
The project is a part of **Udacity Deep Reinforcement Learning Nanodegree Project**. This is the 2nd Project Main Project, where we trained DQN with discrete state space and discrete action.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

<p align=center><img src="images/reacher.gif" alt="scores" width="500"/></p>

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the Environment (Udacity Guideline)

Note that your project submission need only solve one of the two versions of the environment.

#### Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

  - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
  - This yields an average score for each episode (where the average is over all 20 agents).
As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.


## 2. Getting Started:

### Prerequisite

A good understanding of artificial neural network and Q-learning will help you navigate throughout the documentation. In addition, python skillset must be at near intermediate for understanding the structure of the program. 

### Installation (Udacity Guideline)

## 3. Instructions:

## 4. Running the Tests:

## Authors:
- *Wong, John Maverick* 

See also the list of contributors who participated in this project.

## Acknowledgements:
