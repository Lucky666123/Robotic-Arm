# Project Report

## 1. Deep Deterministic Policy Gradient

*Deep Deterministic Policy Gradient or DDPG* is some kind of approximate DQN or an extention of DQN. In DDPG, it employs 2 network model, the actor learns which is the best action, and the critic learns to evaluate the optimal action value function by using the actors best believed action. "The critic in DDPG is used to approximate the maximizer over the Q values of the next state, and not as a learned baseline."


In the light of those network, *the actor* brings the **advantage of learning in the action space**  and *the critic* **supplies the actor with the knowledge of that learning together with its performance.**

"To mitigate the challenge of unstable learning, a number of techniques are applied like Gradient Clipping, Soft Target Update through twin local / target network and Replay Buffer. The most important one is Replay Buffer where it allows the DDPG agent to learn offline by gathering experiences collected from environment agents and sampling experiences from large Replay Memory Buffer across a set of unrelated experiences. This enables a very effective and quicker training process."

Note:
 - DDPG is an off-policy algorithm.
 - DDPG can only be used for environments with continuous action spaces.
 - DDPG can be thought of as being deep Q-learning for continuous action spaces.
 - The Spinning Up implementation of DDPG does not support parallelization.

The *Actor Network:*
```
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```
The *Critic Network:*
```
self.fcs1 = nn.Linear(state_size, fcs1_units)
self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
self.fc3 = nn.Linear(fc2_units, 1)
```

***Process or Structure:*** (Structure of Actor-Critic Method Agents)
1. Input the current state into the actor and get the action to take in that state. 
2. Observe next state and reward to get your experienced tuple (s,a,r,s').
3. Then using the TD estimate, which is r plus the critic's estimate for s prime. Train the critic or the 2nd network.
4. Calculate the following: Also use the critic in the initial s.
5. Finally, we train the actor using the calculated advantage as a baseline. 

## 2. Summary of Params and Hyperparams of the Agent and of the Network
*Network Hyperparameters:*
```
tau (interpolation parameter soft update): 0.001
lr_actor (learning rate actor): 1e-3
lr_critic (learning rate critic): 1e-4
```
*Agent Parameters / Hyperparameters:*
```
Gamma or Discount Rate: 0.99
Buffer Size: 1e5
Batch Size: 100
Update Every: 1
```
*Training Parameters:*
```
Total Number of Episodes: 200
Maximum Number of Timesteps per Episodes: 2000
```

## 3. Final Results and Takeaways:

<p align=center><img src="images/graph.png" width="800"/></p>

## 4. Further Improvements / Further Works

Further improvement towards policy gradient includes:
 - D4PG
 - MADDPG
 - TRPO
 - PPO
 - ACER
 - ACTKR
 - SAC
 - SAC with Automatically Adjusted Temperature
 - TD3

## Acknowledgement
