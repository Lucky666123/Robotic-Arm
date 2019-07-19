from unityagents import UnityEnvironment
import numpy as np
import os
import torch

from collections import deque
from agent import DDPG_Agent

from helper_fn import checkpoint
from tensorboardX import SummaryWriter

# Create the Environment
env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

def check_env(env, prnt=True):
    # get the default brain
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations

    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)

    if prnt==True:
        print('')
        print('State Dimension:\t\t', state_size)
        print('Size of each action:\t\t', action_size)
        print('Number of agents:\t\t', num_agents)
    return num_agents, state_size, action_size


class initialize:
    def __init__(self, agent, n_episodes=3000, max_t=3000, n_step=1):
        self.agent = agent
        self.n_episode = n_episodes
        self.max_tsteps = max_t

        self.n_step = n_step

        self.d_freq = 100

        self.save_at_target = False
        self.save_at_checkpoint = False


    def train(self, exp_number):
        ts = SummaryWriter()
        file_path = 'pth_checkpoint/'

        scores_window = deque(maxlen=self.d_freq)
        scores_agents = []
        moving_avgs = []

        for i_episode in range(1, self.n_episode+1):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations             # get the current state
            self.agent.reset()
            scores = np.zeros(num_agents)

            for t in range(1, self.max_tsteps+1):
                actions = self.agent.act(states)
                env_info = env.step(actions)[brain_name]     # send the action to the environment
                next_states = env_info.vector_observations   # get the next state
                rewards = env_info.rewards                   # get the reward
                dones = env_info.local_done                  # see if episode has finished

                self.agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                scores += rewards

                if np.any(dones):
                    break

            scores_window.append(np.mean(scores))
            scores_agents.append(np.mean(scores))

            ts.add_scalar("Avg.Reward", np.mean(scores), i_episode)
            #ts.add_histogram("Actr_lcl.FC1.weight.grad",     self.agent.actor_local.fc1.weight.grad,     i_episode)
            ts.add_histogram("Actr_lcl.FC1.weight",          self.agent.actor_local.fc1.weight,          i_episode)
            ts.add_histogram("Actr_lcl.FC1.bias",            self.agent.actor_local.fc1.bias,            i_episode)
            #ts.add_histogram("Actr_lcl.FC2.weight.grad",     self.agent.actor_local.fc2.weight.grad,     i_episode)
            ts.add_histogram('Actr_lcl.FC2.weight',          self.agent.actor_local.fc2.weight,          i_episode)
            ts.add_histogram('Actr_lcl.FC2.bias',            self.agent.actor_local.fc2.bias,            i_episode)
            #ts.add_histogram("Actr_lcl.FC3.weight.grad",     self.agent.actor_local.fc3.weight.grad,     i_episode)
            ts.add_histogram('Actr_lcl.FC3.weight',          self.agent.actor_local.fc3.weight,          i_episode)
            ts.add_histogram('Actr_lcl.FC3.bias',            self.agent.actor_local.fc3.bias,            i_episode)

            #ts.add_histogram("Actr_trgt.FC1.weight.grad",    self.agent.actor_target.fc1.weight.grad,    i_episode)
            ts.add_histogram("Actr_trgt.FC1.weight",         self.agent.actor_target.fc1.weight,         i_episode)
            ts.add_histogram("Actr_trgt.FC1.bias",           self.agent.actor_target.fc1.bias,           i_episode)
            #ts.add_histogram("Actr_trgt.FC2.weight.grad",    self.agent.actor_target.fc2.weight.grad,    i_episode)
            ts.add_histogram('Actr_trgt.FC2.weight',         self.agent.actor_target.fc2.weight,         i_episode)
            ts.add_histogram('Actr_trgt.FC2.bias',           self.agent.actor_target.fc2.bias,           i_episode)
            #ts.add_histogram("Actr_trgt.FC3.weight.grad",    self.agent.actor_target.fc3.weight.grad,    i_episode)
            ts.add_histogram('Actr_trgt.FC3.weight',         self.agent.actor_target.fc3.weight,         i_episode)
            ts.add_histogram('Actr_trgt.FC3.bias',           self.agent.actor_target.fc3.bias,           i_episode)

            #ts.add_histogram("Crtc_lcl.FCS1.weight.grad",    self.agent.critic_local.fcs1.weight.grad,   i_episode)
            ts.add_histogram("Crtc_lcl.FCS1.weight",         self.agent.critic_local.fcs1.weight,        i_episode)
            ts.add_histogram("Crtc_lcl.FCS1.bias",           self.agent.critic_local.fcs1.bias,          i_episode)
            #ts.add_histogram("Crtc_lcl.FC2.weight.grad",     self.agent.critic_local.fc2.weight.grad,    i_episode)
            ts.add_histogram('Crtc_lcl.FC2.weight',          self.agent.critic_local.fc2.weight,         i_episode)
            ts.add_histogram('Crtc_lcl.FC2.bias',            self.agent.critic_local.fc2.bias,           i_episode)
            #ts.add_histogram("Crtc_lcl.FC3.weight.grad",     self.agent.critic_local.fc3.weight.grad,    i_episode)
            ts.add_histogram('Crtc_lcl.FC3.weight',          self.agent.critic_local.fc3.weight,         i_episode)
            ts.add_histogram('Crtc_lcl.FC3.bias',            self.agent.critic_local.fc3.bias,           i_episode)

            #ts.add_histogram("Crtc_trgt.FCS1.weight.grad",   self.agent.critic_target.fcs1.weight.grad,  i_episode)
            ts.add_histogram("Crtc_trgt.FCS1.weight",        self.agent.critic_target.fcs1.weight,       i_episode)
            ts.add_histogram("Crtc_trgt.FCS1.bias",          self.agent.critic_target.fcs1.bias,         i_episode)
            #ts.add_histogram("Crtc_trgt.FC2.weight.grad",    self.agent.critic_target.fc2.weight.grad,   i_episode)
            ts.add_histogram('Crtc_trgt.FC2.weight',         self.agent.critic_target.fc2.weight,        i_episode)
            ts.add_histogram('Crtc_trgt.FC2.bias',           self.agent.critic_target.fc2.bias,          i_episode)
            #ts.add_histogram("Crtc_trgt.FC3.weight.grad",    self.agent.critic_target.fc3.weight.grad,   i_episode)
            ts.add_histogram('Crtc_trgt.FC3.weight',         self.agent.critic_target.fc3.weight,        i_episode)
            ts.add_histogram('Crtc_trgt.FC3.bias',           self.agent.critic_target.fc3.bias,          i_episode)

            checkpoint(i_episode, scores_window,
                       scores, file_path, exp_number,
                       display_freq=self.d_freq)

        torch.save(self.agent.actor_local.state_dict(),     file_path + 'ddpg_{}_actor_multiple_agents.pth'.format(str(exp_number)))
        torch.save(self.agent.critic_local.state_dict(),    file_path + 'ddpg_{}_critic_multiple_agents.pth'.format(str(exp_number)))
        ts.close()
        return scores_agents


if __name__ == '__main__':
    num_agents, state_size, action_size = check_env(env)
    '''agent_1 = DDPG_Agent(state_size, action_size, num_agents,
                         lr_critic=0.0004,
                         lr_actor=0.003,
                         gamma = 0.99,
                         tau=0.003,
                         update_every=1,
                         weight_decay=0)
    init_1 =  initialize(agent_1, n_episodes=10000, max_t=3000)
    scores_agent_1 = init_1.train(1)'''

    agent_2 = DDPG_Agent(state_size, action_size, num_agents,
                         lr_critic= 0.00001,
                         lr_actor=  0.0005,
                         tau=0.05,
                         update_every=1,
                         weight_decay=0)
    init_2 =  initialize(agent_2, n_episodes=10000, max_t=3000)
    scores_agent_2 = init_2.train(2)

    agent_3 = DDPG_Agent(state_size, action_size, num_agents,
                         lr_critic=0.00005,
                         lr_actor=0.0001,
                         tau=0.01,
                         update_every=10,
                         weight_decay=0)
    init_3 =  initialize(agent_3, n_episodes=10000, max_t=3000)
    scores_agent_3 = init_3.train(3)

    agent_4 = DDPG_Agent(state_size, action_size, num_agents,
                         lr_critic=0.00005,
                         lr_actor=0.0001,
                         gamma = 0.99,
                         tau=0.10,
                         update_every=20,
                         weight_decay=0)
    init_4 =  initialize(agent_4, n_episodes=10000, max_t=3000)
    scores_agent_4 = init_4.train(4)
