import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorboardX import SummaryWriter


def checkpoint(i_episode, scores_window, scores, filepath, exp_number, display_freq=100):
    print('\rEpisode {}\t\tAvg Score: {:.2f}\t\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), np.mean(scores)), end="")
    if i_episode % display_freq == 0:
        print('\rEpisode {}\t\tAvg Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if i_episode % 2000 == 0:
            torch.save(agent.actor_local.state_dict(), filepath +'checkpoint{}_{}_actor_multiple_agents.pth'.format(i_episode,exp_number))
            torch.save(agent.critic_local.state_dict(), filepath +'checkpoint{}_{}_critic_multiple_agents.pth'.format(i_episode,exp_number))

def plot_training(scores_1, scores_2, scores_3, scores_4):
    fig = plt.figure(figsize=(16,5))
    plt.rcParams.update({'font.size': 11})

    plt.plot(np.arange(len(scores_1)), scores_1, label='Exper#1')
    plt.plot(np.arange(len(scores_2)), scores_2, label='Exper#2')
    plt.plot(np.arange(len(scores_3)), scores_3, label='Exper#3')
    plt.plot(np.arange(len(scores_4)), scores_4, label='Exper#4')

    plt.grid(which="major", alpha=0.30)
    plt.title('Deep Deterministic Policy Gradient')
    plt.ylabel('Scores or Rewards')
    plt.xlabel('# of Episode')
    plt.legend(loc=0)
    plt.show()
