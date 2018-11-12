import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
# %matplotlib inline

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

#define optimal state-value Function

V_opt = np.zeros((env.observation_space.n, env.action_space.n))

# print (V_opt)
# V_opt[0:13][0] = -np.arange(3, 15)[::-1]
# # print (V_opt)
# V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
# V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
# V_opt[3][0] = -13

# # plot_values(V_opt)
#
# # policy = np.random.choice(4)
#
def sarsa(env=gym.make('CliffWalking-v0'), num_episodes=1000, alpha=0.01, gamma=1.0):
    """Saras Algothrim to calculate Q-table.

    Args:
        env: Gym Cliff Walking environment.
        num_episodes: the number of episodes you want to run. Default value is 1000.
        alpha: step size parameter used to update step. Default value is 0.01.
        gamma: discount rate. Default is 1.

    Return:
        Dictionary Q-Table.
    """
    plot_every = 100
    # Q = defaultdict(lambda: np.zeros(env.nA))
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # Use scores and tmp_scores to track rewards update.
    scores = deque(maxlen=num_episodes)
    tmp_scores = deque(maxlen=plot_every)
    for i in range(1, num_episodes + 1):
        if i % plot_every == 0:
            print("\rEpisode {}/{}".format(i, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        score = 0
        done = False
        while not done: # This could be relatively slow if there are many states. Can limite time steps here.
            action = np.random.choice(4) # TODO: change policy e.g. greedy policy
            next_state, reward, done, info = env.step(action)
            score += reward
            next_action = np.random.choice(4)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            if (done):
                tmp_scores.append(score)
        if (i % plot_every == 0):
            scores.append(np.mean(tmp_scores))
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    return Q





sarsa(env)
